"""
Multi-Modal Fusion Model with LightGBM

Combines DeBERTa embeddings (1024), CLIP embeddings (768),
and engineered features (~40) into a single powerful predictor.

Uses LightGBM for robust, efficient training with excellent performance
on mixed feature types and outlier handling.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, EMBEDDINGS_DIR, FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR,
    LIGHTGBM_CONFIG, CV_CONFIG, RANDOM_SEED
)

print("="*80)
print("🚀 MULTI-MODAL FUSION MODEL - LightGBM")
print("="*80)
print("Combining: DeBERTa (1024) + CLIP (768) + Tabular (~40)")
print("="*80)

# ============================================================================
# SMAPE Metric
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (0-200%)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(numerator / denominator) * 100

def smape_objective(y_true, y_pred):
    """SMAPE as LightGBM objective (approximation)"""
    # Use Huber loss as proxy (robust to outliers)
    residual = y_pred - y_true
    grad = np.where(np.abs(residual) < 1, residual, np.sign(residual))
    hess = np.where(np.abs(residual) < 1, 1.0, 0.1)
    return grad, hess

# ============================================================================
# Data Loading
# ============================================================================

def load_and_merge_features():
    """Load and merge all feature sources"""
    
    print("\n📂 Loading features...")
    print("-"*80)
    
    # Load DeBERTa embeddings
    print("Loading DeBERTa embeddings...")
    deberta_train = pd.read_csv(EMBEDDINGS_DIR / 'deberta_train_embeddings.csv')
    deberta_test = pd.read_csv(EMBEDDINGS_DIR / 'deberta_test_embeddings.csv')
    print(f"  ✓ Train: {deberta_train.shape}")
    print(f"  ✓ Test:  {deberta_test.shape}")
    
    # Load CLIP embeddings
    print("\nLoading CLIP embeddings...")
    clip_train = pd.read_csv(EMBEDDINGS_DIR / 'clip_train_embeddings.csv')
    clip_test = pd.read_csv(EMBEDDINGS_DIR / 'clip_test_embeddings.csv')
    print(f"  ✓ Train: {clip_train.shape}")
    print(f"  ✓ Test:  {clip_test.shape}")
    
    # Load tabular features
    print("\nLoading tabular features...")
    tabular_train = pd.read_csv(FEATURES_DIR / 'tabular_train_features.csv')
    tabular_test = pd.read_csv(FEATURES_DIR / 'tabular_test_features.csv')
    print(f"  ✓ Train: {tabular_train.shape}")
    print(f"  ✓ Test:  {tabular_test.shape}")
    
    # Merge all features
    print("\nMerging features...")
    train_df = deberta_train.merge(clip_train, on='sample_id')
    train_df = train_df.merge(tabular_train, on='sample_id')
    
    test_df = deberta_test.merge(clip_test, on='sample_id')
    test_df = test_df.merge(tabular_test, on='sample_id')
    
    print(f"\n✓ Final train shape: {train_df.shape}")
    print(f"✓ Final test shape:  {test_df.shape}")
    
    return train_df, test_df

# ============================================================================
# Outlier-Aware Training
# ============================================================================

def create_sample_weights(df, outlier_results=None):
    """Create sample weights based on outlier confidence"""
    
    if outlier_results is None:
        return np.ones(len(df))
    
    # Merge outlier confidence scores
    df_merged = df.merge(
        outlier_results[['sample_id', 'confidence_score']],
        on='sample_id',
        how='left'
    )
    
    # Use confidence as weight (higher confidence = higher weight)
    weights = df_merged['confidence_score'].fillna(1.0).values
    
    return weights

# ============================================================================
# Training
# ============================================================================

def train_lightgbm_with_cv(train_df, feature_cols, target_col='price'):
    """Train LightGBM with cross-validation"""
    
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM WITH CROSS-VALIDATION")
    print("="*80)
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    # Log transform target
    y_log = np.log1p(y)
    
    # Create price bins for stratification
    price_bins = pd.qcut(y, q=CV_CONFIG['n_price_bins'], labels=False, duplicates='drop')
    
    # Load outlier results if available
    try:
        from pathlib import Path
        analysis_dir = Path(__file__).parent.parent / 'outputs' / 'analysis'
        outlier_results = pd.read_csv(analysis_dir / 'outlier_detection_results.csv')
        weights = create_sample_weights(train_df, outlier_results)
        print("\n✓ Using outlier-aware sample weights")
    except:
        weights = np.ones(len(X))
        print("\n⚠️ Outlier results not found, using uniform weights")
    
    # Cross-validation
    skf = StratifiedKFold(
        n_splits=CV_CONFIG['n_folds'],
        shuffle=CV_CONFIG['shuffle'],
        random_state=RANDOM_SEED
    )
    
    fold_scores = []
    fold_models = []
    oof_predictions = np.zeros(len(X))
    
    # Check for existing trained folds
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing_folds = []
    for fold_num in range(1, CV_CONFIG['n_folds'] + 1):
        model_path = MODELS_DIR / f'lightgbm_fold{fold_num}.txt'
        if model_path.exists():
            existing_folds.append(fold_num)
    
    if existing_folds:
        print(f"\n💾 Found existing trained folds: {existing_folds}")
        print(f"✅ Will resume from Fold {max(existing_folds) + 1}")
        print("="*80)
    
    print(f"\n🚀 Starting {CV_CONFIG['n_folds']}-fold cross-validation...")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_bins), 1):
        model_path = MODELS_DIR / f'lightgbm_fold{fold}.txt'
        
        # Check if fold already trained
        if fold in existing_folds:
            print(f"\n✅ Fold {fold}/{CV_CONFIG['n_folds']} - Already trained! Loading...")
            print("-"*80)
            
            # Load existing model
            model = lgb.Booster(model_file=str(model_path))
            fold_models.append(model)
            
            # Evaluate
            X_val = X[val_idx]
            y_val_log = y_log[val_idx]
            val_pred_log = model.predict(X_val)
            val_pred = np.expm1(val_pred_log)
            val_true = np.expm1(y_log[val_idx])
            fold_smape = calculate_smape(val_true, val_pred)
            
            print(f"✓ Loaded Fold {fold} SMAPE: {fold_smape:.3f}%")
            
            fold_scores.append(fold_smape)
            oof_predictions[val_idx] = val_pred
            continue
        
        print(f"\n📊 Fold {fold}/{CV_CONFIG['n_folds']}")
        print("-"*80)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        w_train = weights[train_idx]
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, y_train, weight=w_train,
            feature_name=feature_cols
        )
        val_data = lgb.Dataset(
            X_val, y_val,
            feature_name=feature_cols,
            reference=train_data
        )
        
        # Train
        model = lgb.train(
            LIGHTGBM_CONFIG,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(LIGHTGBM_CONFIG['early_stopping_rounds']),
                lgb.log_evaluation(100)
            ]
        )
        
        # Predict
        val_pred_log = model.predict(X_val)
        val_pred = np.expm1(val_pred_log)
        val_true = np.expm1(y_log[val_idx])
        
        # Calculate SMAPE
        fold_smape = calculate_smape(val_true, val_pred)
        print(f"\n✓ Fold {fold} SMAPE: {fold_smape:.3f}%")
        
        # Save model
        model.save_model(str(model_path))
        print(f"✓ Model saved: {model_path.name}")
        
        fold_scores.append(fold_smape)
        fold_models.append(model)
        oof_predictions[val_idx] = val_pred
    
    # Overall OOF SMAPE
    oof_smape = calculate_smape(y, oof_predictions)
    
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*80)
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i}: {score:>7.3f}%")
    print("-"*40)
    print(f"Mean:   {np.mean(fold_scores):>7.3f}%")
    print(f"Std:    {np.std(fold_scores):>7.3f}%")
    print(f"OOF:    {oof_smape:>7.3f}%")
    print("="*80)
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price_true': y,
        'price_pred': oof_predictions
    })
    oof_df.to_csv(PREDICTIONS_DIR / 'oof_predictions.csv', index=False)
    print(f"\n✓ OOF predictions saved to: {PREDICTIONS_DIR / 'oof_predictions.csv'}")
    
    return fold_models, fold_scores, oof_smape

# ============================================================================
# Prediction
# ============================================================================

def predict_test(test_df, feature_cols, models):
    """Generate test predictions using ensemble of models"""
    
    print("\n" + "="*80)
    print("🔮 GENERATING TEST PREDICTIONS")
    print("="*80)
    
    X_test = test_df[feature_cols].values
    
    # Average predictions from all folds
    test_preds_log = np.zeros(len(X_test))
    
    for i, model in enumerate(models, 1):
        fold_pred_log = model.predict(X_test)
        test_preds_log += fold_pred_log / len(models)
        print(f"✓ Fold {i} predictions generated")
    
    # Inverse transform
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(test_preds, 0.01)  # Ensure positive
    
    # Save predictions
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    submission.to_csv(PREDICTIONS_DIR / 'test_predictions.csv', index=False)
    
    print(f"\n✓ Test predictions saved to: {PREDICTIONS_DIR / 'test_predictions.csv'}")
    print(f"\nPrediction statistics:")
    print(f"  Count:  {len(test_preds):,}")
    print(f"  Min:    ${test_preds.min():.2f}")
    print(f"  Mean:   ${test_preds.mean():.2f}")
    print(f"  Median: ${np.median(test_preds):.2f}")
    print(f"  Max:    ${test_preds.max():.2f}")
    
    return test_preds

# ============================================================================
# Feature Importance
# ============================================================================

def analyze_feature_importance(models, feature_cols):
    """Analyze and save feature importance"""
    
    print("\n" + "="*80)
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Average importance across folds
    importance_dict = {feat: 0 for feat in feature_cols}
    
    for model in models:
        for feat, imp in zip(feature_cols, model.feature_importance(importance_type='gain')):
            importance_dict[feat] += imp / len(models)
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    # Save
    importance_df.to_csv(PREDICTIONS_DIR / 'feature_importance.csv', index=False)
    
    # Print top 20
    print("\nTop 20 Most Important Features:")
    print("-"*80)
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:>10.2f}")
    
    # Feature type analysis
    print("\n\nFeature Type Analysis:")
    print("-"*80)
    
    deberta_imp = importance_df[importance_df['feature'].str.startswith('deberta_')]['importance'].sum()
    clip_imp = importance_df[importance_df['feature'].str.startswith('clip_')]['importance'].sum()
    tabular_imp = importance_df[~(importance_df['feature'].str.startswith('deberta_') | 
                                   importance_df['feature'].str.startswith('clip_'))]['importance'].sum()
    
    total_imp = deberta_imp + clip_imp + tabular_imp
    
    print(f"  DeBERTa features:  {100*deberta_imp/total_imp:>6.2f}%")
    print(f"  CLIP features:     {100*clip_imp/total_imp:>6.2f}%")
    print(f"  Tabular features:  {100*tabular_imp/total_imp:>6.2f}%")
    
    return importance_df

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main fusion training pipeline"""
    
    # Set seed
    np.random.seed(RANDOM_SEED)
    
    # Load data
    train_df, test_df = load_and_merge_features()
    
    # Feature columns (all except sample_id and price)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['sample_id', 'price']]
    
    print(f"\n✓ Total features: {len(feature_cols)}")
    print(f"  DeBERTa:  {sum(1 for c in feature_cols if c.startswith('deberta_'))}")
    print(f"  CLIP:     {sum(1 for c in feature_cols if c.startswith('clip_'))}")
    print(f"  Tabular:  {len(feature_cols) - sum(1 for c in feature_cols if c.startswith(('deberta_', 'clip_')))}")
    
    # Train
    models, fold_scores, oof_smape = train_lightgbm_with_cv(
        train_df, feature_cols, target_col='price'
    )
    
    # Save models
    print("\n💾 Saving models...")
    for i, model in enumerate(models, 1):
        model.save_model(str(MODELS_DIR / f'lightgbm_fold{i}.txt'))
    print(f"✓ Saved {len(models)} models to: {MODELS_DIR}")
    
    # Predict test
    test_preds = predict_test(test_df, feature_cols, models)
    
    # Feature importance
    importance_df = analyze_feature_importance(models, feature_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🎯 Final OOF SMAPE: {oof_smape:.3f}%")
    print(f"\n📊 Cross-validation scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score:.3f}%")
    print(f"\n  Mean: {np.mean(fold_scores):.3f}% ± {np.std(fold_scores):.3f}%")
    
    print(f"\n📁 Outputs saved to:")
    print(f"  Models:              {MODELS_DIR}")
    print(f"  Predictions:         {PREDICTIONS_DIR}")
    print(f"  Feature importance:  {PREDICTIONS_DIR / 'feature_importance.csv'}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
