"""
MEMORY-EFFICIENT IMAGE-ONLY MODEL TRAINING

Strategy:
1. Train on 2048-dim image features only
2. Use LightGBM with memory-efficient settings
3. 5-fold CV for robust evaluation
4. Save predictions for ensembling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import json

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("IMAGE-ONLY MODEL TRAINING")
print("="*80)
print()

# Configuration
CONFIG = {
    'data_dir': DATA_DIR,
    'image_features_dir': OUTPUT_DIR / "image_features",
    'output_dir': OUTPUT_DIR / "image_model",
    'n_folds': 5,
    'random_state': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load CSV files
train_files = [CONFIG['data_dir'] / 'train1.csv', CONFIG['data_dir'] / 'train2.csv']
test_files = [CONFIG['data_dir'] / 'test1.csv', CONFIG['data_dir'] / 'test2.csv']

train_dfs = []
for f in train_files:
    if f.exists():
        df = pd.read_csv(f)
        train_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

test_dfs = []
for f in test_files:
    if f.exists():
        df = pd.read_csv(f)
        test_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\n✓ Total training samples: {len(train_df):,}")
print(f"✓ Total test samples: {len(test_df):,}")
print()

# ============================================================================
# STEP 2: LOAD IMAGE FEATURES
# ============================================================================
print("="*80)
print("STEP 2: LOADING IMAGE FEATURES")
print("="*80)

train_features_file = CONFIG['image_features_dir'] / 'train_image_features_final.npz'
test_features_file = CONFIG['image_features_dir'] / 'test_image_features_final.npz'

# Load training features
train_data = np.load(train_features_file)
train_features = train_data['features']
train_sample_ids = train_data['sample_ids']

print(f"✓ Training features: {train_features.shape}")

# Load test features
test_data = np.load(test_features_file)
test_features = test_data['features']
test_sample_ids = test_data['sample_ids']

print(f"✓ Test features: {test_features.shape}")
print()

# Verify alignment
assert len(train_features) == len(train_df), "Training features/data mismatch!"
assert len(test_features) == len(test_df), "Test features/data mismatch!"
print("✓ Features aligned with data")
print()

# ============================================================================
# STEP 3: PREPARE TARGET
# ============================================================================
print("="*80)
print("STEP 3: PREPARING TARGET")
print("="*80)

y_train = train_df['price'].values
y_train_log = np.log1p(y_train)

print(f"✓ Target statistics:")
print(f"   Min: ${y_train.min():.2f}")
print(f"   Median: ${np.median(y_train):.2f}")
print(f"   Mean: ${y_train.mean():.2f}")
print(f"   Max: ${y_train.max():.2f}")
print()

# ============================================================================
# STEP 4: SMAPE CALCULATION
# ============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)
    denominator = (y_true + y_pred) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# ============================================================================
# STEP 5: LIGHTGBM TRAINING
# ============================================================================
print("="*80)
print("STEP 5: TRAINING LIGHTGBM")
print("="*80)

# LightGBM parameters (memory-efficient)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': CONFIG['random_state'],
    'n_jobs': -1,
    'verbose': -1,
}

# Cross-validation
kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_state'])

oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))
fold_scores = []
models = []

print(f"✓ {CONFIG['n_folds']}-fold cross-validation")
print()

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    print(f"📊 Fold {fold_idx+1}/{CONFIG['n_folds']}")
    
    # Split data
    X_tr = train_features[train_idx]
    y_tr = y_train_log[train_idx]
    X_val = train_features[val_idx]
    y_val = y_train_log[val_idx]
    
    # Create datasets
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    # Train
    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Predict
    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    val_pred = np.maximum(val_pred, 0.01)  # Clip to minimum price
    
    # Calculate SMAPE
    y_val_original = y_train[val_idx]
    fold_smape = smape(y_val_original, val_pred)
    fold_scores.append(fold_smape)
    
    print(f"   Fold {fold_idx+1} SMAPE: {fold_smape:.3f}%")
    
    # Store OOF predictions
    oof_predictions[val_idx] = val_pred
    
    # Predict on test
    test_pred_log = model.predict(test_features)
    test_predictions += np.expm1(test_pred_log) / CONFIG['n_folds']
    
    # Save model
    models.append(model)
    
    print()

# Clip test predictions
test_predictions = np.maximum(test_predictions, 0.01)

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================
print("="*80)
print("STEP 6: EVALUATION")
print("="*80)

# Calculate overall OOF SMAPE
oof_smape = smape(y_train, oof_predictions)

print(f"📊 Cross-Validation Results:")
print(f"   OOF SMAPE: {oof_smape:.3f}%")
print()
print(f"📊 Fold Breakdown:")
for i, score in enumerate(fold_scores):
    print(f"   Fold {i+1}: {score:.3f}%")
print(f"   Mean:   {np.mean(fold_scores):.3f}%")
print(f"   Std:    {np.std(fold_scores):.3f}%")
print()

# ============================================================================
# STEP 7: SAVE PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 7: SAVING PREDICTIONS")
print("="*80)

# Save OOF predictions
oof_df = train_df[['sample_id', 'price']].copy()
oof_df['predicted_price'] = oof_predictions

oof_file = CONFIG['output_dir'] / 'oof_predictions.csv'
oof_df.to_csv(oof_file, index=False)
print(f"✓ Saved OOF predictions: {oof_file}")

# Save test predictions
submission = test_df[['sample_id']].copy()
submission['price'] = test_predictions

submission_file = CONFIG['output_dir'] / 'test_predictions.csv'
submission.to_csv(submission_file, index=False)
print(f"✓ Saved test predictions: {submission_file}")
print()

# Save predictions as numpy arrays (for ensembling)
np.savez_compressed(
    CONFIG['output_dir'] / 'predictions.npz',
    oof=oof_predictions,
    test=test_predictions,
    train_ids=train_sample_ids,
    test_ids=test_sample_ids
)
print(f"✓ Saved numpy arrays for ensembling")
print()

# ============================================================================
# STEP 8: PREDICTION STATISTICS
# ============================================================================
print("="*80)
print("STEP 8: PREDICTION STATISTICS")
print("="*80)

print(f"📊 Test Predictions:")
print(f"   Min:    ${test_predictions.min():.2f}")
print(f"   Median: ${np.median(test_predictions):.2f}")
print(f"   Mean:   ${test_predictions.mean():.2f}")
print(f"   Max:    ${test_predictions.max():.2f}")
print()

print(f"📊 Training Prices:")
print(f"   Mean:   ${y_train.mean():.2f}")
print()

print(f"📊 Distribution Comparison:")
print(f"   Test/Train mean ratio: {test_predictions.mean() / y_train.mean():.3f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ IMAGE-ONLY MODEL COMPLETE!")
print("="*80)
print()
print(f"🎯 Performance:")
print(f"   OOF SMAPE: {oof_smape:.3f}%")
print(f"   Features: {train_features.shape[1]} (ResNet50)")
print(f"   Folds: {CONFIG['n_folds']}")
print()
print(f"📁 Output files:")
print(f"   {oof_file}")
print(f"   {submission_file}")
print()
print(f"🚀 Next step: Ensemble with text model")
print(f"   python ensemble_text_image.py")
print()
