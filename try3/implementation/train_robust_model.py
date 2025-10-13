"""
ROBUST MODEL: Conservative Approach to Avoid Overfitting

Key Changes:
1. Remove suspicious high-correlation features (price_per_unit)
2. Use only stable, generalizable features
3. Simpler model (less risk of overfitting)
4. More conservative cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

# ML libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

# Try importing transformers
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

print("="*80)
print("ROBUST MODEL - CONSERVATIVE APPROACH")
print("="*80)
print()

CONFIG = {
    'advanced_features_dir': OUTPUT_DIR / "phase1_advanced_features",
    'output_dir': OUTPUT_DIR / "robust_model",
    'use_distilbert': False,  # Disable for now - too complex
    'n_folds': 10,  # More folds for better validation
    'random_seed': 42,
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced from 31
        'learning_rate': 0.01,  # Lower learning rate
        'feature_fraction': 0.7,  # More regularization
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,  # Prevent overfitting
        'max_depth': 5,  # Limit tree depth
        'verbose': -1,
        'random_state': 42,
    }
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# SMAPE METRIC
# ============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

train_df = pd.read_csv(CONFIG['advanced_features_dir'] / 'train_with_advanced_features.csv')
test_df = pd.read_csv(CONFIG['advanced_features_dir'] / 'test_with_advanced_features.csv')

print(f"✓ Loaded {len(train_df):,} training samples")
print(f"✓ Loaded {len(test_df):,} test samples")
print()

# ============================================================================
# STEP 2: SELECT ONLY SAFE FEATURES (NO LEAKAGE)
# ============================================================================
print("="*80)
print("STEP 2: SELECTING SAFE FEATURES")
print("="*80)

# EXCLUDE high-correlation features that might cause overfitting
EXCLUDED_FEATURES = [
    'price_per_unit',  # 0.925 correlation - too high!
    'price_per_char',  # 0.503 - derived from price patterns
]

# Safe numeric features (low to moderate correlation)
safe_numeric_features = [
    # Unit features (basic extraction only)
    'qty', 'total_qty', 'multiplier',
    # Premium/Budget signals
    'premium_count', 'budget_count', 'premium_material_count', 
    'budget_material_count', 'premium_signal',
    # Text complexity (stable across datasets)
    'text_char_count', 'text_word_count', 'text_avg_word_length',
    'text_unique_word_ratio', 'text_digit_ratio',
]

# Safe categorical features
safe_categorical_features = [
    'unit', 'unit_category',
    'brand_tier', 'category',
]

# Filter to existing columns
safe_numeric_features = [f for f in safe_numeric_features if f in train_df.columns]
safe_categorical_features = [f for f in safe_categorical_features if f in train_df.columns]

print(f"✓ Selected {len(safe_numeric_features)} numeric features")
print(f"✓ Selected {len(safe_categorical_features)} categorical features")
print(f"✓ Excluded {len(EXCLUDED_FEATURES)} suspicious features:")
for feat in EXCLUDED_FEATURES:
    print(f"   - {feat}")
print()

# ============================================================================
# STEP 3: HANDLE MISSING VALUES CONSERVATIVELY
# ============================================================================
print("="*80)
print("STEP 3: HANDLING MISSING VALUES")
print("="*80)

for feat in safe_numeric_features:
    train_df[feat] = train_df[feat].fillna(train_df[feat].median())
    test_df[feat] = test_df[feat].fillna(train_df[feat].median())  # Use train median!

for feat in safe_categorical_features:
    train_df[feat] = train_df[feat].fillna('unknown').astype(str)
    test_df[feat] = test_df[feat].fillna('unknown').astype(str)

print(f"✓ Filled missing values")
print()

# ============================================================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ============================================================================
print("="*80)
print("STEP 4: ENCODING CATEGORICAL FEATURES")
print("="*80)

# Use frequency encoding (more robust than label encoding)
for feat in safe_categorical_features:
    freq_map = train_df[feat].value_counts().to_dict()
    train_df[f'{feat}_freq'] = train_df[feat].map(freq_map).fillna(0)
    test_df[f'{feat}_freq'] = test_df[feat].map(freq_map).fillna(0)
    safe_numeric_features.append(f'{feat}_freq')

print(f"✓ Encoded {len(safe_categorical_features)} categorical features")
print()

# ============================================================================
# STEP 5: PREPARE FEATURES
# ============================================================================
X_train = train_df[safe_numeric_features].copy()
y_train = train_df['price'].values
X_test = test_df[safe_numeric_features].copy()

# Apply log transform to target (helps with skewed distribution)
y_train_log = np.log1p(y_train)

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Features: {len(safe_numeric_features)}")
print()

# ============================================================================
# STEP 6: TRAIN WITH MORE ROBUST CROSS-VALIDATION
# ============================================================================
print("="*80)
print("STEP 6: TRAINING LIGHTGBM (CONSERVATIVE)")
print("="*80)

# Create stratified folds
train_df['price_bin'] = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])

oof_predictions_log = np.zeros(len(train_df))
test_predictions_log = np.zeros(len(test_df))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bin']), 1):
    print(f"📊 Fold {fold}/{CONFIG['n_folds']}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(X_tr, y_tr)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        CONFIG['lgb_params'],
        train_data,
        num_boost_round=500,  # Reduced from 1000
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Predict in log space
    oof_predictions_log[val_idx] = model.predict(X_val)
    test_predictions_log += model.predict(X_test) / CONFIG['n_folds']
    
    # Transform back and calculate SMAPE
    oof_pred = np.expm1(oof_predictions_log[val_idx])
    y_val_original = y_train[val_idx]
    
    fold_smape = smape(y_val_original, oof_pred)
    fold_scores.append(fold_smape)
    print(f"   Fold {fold} SMAPE: {fold_smape:.3f}%")

print()

# ============================================================================
# STEP 7: EVALUATE
# ============================================================================
print("="*80)
print("STEP 7: FINAL EVALUATION")
print("="*80)

# Transform predictions back from log space
oof_predictions = np.expm1(oof_predictions_log)
test_predictions = np.expm1(test_predictions_log)

# Clip to reasonable range (prevent extreme values)
oof_predictions = np.clip(oof_predictions, 0.01, 1000)
test_predictions = np.clip(test_predictions, 0.01, 1000)

oof_smape = smape(y_train, oof_predictions)

print(f"📊 Cross-Validation Results:")
print(f"   Baseline SMAPE:     53.636%")
print(f"   OOF SMAPE:          {oof_smape:.3f}%")
print(f"   Improvement:        {53.636 - oof_smape:.3f} points")
print()
print(f"📊 Fold Scores:")
for i, score in enumerate(fold_scores, 1):
    print(f"   Fold {i}: {score:.3f}%")
print(f"   Mean:   {np.mean(fold_scores):.3f}%")
print(f"   Std:    {np.std(fold_scores):.3f}%")
print()

# ============================================================================
# STEP 8: SAVE PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 8: SAVING PREDICTIONS")
print("="*80)

# Save OOF
oof_df = train_df[['sample_id', 'price']].copy()
oof_df['PREDICTED_PRICE'] = oof_predictions
oof_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)

# Save submission (correct format)
submission = test_df[['sample_id']].copy()
submission['price'] = test_predictions
submission.to_csv(CONFIG['output_dir'] / 'submission.csv', index=False)

print(f"✓ Saved OOF: {CONFIG['output_dir'] / 'oof_predictions.csv'}")
print(f"✓ Saved submission: {CONFIG['output_dir'] / 'submission.csv'}")
print()

# ============================================================================
# STEP 9: PREDICTION SANITY CHECKS
# ============================================================================
print("="*80)
print("STEP 9: SANITY CHECKS")
print("="*80)

print(f"📊 Test Predictions Statistics:")
print(f"   Min:    ${test_predictions.min():.2f}")
print(f"   25%:    ${np.percentile(test_predictions, 25):.2f}")
print(f"   Median: ${np.median(test_predictions):.2f}")
print(f"   75%:    ${np.percentile(test_predictions, 75):.2f}")
print(f"   Max:    ${test_predictions.max():.2f}")
print(f"   Mean:   ${test_predictions.mean():.2f}")
print()

print(f"   All positive: {(test_predictions > 0).all()}")
print(f"   All < $1000: {(test_predictions < 1000).all()}")
print()

# Compare with training distribution
print(f"📊 Comparison with Training Data:")
print(f"   Train mean:  ${y_train.mean():.2f}")
print(f"   Test mean:   ${test_predictions.mean():.2f}")
print(f"   Ratio:       {test_predictions.mean() / y_train.mean():.2f}x")
print()

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'baseline_smape': 53.636,
    'oof_smape': float(oof_smape),
    'improvement': float(53.636 - oof_smape),
    'fold_scores': [float(s) for s in fold_scores],
    'n_folds': CONFIG['n_folds'],
    'n_features': len(safe_numeric_features),
    'excluded_features': EXCLUDED_FEATURES,
    'test_pred_mean': float(test_predictions.mean()),
    'test_pred_median': float(np.median(test_predictions)),
}

with open(CONFIG['output_dir'] / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print()
print(f"🎯 Results:")
print(f"   OOF SMAPE: {oof_smape:.3f}%")
print(f"   Features used: {len(safe_numeric_features)} (excluded {len(EXCLUDED_FEATURES)} suspicious)")
print(f"   More conservative model should generalize better")
print()
print(f"📁 Output: {CONFIG['output_dir'] / 'submission.csv'}")
print()
