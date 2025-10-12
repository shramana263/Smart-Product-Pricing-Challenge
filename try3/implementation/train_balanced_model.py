"""
BALANCED MODEL: Best of Both Worlds

Strategy:
1. Use DistilBERT embeddings (powerful but not overfitting)
2. Add SOME engineered features (but not price_per_unit)
3. Moderate regularization
4. Target scaling to match training distribution
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
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Try importing transformers
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    HAS_TRANSFORMERS = True
    print("✅ DistilBERT available")
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  DistilBERT not available")

print("="*80)
print("BALANCED MODEL - DISTILBERT + SAFE FEATURES")
print("="*80)
print()

CONFIG = {
    'data_dir': DATA_DIR,
    'advanced_features_dir': OUTPUT_DIR / "phase1_advanced_features",
    'output_dir': OUTPUT_DIR / "balanced_model",
    'embeddings_cache': OUTPUT_DIR / "final_model" / "embeddings_cache",
    'use_distilbert': HAS_TRANSFORMERS,
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'batch_size': 32,
    'n_folds': 5,
    'random_seed': 42,
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 20,  # Moderate
        'learning_rate': 0.03,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'max_depth': 6,
        'verbose': -1,
        'random_state': 42,
    }
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

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
# STEP 2: LOAD OR CREATE EMBEDDINGS
# ============================================================================
print("="*80)
print("STEP 2: LOADING EMBEDDINGS")
print("="*80)

if CONFIG['embeddings_cache'].exists():
    print("✅ Loading cached embeddings...")
    train_embeddings = np.load(CONFIG['embeddings_cache'] / 'train_embeddings.npy')
    test_embeddings = np.load(CONFIG['embeddings_cache'] / 'test_embeddings.npy')
    print(f"✓ Train embeddings: {train_embeddings.shape}")
    print(f"✓ Test embeddings: {test_embeddings.shape}")
    
    # Add to dataframes
    for i in range(train_embeddings.shape[1]):
        train_df[f'emb_{i}'] = train_embeddings[:, i]
        test_df[f'emb_{i}'] = test_embeddings[:, i]
    
    HAS_EMBEDDINGS = True
else:
    print("⚠️  No cached embeddings found")
    HAS_EMBEDDINGS = False

print()

# ============================================================================
# STEP 3: SELECT SAFE FEATURES (NO LEAKAGE)
# ============================================================================
print("="*80)
print("STEP 3: SELECTING FEATURES")
print("="*80)

# EXCLUDE features that caused overfitting
EXCLUDED_FEATURES = [
    'price_per_unit',  # 0.925 correlation - DEFINITE LEAKAGE
    'price_per_char',  # 0.503 - Derived from price patterns
]

# Safe features with moderate correlation
safe_features = [
    # Unit basics (no derived prices)
    'qty', 'total_qty', 'multiplier',
    # Signals
    'premium_count', 'budget_count', 
    'premium_signal',
    # Text features
    'text_char_count', 'text_word_count',
    'text_unique_word_ratio',
]

# Categorical
categorical_features = [
    'unit_category', 'brand_tier', 'category',
]

# Filter existing
safe_features = [f for f in safe_features if f in train_df.columns]
categorical_features = [f for f in categorical_features if f in train_df.columns]

# Add embeddings if available
if HAS_EMBEDDINGS:
    embedding_features = [f'emb_{i}' for i in range(768)]
    safe_features.extend(embedding_features)
    print(f"✓ Using {len(embedding_features)} DistilBERT embeddings")

print(f"✓ Selected {len(safe_features)} numeric features")
print(f"✓ Selected {len(categorical_features)} categorical features")
print(f"✓ Excluded {len(EXCLUDED_FEATURES)} high-risk features")
print()

# ============================================================================
# STEP 4: PREPARE FEATURES
# ============================================================================
print("="*80)
print("STEP 4: PREPARING FEATURES")
print("="*80)

# Fill missing
for feat in safe_features:
    if feat in train_df.columns:
        train_df[feat] = train_df[feat].fillna(0)
        test_df[feat] = test_df[feat].fillna(0)

for feat in categorical_features:
    train_df[feat] = train_df[feat].fillna('unknown').astype(str)
    test_df[feat] = test_df[feat].fillna('unknown').astype(str)

# Encode categoricals with frequency
for feat in categorical_features:
    freq_map = train_df[feat].value_counts().to_dict()
    train_df[f'{feat}_freq'] = train_df[feat].map(freq_map).fillna(0)
    test_df[f'{feat}_freq'] = test_df[feat].map(freq_map).fillna(0)
    safe_features.append(f'{feat}_freq')

X_train = train_df[safe_features].copy()
y_train = train_df['PRICE'].values
X_test = test_df[safe_features].copy()

# Log transform (helps with skewed distribution)
y_train_log = np.log1p(y_train)

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Total features: {len(safe_features)}")
print()

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
print("="*80)
print("STEP 5: TRAINING LIGHTGBM")
print("="*80)

train_df['price_bin'] = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])

oof_predictions_log = np.zeros(len(train_df))
test_predictions_log = np.zeros(len(test_df))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bin']), 1):
    print(f"📊 Fold {fold}/{CONFIG['n_folds']}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    
    train_data = lgb.Dataset(X_tr, y_tr)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    model = lgb.train(
        CONFIG['lgb_params'],
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    oof_predictions_log[val_idx] = model.predict(X_val)
    test_predictions_log += model.predict(X_test) / CONFIG['n_folds']
    
    # Calculate SMAPE on original scale
    oof_pred = np.expm1(oof_predictions_log[val_idx])
    y_val_original = y_train[val_idx]
    
    fold_smape = smape(y_val_original, oof_pred)
    fold_scores.append(fold_smape)
    print(f"   Fold {fold} SMAPE: {fold_smape:.3f}%")

print()

# ============================================================================
# STEP 6: POST-PROCESS PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 6: POST-PROCESSING")
print("="*80)

# Transform back from log
oof_predictions = np.expm1(oof_predictions_log)
test_predictions = np.expm1(test_predictions_log)

# Clip to reasonable range
test_predictions = np.clip(test_predictions, 0.01, 500)

# Scale predictions to match training mean (conservative adjustment)
train_mean = y_train.mean()
test_mean = test_predictions.mean()
scale_factor = train_mean / test_mean

print(f"📊 Distribution Alignment:")
print(f"   Train mean: ${train_mean:.2f}")
print(f"   Test mean (raw): ${test_mean:.2f}")
print(f"   Scale factor: {scale_factor:.3f}")

# Apply mild scaling (50% of the gap)
adjusted_scale = 1.0 + (scale_factor - 1.0) * 0.5
test_predictions_scaled = test_predictions * adjusted_scale
test_predictions_scaled = np.clip(test_predictions_scaled, 0.01, 500)

print(f"   Test mean (adjusted): ${test_predictions_scaled.mean():.2f}")
print()

# ============================================================================
# STEP 7: EVALUATE
# ============================================================================
print("="*80)
print("STEP 7: EVALUATION")
print("="*80)

oof_smape = smape(y_train, oof_predictions)

print(f"📊 Cross-Validation Results:")
print(f"   Baseline SMAPE:     53.636%")
print(f"   OOF SMAPE:          {oof_smape:.3f}%")
if oof_smape < 53.636:
    print(f"   Improvement:        {53.636 - oof_smape:.3f} points ✅")
else:
    print(f"   Difference:         {oof_smape - 53.636:.3f} points (higher)")
print()

print(f"📊 Fold Breakdown:")
for i, score in enumerate(fold_scores, 1):
    print(f"   Fold {i}: {score:.3f}%")
print(f"   Mean:   {np.mean(fold_scores):.3f}%")
print(f"   Std:    {np.std(fold_scores):.3f}%")
print()

# ============================================================================
# STEP 8: SAVE BOTH VERSIONS
# ============================================================================
print("="*80)
print("STEP 8: SAVING PREDICTIONS")
print("="*80)

# Version 1: Raw predictions
submission_raw = test_df[['ITEM_ID']].copy()
submission_raw.columns = ['sample_id']
submission_raw['price'] = test_predictions
submission_raw.to_csv(CONFIG['output_dir'] / 'submission_raw.csv', index=False)
print(f"✓ Saved raw: {CONFIG['output_dir'] / 'submission_raw.csv'}")

# Version 2: Scaled predictions (RECOMMENDED)
submission_scaled = test_df[['ITEM_ID']].copy()
submission_scaled.columns = ['sample_id']
submission_scaled['price'] = test_predictions_scaled
submission_scaled.to_csv(CONFIG['output_dir'] / 'submission.csv', index=False)
print(f"✓ Saved scaled: {CONFIG['output_dir'] / 'submission.csv'} ⭐")

print()
print(f"📊 Prediction Statistics (Scaled):")
print(f"   Min:    ${test_predictions_scaled.min():.2f}")
print(f"   Median: ${np.median(test_predictions_scaled):.2f}")
print(f"   Mean:   ${test_predictions_scaled.mean():.2f}")
print(f"   Max:    ${test_predictions_scaled.max():.2f}")
print()

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'oof_smape': float(oof_smape),
    'fold_scores': [float(s) for s in fold_scores],
    'n_features': len(safe_features),
    'excluded_features': EXCLUDED_FEATURES,
    'has_embeddings': HAS_EMBEDDINGS,
    'test_mean_raw': float(test_predictions.mean()),
    'test_mean_scaled': float(test_predictions_scaled.mean()),
    'scale_factor_applied': float(adjusted_scale),
}

with open(CONFIG['output_dir'] / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print()
print(f"🎯 Performance:")
print(f"   OOF SMAPE: {oof_smape:.3f}%")
print(f"   Features: {len(safe_features)} (safe features only)")
print(f"   Embeddings: {'✅ Yes (768-dim)' if HAS_EMBEDDINGS else '❌ No'}")
print()
print(f"📁 Recommended submission:")
print(f"   {CONFIG['output_dir'] / 'submission.csv'}")
print()
print(f"💡 This model should generalize better:")
print(f"   - Removed high-correlation features")
print(f"   - Used DistilBERT for text understanding")
print(f"   - Applied mild distribution alignment")
print(f"   - More realistic CV score ({oof_smape:.1f}%)")
print()
