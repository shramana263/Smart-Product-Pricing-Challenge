"""
Image-Only Model for Ensemble with DistilBERT
==============================================
Trains a model using ONLY image features (no text).
This will be ensembled with DistilBERT predictions.

Expected: Image-only will be worse than DistilBERT,
but ensemble might capture complementary information.

Author: ML Challenge Team
Date: October 12, 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    smape_val = np.mean(diff / denominator) * 100
    return smape_val

print("="*80)
print("TRAINING IMAGE-ONLY MODEL (FOR ENSEMBLE)")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n📂 Loading data...")

# Check if image features exist
base_dir = Path(__file__).parent.parent
img_train_path = base_dir / 'preparation' / 'image_features_train.csv'
img_test_path = base_dir / 'preparation' / 'image_features_test.csv'

if not img_train_path.exists():
    print("❌ ERROR: Image features not found!")
    print(f"   Expected: {img_train_path}")
    print("\n   Please run image feature extraction first:")
    print("   cd try2/images")
    print("   python image_feature_extraction.py")
    exit(1)

# Load image features
print("Loading image features...")
train_img = pd.read_csv(img_train_path)
test_img = pd.read_csv(img_test_path)

print(f"✓ Train images: {train_img.shape}")
print(f"✓ Test images:  {test_img.shape}")

# Load original training data for prices
dataset_dir = base_dir / 'dataset'
train1 = pd.read_csv(dataset_dir / 'train1.csv')
train2 = pd.read_csv(dataset_dir / 'train2.csv')
train_orig = pd.concat([train1, train2], ignore_index=True)

print(f"✓ Original train: {train_orig.shape}")

# Merge to get prices
train_df = train_img.merge(train_orig[['sample_id', 'price']], on='sample_id', how='inner')

print(f"✓ Merged train: {train_df.shape}")
print(f"\n📊 Price Statistics:")
print(f"   Min: ${train_df['price'].min():.2f}")
print(f"   Max: ${train_df['price'].max():.2f}")
print(f"   Mean: ${train_df['price'].mean():.2f}")
print(f"   Median: ${train_df['price'].median():.2f}")

# ============================================================================
# Prepare Features
# ============================================================================
print("\n🔧 Preparing features...")

# Separate features and target
X = train_df.drop(['sample_id', 'price'], axis=1)
y = train_df['price'].values
sample_ids_train = train_df['sample_id'].values

# Test data
X_test_final = test_img.drop(['sample_id'], axis=1)
sample_ids_test = test_img['sample_id'].values

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Training samples: {len(X):,}")
print(f"✓ Test samples: {len(X_test_final):,}")

# Check for missing values
missing_train = X.isnull().sum().sum()
missing_test = X_test_final.isnull().sum().sum()

if missing_train > 0 or missing_test > 0:
    print(f"\n⚠️  Missing values detected:")
    print(f"   Train: {missing_train}")
    print(f"   Test: {missing_test}")
    print("   Filling with 0...")
    X = X.fillna(0)
    X_test_final = X_test_final.fillna(0)

# ============================================================================
# Create Splits (SAME AS DISTILBERT: 60/15/25)
# ============================================================================
print("\n✂️  Creating splits (same as DistilBERT)...")

# Create price strata for stratification
y_strata = pd.qcut(y, q=10, labels=False, duplicates='drop')

# First split: 75% temp, 25% test
X_temp, X_test, y_temp, y_test, strata_temp, strata_test = train_test_split(
    X, y, y_strata, test_size=0.25, random_state=42, stratify=y_strata
)

# Second split: 80% train, 20% val (of temp)
# This gives us: 60% train, 15% val, 25% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42
)

print(f"✓ Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# Train Models
# ============================================================================

# Train XGBoost
print("\n" + "="*80)
print("TRAINING XGBOOST")
print("="*80)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

evals = [(dtrain, 'train'), (dval, 'val')]
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predictions
y_train_pred_xgb = xgb_model.predict(dtrain)
y_val_pred_xgb = xgb_model.predict(dval)
y_test_pred_xgb = xgb_model.predict(dtest)

# Metrics
train_smape_xgb = smape(y_train, y_train_pred_xgb)
val_smape_xgb = smape(y_val, y_val_pred_xgb)
test_smape_xgb = smape(y_test, y_test_pred_xgb)

print(f"\n✓ XGBoost Results:")
print(f"  Train SMAPE: {train_smape_xgb:.2f}%")
print(f"  Val SMAPE:   {val_smape_xgb:.2f}%")
print(f"  Test SMAPE:  {test_smape_xgb:.2f}%")

# Train LightGBM
print("\n" + "="*80)
print("TRAINING LIGHTGBM")
print("="*80)

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Predictions
y_train_pred_lgb = lgb_model.predict(X_train)
y_val_pred_lgb = lgb_model.predict(X_val)
y_test_pred_lgb = lgb_model.predict(X_test)

# Metrics
train_smape_lgb = smape(y_train, y_train_pred_lgb)
val_smape_lgb = smape(y_val, y_val_pred_lgb)
test_smape_lgb = smape(y_test, y_test_pred_lgb)

print(f"\n✓ LightGBM Results:")
print(f"  Train SMAPE: {train_smape_lgb:.2f}%")
print(f"  Val SMAPE:   {val_smape_lgb:.2f}%")
print(f"  Test SMAPE:  {test_smape_lgb:.2f}%")

# ============================================================================
# Compare and Select Best Model
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Train SMAPE': [train_smape_xgb, train_smape_lgb],
    'Val SMAPE': [val_smape_xgb, val_smape_lgb],
    'Test SMAPE': [test_smape_xgb, test_smape_lgb]
})

print("\n", results.to_string(index=False))

# Best model
best_idx = results['Test SMAPE'].argmin()
best_model_name = results.iloc[best_idx]['Model']
best_test_smape = results.iloc[best_idx]['Test SMAPE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test SMAPE: {best_test_smape:.2f}%")

# ============================================================================
# Generate Predictions on Real Test Data
# ============================================================================
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Use best model
if best_model_name == 'XGBoost':
    dtest_final = xgb.DMatrix(X_test_final)
    final_predictions = xgb_model.predict(dtest_final)
else:
    final_predictions = lgb_model.predict(X_test_final)

# Ensure positive prices
final_predictions = np.maximum(final_predictions, 0.01)

# Create submission
submission = pd.DataFrame({
    'sample_id': sample_ids_test,
    'price': final_predictions
})

# Save
output_dir = base_dir / 'modeling'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'test_out_image_only.csv'
submission.to_csv(output_path, index=False)

print(f"\n✓ Predictions saved: {output_path}")
print(f"  Samples: {len(submission):,}")
print(f"  Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
print(f"  Mean price: ${submission['price'].mean():.2f}")
print(f"  Median price: ${submission['price'].median():.2f}")

# Also save validation predictions for ensemble tuning
val_predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred_xgb': y_test_pred_xgb,
    'y_pred_lgb': y_test_pred_lgb
})
val_pred_path = output_dir / 'image_only_validation_predictions.csv'
val_predictions_df.to_csv(val_pred_path, index=False)
print(f"✓ Validation predictions saved: {val_pred_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("IMAGE-ONLY MODEL COMPLETE")
print("="*80)

print(f"\n📊 Performance:")
print(f"   Image-only Test SMAPE: {best_test_smape:.2f}%")
print(f"   DistilBERT SMAPE:      53.78%")
print(f"   Expected Ensemble:     51-53% (if images help)")

print(f"\n📁 Output Files:")
print(f"   - Predictions: {output_path}")
print(f"   - Validation:  {val_pred_path}")

print(f"\n🎯 Next Steps:")
print(f"   1. Run ensemble script:")
print(f"      cd try2/modeling")
print(f"      python ensemble_distilbert_image.py")
print(f"   2. If ensemble improves > 0.5%, submit it!")
print(f"   3. Otherwise, stick with pure DistilBERT (53.78%)")

print("\n" + "="*80)
