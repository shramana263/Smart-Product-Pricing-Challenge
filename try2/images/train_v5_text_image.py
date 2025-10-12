"""
Train Models with Text + Image Features (V5)
============================================
Combines V3 clean text features with extracted image features.
Uses same 60/15/25 split as baseline for fair comparison.

Expected improvement: 5-10% SMAPE reduction
Target: 55-60% SMAPE (down from 63.28%)

Author: ML Challenge Team
Date: October 12, 2025
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    smape_val = np.mean(diff / denominator) * 100
    return smape_val

print("\n" + "="*70)
print("TRAINING WITH TEXT + IMAGE FEATURES (V5)")
print("="*70)

# ============================================================================
# Load Features
# ============================================================================
print("\nLoading combined features...")

train_df = pd.read_csv('./preparation/features_v5_text_image_train.csv')
test_df = pd.read_csv('./preparation/features_v5_text_image_test.csv')

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")

# Prepare data
X = train_df.drop(['sample_id', 'price'], axis=1)
y = train_df['price'].values
sample_ids_train = train_df['sample_id'].values

X_test_final = test_df.drop(['sample_id'], axis=1)
sample_ids_test = test_df['sample_id'].values

print(f"\nFeatures: {X.shape[1]}")
print(f"  Text features: ~47")
print(f"  Image features: ~157")
print(f"\nPrice range: ${y.min():.2f} - ${y.max():.2f}")
print(f"Price mean: ${y.mean():.2f}")
print(f"Price median: ${y.median():.2f}")

# ============================================================================
# Create Stratified Splits (SAME AS BASELINE: 60/15/25)
# ============================================================================
print("\n" + "="*70)
print("CREATING STRATIFIED SPLITS")
print("="*70)

# Create price strata for stratification
y_strata = pd.qcut(y, q=10, labels=False, duplicates='drop')

# First split: 75% temp, 25% test (holdout)
X_temp, X_test, y_temp, y_test, strata_temp, strata_test = train_test_split(
    X, y, y_strata, test_size=0.25, random_state=42, stratify=y_strata
)

# Second split: 80% of temp for train, 20% for val
# This gives us: 60% train, 15% val, 25% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42
)

print(f"✓ Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test (holdout): {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# Train XGBoost (Best performer in V3)
# ============================================================================
print("\n" + "-"*70)
print("Training XGBoost...")
print("-"*70)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 7,
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
    num_boost_round=2000,
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
print(f"  Val SMAPE: {val_smape_xgb:.2f}%")
print(f"  Test SMAPE: {test_smape_xgb:.2f}%")

# ============================================================================
# Train LightGBM
# ============================================================================
print("\n" + "-"*70)
print("Training LightGBM...")
print("-"*70)

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
    num_boost_round=2000,
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
print(f"  Val SMAPE: {val_smape_lgb:.2f}%")
print(f"  Test SMAPE: {test_smape_lgb:.2f}%")

# ============================================================================
# Train CatBoost
# ============================================================================
print("\n" + "-"*70)
print("Training CatBoost...")
print("-"*70)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3,
    random_state=42,
    verbose=100,
    early_stopping_rounds=50
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Predictions
y_train_pred_cat = cat_model.predict(X_train)
y_val_pred_cat = cat_model.predict(X_val)
y_test_pred_cat = cat_model.predict(X_test)

# Metrics
train_smape_cat = smape(y_train, y_train_pred_cat)
val_smape_cat = smape(y_val, y_val_pred_cat)
test_smape_cat = smape(y_test, y_test_pred_cat)

print(f"\n✓ CatBoost Results:")
print(f"  Train SMAPE: {train_smape_cat:.2f}%")
print(f"  Val SMAPE: {val_smape_cat:.2f}%")
print(f"  Test SMAPE: {test_smape_cat:.2f}%")

# ============================================================================
# Compare Results
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'Train SMAPE': [train_smape_xgb, train_smape_lgb, train_smape_cat],
    'Val SMAPE': [val_smape_xgb, val_smape_lgb, val_smape_cat],
    'Test SMAPE': [test_smape_xgb, test_smape_lgb, test_smape_cat]
})

print("\n", results.to_string(index=False))

# Best model
best_idx = results['Test SMAPE'].argmin()
best_model_name = results.iloc[best_idx]['Model']
best_test_smape = results.iloc[best_idx]['Test SMAPE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test SMAPE: {best_test_smape:.2f}%")

# ============================================================================
# Generate Predictions for Submission
# ============================================================================
print("\n" + "-"*70)
print("Generating predictions for submission...")
print("-"*70)

# Use best model (typically XGBoost)
if best_model_name == 'XGBoost':
    best_model = xgb_model
    dtest_final = xgb.DMatrix(X_test_final)
    final_predictions = best_model.predict(dtest_final)
elif best_model_name == 'LightGBM':
    best_model = lgb_model
    final_predictions = best_model.predict(X_test_final)
else:
    best_model = cat_model
    final_predictions = best_model.predict(X_test_final)

# Create submission file
submission_df = pd.DataFrame({
    'sample_id': sample_ids_test,
    'price': final_predictions
})

# Ensure positive prices
submission_df['price'] = submission_df['price'].clip(lower=0.01)

# Save
output_dir = './modeling'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'test_out_v5_text_image.csv')
submission_df.to_csv(output_path, index=False)

print(f"✓ Predictions saved: {output_path}")
print(f"  Samples: {len(submission_df)}")
print(f"  Price range: ${submission_df['price'].min():.2f} - ${submission_df['price'].max():.2f}")

# ============================================================================
# Feature Importance Analysis
# ============================================================================
print("\n" + "-"*70)
print("Analyzing feature importance...")
print("-"*70)

if best_model_name == 'XGBoost':
    importance = xgb_model.get_score(importance_type='weight')
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    })
elif best_model_name == 'LightGBM':
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_model.feature_importance(importance_type='gain')
    })
else:
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': cat_model.feature_importances_
    })

importance_df = importance_df.sort_values('importance', ascending=False)

# Save feature importance
importance_path = os.path.join(output_dir, 'feature_importance_v5_text_image.csv')
importance_df.to_csv(importance_path, index=False)

print(f"✓ Feature importance saved: {importance_path}")

# Show top features
print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Categorize features
img_features = importance_df[importance_df['feature'].str.contains('img_|image_|mean_|std_|dominant_|sharpness|contrast|brightness|aspect')]
text_features = importance_df[~importance_df['feature'].str.contains('img_|image_|mean_|std_|dominant_|sharpness|contrast|brightness|aspect')]

print(f"\nFeature contribution:")
print(f"  Image features importance: {img_features['importance'].sum():.0f} ({img_features['importance'].sum()/importance_df['importance'].sum()*100:.1f}%)")
print(f"  Text features importance: {text_features['importance'].sum():.0f} ({text_features['importance'].sum()/importance_df['importance'].sum()*100:.1f}%)")

# ============================================================================
# Comparison with Previous Versions
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE COMPARISON WITH PREVIOUS VERSIONS")
print("="*70)

comparison = pd.DataFrame({
    'Version': [
        'V3 Clean (Text only)',
        'V5 (Text + Image)',
        'Improvement'
    ],
    'Features': [
        '47',
        f'{X.shape[1]}',
        f'+{X.shape[1] - 47}'
    ],
    'Test SMAPE': [
        '63.28%',
        f'{best_test_smape:.2f}%',
        f'{63.28 - best_test_smape:.2f}% ↓'
    ]
})

print("\n", comparison.to_string(index=False))

if best_test_smape < 63.28:
    improvement_pct = (63.28 - best_test_smape) / 63.28 * 100
    print(f"\n🎉 SUCCESS! {improvement_pct:.1f}% relative improvement!")
    print(f"   Image features helped reduce SMAPE by {63.28 - best_test_smape:.2f} percentage points")
else:
    print(f"\n⚠️  No improvement detected. Image features may need tuning.")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Test SMAPE: {best_test_smape:.2f}%")
print(f"Submission file: {output_path}")
print("\nNext steps:")
print("1. If score improved: Consider ensemble with V3 clean")
print("2. If score didn't improve: Try different image model (EfficientNet, ViT)")
print("3. Consider fine-tuning image model on product images")
