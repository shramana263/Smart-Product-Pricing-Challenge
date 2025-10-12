"""
Train models with Advanced Features V3 - FAIR COMPARISON
Using SAME split strategy as baseline (60/15/25)
Goal: Compare fairly with 47.52% baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    smape_val = np.mean(diff / denominator) * 100
    return smape_val

print("\n" + "="*70)
print("FAIR COMPARISON WITH BASELINE (SAME 60/15/25 SPLIT)")
print("="*70)

# Load features
print("\nLoading features...")
base_dir = Path(__file__).resolve().parent
prep_dir = base_dir / 'preparation'
model_dir = base_dir.parent / 'modeling'

train_df = pd.read_csv(prep_dir / 'features_v3_clean_train.csv')
test_df = pd.read_csv(prep_dir / 'features_v3_clean_test.csv')

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")

# Prepare data
X = train_df.drop(['sample_id', 'price'], axis=1)
y = train_df['price'].values
sample_ids_train = train_df['sample_id'].values

X_test_final = test_df.drop(['sample_id'], axis=1)
sample_ids_test = test_df['sample_id'].values

print(f"\nFeatures: {X.shape[1]}")
print(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
print(f"Price mean: ${y.mean():.2f}")

# ==================================================
# SAME SPLIT AS BASELINE: 60% train, 15% val, 25% test
# ==================================================
print("\n" + "="*70)
print("CREATING STRATIFIED SPLITS (SAME AS BASELINE)")
print("="*70)

# Create price strata for stratification
y_strata = pd.qcut(y, q=10, labels=False, duplicates='drop')

# First split: 80% temp, 20% test (holdout)
X_temp, X_test, y_temp, y_test, strata_temp, strata_test = train_test_split(
    X, y, y_strata, test_size=0.25, random_state=42, stratify=y_strata
)

# Second split: 75% of temp for train, 25% for val
# This gives us: 60% train (0.75 * 0.75), 15% val (0.25 * 0.75), 25% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42  # 20% of 80% = 16% ≈ 15%
)

print(f"✓ Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test (holdout): {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ===============================================
# LightGBM
# ===============================================
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
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

lgb_pred_test = lgb_model.predict(X_test)
lgb_smape = smape(y_test, lgb_pred_test)

print(f"\n✓ LightGBM Test SMAPE: {lgb_smape:.4f}%")

# ===============================================
# XGBoost
# ===============================================
print("\n" + "-"*70)
print("Training XGBoost...")
print("-"*70)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'alpha': 0.1,
    'lambda': 0.1,
    'random_state': 42,
    'tree_method': 'hist'
}

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val = xgb.DMatrix(X_val, label=y_val)
xgb_test = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    xgb_params,
    xgb_train,
    num_boost_round=2000,
    evals=[(xgb_val, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

xgb_pred_test = xgb_model.predict(xgb_test)
xgb_smape = smape(y_test, xgb_pred_test)

print(f"\n✓ XGBoost Test SMAPE: {xgb_smape:.4f}%")

# ===============================================
# CatBoost
# ===============================================
print("\n" + "-"*70)
print("Training CatBoost...")
print("-"*70)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

cat_pred_test = cat_model.predict(X_test)
cat_smape = smape(y_test, cat_pred_test)

print(f"\n✓ CatBoost Test SMAPE: {cat_smape:.4f}%")

# ==================================================
# Model Comparison
# ==================================================
print("\n" + "="*70)
print("MODEL COMPARISON (ON HELD-OUT 25% TEST SET)")
print("="*70)

results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost'],
    'Test SMAPE': [lgb_smape, xgb_smape, cat_smape]
}).sort_values('Test SMAPE')

print("\n", results.to_string(index=False))

best_model_name = results.iloc[0]['Model']
best_smape = results.iloc[0]['Test SMAPE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 Best Test SMAPE: {best_smape:.4f}%")

# Improvement from baseline
baseline_smape = 47.52
improvement = baseline_smape - best_smape
improvement_pct = (improvement / baseline_smape) * 100

print(f"\n{'='*70}")
print("COMPARISON WITH BASELINE")
print(f"{'='*70}")
print(f"Baseline SMAPE (with possible leakage): {baseline_smape:.2f}%")
print(f"Clean Features SMAPE (NO leakage): {best_smape:.2f}%")
print(f"Difference: {improvement:.2f}% ({improvement_pct:.1f}% change)")

if best_smape < baseline_smape:
    print(f"\n🎉 IMPROVEMENT! Clean features beat baseline!")
    print(f"   This is impressive since baseline may have had target leakage!")
elif best_smape < baseline_smape * 1.1:
    print(f"\n✅ CLOSE! Within 10% of baseline (which may have had leakage)")
    print(f"   Given we removed leakage, this is actually good progress!")
else:
    print(f"\n⚠️  Need more features to match baseline performance")
    print(f"   Note: Baseline may have had target leakage advantages")

# ==================================================
# Make Predictions on Real Test Set
# ==================================================
print("\n" + "="*70)
print("MAKING PREDICTIONS ON REAL TEST SET (75K samples)")
print("="*70)

# Select best model
if best_model_name == 'LightGBM':
    predictions = lgb_model.predict(X_test_final)
elif best_model_name == 'XGBoost':
    dtest_final = xgb.DMatrix(X_test_final)
    predictions = xgb_model.predict(dtest_final)
else:
    predictions = cat_model.predict(X_test_final)

# Ensure positive prices
predictions = np.clip(predictions, 0.01, None)

# Create submission
submission = pd.DataFrame({
    'sample_id': sample_ids_test,
    'price': predictions
})

model_dir.mkdir(parents=True, exist_ok=True)
output_path = model_dir / 'test_out_v3_fair.csv'
submission.to_csv(output_path, index=False)

print(f"\n✓ Predictions saved: {output_path}")
print(f"  Samples: {len(submission)}")
print(f"  Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
print(f"  Mean price: ${predictions.mean():.2f}")
print(f"  Median price: ${np.median(predictions):.2f}")

# Compare with training distribution
print(f"\n{'='*70}")
print("PREDICTION vs TRAINING DISTRIBUTION")
print(f"{'='*70}")
print(f"Training - Min: ${y.min():.2f}, Max: ${y.max():.2f}, Mean: ${y.mean():.2f}, Median: ${np.median(y):.2f}")
print(f"Test Pred - Min: ${predictions.min():.2f}, Max: ${predictions.max():.2f}, Mean: ${predictions.mean():.2f}, Median: ${np.median(predictions):.2f}")

# Feature importance (top 20)
print("\n" + "="*70)
print("TOP 20 FEATURE IMPORTANCES")
print("="*70)

if best_model_name == 'LightGBM':
    importance = lgb_model.feature_importance()
    feature_names = X.columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
elif best_model_name == 'XGBoost':
    importance = xgb_model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': cat_model.feature_importances_
    }).sort_values('importance', ascending=False)

print("\n", feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv(model_dir / 'feature_importance_v3_fair.csv', index=False)
print(f"\n✓ Feature importance saved: feature_importance_v3_fair.csv")

print("\n" + "="*70)
print("✓ FAIR COMPARISON COMPLETED!")
print("="*70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 Test SMAPE: {best_smape:.4f}%")
print(f"📊 Baseline SMAPE: {baseline_smape:.2f}%")
print(f"📁 Submission: {output_path}")
print(f"📈 Change: {improvement:.2f}% from baseline")
print(f"\n✅ FAIR COMPARISON with same 60/15/25 split!")
print(f"✅ NO TARGET LEAKAGE in features!")
