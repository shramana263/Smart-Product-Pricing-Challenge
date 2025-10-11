"""
Optimized Training with Text Embeddings + Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    smape_val = np.mean(diff / denominator) * 100
    return smape_val

print("\n" + "="*70)
print("OPTIMIZED TRAINING WITH TEXT EMBEDDINGS")
print("="*70)

# Load embeddings
print("\nLoading training embeddings...")
train_df = pd.read_csv('./embeddings_data/train_embeddings.csv')
test_df = pd.read_csv('./embeddings_data/test_embeddings.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Prepare data
X = train_df.drop(['sample_id', 'price'], axis=1)
y = np.log1p(train_df['price'].values)  # Log transform target
sample_ids_train = train_df['sample_id'].values

X_test_final = test_df.drop(['sample_id'], axis=1)
sample_ids_test = test_df['sample_id'].values

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
print(f"Target (log price) range: {y.min():.2f} - {y.max():.2f}")

# ==================================================
# Optuna optimization for LightGBM
# ==================================================
print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION (Optuna)")
print("="*70)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    preds = model.predict(X_val)
    score = smape(np.expm1(y_val), np.expm1(preds))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✓ Best SMAPE: {study.best_value:.4f}%")
print(f"✓ Best params: {study.best_params}")

# ==================================================
# Train final model with best params
# ==================================================
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
})

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# Evaluate
val_preds = final_model.predict(X_val)
val_smape = smape(np.expm1(y_val), np.expm1(val_preds))

print(f"\n✓ Final Validation SMAPE: {val_smape:.4f}%")

# ==================================================
# Make predictions
# ==================================================
print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

test_preds = final_model.predict(X_test_final)
test_preds = np.expm1(test_preds)  # Reverse log transform
test_preds = np.clip(test_preds, 0.01, None)  # Ensure positive

# Create submission
submission = pd.DataFrame({
    'sample_id': sample_ids_test,
    'price': test_preds
})

output_path = './modeling/test_out_embeddings_optimized.csv'
submission.to_csv(output_path, index=False)

print(f"\n✓ Predictions saved: {output_path}")
print(f"  Samples: {len(submission)}")
print(f"  Price range: ${test_preds.min():.2f} - ${test_preds.max():.2f}")
print(f"  Mean price: ${test_preds.mean():.2f}")

print("\n" + "="*70)
print("✓ OPTIMIZATION COMPLETED!")
print("="*70)
print(f"\n🏆 Best Validation SMAPE: {val_smape:.4f}%")
print(f"📁 Submission: {output_path}")
