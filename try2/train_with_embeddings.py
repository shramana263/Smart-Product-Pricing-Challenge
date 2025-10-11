"""
Model Training with Text Embeddings
Train XGBoost/LightGBM/CatBoost using sentence transformer embeddings
"""

import pandas as pd
import numpy as np
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

def train_with_embeddings(embeddings_path, existing_features_path=None, 
                          use_existing_features=True, test_size=0.15, random_state=42):
    """
    Train models using text embeddings (optionally combined with existing features)
    
    Args:
        embeddings_path: Path to embeddings CSV
        existing_features_path: Path to existing features CSV (optional)
        use_existing_features: Whether to combine with existing features
        test_size: Test split ratio
        random_state: Random seed
    """
    print(f"\n{'='*70}")
    print("MODEL TRAINING WITH TEXT EMBEDDINGS")
    print(f"{'='*70}")
    
    # Load embeddings
    print(f"\nLoading embeddings: {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)
    print(f"✓ Embeddings loaded: {embeddings_df.shape}")
    
    # Optionally combine with existing features
    if use_existing_features and existing_features_path:
        print(f"\nLoading existing features: {existing_features_path}")
        existing_df = pd.read_csv(existing_features_path)
        print(f"✓ Existing features loaded: {existing_df.shape}")
        
        # Merge
        print("\nCombining embeddings with existing features...")
        data = embeddings_df.merge(existing_df, on='sample_id', how='inner')
        
        # Handle duplicate price columns
        if 'price_x' in data.columns and 'price_y' in data.columns:
            data['price'] = data['price_x']
            data = data.drop(['price_x', 'price_y'], axis=1)
        
        print(f"✓ Combined data shape: {data.shape}")
        
    else:
        data = embeddings_df.copy()
        print(f"\nUsing embeddings only: {data.shape}")
    
    # Prepare features and target
    X = data.drop(['sample_id', 'price'], axis=1)
    y = data['price'].values
    sample_ids = data['sample_id'].values
    
    print(f"\n{'='*70}")
    print("DATA SPLITS")
    print(f"{'='*70}")
    print(f"Total samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")
    print(f"Target (price) range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, sample_ids, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size/(1-test_size), random_state=random_state
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Store results
    results = {}
    
    # ===============================================
    # 1. LightGBM
    # ===============================================
    print(f"\n{'='*70}")
    print("TRAINING LIGHTGBM")
    print(f"{'='*70}")
    
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
        'random_state': random_state,
        'n_jobs': -1,
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
    
    # Predictions
    lgb_pred_val = lgb_model.predict(X_val)
    lgb_pred_test = lgb_model.predict(X_test)
    
    # Metrics
    lgb_smape_val = smape(y_val, lgb_pred_val)
    lgb_smape_test = smape(y_test, lgb_pred_test)
    lgb_mae_test = mean_absolute_error(y_test, lgb_pred_test)
    lgb_rmse_test = np.sqrt(mean_squared_error(y_test, lgb_pred_test))
    
    results['LightGBM'] = {
        'val_smape': lgb_smape_val,
        'test_smape': lgb_smape_test,
        'test_mae': lgb_mae_test,
        'test_rmse': lgb_rmse_test,
        'model': lgb_model
    }
    
    print(f"\n✓ LightGBM Results:")
    print(f"  Val SMAPE: {lgb_smape_val:.4f}%")
    print(f"  Test SMAPE: {lgb_smape_test:.4f}%")
    print(f"  Test MAE: ${lgb_mae_test:.2f}")
    print(f"  Test RMSE: ${lgb_rmse_test:.2f}")
    
    # ===============================================
    # 2. XGBoost
    # ===============================================
    print(f"\n{'='*70}")
    print("TRAINING XGBOOST")
    print(f"{'='*70}")
    
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
        'random_state': random_state,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)
    xgb_test_dm = xgb.DMatrix(X_test)
    
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=2000,
        evals=[(xgb_val, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predictions
    xgb_pred_val = xgb_model.predict(xgb_val)
    xgb_pred_test = xgb_model.predict(xgb_test_dm)
    
    # Metrics
    xgb_smape_val = smape(y_val, xgb_pred_val)
    xgb_smape_test = smape(y_test, xgb_pred_test)
    xgb_mae_test = mean_absolute_error(y_test, xgb_pred_test)
    xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
    
    results['XGBoost'] = {
        'val_smape': xgb_smape_val,
        'test_smape': xgb_smape_test,
        'test_mae': xgb_mae_test,
        'test_rmse': xgb_rmse_test,
        'model': xgb_model
    }
    
    print(f"\n✓ XGBoost Results:")
    print(f"  Val SMAPE: {xgb_smape_val:.4f}%")
    print(f"  Test SMAPE: {xgb_smape_test:.4f}%")
    print(f"  Test MAE: ${xgb_mae_test:.2f}")
    print(f"  Test RMSE: ${xgb_rmse_test:.2f}")
    
    # ===============================================
    # 3. CatBoost
    # ===============================================
    print(f"\n{'='*70}")
    print("TRAINING CATBOOST")
    print(f"{'='*70}")
    
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=random_state,
        verbose=100,
        early_stopping_rounds=50,
        task_type='CPU'
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    # Predictions
    cat_pred_val = cat_model.predict(X_val)
    cat_pred_test = cat_model.predict(X_test)
    
    # Metrics
    cat_smape_val = smape(y_val, cat_pred_val)
    cat_smape_test = smape(y_test, cat_pred_test)
    cat_mae_test = mean_absolute_error(y_test, cat_pred_test)
    cat_rmse_test = np.sqrt(mean_squared_error(y_test, cat_pred_test))
    
    results['CatBoost'] = {
        'val_smape': cat_smape_val,
        'test_smape': cat_smape_test,
        'test_mae': cat_mae_test,
        'test_rmse': cat_rmse_test,
        'model': cat_model
    }
    
    print(f"\n✓ CatBoost Results:")
    print(f"  Val SMAPE: {cat_smape_val:.4f}%")
    print(f"  Test SMAPE: {cat_smape_test:.4f}%")
    print(f"  Test MAE: ${cat_mae_test:.2f}")
    print(f"  Test RMSE: ${cat_rmse_test:.2f}")
    
    # ===============================================
    # SUMMARY
    # ===============================================
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Val SMAPE': [results[m]['val_smape'] for m in results],
        'Test SMAPE': [results[m]['test_smape'] for m in results],
        'Test MAE': [results[m]['test_mae'] for m in results],
        'Test RMSE': [results[m]['test_rmse'] for m in results]
    })
    
    summary_df = summary_df.sort_values('Test SMAPE')
    print("\n", summary_df.to_string(index=False))
    
    best_model_name = summary_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test SMAPE: {results[best_model_name]['test_smape']:.4f}%")
    
    # Save results
    summary_df.to_csv('./modeling/embedding_model_results.csv', index=False)
    print(f"\n✓ Results saved to: ./modeling/embedding_model_results.csv")
    
    return results, best_model_name, best_model, X_test, y_test, ids_test


def make_predictions(model, test_embeddings_path, model_type='lightgbm', 
                     existing_features_path=None, use_existing_features=True):
    """
    Make predictions on test set
    
    Args:
        model: Trained model
        test_embeddings_path: Path to test embeddings
        model_type: 'lightgbm', 'xgboost', or 'catboost'
        existing_features_path: Path to existing test features (optional)
        use_existing_features: Whether to use existing features
    """
    print(f"\n{'='*70}")
    print("MAKING PREDICTIONS ON TEST SET")
    print(f"{'='*70}")
    
    # Load test embeddings
    print(f"\nLoading test embeddings: {test_embeddings_path}")
    test_df = pd.read_csv(test_embeddings_path)
    print(f"✓ Test embeddings loaded: {test_df.shape}")
    
    # Optionally combine with existing features
    if use_existing_features and existing_features_path:
        print(f"\nLoading test features: {existing_features_path}")
        existing_test = pd.read_csv(existing_features_path)
        print(f"✓ Test features loaded: {existing_test.shape}")
        
        test_df = test_df.merge(existing_test, on='sample_id', how='inner')
        print(f"✓ Combined test data: {test_df.shape}")
    
    # Prepare features
    X_test = test_df.drop(['sample_id'], axis=1)
    sample_ids = test_df['sample_id'].values
    
    print(f"\nMaking predictions using {model_type}...")
    
    # Make predictions
    if model_type.lower() == 'lightgbm':
        predictions = model.predict(X_test)
    elif model_type.lower() == 'xgboost':
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
    elif model_type.lower() == 'catboost':
        predictions = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Ensure positive prices
    submission['price'] = submission['price'].clip(lower=0.01)
    
    print(f"\n✓ Predictions completed")
    print(f"  Samples: {len(submission)}")
    print(f"  Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"  Mean price: ${predictions.mean():.2f}")
    
    # Save submission
    output_path = './modeling/test_out_embeddings.csv'
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved: {output_path}")
    
    return submission


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL TRAINING WITH TEXT EMBEDDINGS - MAIN SCRIPT")
    print("="*70)
    
    # Configuration
    EMBEDDINGS_PATH = './embeddings_data/train_embeddings.csv'
    TEST_EMBEDDINGS_PATH = './embeddings_data/test_embeddings.csv'
    EXISTING_FEATURES_PATH = './preparation/features_clean.csv'
    
    # Option 1: Use embeddings only
    # USE_EXISTING_FEATURES = False
    
    # Option 2: Use embeddings + existing features (RECOMMENDED)
    USE_EXISTING_FEATURES = True
    
    print(f"\nConfiguration:")
    print(f"  Embeddings: {EMBEDDINGS_PATH}")
    print(f"  Existing features: {EXISTING_FEATURES_PATH if USE_EXISTING_FEATURES else 'Not used'}")
    
    # Train models
    results, best_model_name, best_model, X_test, y_test, ids_test = train_with_embeddings(
        embeddings_path=EMBEDDINGS_PATH,
        existing_features_path=EXISTING_FEATURES_PATH if USE_EXISTING_FEATURES else None,
        use_existing_features=USE_EXISTING_FEATURES
    )
    
    # Make predictions on test set
    if best_model_name == 'LightGBM':
        model_type = 'lightgbm'
    elif best_model_name == 'XGBoost':
        model_type = 'xgboost'
    else:
        model_type = 'catboost'
    
    submission = make_predictions(
        model=best_model,
        test_embeddings_path=TEST_EMBEDDINGS_PATH,
        model_type=model_type,
        existing_features_path=EXISTING_FEATURES_PATH if USE_EXISTING_FEATURES else None,
        use_existing_features=USE_EXISTING_FEATURES
    )
    
    print("\n" + "="*70)
    print("✓ EMBEDDING-BASED MODELING COMPLETED!")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Test SMAPE: {results[best_model_name]['test_smape']:.4f}%")
    print(f"📁 Submission: ./modeling/test_out_embeddings.csv")
