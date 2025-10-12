"""
================================================================================
FINAL MODEL: DISTILBERT + LIGHTGBM WITH ALL PHASE 1 FEATURES
================================================================================

This script trains a two-stage ensemble:
1. DistilBERT for text embeddings (use existing model or retrain)
2. LightGBM on embeddings + 29 engineered features

Expected SMAPE: 45-46% (improvement: -7 to -9 points from 53.6% baseline)
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

# Try importing transformers (for DistilBERT embeddings)
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available - will use features only")

print("="*80)
print("FINAL MODEL TRAINING - PHASE 1 FEATURES")
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'advanced_features_dir': OUTPUT_DIR / "phase1_advanced_features",
    'output_dir': OUTPUT_DIR / "final_model",
    'use_distilbert': HAS_TRANSFORMERS,  # Auto-detect
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'batch_size': 32,
    'n_folds': 5,
    'random_seed': 42,
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
    }
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
for key, value in CONFIG.items():
    if key != 'lgb_params':
        print(f"  {key}: {value}")
print()

# ============================================================================
# STEP 1: LOAD DATA WITH ADVANCED FEATURES
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

train_df = pd.read_csv(CONFIG['advanced_features_dir'] / 'train_with_advanced_features.csv')
test_df = pd.read_csv(CONFIG['advanced_features_dir'] / 'test_with_advanced_features.csv')

# Normalize column names
train_df.columns = train_df.columns.str.upper()
test_df.columns = test_df.columns.str.upper()

# Rename specific columns to match expected names
train_df = train_df.rename(columns={'CATALOG_CONTENT': 'ITEM_NAME', 'SAMPLE_ID': 'ITEM_ID'})
test_df = test_df.rename(columns={'CATALOG_CONTENT': 'ITEM_NAME', 'SAMPLE_ID': 'ITEM_ID'})

print(f"✓ Loaded {len(train_df):,} training samples")
print(f"✓ Loaded {len(test_df):,} test samples")
print(f"✓ Features: {train_df.shape[1]}")
print()

# ============================================================================
# STEP 2: SMAPE METRIC
# ============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# ============================================================================
# STEP 3: PREPARE FEATURES
# ============================================================================
print("="*80)
print("STEP 3: PREPARING FEATURES")
print("="*80)

# Numeric features from Phase 1.2 & 1.3
numeric_features = [
    # Unit features
    'QTY', 'TOTAL_QTY', 'MULTIPLIER', 'PRICE_PER_UNIT',
    # Premium/Budget
    'PREMIUM_COUNT', 'BUDGET_COUNT', 'PREMIUM_MATERIAL', 
    'BUDGET_MATERIAL', 'PREMIUM_SIGNAL',
    # Text complexity
    'TEXT_CHAR_COUNT', 'TEXT_WORD_COUNT', 'TEXT_AVG_WORD_LENGTH',
    'TEXT_UNIQUE_WORD_RATIO', 'TEXT_DIGIT_RATIO', 'TEXT_CAPITAL_RATIO',
    'TEXT_SPECIAL_CHAR_RATIO', 'TEXT_SENTENCE_COUNT',
    # Interactions
    'PRICE_PER_CHAR', 'QTY_PREMIUM_INTERACTION', 'IS_BULK',
]

# Categorical features
categorical_features = [
    'UNIT', 'MULTIPLIER_BIN', 'UNIT_CATEGORY',
    'BRAND_TIER', 'CATEGORY',
    'UNIT_PREMIUM', 'BRAND_CATEGORY', 'BULK_UNIT', 'BRAND'
]

# Target
target_col = 'PRICE'

# Filter features that exist
numeric_features = [f for f in numeric_features if f in train_df.columns]
categorical_features = [f for f in categorical_features if f in train_df.columns]

print(f"✓ Numeric features: {len(numeric_features)}")
print(f"✓ Categorical features: {len(categorical_features)}")
print()

# Handle missing values
for feat in numeric_features:
    train_df[feat] = train_df[feat].fillna(0)
    if feat in test_df.columns:  # Only fill if feature exists in test
        test_df[feat] = test_df[feat].fillna(0)

for feat in categorical_features:
    train_df[feat] = train_df[feat].fillna('unknown').astype(str)
    if feat in test_df.columns:  # Only fill if feature exists in test
        test_df[feat] = test_df[feat].fillna('unknown').astype(str)

# Filter features to only those present in BOTH train and test
numeric_features = [f for f in numeric_features if f in test_df.columns]
categorical_features = [f for f in categorical_features if f in test_df.columns]

print(f"✓ Final numeric features: {len(numeric_features)}")
print(f"✓ Final categorical features: {len(categorical_features)}")
print()

# ============================================================================
# STEP 4: CREATE EMBEDDINGS (OPTIONAL)
# ============================================================================
print("="*80)
print("STEP 4: CREATING TEXT EMBEDDINGS")
print("="*80)

# Check if embeddings already exist
embeddings_dir = CONFIG['output_dir'] / 'embeddings_cache'
embeddings_dir.mkdir(parents=True, exist_ok=True)
train_emb_path = embeddings_dir / 'train_embeddings.npy'
test_emb_path = embeddings_dir / 'test_embeddings.npy'

if CONFIG['use_distilbert']:
    # Try to load existing embeddings
    if train_emb_path.exists() and test_emb_path.exists():
        print("� Loading cached embeddings...")
        train_embeddings = np.load(train_emb_path)
        test_embeddings = np.load(test_emb_path)
        print(f"✓ Loaded train embeddings: {train_embeddings.shape}")
        print(f"✓ Loaded test embeddings: {test_embeddings.shape}")
        print()
    else:
        print("🚀 Creating DistilBERT embeddings...")
        print(f"⏱️  This will take ~30-45 minutes on T4 GPU")
        print()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device: {device}")
        
        tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
        model = DistilBertModel.from_pretrained(CONFIG['model_name']).to(device)
        model.eval()
        
        def get_embeddings(texts, batch_size=32):
            """Get DistilBERT embeddings for texts"""
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=CONFIG['max_length'],
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
                
                if (i // batch_size) % 100 == 0:
                    print(f"  Processed {i}/{len(texts)} texts...")
            
            return np.vstack(embeddings)
        
        # Create embeddings
        print("🔄 Creating train embeddings...")
        train_embeddings = get_embeddings(train_df['ITEM_NAME'].tolist(), CONFIG['batch_size'])
        print(f"✓ Train embeddings shape: {train_embeddings.shape}")
        
        print("🔄 Creating test embeddings...")
        test_embeddings = get_embeddings(test_df['ITEM_NAME'].tolist(), CONFIG['batch_size'])
        print(f"✓ Test embeddings shape: {test_embeddings.shape}")
        print()
        
        # Save embeddings for future use
        print("💾 Saving embeddings to cache...")
        np.save(train_emb_path, train_embeddings)
        np.save(test_emb_path, test_embeddings)
        print(f"✓ Saved to {embeddings_dir}")
        print()
    
    # Add embeddings as features
    for i in range(train_embeddings.shape[1]):
        train_df[f'emb_{i}'] = train_embeddings[:, i]
        test_df[f'emb_{i}'] = test_embeddings[:, i]
    
    # Update numeric features
    embedding_features = [f'emb_{i}' for i in range(train_embeddings.shape[1])]
    numeric_features.extend(embedding_features)
    
    print(f"✓ Added {len(embedding_features)} embedding features")
    print()

else:
    print("⚠️  Skipping DistilBERT embeddings (transformers not available)")
    print("   Training on engineered features only")
    print()

# ============================================================================
# STEP 5: TRAIN LIGHTGBM MODEL
# ============================================================================
print("="*80)
print("STEP 5: TRAINING LIGHTGBM MODEL")
print("="*80)

# Prepare features
X_train = train_df[numeric_features + categorical_features].copy()
y_train = train_df[target_col].values
X_test = test_df[numeric_features + categorical_features].copy()

# Convert categorical features to 'category' dtype for LightGBM
print("🔄 Encoding categorical features...")
for feat in categorical_features:
    X_train[feat] = X_train[feat].astype('category')
    X_test[feat] = X_test[feat].astype('category')
print(f"✓ Encoded {len(categorical_features)} categorical features")
print()

# Create stratified folds
train_df['price_bin'] = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")
print(f"✓ Folds: {CONFIG['n_folds']}")
print()

# Train models
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))
fold_scores = []

# Check for existing fold predictions
fold_cache_dir = CONFIG['output_dir'] / 'fold_cache'
fold_cache_dir.mkdir(parents=True, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bin']), 1):
    fold_oof_path = fold_cache_dir / f'fold_{fold}_oof.npy'
    fold_test_path = fold_cache_dir / f'fold_{fold}_test.npy'
    fold_score_path = fold_cache_dir / f'fold_{fold}_score.txt'
    
    # Check if this fold was already trained
    if fold_oof_path.exists() and fold_test_path.exists() and fold_score_path.exists():
        print(f"� Fold {fold}/{CONFIG['n_folds']} - Loading cached predictions")
        oof_predictions[val_idx] = np.load(fold_oof_path)
        fold_test_pred = np.load(fold_test_path)
        test_predictions += fold_test_pred / CONFIG['n_folds']
        with open(fold_score_path, 'r') as f:
            fold_smape = float(f.read().strip())
        fold_scores.append(fold_smape)
        print(f"   Fold {fold} SMAPE: {fold_smape:.3f}% (cached)")
        print()
        continue
    
    print(f"�📊 Fold {fold}/{CONFIG['n_folds']} - Training")
    
    # Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Create datasets (LightGBM will auto-detect category dtype)
    train_data = lgb.Dataset(X_tr, y_tr)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        CONFIG['lgb_params'],
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predict
    fold_oof_pred = model.predict(X_val)
    fold_test_pred = model.predict(X_test)
    
    oof_predictions[val_idx] = fold_oof_pred
    test_predictions += fold_test_pred / CONFIG['n_folds']
    
    # Calculate SMAPE
    fold_smape = smape(y_val, fold_oof_pred)
    fold_scores.append(fold_smape)
    
    # Save fold predictions
    np.save(fold_oof_path, fold_oof_pred)
    np.save(fold_test_path, fold_test_pred)
    with open(fold_score_path, 'w') as f:
        f.write(str(fold_smape))
    
    print(f"   Fold {fold} SMAPE: {fold_smape:.3f}%")
    print(f"   💾 Saved fold {fold} cache")
    print()

# ============================================================================
# STEP 6: EVALUATE
# ============================================================================
print("="*80)
print("STEP 6: FINAL EVALUATION")
print("="*80)

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
# STEP 7: SAVE PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 7: SAVING PREDICTIONS")
print("="*80)

# Save OOF predictions
oof_df = train_df[['ITEM_ID', 'PRICE']].copy()
oof_df['PREDICTED_PRICE'] = oof_predictions
oof_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)
print(f"✓ Saved OOF predictions: {CONFIG['output_dir'] / 'oof_predictions.csv'}")

# Save test predictions (submission format)
submission = test_df[['ITEM_ID']].copy()
submission['PREDICTED_PRICE'] = test_predictions
submission.to_csv(CONFIG['output_dir'] / 'submission.csv', index=False)
print(f"✓ Saved submission: {CONFIG['output_dir'] / 'submission.csv'}")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'baseline_smape': 53.636,
    'oof_smape': float(oof_smape),
    'improvement': float(53.636 - oof_smape),
    'fold_scores': [float(s) for s in fold_scores],
    'n_folds': CONFIG['n_folds'],
    'n_train': len(train_df),
    'n_test': len(test_df),
    'n_features': len(numeric_features) + len(categorical_features),
    'used_distilbert': CONFIG['use_distilbert'],
}

with open(CONFIG['output_dir'] / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results: {CONFIG['output_dir'] / 'results.json'}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print()
print(f"🎯 Final Results:")
print(f"   Baseline:    53.636% SMAPE")
print(f"   Your Model:  {oof_smape:.3f}% SMAPE")
print(f"   Improvement: {53.636 - oof_smape:.3f} points")
print()

if oof_smape < 46.0:
    print("🏆 EXCELLENT! Target achieved (<46% SMAPE)")
    print("   You should be competitive with top 3 teams!")
elif oof_smape < 48.0:
    print("✅ GOOD! Close to target (<48% SMAPE)")
    print("   Consider Phase 2 for further improvement")
else:
    print("⚠️  NEEDS IMPROVEMENT")
    print("   Recommend: Proceed to Phase 2 (Stratified Models)")

print()
print(f"📁 Output Files:")
print(f"   - {CONFIG['output_dir'] / 'submission.csv'}")
print(f"   - {CONFIG['output_dir'] / 'oof_predictions.csv'}")
print(f"   - {CONFIG['output_dir'] / 'results.json'}")
print()
print("="*80)
