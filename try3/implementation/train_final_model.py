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
    'qty', 'total_qty', 'multiplier', 'price_per_unit',
    # Premium/Budget
    'premium_count', 'budget_count', 'premium_material_count', 
    'budget_material_count', 'premium_signal',
    # Text complexity
    'text_char_count', 'text_word_count', 'text_avg_word_length',
    'text_unique_word_ratio', 'text_digit_ratio', 'text_upper_ratio',
    'text_special_char_ratio', 'text_bullet_count',
    # Interactions
    'price_per_char', 'qty_premium_interaction',
]

# Categorical features
categorical_features = [
    'unit', 'multiplier_bin', 'unit_category',
    'brand_tier', 'category',
    'unit_premium', 'brand_category', 'bulk_unit'
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
    test_df[feat] = test_df[feat].fillna(0)

for feat in categorical_features:
    train_df[feat] = train_df[feat].fillna('unknown').astype(str)
    test_df[feat] = test_df[feat].fillna('unknown').astype(str)

# ============================================================================
# STEP 4: CREATE EMBEDDINGS (OPTIONAL)
# ============================================================================
print("="*80)
print("STEP 4: CREATING TEXT EMBEDDINGS")
print("="*80)

if CONFIG['use_distilbert']:
    print("🚀 Using DistilBERT for text embeddings...")
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

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bin']), 1):
    print(f"📊 Fold {fold}/{CONFIG['n_folds']}")
    
    # Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(
        X_tr, y_tr,
        categorical_feature=categorical_features
    )
    val_data = lgb.Dataset(
        X_val, y_val,
        categorical_feature=categorical_features,
        reference=train_data
    )
    
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
    oof_predictions[val_idx] = model.predict(X_val)
    test_predictions += model.predict(X_test) / CONFIG['n_folds']
    
    # Calculate SMAPE
    fold_smape = smape(y_val, oof_predictions[val_idx])
    fold_scores.append(fold_smape)
    print(f"   Fold {fold} SMAPE: {fold_smape:.3f}%")
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
