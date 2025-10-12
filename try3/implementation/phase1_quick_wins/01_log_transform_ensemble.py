"""
PHASE 1.1: LOG TRANSFORM ENSEMBLE
==================================
Expected Improvement: 53.6% → 49-50% SMAPE

Research Finding:
- Price distribution is HIGHLY skewed (skewness=13.60, kurtosis=736)
- Log transformation makes distribution nearly normal
- Q-Q plot shows clear log-normal pattern

Strategy:
1. Train 3 models on different target transformations
2. Ensemble predictions with optimized weights
3. Validate on stratified holdout set

Time Estimate: 4-6 hours
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': Path("../../try2/dataset"),
    'output_dir': Path("../outputs/phase1_log_transform"),
    'model_name': 'distilbert-base-uncased',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 3,
    'n_folds': 5,
    'random_seed': 42
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1.1: LOG TRANSFORM ENSEMBLE")
print("="*80)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

print("\n📂 Loading training data...")
train1 = pd.read_csv(CONFIG['data_dir'] / "train1.csv")
train2 = pd.read_csv(CONFIG['data_dir'] / "train2.csv")
train = pd.concat([train1, train2], ignore_index=True)
print(f"   ✓ Loaded {len(train):,} training samples")

print("\n📂 Loading test data...")
test1 = pd.read_csv(CONFIG['data_dir'] / "test1.csv")
test2 = pd.read_csv(CONFIG['data_dir'] / "test2.csv")
test = pd.concat([test1, test2], ignore_index=True)
print(f"   ✓ Loaded {len(test):,} test samples")

# ============================================================================
# TARGET TRANSFORMATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: APPLYING TARGET TRANSFORMATIONS")
print("="*80)

# Original price statistics
print("\n📊 Original Price Statistics:")
print(f"   Mean:     ${train['price'].mean():.2f}")
print(f"   Median:   ${train['price'].median():.2f}")
print(f"   Std:      ${train['price'].std():.2f}")
print(f"   Min:      ${train['price'].min():.2f}")
print(f"   Max:      ${train['price'].max():.2f}")
print(f"   Skewness: {train['price'].skew():.2f}")
print(f"   Kurtosis: {train['price'].kurtosis():.2f}")

# Apply transformations
print("\n🔄 Creating transformed targets...")

# 1. Log transform: log(price + 1)
train['target_log'] = np.log1p(train['price'])
print(f"   ✓ Log transform: skewness = {train['target_log'].skew():.2f}")

# 2. Square root transform
train['target_sqrt'] = np.sqrt(train['price'])
print(f"   ✓ Sqrt transform: skewness = {train['target_sqrt'].skew():.2f}")

# 3. Box-Cox-like transform (Yeo-Johnson for handling zeros)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=False)
train['target_boxcox'] = pt.fit_transform(train[['price']])
print(f"   ✓ Box-Cox transform: skewness = {train['target_boxcox'].skew():.2f}")

# 4. Keep raw price
train['target_raw'] = train['price']

# ============================================================================
# CREATE PRICE BINS FOR STRATIFICATION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CREATE STRATIFIED FOLDS")
print("="*80)

# Create price bins for stratified splitting
bins = [0, 10, 20, 30, 50, 100, float('inf')]
labels = ['0-10', '10-20', '20-30', '30-50', '50-100', '100+']
train['price_bin'] = pd.cut(train['price'], bins=bins, labels=labels)

print("\n📊 Price Bin Distribution:")
for bin_label, count in train['price_bin'].value_counts().sort_index().items():
    pct = (count / len(train)) * 100
    print(f"   ${bin_label:10s}: {count:>8,} ({pct:5.2f}%)")

# Create stratified folds
skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['price_bin'])):
    train.loc[val_idx, 'fold'] = fold

print(f"\n✓ Created {CONFIG['n_folds']} stratified folds")

# ============================================================================
# DEFINE SMAPE METRIC
# ============================================================================

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def smape_score(y_true, y_pred):
    """Wrapper for consistency"""
    return smape(y_true, y_pred)

print("\n✓ SMAPE metric defined")

# ============================================================================
# TRAINING STRATEGY
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING STRATEGY")
print("="*80)

print("""
We will train 3 separate DistilBERT models:

Model 1: Predict log(price + 1)
   - Transform: y_pred = exp(model_output) - 1
   - Best for: Handling extreme skewness
   - Expected: ~48-50% SMAPE

Model 2: Predict sqrt(price)
   - Transform: y_pred = model_output ^ 2
   - Best for: Moderate outlier handling
   - Expected: ~50-52% SMAPE

Model 3: Predict Box-Cox transformed price
   - Transform: y_pred = inverse_transform(model_output)
   - Best for: Optimal normalization
   - Expected: ~49-51% SMAPE

Final Ensemble:
   - Weighted average: 0.5*log + 0.3*boxcox + 0.2*sqrt
   - Weights optimized on validation set
   - Expected: ~48-50% SMAPE
""")

# ============================================================================
# PLACEHOLDER: ACTUAL MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("STEP 5: MODEL TRAINING (PLACEHOLDER)")
print("="*80)

print("""
⚠️ IMPLEMENTATION NOTE:

Due to computational requirements, the actual DistilBERT training 
requires GPU resources and significant time (4-6 hours per model).

This script provides the FRAMEWORK. To complete:

1. Set up GPU environment (CUDA-enabled)
2. Install: transformers, torch, accelerate
3. Implement the training loops below
4. Run: python 01_log_transform_ensemble.py --train

For now, we'll create the training structure and save it.
""")

# ============================================================================
# TRAINING TEMPLATE
# ============================================================================

training_template = """
# Training Template (to be executed with GPU)

def train_distilbert_model(train_df, target_col, output_dir, config):
    '''Train DistilBERT for regression on transformed target'''
    
    # 1. Prepare data
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['catalog_content'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=config['max_length']
        )
    
    # 2. Create dataset
    from datasets import Dataset
    dataset = Dataset.from_pandas(train_df[['catalog_content', target_col]])
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 3. Define model
    from transformers import DistilBertForSequenceClassification
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=1  # Regression
    )
    
    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=True,  # Mixed precision
        dataloader_num_workers=4,
    )
    
    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics_regression
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
    return trainer

# Execute training for each transformation
for target_name in ['target_log', 'target_sqrt', 'target_boxcox']:
    print(f'\\nTraining model for {target_name}...')
    output_dir = CONFIG['output_dir'] / target_name
    
    # Train on 4 folds, validate on 1
    for fold in range(CONFIG['n_folds']):
        train_df = train[train['fold'] != fold]
        val_df = train[train['fold'] == fold]
        
        trainer = train_distilbert_model(
            train_df, 
            target_name, 
            output_dir / f'fold_{fold}',
            CONFIG
        )
        
        # Validate
        predictions = trainer.predict(val_df)
        
        # Inverse transform
        if target_name == 'target_log':
            y_pred = np.expm1(predictions)
        elif target_name == 'target_sqrt':
            y_pred = predictions ** 2
        else:  # boxcox
            y_pred = pt.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Calculate SMAPE
        fold_smape = smape(val_df['price'].values, y_pred)
        print(f'  Fold {fold} SMAPE: {fold_smape:.2f}%')
"""

# Save training template
with open(CONFIG['output_dir'] / 'training_template.py', 'w') as f:
    f.write(training_template)

print(f"\n✓ Training template saved to: {CONFIG['output_dir'] / 'training_template.py'}")

# ============================================================================
# ENSEMBLE STRATEGY
# ============================================================================

print("\n" + "="*80)
print("STEP 6: ENSEMBLE STRATEGY")
print("="*80)

ensemble_template = """
# Ensemble Strategy

# After training all 3 models, ensemble their predictions:

def ensemble_predictions(pred_log, pred_sqrt, pred_boxcox, weights=(0.5, 0.2, 0.3)):
    '''
    Combine predictions from 3 models with optimal weights
    
    Args:
        pred_log: Predictions from log-transformed model (already inverse-transformed)
        pred_sqrt: Predictions from sqrt-transformed model (already inverse-transformed)
        pred_boxcox: Predictions from boxcox model (already inverse-transformed)
        weights: Tuple of (w_log, w_sqrt, w_boxcox)
    
    Returns:
        final_predictions: Weighted ensemble
    '''
    w_log, w_sqrt, w_boxcox = weights
    final = w_log * pred_log + w_sqrt * pred_sqrt + w_boxcox * pred_boxcox
    return final

# Optimize weights on validation set
from scipy.optimize import minimize

def optimize_weights(pred_log, pred_sqrt, pred_boxcox, y_true):
    '''Find optimal ensemble weights'''
    
    def objective(weights):
        w_log, w_sqrt, w_boxcox = weights
        ensemble = w_log * pred_log + w_sqrt * pred_sqrt + w_boxcox * pred_boxcox
        return smape(y_true, ensemble)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    result = minimize(
        objective,
        x0=[0.5, 0.2, 0.3],  # Initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# Example usage:
# optimal_weights = optimize_weights(val_pred_log, val_pred_sqrt, val_pred_boxcox, val_y_true)
# final_pred = ensemble_predictions(test_pred_log, test_pred_sqrt, test_pred_boxcox, optimal_weights)
"""

with open(CONFIG['output_dir'] / 'ensemble_strategy.py', 'w') as f:
    f.write(ensemble_template)

print(f"✓ Ensemble strategy saved to: {CONFIG['output_dir'] / 'ensemble_strategy.py'}")

# ============================================================================
# CREATE SUBMISSION TEMPLATE
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SUBMISSION TEMPLATE")
print("="*80)

submission_template = """
# Generate Final Submission

# After training and ensembling:

# 1. Load trained models
model_log = load_model('phase1_log_transform/target_log')
model_sqrt = load_model('phase1_log_transform/target_sqrt')
model_boxcox = load_model('phase1_log_transform/target_boxcox')

# 2. Generate predictions on test set
test_pred_log = model_log.predict(test['catalog_content'])
test_pred_log = np.expm1(test_pred_log)  # Inverse transform

test_pred_sqrt = model_sqrt.predict(test['catalog_content'])
test_pred_sqrt = test_pred_sqrt ** 2  # Inverse transform

test_pred_boxcox = model_boxcox.predict(test['catalog_content'])
test_pred_boxcox = pt.inverse_transform(test_pred_boxcox)  # Inverse transform

# 3. Ensemble with optimal weights
optimal_weights = (0.5, 0.2, 0.3)  # From validation
final_predictions = ensemble_predictions(
    test_pred_log, 
    test_pred_sqrt, 
    test_pred_boxcox, 
    optimal_weights
)

# 4. Create submission file
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_predictions
})

# 5. Ensure positive prices
submission['price'] = submission['price'].clip(lower=0.01)

# 6. Save
submission.to_csv('test_out_phase1_log_ensemble.csv', index=False)

print(f'Submission saved: {len(submission)} predictions')
print(f'Price range: ${submission["price"].min():.2f} - ${submission["price"].max():.2f}')
"""

with open(CONFIG['output_dir'] / 'create_submission.py', 'w') as f:
    f.write(submission_template)

print(f"✓ Submission template saved to: {CONFIG['output_dir'] / 'create_submission.py'}")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("✅ PHASE 1.1 SETUP COMPLETE")
print("="*80)

print(f"""
📁 Output Directory: {CONFIG['output_dir']}

📄 Files Created:
   1. training_template.py     - DistilBERT training code
   2. ensemble_strategy.py     - Ensemble & weight optimization
   3. create_submission.py     - Generate final predictions

📊 Data Prepared:
   - Training samples: {len(train):,}
   - Test samples: {len(test):,}
   - Stratified folds: {CONFIG['n_folds']}
   - Target transformations: 3 (log, sqrt, box-cox)

🎯 Expected Results:
   Model 1 (Log):     ~48-50% SMAPE
   Model 2 (Sqrt):    ~50-52% SMAPE
   Model 3 (BoxCox):  ~49-51% SMAPE
   Ensemble:          ~48-50% SMAPE
   
   Improvement from baseline: -4 to -6 SMAPE points

⚠️ Next Steps:
   1. Set up GPU environment
   2. Install: pip install transformers torch accelerate datasets
   3. Run training (4-6 hours per model)
   4. Validate on holdout set
   5. Optimize ensemble weights
   6. Generate submission

💡 Tips:
   - Use fp16 (mixed precision) for faster training
   - Monitor validation SMAPE during training
   - Save best checkpoint based on validation loss
   - Try different learning rates if needed
   - Ensure no data leakage in folds

🚀 After This Phase:
   Expected SMAPE: 48-50%
   Next Phase: Unit Standardization (47-48%)
""")

# Save config for reference
import json
config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}
with open(CONFIG['output_dir'] / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"\n✓ Configuration saved to: {CONFIG['output_dir'] / 'config.json'}")
print("\n" + "="*80)
print("Ready to proceed with training! 🎯")
print("="*80)
