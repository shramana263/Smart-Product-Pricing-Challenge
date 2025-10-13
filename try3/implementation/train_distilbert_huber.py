"""
DistilBERT Fine-Tuning with Huber Loss - Outlier Robust

KEY IMPROVEMENTS:
1. Huber Loss instead of MSE (robust to outliers)
2. Log1p transform on target (handles skewed distribution)
3. Price range stratified training
4. Conservative regularization

Expected: 53.6% → 49-50% SMAPE

Strategy Reference: try3/STRATEGY_TO_40_PERCENT.md
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("🚀 DistilBERT + Huber Loss - Outlier Robust Training")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'distilbert_huber',
    
    # Model
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    
    # Training
    'n_folds': 5,
    'batch_size': 32,
    'gradient_accumulation_steps': 2,  # Effective batch = 64
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Huber Loss
    'huber_delta': 1.0,  # Threshold for quadratic→linear transition
    
    # Target transform
    'use_log_transform': True,  # Train on log1p(price)
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16': True if torch.cuda.is_available() else False,
    
    # Misc
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"📍 Device: {CONFIG['device']}")
print(f"📍 Model: {CONFIG['model_name']}")
print(f"📍 Huber Delta: {CONFIG['huber_delta']}")
print(f"📍 Log Transform: {CONFIG['use_log_transform']}")
print(f"📍 Mixed Precision: {CONFIG['fp16']}")
print("="*80)
print()

# ============================================================================
# Huber Loss
# ============================================================================

class HuberLoss(nn.Module):
    """
    Huber Loss: Combination of L2 (quadratic) and L1 (linear)
    
    For error |e| < δ: Loss = 0.5 * e²
    For error |e| ≥ δ: Loss = δ * (|e| - 0.5 * δ)
    
    Benefits:
    - Quadratic for small errors → accurate predictions
    - Linear for large errors → robust to outliers
    - Reduces catastrophic SMAPE on $20→$500 predictions
    """
    
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        
        # Quadratic part (small errors)
        quadratic_mask = abs_error <= self.delta
        quadratic_loss = 0.5 * error[quadratic_mask] ** 2
        
        # Linear part (large errors)
        linear_mask = abs_error > self.delta
        linear_loss = self.delta * (abs_error[linear_mask] - 0.5 * self.delta)
        
        # Combine
        total_loss = torch.cat([quadratic_loss, linear_loss])
        return total_loss.mean()

# ============================================================================
# SMAPE Metric
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (0-200%)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# Dataset
# ============================================================================

class PriceDataset(Dataset):
    """PyTorch Dataset for text→price regression"""
    
    def __init__(self, texts, prices, tokenizer, max_length, use_log=False):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Apply log transform if enabled
        if use_log:
            prices = np.log1p(prices)  # log(1 + price)
        
        self.labels = torch.tensor(prices, dtype=torch.float)
        self.use_log = use_log
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# ============================================================================
# Model
# ============================================================================

class DistilBERTForRegression(nn.Module):
    """DistilBERT with regression head"""
    
    def __init__(self, model_name):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        prediction = self.regressor(cls_output)
        
        return prediction.squeeze(-1)

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, fp16=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if fp16:
            with torch.cuda.amp.autocast():
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
        else:
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
        
        if fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, use_log=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Inverse transform if using log
    if use_log:
        all_preds = np.expm1(all_preds)  # exp(x) - 1
        all_labels = np.expm1(all_labels)
    
    # Ensure positive predictions
    all_preds = np.maximum(all_preds, 0.01)
    
    smape = calculate_smape(all_labels, all_preds)
    
    return total_loss / len(dataloader), smape, all_preds, all_labels

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n📂 Loading data...")
    train1 = pd.read_csv(CONFIG['data_dir'] / 'train1.csv')
    train2 = pd.read_csv(CONFIG['data_dir'] / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    test1 = pd.read_csv(CONFIG['data_dir'] / 'test1.csv')
    test2 = pd.read_csv(CONFIG['data_dir'] / 'test2.csv')
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Training samples: {len(train_df):,}")
    print(f"✓ Test samples:     {len(test_df):,}")
    
    # Clean text
    train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
    
    # Stratified CV by price quantiles
    train_df['price_bin'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    
    print("\n🚀 Starting cross-validation...")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_bin']), 1):
        print(f"\n📊 Fold {fold}/{CONFIG['n_folds']}")
        print("-"*80)
        
        # Split data
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Create datasets
        train_dataset = PriceDataset(
            train_fold['catalog_content'].values,
            train_fold['price'].values,
            tokenizer,
            CONFIG['max_length'],
            use_log=CONFIG['use_log_transform']
        )
        
        val_dataset = PriceDataset(
            val_fold['catalog_content'].values,
            val_fold['price'].values,
            tokenizer,
            CONFIG['max_length'],
            use_log=CONFIG['use_log_transform']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'] * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize model
        model = DistilBERTForRegression(CONFIG['model_name'])
        model = model.to(CONFIG['device'])
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * CONFIG['num_epochs']
        num_warmup_steps = int(num_training_steps * CONFIG['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function
        criterion = HuberLoss(delta=CONFIG['huber_delta'])
        
        # Training loop
        best_smape = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, 
                criterion, CONFIG['device'], CONFIG['fp16']
            )
            
            # Evaluate
            val_loss, val_smape, _, _ = evaluate(
                model, val_loader, criterion, CONFIG['device'],
                use_log=CONFIG['use_log_transform']
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val SMAPE:  {val_smape:.3f}%")
            
            # Early stopping
            if val_smape < best_smape:
                best_smape = val_smape
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), CONFIG['output_dir'] / f'best_model_fold{fold}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered (patience={patience})")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        
        # Final evaluation on validation fold
        _, fold_smape, fold_preds, _ = evaluate(
            model, val_loader, criterion, CONFIG['device'],
            use_log=CONFIG['use_log_transform']
        )
        
        print(f"\n✓ Fold {fold} SMAPE: {fold_smape:.3f}%")
        
        # Store OOF predictions
        oof_predictions[val_idx] = fold_preds
        fold_scores.append(fold_smape)
    
    # Overall OOF score
    oof_smape = calculate_smape(train_df['price'].values, oof_predictions)
    
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"\n{'Fold':<8} {'SMAPE':<10}")
    print("-"*20)
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i:<3} {score:>7.3f}%")
    print("-"*20)
    print(f"Mean     {np.mean(fold_scores):>7.3f}%")
    print(f"Std      {np.std(fold_scores):>7.3f}%")
    print(f"OOF      {oof_smape:>7.3f}%")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price_true': train_df['price'],
        'price_pred': oof_predictions
    })
    oof_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)
    
    print("\n✓ OOF predictions saved")
    
    # Generate test predictions (using average of all folds)
    print("\n🔮 Generating test predictions...")
    
    test_dataset = PriceDataset(
        test_df['catalog_content'].values,
        np.zeros(len(test_df)),  # Dummy labels
        tokenizer,
        CONFIG['max_length'],
        use_log=False  # Don't transform dummy labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Average predictions from all folds
    test_predictions = np.zeros(len(test_df))
    
    for fold in range(1, CONFIG['n_folds'] + 1):
        model = DistilBERTForRegression(CONFIG['model_name'])
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        model = model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                
                predictions = model(input_ids, attention_mask)
                fold_preds.extend(predictions.cpu().numpy())
        
        fold_preds = np.array(fold_preds)
        
        # Inverse transform if using log
        if CONFIG['use_log_transform']:
            fold_preds = np.expm1(fold_preds)
        
        # Ensure positive
        fold_preds = np.maximum(fold_preds, 0.01)
        
        test_predictions += fold_preds / CONFIG['n_folds']
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_predictions
    })
    
    submission.to_csv(CONFIG['output_dir'] / 'test_predictions.csv', index=False)
    
    print("\n✓ Test predictions saved")
    print(f"  Min:    ${test_predictions.min():.2f}")
    print(f"  Median: ${np.median(test_predictions):.2f}")
    print(f"  Mean:   ${test_predictions.mean():.2f}")
    print(f"  Max:    ${test_predictions.max():.2f}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🎯 OOF SMAPE: {oof_smape:.3f}%")
    print(f"📊 Baseline:  53.636%")
    if oof_smape < 53.636:
        print(f"🚀 Improvement: {53.636 - oof_smape:.3f}% ✓")
    
    print(f"\n📁 Output files:")
    print(f"  - OOF predictions: {CONFIG['output_dir'] / 'oof_predictions.csv'}")
    print(f"  - Test predictions: {CONFIG['output_dir'] / 'test_predictions.csv'}")
    print(f"  - Models: {CONFIG['output_dir'] / 'best_model_fold*.pt'}")

if __name__ == "__main__":
    main()
