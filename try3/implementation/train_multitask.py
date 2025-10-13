"""
DistilBERT Multi-Task Learning: Price Prediction + Range Classification

INNOVATION:
- Auxiliary range classification task helps price regression
- Soft probabilities (not hard like two-stage!)
- Shared representation learns price structure better

Expected: 45-46% → 42-43% SMAPE (-3 points improvement!)

Based on: train_distilbert_huber.py (47.378% SMAPE)
Key difference from two-stage: Uses soft range probabilities to guide prediction,
not hard classification + conditional regression (which failed at 52-54%)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("🚀 DistilBERT Multi-Task Learning: Price + Range")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'distilbert_multitask',
    
    # Model
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'hidden_dropout_prob': 0.1,
    
    # Training
    'n_folds': 5,
    'batch_size': 32,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Multi-task parameters
    'n_ranges': 3,  # Budget, Mid-Range, Premium
    'range_thresholds': [20.0, 50.0],  # Split at $20 and $50
    'task_weights': {
        'regression': 0.9,      # Primary task
        'classification': 0.1   # Auxiliary task
    },
    
    # Loss parameters
    'huber_delta': 1.0,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16': True if torch.cuda.is_available() else False,
    
    # Misc
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"📍 Device: {CONFIG['device']}")
print(f"📍 Model: {CONFIG['model_name']}")
print(f"📍 Multi-Task: Regression (90%) + Classification (10%)")
print(f"📍 Price Ranges: Budget (<$20), Mid ($20-$50), Premium (>$50)")
print(f"📍 Mixed Precision: {CONFIG['fp16']}")
print("="*80)
print()

# ============================================================================
# Price Range Helper
# ============================================================================

def price_to_range(price):
    """Convert price to range label"""
    if price < CONFIG['range_thresholds'][0]:
        return 0  # Budget
    elif price < CONFIG['range_thresholds'][1]:
        return 1  # Mid-Range
    else:
        return 2  # Premium

def price_to_ranges_batch(prices):
    """Batch version"""
    ranges = np.zeros(len(prices), dtype=np.int64)
    ranges[prices >= CONFIG['range_thresholds'][0]] = 1
    ranges[prices >= CONFIG['range_thresholds'][1]] = 2
    return ranges

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

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Log transform prices for regression
        self.prices = torch.tensor(np.log1p(prices), dtype=torch.float)
        self.prices_original = torch.tensor(prices, dtype=torch.float)
        
        # Range labels for classification
        self.ranges = torch.tensor(price_to_ranges_batch(prices), dtype=torch.long)
    
    def __len__(self):
        return len(self.prices)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['price'] = self.prices[idx]
        item['price_original'] = self.prices_original[idx]
        item['range'] = self.ranges[idx]
        return item

# ============================================================================
# Multi-Task Model
# ============================================================================

class MultiTaskPricePredictor(nn.Module):
    """
    Multi-Task Learning Model
    
    Architecture:
    - Shared DistilBERT encoder
    - Dual heads:
      1. Range classifier (Budget/Mid/Premium)
      2. Price regressor (continuous value)
    
    Key Insight:
    Range classification helps model learn price structure,
    improving regression accuracy!
    """
    
    def __init__(self, model_name, n_ranges=3, dropout=0.1):
        super().__init__()
        
        # Shared encoder
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Range classifier head
        self.range_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_ranges)
        )
        
        # Price regressor head
        self.price_regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Shared encoding
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # Range classification logits
        range_logits = self.range_classifier(cls_output)
        
        # Price prediction (log-space)
        price_pred = self.price_regressor(cls_output)
        
        return {
            'range_logits': range_logits,
            'price_pred': price_pred.squeeze(-1)
        }

# ============================================================================
# Multi-Task Loss
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    
    Loss = α * Regression Loss + β * Classification Loss
    
    - Regression: Huber Loss (robust to outliers)
    - Classification: Cross Entropy (soft probabilities)
    - Weights: 90% regression, 10% classification
    """
    
    def __init__(self, regression_weight=0.9, classification_weight=0.1, huber_delta=1.0):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # Regression loss
        reg_loss = self.huber_loss(predictions['price_pred'], targets['price'])
        
        # Classification loss
        cls_loss = self.ce_loss(predictions['range_logits'], targets['range'])
        
        # Combined loss
        total_loss = (
            self.regression_weight * reg_loss +
            self.classification_weight * cls_loss
        )
        
        return total_loss, reg_loss, cls_loss

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, fp16=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prices = batch['price'].to(device)
        ranges = batch['range'].to(device)
        
        targets = {'price': prices, 'range': ranges}
        
        optimizer.zero_grad()
        
        if fp16:
            with torch.cuda.amp.autocast():
                predictions = model(input_ids, attention_mask)
                loss, reg_loss, cls_loss = criterion(predictions, targets)
        else:
            predictions = model(input_ids, attention_mask)
            loss, reg_loss, cls_loss = criterion(predictions, targets)
        
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
        total_reg_loss += reg_loss.item()
        total_cls_loss += cls_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, total_reg_loss / n, total_cls_loss / n

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    all_preds = []
    all_labels = []
    all_range_preds = []
    all_range_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prices = batch['price'].to(device)
            prices_original = batch['price_original'].cpu().numpy()
            ranges = batch['range'].to(device)
            
            targets = {'price': prices, 'range': ranges}
            
            predictions = model(input_ids, attention_mask)
            loss, reg_loss, cls_loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Inverse transform predictions
            preds = predictions['price_pred'].cpu().numpy()
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0.01)
            
            # Range predictions
            range_preds = predictions['range_logits'].argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(prices_original)
            all_range_preds.extend(range_preds)
            all_range_labels.extend(ranges.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_range_preds = np.array(all_range_preds)
    all_range_labels = np.array(all_range_labels)
    
    smape = calculate_smape(all_labels, all_preds)
    range_acc = (all_range_preds == all_range_labels).mean() * 100
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'reg_loss': total_reg_loss / n,
        'cls_loss': total_cls_loss / n,
        'smape': smape,
        'range_acc': range_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

# ============================================================================
# Main
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
    
    # Price statistics
    print(f"\n📊 Price Statistics:")
    print(f"  Mean:     ${train_df['price'].mean():.2f}")
    print(f"  Median:   ${train_df['price'].median():.2f}")
    print(f"  Min:      ${train_df['price'].min():.2f}")
    print(f"  Max:      ${train_df['price'].max():.2f}")
    
    # Range distribution
    train_df['range'] = train_df['price'].apply(price_to_range)
    range_counts = train_df['range'].value_counts().sort_index()
    print(f"\n📊 Range Distribution:")
    range_names = ['Budget (<$20)', 'Mid-Range ($20-$50)', 'Premium (>$50)']
    for i, name in enumerate(range_names):
        count = range_counts.get(i, 0)
        pct = count / len(train_df) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Stratified CV by price quantiles
    train_df['price_bin'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    # Initialize loss function
    criterion = MultiTaskLoss(
        regression_weight=CONFIG['task_weights']['regression'],
        classification_weight=CONFIG['task_weights']['classification'],
        huber_delta=CONFIG['huber_delta']
    )
    
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    fold_range_accs = []
    
    print("\n🚀 Starting cross-validation...")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_bin']), 1):
        print(f"\n📊 Fold {fold}/{CONFIG['n_folds']}")
        print("-"*80)
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_dataset = MultiTaskDataset(
            train_fold['catalog_content'].values,
            train_fold['price'].values,
            tokenizer,
            CONFIG['max_length']
        )
        
        val_dataset = MultiTaskDataset(
            val_fold['catalog_content'].values,
            val_fold['price'].values,
            tokenizer,
            CONFIG['max_length']
        )
        
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
        model = MultiTaskPricePredictor(
            CONFIG['model_name'],
            n_ranges=CONFIG['n_ranges'],
            dropout=CONFIG['hidden_dropout_prob']
        )
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
        
        # Training loop
        best_smape = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            
            train_loss, train_reg, train_cls = train_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, CONFIG['device'], CONFIG['fp16']
            )
            
            val_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])
            
            print(f"Train - Loss: {train_loss:.4f}, Reg: {train_reg:.4f}, Cls: {train_cls:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f}, Cls: {val_metrics['cls_loss']:.4f}")
            print(f"Val   - SMAPE: {val_metrics['smape']:.3f}%, Range Acc: {val_metrics['range_acc']:.2f}%")
            
            if val_metrics['smape'] < best_smape:
                best_smape = val_metrics['smape']
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG['output_dir'] / f'best_model_fold{fold}.pt')
                print(f"✓ Saved best model (SMAPE: {val_metrics['smape']:.3f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        
        val_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])
        
        print(f"\n✓ Fold {fold} Final SMAPE: {val_metrics['smape']:.3f}%")
        print(f"✓ Fold {fold} Range Accuracy: {val_metrics['range_acc']:.2f}%")
        
        oof_predictions[val_idx] = val_metrics['predictions']
        fold_scores.append(val_metrics['smape'])
        fold_range_accs.append(val_metrics['range_acc'])
    
    # Overall OOF score
    oof_smape = calculate_smape(train_df['price'].values, oof_predictions)
    
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*80)
    print("\nSMAPE:")
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i:<3} {score:>7.3f}%")
    print("-"*20)
    print(f"Mean     {np.mean(fold_scores):>7.3f}%")
    print(f"Std      {np.std(fold_scores):>7.3f}%")
    print(f"OOF      {oof_smape:>7.3f}%")
    
    print("\nRange Classification Accuracy:")
    for i, acc in enumerate(fold_range_accs, 1):
        print(f"Fold {i:<3} {acc:>7.2f}%")
    print("-"*20)
    print(f"Mean     {np.mean(fold_range_accs):>7.2f}%")
    
    # Compare to baselines
    print("\n📊 COMPARISON:")
    print(f"Huber Loss Baseline:     47.378%")
    print(f"Quantile-Huber:          ~45-46%")
    print(f"Multi-Task (THIS):       {oof_smape:.3f}%", end="")
    if oof_smape < 45:
        print(f" (EXCELLENT! ✅✅)")
    elif oof_smape < 47.378:
        print(f" (BETTER than baseline ✅)")
    else:
        print(f" (Need investigation ❌)")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price_true': train_df['price'],
        'price_pred': oof_predictions
    })
    oof_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)
    
    print("\n✓ OOF predictions saved")
    
    # Test predictions
    print("\n🔮 Generating test predictions...")
    
    test_dataset = MultiTaskDataset(
        test_df['catalog_content'].values,
        np.zeros(len(test_df)),
        tokenizer,
        CONFIG['max_length']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_predictions = np.zeros(len(test_df))
    
    for fold in range(1, CONFIG['n_folds'] + 1):
        model = MultiTaskPricePredictor(
            CONFIG['model_name'],
            n_ranges=CONFIG['n_ranges'],
            dropout=CONFIG['hidden_dropout_prob']
        )
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        model = model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                
                predictions = model(input_ids, attention_mask)
                
                # Inverse transform
                preds = predictions['price_pred'].cpu().numpy()
                preds = np.expm1(preds)
                preds = np.maximum(preds, 0.01)
                
                fold_preds.extend(preds)
        
        test_predictions += np.array(fold_preds) / CONFIG['n_folds']
    
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
    print(f"\n🎯 Final OOF SMAPE: {oof_smape:.3f}%")
    if oof_smape < 43:
        print(f"🚀 Excellent! Ready for ensemble!")
        print("\n👉 Next: Create ensemble of all three models to reach <40% SMAPE!")

if __name__ == "__main__":
    main()
