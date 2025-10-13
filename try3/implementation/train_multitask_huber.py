"""
Multi-Task DistilBERT with Huber Loss

APPROACH:
- Main task: Price regression (Huber Loss)
- Auxiliary task: Price range classification (helps learning)
- Joint training with shared representations

Expected: 47.4% → 43-44% SMAPE (~3-4 point improvement)

Architecture:
    DistilBERT (768-dim)
           ↓
    Shared Features (256-dim)
           ↓
    ┌──────┴───────┐
    │              │
Range (soft)   Price (Huber)
  0.1×          0.9×
    │              │
    └──────┬───────┘
           ↓
    Combined Loss
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("🚀 Multi-Task DistilBERT: Price Regression + Range Classification")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'multitask_huber',
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    
    # Price ranges for auxiliary task
    'price_ranges': [
        (0, 20, 'Budget_Economy'),
        (20, 100, 'Mid_Premium'),
        (100, np.inf, 'Luxury'),
    ],
    
    # Training
    'n_folds': 5,
    'batch_size': 32,
    'gradient_accumulation_steps': 2,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Multi-task loss weights
    'range_weight': 0.1,    # Auxiliary task (help learning)
    'price_weight': 0.9,    # Main task (primary objective)
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16': True if torch.cuda.is_available() else False,
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"📍 Device: {CONFIG['device']}")
print(f"📍 Model: {CONFIG['model_name']}")
print(f"📍 Architecture: Multi-Task (Range + Price)")
print(f"📍 Loss Weights: Range={CONFIG['range_weight']}, Price={CONFIG['price_weight']}")
print("="*80)
print()

# ============================================================================
# Helper Functions
# ============================================================================

def assign_price_range(price):
    """Assign price to range for auxiliary task"""
    for i, (low, high, _) in enumerate(CONFIG['price_ranges']):
        if low <= price < high:
            return i
    return len(CONFIG['price_ranges']) - 1

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(numerator / denominator) * 100

# ============================================================================
# Loss Functions
# ============================================================================

class HuberLoss(nn.Module):
    """Huber loss for price regression"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        
        quadratic_mask = abs_error <= self.delta
        linear_mask = abs_error > self.delta
        
        loss = torch.where(
            quadratic_mask,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta)
        )
        
        return loss.mean()

# ============================================================================
# Dataset
# ============================================================================

class MultiTaskDataset(Dataset):
    """Dataset with price and range labels"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        self.price_ranges = torch.tensor([assign_price_range(p) for p in prices], dtype=torch.long)
        self.prices_log = torch.tensor(np.log1p(prices), dtype=torch.float)
        self.prices_original = torch.tensor(prices, dtype=torch.float)
    
    def __len__(self):
        return len(self.prices_log)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['price_range'] = self.price_ranges[idx]
        item['price_log'] = self.prices_log[idx]
        item['price_original'] = self.prices_original[idx]
        return item

# ============================================================================
# Multi-Task Model
# ============================================================================

class MultiTaskModel(nn.Module):
    """
    Multi-Task Learning:
    - Main: Price regression (Huber Loss)
    - Auxiliary: Range classification (helps feature learning)
    """
    
    def __init__(self, model_name, n_ranges):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Auxiliary task: Range classification (soft probabilities)
        self.range_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_ranges)
        )
        
        # Main task: Price regression
        self.price_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get DistilBERT embeddings
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # Shared features
        shared_features = self.shared(cls_output)
        
        # Two heads
        range_logits = self.range_head(shared_features)
        price_pred = self.price_head(shared_features).squeeze(-1)
        
        return range_logits, price_pred

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, range_criterion, 
                price_criterion, device, fp16=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_range_loss = 0
    total_price_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        price_range = batch['price_range'].to(device)
        price_log = batch['price_log'].to(device)
        
        optimizer.zero_grad()
        
        if fp16:
            with torch.cuda.amp.autocast():
                range_logits, price_pred = model(input_ids, attention_mask)
                
                # Auxiliary loss (range classification)
                range_loss = range_criterion(range_logits, price_range)
                
                # Main loss (price regression)
                price_loss = price_criterion(price_pred, price_log)
                
                # Combined loss
                loss = CONFIG['range_weight'] * range_loss + \
                       CONFIG['price_weight'] * price_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            range_logits, price_pred = model(input_ids, attention_mask)
            
            range_loss = range_criterion(range_logits, price_range)
            price_loss = price_criterion(price_pred, price_log)
            
            loss = CONFIG['range_weight'] * range_loss + \
                   CONFIG['price_weight'] * price_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        total_range_loss += range_loss.item()
        total_price_loss += price_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'range': f'{range_loss.item():.4f}',
            'price': f'{price_loss.item():.4f}'
        })
    
    return (total_loss / len(dataloader),
            total_range_loss / len(dataloader),
            total_price_loss / len(dataloader))

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_range_correct = 0
    all_range_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            price_range = batch['price_range'].to(device)
            price_original = batch['price_original'].cpu().numpy()
            
            range_logits, price_pred = model(input_ids, attention_mask)
            
            # Range accuracy
            pred_range = torch.argmax(range_logits, dim=1)
            all_range_correct += (pred_range == price_range).sum().item()
            all_range_total += len(price_range)
            
            # Price predictions
            price_pred = price_pred.cpu().numpy()
            price_pred = np.expm1(price_pred)
            price_pred = np.maximum(price_pred, 0.01)
            
            all_preds.extend(price_pred)
            all_labels.extend(price_original)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    smape = calculate_smape(all_labels, all_preds)
    range_accuracy = 100 * all_range_correct / all_range_total
    
    return smape, range_accuracy, all_preds, all_labels

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
    
    train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
    
    train_df['price_range'] = train_df['price'].apply(assign_price_range)
    
    print("\n📊 Price Range Distribution:")
    for i, (low, high, name) in enumerate(CONFIG['price_ranges']):
        count = (train_df['price_range'] == i).sum()
        pct = 100 * count / len(train_df)
        print(f"  {name:20s} (${low:>3.0f}-${high if high != np.inf else 999:>3.0f}): {count:>6,} ({pct:>5.2f}%)")
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Loss functions
    range_criterion = nn.CrossEntropyLoss()
    price_criterion = HuberLoss(delta=1.0)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    
    # Check existing folds
    existing_folds = []
    for fold in range(1, CONFIG['n_folds'] + 1):
        if (CONFIG['output_dir'] / f'best_model_fold{fold}.pt').exists():
            existing_folds.append(fold)
    
    print("\n🚀 Starting Multi-Task Training...")
    print("="*80)
    
    if existing_folds:
        print(f"\n💾 Found existing folds: {existing_folds}")
        print(f"✅ Will resume from Fold {max(existing_folds) + 1}")
        print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_range']), 1):
        
        fold_model_path = CONFIG['output_dir'] / f'best_model_fold{fold}.pt'
        if fold_model_path.exists():
            print(f"\n✅ Fold {fold}/{CONFIG['n_folds']} - Already trained!")
            print("-"*80)
            
            val_fold = train_df.iloc[val_idx]
            
            val_dataset = MultiTaskDataset(
                val_fold['catalog_content'].values,
                val_fold['price'].values,
                tokenizer,
                CONFIG['max_length']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=CONFIG['batch_size'] * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            model = MultiTaskModel(CONFIG['model_name'], len(CONFIG['price_ranges']))
            model.load_state_dict(torch.load(fold_model_path))
            model = model.to(CONFIG['device'])
            
            fold_smape, fold_range_acc, fold_preds, _ = evaluate(model, val_loader, CONFIG['device'])
            
            print(f"✓ Fold {fold} SMAPE: {fold_smape:.3f}% (Range Acc: {fold_range_acc:.2f}%)")
            
            oof_predictions[val_idx] = fold_preds
            fold_scores.append(fold_smape)
            
            continue
        
        print(f"\n📊 Fold {fold}/{CONFIG['n_folds']} - Training...")
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
        
        model = MultiTaskModel(CONFIG['model_name'], len(CONFIG['price_ranges']))
        model = model.to(CONFIG['device'])
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        num_training_steps = len(train_loader) * CONFIG['num_epochs']
        num_warmup_steps = int(num_training_steps * CONFIG['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        best_smape = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            
            train_loss, train_range_loss, train_price_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                range_criterion, price_criterion, CONFIG['device'], CONFIG['fp16']
            )
            
            val_smape, val_range_acc, _, _ = evaluate(model, val_loader, CONFIG['device'])
            
            print(f"Train Loss: {train_loss:.4f} (range: {train_range_loss:.4f}, price: {train_price_loss:.4f})")
            print(f"Val SMAPE:  {val_smape:.3f}%")
            print(f"Val Range Acc: {val_range_acc:.2f}%")
            
            if val_smape < best_smape:
                best_smape = val_smape
                patience_counter = 0
                torch.save(model.state_dict(), fold_model_path)
                print(f"✓ New best: {best_smape:.3f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered")
                    break
        
        model.load_state_dict(torch.load(fold_model_path))
        
        fold_smape, fold_range_acc, fold_preds, _ = evaluate(model, val_loader, CONFIG['device'])
        
        print(f"\n✓ Fold {fold} Final SMAPE: {fold_smape:.3f}% (Range Acc: {fold_range_acc:.2f}%)")
        
        oof_predictions[val_idx] = fold_preds
        fold_scores.append(fold_smape)
    
    oof_smape = calculate_smape(train_df['price'].values, oof_predictions)
    
    print("\n" + "="*80)
    print("📊 MULTI-TASK RESULTS")
    print("="*80)
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i:<3} {score:>7.3f}%")
    print("-"*20)
    print(f"Mean     {np.mean(fold_scores):>7.3f}%")
    print(f"Std      {np.std(fold_scores):>7.3f}%")
    print(f"OOF      {oof_smape:>7.3f}%")
    
    print("\n📊 COMPARISON:")
    print(f"Huber Baseline:  47.378%")
    print(f"Multi-Task:      {oof_smape:.3f}%", end="")
    if oof_smape < 47.378:
        print(f" (✅ BETTER by {47.378 - oof_smape:.3f}%)")
    else:
        print(f" (❌ WORSE by {oof_smape - 47.378:.3f}%)")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price_true': train_df['price'],
        'price_pred': oof_predictions
    })
    oof_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)
    
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
        model = MultiTaskModel(CONFIG['model_name'], len(CONFIG['price_ranges']))
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        model = model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                
                _, price_pred = model(input_ids, attention_mask)
                
                price_pred = price_pred.cpu().numpy()
                price_pred = np.expm1(price_pred)
                price_pred = np.maximum(price_pred, 0.01)
                
                fold_preds.extend(price_pred)
        
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
    print("✅ MULTI-TASK TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
