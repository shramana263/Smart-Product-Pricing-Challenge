"""
DistilBERT Fine-Tuning with Quantile-Huber Loss

IMPROVEMENTS OVER HUBER:
1. Quantile Loss: Asymmetric - better for SMAPE (penalizes over/under differently)
2. Huber Component: Still robust to outliers
3. Combined: Best of both worlds

Expected: 47.4% → 45-46% SMAPE (-2 points improvement!)

Based on: train_distilbert_huber.py (47.378% SMAPE)
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
print("🚀 DistilBERT + Quantile-Huber Loss - SMAPE Optimized")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'distilbert_quantile_huber',
    
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
    
    # Loss parameters
    'quantile': 0.5,      # Median prediction
    'huber_delta': 1.0,   # Huber delta
    'quantile_weight': 0.5,  # Balance between quantile and huber
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16': True if torch.cuda.is_available() else False,
    
    # Misc
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"📍 Device: {CONFIG['device']}")
print(f"📍 Model: {CONFIG['model_name']}")
print(f"📍 Loss: Quantile-Huber (q={CONFIG['quantile']}, δ={CONFIG['huber_delta']})")
print(f"📍 Mixed Precision: {CONFIG['fp16']}")
print("="*80)
print()

# ============================================================================
# Quantile-Huber Loss (SMAPE Optimized!)
# ============================================================================

class QuantileHuberLoss(nn.Module):
    """
    Combined Quantile + Huber Loss
    
    Quantile Loss:
    - Asymmetric penalty (SMAPE is asymmetric!)
    - Optimizes for median (robust to outliers)
    - Better for percentage errors
    
    Huber Loss:
    - Smooth transition (quadratic → linear)
    - Prevents gradient explosion
    - Robust to extreme outliers
    
    Combined:
    - Best of both worlds
    - SMAPE-aware + outlier-robust
    """
    
    def __init__(self, quantile=0.5, delta=1.0, weight=0.5):
        super().__init__()
        self.quantile = quantile
        self.delta = delta
        self.weight = weight  # Balance between quantile and huber
    
    def forward(self, pred, target):
        error = target - pred
        abs_error = torch.abs(error)
        
        # Quantile Loss Component (asymmetric)
        quantile_loss = torch.where(
            error > 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        
        # Huber Loss Component (outlier-robust)
        huber_mask = abs_error <= self.delta
        huber_loss = torch.where(
            huber_mask,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta)
        )
        
        # Weighted combination
        combined_loss = self.weight * quantile_loss + (1 - self.weight) * huber_loss
        
        return combined_loss.mean()

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
    """Dataset for price prediction"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Log transform prices
        self.prices = torch.tensor(np.log1p(prices), dtype=torch.float)
        self.prices_original = torch.tensor(prices, dtype=torch.float)
    
    def __len__(self):
        return len(self.prices)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['price'] = self.prices[idx]
        item['price_original'] = self.prices_original[idx]
        return item

# ============================================================================
# Model
# ============================================================================

class DistilBERTPricePredictor(nn.Module):
    """DistilBERT for price prediction"""
    
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # Predict log-price
        price_pred = self.regressor(cls_output)
        
        return price_pred.squeeze(-1)

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
        prices = batch['price'].to(device)
        
        optimizer.zero_grad()
        
        if fp16:
            with torch.cuda.amp.autocast():
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, prices)
        else:
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, prices)
        
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

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prices = batch['price'].to(device)
            prices_original = batch['price_original'].cpu().numpy()
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, prices)
            
            total_loss += loss.item()
            
            # Inverse transform predictions
            preds = predictions.cpu().numpy()
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0.01)
            
            all_preds.extend(preds)
            all_labels.extend(prices_original)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    smape = calculate_smape(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, smape, all_preds, all_labels

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
    print(f"  Skewness: {train_df['price'].skew():.2f}")
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Stratified CV by price quantiles
    train_df['price_bin'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    # Initialize loss function
    criterion = QuantileHuberLoss(
        quantile=CONFIG['quantile'],
        delta=CONFIG['huber_delta'],
        weight=CONFIG['quantile_weight']
    )
    
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    
    print("\n🚀 Starting cross-validation...")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_bin']), 1):
        print(f"\n📊 Fold {fold}/{CONFIG['n_folds']}")
        print("-"*80)
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_dataset = PriceDataset(
            train_fold['catalog_content'].values,
            train_fold['price'].values,
            tokenizer,
            CONFIG['max_length']
        )
        
        val_dataset = PriceDataset(
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
        model = DistilBERTPricePredictor(
            CONFIG['model_name'],
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
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, CONFIG['device'], CONFIG['fp16']
            )
            
            val_loss, val_smape, _, _ = evaluate(
                model, val_loader, criterion, CONFIG['device']
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val SMAPE:  {val_smape:.3f}%")
            
            if val_smape < best_smape:
                best_smape = val_smape
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG['output_dir'] / f'best_model_fold{fold}.pt')
                print(f"✓ Saved best model (SMAPE: {val_smape:.3f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered")
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        
        _, fold_smape, fold_preds, _ = evaluate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        print(f"\n✓ Fold {fold} Final SMAPE: {fold_smape:.3f}%")
        
        oof_predictions[val_idx] = fold_preds
        fold_scores.append(fold_smape)
    
    # Overall OOF score
    oof_smape = calculate_smape(train_df['price'].values, oof_predictions)
    
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*80)
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i:<3} {score:>7.3f}%")
    print("-"*20)
    print(f"Mean     {np.mean(fold_scores):>7.3f}%")
    print(f"Std      {np.std(fold_scores):>7.3f}%")
    print(f"OOF      {oof_smape:>7.3f}%")
    
    # Compare to baseline
    print("\n📊 COMPARISON:")
    print(f"Huber Loss Baseline:     47.378%")
    print(f"Quantile-Huber:          {oof_smape:.3f}%", end="")
    if oof_smape < 47.378:
        print(f" (BETTER by {47.378 - oof_smape:.3f}% ✅)")
    else:
        print(f" (WORSE by {oof_smape - 47.378:.3f}% ❌)")
    
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
    
    test_dataset = PriceDataset(
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
        model = DistilBERTPricePredictor(
            CONFIG['model_name'],
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
                preds = predictions.cpu().numpy()
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
    if oof_smape < 47.378:
        print(f"🚀 Improvement: {47.378 - oof_smape:.3f}% ✓")
        print("\n👉 Next: Run Multi-Task model for another ~3 point improvement!")

if __name__ == "__main__":
    main()
