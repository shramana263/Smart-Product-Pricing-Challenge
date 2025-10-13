"""
Two-Stage DistilBERT: Classification + Conditional Regression

APPROACH:
Stage 1: Classify product into price range (5 classes)
Stage 2: Use range-specific regression head for precise prediction

KEY IMPROVEMENTS:
1. Separate handling for Budget ($0-10) vs Luxury ($100+)
2. Range-specific loss functions (Huber for budget, Log-Cosh for luxury)
3. Focal Loss for classification (handles class imbalance)
4. Cross-entropy regularization between stages

Expected: 47.4% → 41-42% SMAPE

Based on error analysis:
- Budget ($0-$10):    122.6% SMAPE → Target: 40%
- Economy ($10-$20):   46.5% SMAPE → Target: 30%
- Mid-Range ($20-$30): 17.6% SMAPE → Target: 15%
- Premium ($30-$50):   45.9% SMAPE → Target: 35%
- High-End ($50-$100): 91.2% SMAPE → Target: 50%
- Luxury ($100+):     140.8% SMAPE → Target: 70%

Weighted Average: Should achieve ~41-42% SMAPE
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
print("🚀 Two-Stage Model: Classification + Conditional Regression")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Paths
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'two_stage_model',
    
    # Model
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    
    # Price ranges (from error analysis)
    'price_ranges': [
        (0, 10, 'Budget'),        # 38% of samples, 122.6% SMAPE
        (10, 20, 'Economy'),       # 26% of samples, 46.5% SMAPE
        (20, 50, 'Mid_Premium'),   # 25% of samples, avg 30% SMAPE
        (50, 100, 'High_End'),     # 8% of samples, 91.2% SMAPE
        (100, np.inf, 'Luxury'),   # 3% of samples, 140.8% SMAPE
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
    
    # Loss weights
    'classification_weight': 0.3,  # Weight for classification loss
    'regression_weight': 0.7,      # Weight for regression loss
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fp16': True if torch.cuda.is_available() else False,
    
    # Misc
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"📍 Device: {CONFIG['device']}")
print(f"📍 Model: {CONFIG['model_name']}")
print(f"📍 Price Ranges: {len(CONFIG['price_ranges'])}")
print(f"📍 Mixed Precision: {CONFIG['fp16']}")
print("="*80)
print()

# ============================================================================
# Price Range Assignment
# ============================================================================

def assign_price_range(price):
    """Assign price to range category"""
    for i, (low, high, name) in enumerate(CONFIG['price_ranges']):
        if low <= price < high:
            return i
    return len(CONFIG['price_ranges']) - 1  # Default to last range

# ============================================================================
# Focal Loss for Classification
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================================================
# Range-Specific Loss Functions
# ============================================================================

class HuberLoss(nn.Module):
    """Huber loss for budget/economy ranges"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        
        quadratic_mask = abs_error <= self.delta
        quadratic_loss = 0.5 * error[quadratic_mask] ** 2
        
        linear_mask = abs_error > self.delta
        linear_loss = self.delta * (abs_error[linear_mask] - 0.5 * self.delta)
        
        if len(quadratic_loss) == 0:
            return linear_loss.mean()
        if len(linear_loss) == 0:
            return quadratic_loss.mean()
        
        total_loss = torch.cat([quadratic_loss, linear_loss])
        return total_loss.mean()

class LogCoshLoss(nn.Module):
    """Log-Cosh loss for premium/luxury ranges"""
    def forward(self, pred, target):
        error = pred - target
        return torch.mean(torch.log(torch.cosh(error + 1e-12)))

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

class TwoStageDataset(Dataset):
    """Dataset for two-stage model"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Price range labels
        self.price_classes = torch.tensor([assign_price_range(p) for p in prices], dtype=torch.long)
        
        # Log-transformed prices for regression
        self.prices_log = torch.tensor(np.log1p(prices), dtype=torch.float)
        self.prices_original = torch.tensor(prices, dtype=torch.float)
    
    def __len__(self):
        return len(self.price_classes)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['price_class'] = self.price_classes[idx]
        item['price_log'] = self.prices_log[idx]
        item['price_original'] = self.prices_original[idx]
        return item

# ============================================================================
# Two-Stage Model
# ============================================================================

class TwoStageModel(nn.Module):
    """
    Two-Stage Model:
    1. Classification head: Predict price range
    2. Conditional regression heads: Range-specific price prediction
    """
    
    def __init__(self, model_name, n_ranges):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Stage 1: Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_ranges)
        )
        
        # Stage 2: Range-specific regression heads
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(n_ranges)
        ])
        
        self.n_ranges = n_ranges
    
    def forward(self, input_ids, attention_mask, price_class=None):
        # Get DistilBERT embeddings
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        cls_output = self.dropout(cls_output)
        
        # Stage 1: Classification
        class_logits = self.classifier(cls_output)
        
        # Stage 2: Conditional regression
        if price_class is not None:
            # Training: Use true class
            regression_preds = []
            for i, head in enumerate(self.regression_heads):
                mask = (price_class == i)
                if mask.any():
                    head_pred = head(cls_output[mask])
                    regression_preds.append((mask, head_pred.squeeze(-1)))
            
            # Reconstruct full prediction tensor (match dtype for FP16 compatibility)
            batch_preds = torch.zeros(len(price_class), device=cls_output.device, dtype=cls_output.dtype)
            for mask, pred in regression_preds:
                batch_preds[mask] = pred.to(dtype=batch_preds.dtype)
            
            return class_logits, batch_preds
        
        else:
            # Inference: Use predicted class
            pred_class = torch.argmax(class_logits, dim=1)
            
            # Get predictions from appropriate heads (match dtype for FP16)
            batch_preds = torch.zeros(len(pred_class), device=cls_output.device, dtype=cls_output.dtype)
            for i in range(self.n_ranges):
                mask = (pred_class == i)
                if mask.any():
                    head_output = self.regression_heads[i](cls_output[mask]).squeeze(-1)
                    batch_preds[mask] = head_output.to(dtype=batch_preds.dtype)
            
            return class_logits, batch_preds

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, class_criterion, 
                reg_criteria, device, fp16=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        price_class = batch['price_class'].to(device)
        price_log = batch['price_log'].to(device)
        
        optimizer.zero_grad()
        
        if fp16:
            with torch.cuda.amp.autocast():
                class_logits, reg_preds = model(input_ids, attention_mask, price_class)
                
                # Classification loss
                class_loss = class_criterion(class_logits, price_class)
                
                # Range-specific regression loss
                reg_loss = 0
                for i, criterion in enumerate(reg_criteria):
                    mask = (price_class == i)
                    if mask.any():
                        reg_loss += criterion(reg_preds[mask], price_log[mask])
                reg_loss = reg_loss / len(reg_criteria)
                
                # Combined loss
                loss = CONFIG['classification_weight'] * class_loss + \
                       CONFIG['regression_weight'] * reg_loss
        else:
            class_logits, reg_preds = model(input_ids, attention_mask, price_class)
            
            class_loss = class_criterion(class_logits, price_class)
            
            reg_loss = 0
            for i, criterion in enumerate(reg_criteria):
                mask = (price_class == i)
                if mask.any():
                    reg_loss += criterion(reg_preds[mask], price_log[mask])
            reg_loss = reg_loss / len(reg_criteria)
            
            loss = CONFIG['classification_weight'] * class_loss + \
                   CONFIG['regression_weight'] * reg_loss
        
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
        total_class_loss += class_loss.item()
        total_reg_loss += reg_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'cls': class_loss.item(),
            'reg': reg_loss.item()
        })
    
    return (total_loss / len(dataloader),
            total_class_loss / len(dataloader),
            total_reg_loss / len(dataloader))

def evaluate(model, dataloader, class_criterion, reg_criteria, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_class_correct = 0
    all_class_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            price_class = batch['price_class'].to(device)
            price_log = batch['price_log'].to(device)
            price_original = batch['price_original'].cpu().numpy()
            
            class_logits, reg_preds = model(input_ids, attention_mask, None)
            
            # Classification accuracy
            pred_class = torch.argmax(class_logits, dim=1)
            all_class_correct += (pred_class == price_class).sum().item()
            all_class_total += len(price_class)
            
            # Inverse transform predictions
            reg_preds = reg_preds.cpu().numpy()
            reg_preds = np.expm1(reg_preds)  # Inverse log1p
            reg_preds = np.maximum(reg_preds, 0.01)  # Ensure positive
            
            all_preds.extend(reg_preds)
            all_labels.extend(price_original)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    smape = calculate_smape(all_labels, all_preds)
    class_accuracy = 100 * all_class_correct / all_class_total
    
    return smape, class_accuracy, all_preds, all_labels

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
    
    # Assign price ranges
    train_df['price_range'] = train_df['price'].apply(assign_price_range)
    
    # Print range distribution
    print("\n📊 Price Range Distribution:")
    for i, (low, high, name) in enumerate(CONFIG['price_ranges']):
        count = (train_df['price_range'] == i).sum()
        pct = 100 * count / len(train_df)
        print(f"  {name:15s} (${low:>3.0f}-${high if high != np.inf else 999:>3.0f}): {count:>6,} ({pct:>5.2f}%)")
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Compute class weights for Focal Loss
    class_counts = train_df['price_range'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])
    
    print(f"✓ Class weights: {class_weights.cpu().numpy()}")
    
    # Initialize loss functions
    class_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Range-specific regression losses
    reg_criteria = [
        HuberLoss(delta=0.5),  # Budget - sensitive to small errors
        HuberLoss(delta=1.0),  # Economy
        HuberLoss(delta=1.5),  # Mid-Premium
        LogCoshLoss(),         # High-End - robust to outliers
        LogCoshLoss(),         # Luxury - robust to outliers
    ]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    
    print("\n🚀 Starting cross-validation...")
    print("="*80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_range']), 1):
        print(f"\n📊 Fold {fold}/{CONFIG['n_folds']}")
        print("-"*80)
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_dataset = TwoStageDataset(
            train_fold['catalog_content'].values,
            train_fold['price'].values,
            tokenizer,
            CONFIG['max_length']
        )
        
        val_dataset = TwoStageDataset(
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
        model = TwoStageModel(CONFIG['model_name'], len(CONFIG['price_ranges']))
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
            
            train_loss, train_class_loss, train_reg_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                class_criterion, reg_criteria, CONFIG['device'], CONFIG['fp16']
            )
            
            val_smape, val_class_acc, _, _ = evaluate(
                model, val_loader, class_criterion, reg_criteria, CONFIG['device']
            )
            
            print(f"Train Loss: {train_loss:.4f} (cls: {train_class_loss:.4f}, reg: {train_reg_loss:.4f})")
            print(f"Val SMAPE:  {val_smape:.3f}%")
            print(f"Val Class Acc: {val_class_acc:.2f}%")
            
            if val_smape < best_smape:
                best_smape = val_smape
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG['output_dir'] / f'best_model_fold{fold}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        
        # Final evaluation
        fold_smape, fold_class_acc, fold_preds, _ = evaluate(
            model, val_loader, class_criterion, reg_criteria, CONFIG['device']
        )
        
        print(f"\n✓ Fold {fold} SMAPE: {fold_smape:.3f}% (Class Acc: {fold_class_acc:.2f}%)")
        
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
    
    test_dataset = TwoStageDataset(
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
        model = TwoStageModel(CONFIG['model_name'], len(CONFIG['price_ranges']))
        model.load_state_dict(torch.load(CONFIG['output_dir'] / f'best_model_fold{fold}.pt'))
        model = model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                
                _, reg_preds = model(input_ids, attention_mask, None)
                
                reg_preds = reg_preds.cpu().numpy()
                reg_preds = np.expm1(reg_preds)
                reg_preds = np.maximum(reg_preds, 0.01)
                
                fold_preds.extend(reg_preds)
        
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
    print(f"\n🎯 OOF SMAPE: {oof_smape:.3f}%")
    print(f"📊 Previous:  47.378%")
    if oof_smape < 47.378:
        print(f"🚀 Improvement: {47.378 - oof_smape:.3f}% ✓")

if __name__ == "__main__":
    main()
