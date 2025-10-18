"""
DeBERTa-v3-large Text Embedding Extraction

Fine-tunes DeBERTa-v3-large on product descriptions and extracts
1024-dimensional embeddings for fusion with image and tabular features.

Key Features:
- Fine-tuning with regression head
- Mixed precision training (FP16)
- Gradient accumulation for large batch sizes
- Early stopping with best model checkpointing
- Embedding extraction and caching
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR,
    TEXT_MODEL, DEVICE, USE_FP16, RANDOM_SEED, CV_CONFIG
)

NUM_AVAILABLE_GPUS = max(1, torch.cuda.device_count())
PIN_MEMORY = DEVICE.startswith('cuda')

print("="*80)
print("🔤 DeBERTa-v3-large Embedding Extraction")
print("="*80)
print(f"Model: {TEXT_MODEL['name']}")
print(f"Device: {DEVICE}")
print(f"FP16: {USE_FP16}")
print("="*80)

# ============================================================================
# Dataset
# ============================================================================

class ProductTextDataset(Dataset):
    """Dataset for product text"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.prices = torch.tensor(np.log1p(prices), dtype=torch.float) if prices is not None else None
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.prices is not None:
            item['price'] = self.prices[idx]
        return item

# ============================================================================
# Model
# ============================================================================

class DeBERTaRegressor(nn.Module):
    """DeBERTa with regression head for price prediction"""
    
    def __init__(self, model_name, dropout=0.1, enable_multi_gpu=True):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        
        # ⚡ Enable multi-GPU if available
        if enable_multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
            self.deberta = nn.DataParallel(self.deberta)
        
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0]  # (batch, 1024)
        
        if return_embeddings:
            return cls_output
        
        cls_output = self.dropout(cls_output)
        logits = self.regressor(cls_output).squeeze(-1)
        return logits, cls_output

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, use_fp16=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prices = batch['price'].to(device)
        
        optimizer.zero_grad()
        
        if use_fp16:
            with torch.cuda.amp.autocast():
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, prices)
        else:
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, prices)
        
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, prices)
            
            total_loss += loss.item()
            
            # Inverse transform
            preds = np.expm1(logits.cpu().numpy())
            labels = np.expm1(prices.cpu().numpy())
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate SMAPE
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    smape = calculate_smape(all_labels, all_preds)
    
    return total_loss / len(dataloader), smape

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(numerator / denominator) * 100

# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from trained model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = model(input_ids, attention_mask, return_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training and extraction pipeline"""
    
    # Set seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("\n📂 Loading data...")
    train1 = pd.read_csv(DATA_DIR / 'train1.csv')
    train2 = pd.read_csv(DATA_DIR / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    test1 = pd.read_csv(DATA_DIR / 'test1.csv')
    test2 = pd.read_csv(DATA_DIR / 'test2.csv')
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")
    print(f"✓ GPUs detected: {NUM_AVAILABLE_GPUS}")

    per_device_batch = TEXT_MODEL['batch_size']
    scale_batch = TEXT_MODEL.get('scale_batch_by_gpu', True)
    train_batch_size = max(1, per_device_batch * NUM_AVAILABLE_GPUS) if scale_batch else per_device_batch
    eval_batch_size = max(1, int(train_batch_size * TEXT_MODEL.get('eval_batch_multiplier', 2)))
    dataloader_workers = max(2, TEXT_MODEL.get('num_workers', 2))
    print(f"✓ Effective train batch size: {train_batch_size}")
    print(f"✓ Effective eval batch size:  {eval_batch_size}")
    print(f"✓ DataLoader workers:        {dataloader_workers}")
    
    # Clean text
    train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL['name'])
    
    # Create price bins for stratification
    train_df['price_bin'] = pd.qcut(
        train_df['price'], 
        q=CV_CONFIG['n_price_bins'], 
        labels=False, 
        duplicates='drop'
    )
    
    print(f"\n🚀 Starting {CV_CONFIG['n_folds']}-fold cross-validation...")
    print("="*80)
    
    # Cross-validation
    kf = KFold(
        n_splits=CV_CONFIG['n_folds'],
        shuffle=CV_CONFIG['shuffle'],
        random_state=RANDOM_SEED
    )
    
    fold_scores = []
    train_embeddings_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
        print(f"\n📊 Fold {fold}/{CV_CONFIG['n_folds']}")
        print("-"*80)
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Datasets
        train_dataset = ProductTextDataset(
            train_fold['catalog_content'].values,
            train_fold['price'].values,
            tokenizer,
            TEXT_MODEL['max_length']
        )
        
        val_dataset = ProductTextDataset(
            val_fold['catalog_content'].values,
            val_fold['price'].values,
            tokenizer,
            TEXT_MODEL['max_length']
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False
        )
        
        # Model
        model = DeBERTaRegressor(TEXT_MODEL['name'])
        model = model.to(DEVICE)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TEXT_MODEL['learning_rate'],
            weight_decay=TEXT_MODEL['weight_decay']
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * TEXT_MODEL['num_epochs']
        num_warmup_steps = int(num_training_steps * TEXT_MODEL['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss
        criterion = nn.HuberLoss(delta=1.0)
        
        # Training loop
        best_smape = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(TEXT_MODEL['num_epochs']):
            print(f"\nEpoch {epoch+1}/{TEXT_MODEL['num_epochs']}")
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, 
                criterion, DEVICE, USE_FP16
            )
            
            val_loss, val_smape = evaluate(model, val_loader, criterion, DEVICE)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val SMAPE:  {val_smape:.3f}%")
            
            if val_smape < best_smape:
                best_smape = val_smape
                patience_counter = 0
                torch.save(
                    model.state_dict(), 
                    MODELS_DIR / f'deberta_fold{fold}.pt'
                )
                print("✓ Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        print(f"\n✓ Fold {fold} Best SMAPE: {best_smape:.3f}%")
        fold_scores.append(best_smape)
        
        # Load best model for embedding extraction
        model.load_state_dict(torch.load(MODELS_DIR / f'deberta_fold{fold}.pt'))
        
        # Extract embeddings for this fold
        full_train_dataset = ProductTextDataset(
            train_df['catalog_content'].values,
            train_df['price'].values,
            tokenizer,
            TEXT_MODEL['max_length']
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=PIN_MEMORY,
            drop_last=False
        )
        
        train_emb = extract_embeddings(model, full_train_loader, DEVICE)
        train_embeddings_list.append(train_emb)
    
    # Average embeddings from all folds
    print("\n📊 Averaging embeddings from all folds...")
    train_embeddings = np.mean(train_embeddings_list, axis=0)
    
    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*80)
    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i}: {score:.3f}%")
    print("-"*40)
    print(f"Mean:   {np.mean(fold_scores):.3f}%")
    print(f"Std:    {np.std(fold_scores):.3f}%")
    print("="*80)
    
    # Extract test embeddings using ensemble
    print("\n🔮 Extracting test embeddings...")
    test_dataset = ProductTextDataset(
        test_df['catalog_content'].values,
        None,
        tokenizer,
        TEXT_MODEL['max_length']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    
    test_embeddings_list = []
    for fold in range(1, CV_CONFIG['n_folds'] + 1):
        model = DeBERTaRegressor(TEXT_MODEL['name'])
        model.load_state_dict(torch.load(MODELS_DIR / f'deberta_fold{fold}.pt'))
        model = model.to(DEVICE)
        
        test_emb = extract_embeddings(model, test_loader, DEVICE)
        test_embeddings_list.append(test_emb)
    
    test_embeddings = np.mean(test_embeddings_list, axis=0)
    
    # Save embeddings
    print("\n💾 Saving embeddings...")
    
    train_emb_df = pd.DataFrame(
        train_embeddings,
        columns=[f'deberta_{i}' for i in range(TEXT_MODEL['embedding_dim'])]
    )
    train_emb_df.insert(0, 'sample_id', train_df['sample_id'].values)
    train_emb_df.to_csv(EMBEDDINGS_DIR / 'deberta_train_embeddings.csv', index=False)
    
    test_emb_df = pd.DataFrame(
        test_embeddings,
        columns=[f'deberta_{i}' for i in range(TEXT_MODEL['embedding_dim'])]
    )
    test_emb_df.insert(0, 'sample_id', test_df['sample_id'].values)
    test_emb_df.to_csv(EMBEDDINGS_DIR / 'deberta_test_embeddings.csv', index=False)
    
    print(f"✓ Train embeddings: {train_embeddings.shape}")
    print(f"✓ Test embeddings:  {test_embeddings.shape}")
    print(f"✓ Saved to: {EMBEDDINGS_DIR}")
    
    print("\n" + "="*80)
    print("✅ DeBERTa EMBEDDING EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
