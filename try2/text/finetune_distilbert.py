"""
Fine-tune DistilBERT for Product Price Prediction
Amazon ML Hackathon - Smart Product Pricing Challenge

This script fine-tunes DistilBERT on catalog_content to predict product prices.
Expected SMAPE improvement: 63.28% → 50-55%

Author: Generated for Amazon ML Challenge 2025
Date: October 12, 2025
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 DistilBERT Fine-Tuning for Product Price Prediction")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR.parent / 'dataset'
    OUTPUT_DIR = BASE_DIR.parent / 'modeling' / 'distilbert_model'
    
    # Data files
    TRAIN_FILES = ['train1.csv', 'train2.csv']
    TEST_FILES = ['test1.csv', 'test2.csv']
    
    # Model settings
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 256  # Reduced for faster training
    BATCH_SIZE = 16  # Adjust based on GPU memory
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data split
    VAL_SIZE = 0.15
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

print(f"📍 Device: {config.DEVICE}")
print(f"📍 Model: {config.MODEL_NAME}")
print(f"📍 Max Length: {config.MAX_LENGTH}")
print(f"📍 Batch Size: {config.BATCH_SIZE}")
print("="*80)

# ============================================================================
# Custom SMAPE Metric
# ============================================================================

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    SMAPE = (1/n) * Σ |y_pred - y_true| / ((|y_true| + |y_pred|) / 2)
    
    Returns SMAPE as percentage (0-200%)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# Custom Dataset Class
# ============================================================================

class ProductPriceDataset(Dataset):
    """PyTorch Dataset for product catalog → price regression"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.texts = texts
        self.prices = prices
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        price = float(self.prices[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(price, dtype=torch.float)
        }

# ============================================================================
# Custom Model (DistilBERT for Regression)
# ============================================================================

class DistilBERTForRegression(torch.nn.Module):
    """DistilBERT model adapted for price regression"""
    
    def __init__(self, model_name):
        super(DistilBERTForRegression, self).__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = torch.hub.load(
            'huggingface/pytorch-transformers', 
            'model', 
            model_name
        )
        
        # Regression head
        self.dropout = torch.nn.Dropout(0.3)
        self.regressor = torch.nn.Linear(768, 1)  # 768 = DistilBERT hidden size
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and regression head
        cls_output = self.dropout(cls_output)
        logits = self.regressor(cls_output).squeeze(-1)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load and combine train/test datasets"""
    
    print("\n📂 Loading datasets...")
    
    # Load training data
    train_dfs = []
    for file in config.TRAIN_FILES:
        path = config.DATASET_DIR / file
        df = pd.read_csv(path)
        train_dfs.append(df)
        print(f"✓ Loaded {file}: {len(df):,} samples")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\n📊 Total training samples: {len(train_df):,}")
    
    # Load test data
    test_dfs = []
    for file in config.TEST_FILES:
        path = config.DATASET_DIR / file
        df = pd.read_csv(path)
        test_dfs.append(df)
        print(f"✓ Loaded {file}: {len(df):,} samples")
    
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"\n📊 Total test samples: {len(test_df):,}")
    
    return train_df, test_df

def preprocess_text(df):
    """Clean and preprocess catalog_content"""
    
    print("\n🧹 Preprocessing text...")
    
    # Fill NaN values
    df['catalog_content'] = df['catalog_content'].fillna('')
    
    # Basic cleaning
    df['catalog_content'] = df['catalog_content'].str.replace('\n', ' ')
    df['catalog_content'] = df['catalog_content'].str.replace('\r', ' ')
    df['catalog_content'] = df['catalog_content'].str.replace('<br>', ' ')
    df['catalog_content'] = df['catalog_content'].str.strip()
    
    # Limit length for efficiency (keep first 512 chars)
    df['catalog_content'] = df['catalog_content'].str[:512]
    
    print(f"✓ Text preprocessing complete")
    print(f"  Average length: {df['catalog_content'].str.len().mean():.0f} characters")
    
    return df

# ============================================================================
# Data Splitting
# ============================================================================

def split_data(train_df):
    """Split data into train/val/test sets (stratified by price quantiles)"""
    
    print("\n✂️  Splitting data...")
    
    # Create price quantiles for stratification
    train_df['price_quantile'] = pd.qcut(
        train_df['price'], 
        q=10, 
        labels=False, 
        duplicates='drop'
    )
    
    # First split: train+val vs test (75% vs 25%)
    train_val_df, test_df = train_test_split(
        train_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_df['price_quantile']
    )
    
    # Second split: train vs val (85% vs 15% of remaining)
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train_df_split, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=train_val_df['price_quantile']
    )
    
    print(f"✓ Train set: {len(train_df_split):,} samples")
    print(f"✓ Val set:   {len(val_df):,} samples")
    print(f"✓ Test set:  {len(test_df):,} samples")
    
    return train_df_split, val_df, test_df

# ============================================================================
# Model Training
# ============================================================================

def train_model(train_dataset, val_dataset):
    """Fine-tune DistilBERT on training data"""
    
    print("\n🔥 Initializing model...")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Initialize model
    model = DistilBERTForRegression(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    print(f"✓ Model loaded: {config.MODEL_NAME}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        learning_rate=config.LEARNING_RATE,
        logging_dir=str(config.OUTPUT_DIR / 'logs'),
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to='tensorboard'
    )
    
    # Custom metric computation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        smape = calculate_smape(labels, predictions)
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            'smape': smape,
            'mse': mse,
            'mae': mae
        }
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\n🚀 Starting training...")
    print("="*80)
    
    # Train model
    trainer.train()
    
    print("\n✓ Training complete!")
    
    return model, tokenizer

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, tokenizer, test_df):
    """Evaluate model on test set"""
    
    print("\n📊 Evaluating model on test set...")
    
    # Create test dataset
    test_dataset = ProductPriceDataset(
        texts=test_df['catalog_content'].values,
        prices=test_df['price'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Prediction mode
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            preds = outputs['logits'].cpu().numpy()
            
            predictions.extend(preds)
            actuals.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    smape = calculate_smape(actuals, predictions)
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"\n📈 Test Results:")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"  MSE:   {mse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"\n  Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"  Actual range:     ${actuals.min():.2f} - ${actuals.max():.2f}")
    
    return predictions, actuals, smape

# ============================================================================
# Generate Test Predictions
# ============================================================================

def generate_test_predictions(model, tokenizer, test_df):
    """Generate predictions for final test set"""
    
    print("\n🔮 Generating test predictions...")
    
    # Create dummy prices (not used, but needed for dataset)
    dummy_prices = np.zeros(len(test_df))
    
    # Create dataset
    test_dataset = ProductPriceDataset(
        texts=test_df['catalog_content'].values,
        prices=dummy_prices,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Prediction mode
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            preds = outputs['logits'].cpu().numpy()
            
            predictions.extend(preds)
    
    predictions = np.array(predictions)
    
    # Ensure positive prices
    predictions = np.maximum(predictions, 0.01)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_dir = config.BASE_DIR.parent / 'modeling'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'test_out_distilbert.csv'
    submission_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved to: {output_path}")
    print(f"  Sample count: {len(submission_df):,}")
    print(f"  Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    
    return submission_df

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("STARTING DISTILBERT FINE-TUNING PIPELINE")
    print("="*80)
    
    # 1. Load data
    train_df, test_df = load_data()
    
    # 2. Preprocess text
    train_df = preprocess_text(train_df)
    test_df = preprocess_text(test_df)
    
    # 3. Split data
    train_split, val_split, test_split = split_data(train_df)
    
    # 4. Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    print(f"✓ Tokenizer loaded: {config.MODEL_NAME}")
    
    # 5. Create datasets
    print("\n📦 Creating PyTorch datasets...")
    train_dataset = ProductPriceDataset(
        texts=train_split['catalog_content'].values,
        prices=train_split['price'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = ProductPriceDataset(
        texts=val_split['catalog_content'].values,
        prices=val_split['price'].values,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    print(f"✓ Train dataset: {len(train_dataset):,} samples")
    print(f"✓ Val dataset:   {len(val_dataset):,} samples")
    
    # 6. Train model
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = train_model(train_dataset, val_dataset)
    
    # 7. Evaluate on test split
    predictions, actuals, smape = evaluate_model(model, tokenizer, test_split)
    
    # 8. Generate final test predictions
    submission_df = generate_test_predictions(model, tokenizer, test_df)
    
    # 9. Save model
    print("\n💾 Saving model...")
    model_path = config.OUTPUT_DIR / 'final_model'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n🎯 Final Test SMAPE: {smape:.2f}%")
    print(f"📊 Baseline SMAPE:   63.28%")
    print(f"🚀 Improvement:      {63.28 - smape:.2f}%")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
