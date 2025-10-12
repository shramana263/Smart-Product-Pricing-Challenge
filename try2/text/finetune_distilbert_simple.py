"""
Simplified DistilBERT Fine-Tuning for Product Price Prediction
Amazon ML Hackathon - Smart Product Pricing Challenge

This is a simplified version using Hugging Face's AutoModel with regression head.
Expected improvement: 63.28% → 50-55% SMAPE

Author: Generated for Amazon ML Challenge 2025
Date: October 12, 2025
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 DistilBERT Fine-Tuning - Simplified Version")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR.parent / 'dataset'
    OUTPUT_DIR = BASE_DIR.parent / 'modeling' / 'distilbert_simple'
    
    # Model
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128  # Shorter for faster training
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data split (same as V3 for fair comparison)
    VAL_SIZE = 0.15
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"📍 Device: {config.DEVICE}")
print(f"📍 Model: {config.MODEL_NAME}")
print("="*80)

# ============================================================================
# SMAPE Metric
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (0-200%)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# Dataset Class
# ============================================================================

class PriceDataset(Dataset):
    """Simple PyTorch Dataset for text→price regression"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(prices, dtype=torch.float)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# ============================================================================
# Load & Prepare Data
# ============================================================================

def load_and_prepare_data():
    """Load datasets and split into train/val/test"""
    
    print("\n📂 Loading datasets...")
    
    # Load training data
    train1 = pd.read_csv(config.DATASET_DIR / 'train1.csv')
    train2 = pd.read_csv(config.DATASET_DIR / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    print(f"✓ Training samples: {len(train_df):,}")
    
    # Load test data
    test1 = pd.read_csv(config.DATASET_DIR / 'test1.csv')
    test2 = pd.read_csv(config.DATASET_DIR / 'test2.csv')
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Test samples:     {len(test_df):,}")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
    
    # Remove line breaks
    train_df['catalog_content'] = train_df['catalog_content'].str.replace(r'[\n\r]+', ' ', regex=True)
    test_df['catalog_content'] = test_df['catalog_content'].str.replace(r'[\n\r]+', ' ', regex=True)
    
    # Stratified split by price quantiles
    print("\n✂️  Splitting data...")
    train_df['price_quantile'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    
    # Split: 60% train, 15% val, 25% test
    train_val, test_split = train_test_split(
        train_df, 
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_df['price_quantile']
    )
    
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train_split, val_split = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=train_val['price_quantile']
    )
    
    print(f"✓ Train: {len(train_split):,} samples ({len(train_split)/len(train_df)*100:.1f}%)")
    print(f"✓ Val:   {len(val_split):,} samples ({len(val_split)/len(train_df)*100:.1f}%)")
    print(f"✓ Test:  {len(test_split):,} samples ({len(test_split)/len(train_df)*100:.1f}%)")
    
    return train_split, val_split, test_split, test_df

# ============================================================================
# Training
# ============================================================================

def train_model(train_df, val_df):
    """Fine-tune DistilBERT"""
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print("📦 Creating datasets...")
    train_dataset = PriceDataset(
        train_df['catalog_content'].values,
        train_df['price'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    
    val_dataset = PriceDataset(
        val_df['catalog_content'].values,
        val_df['price'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    
    print("\n🔥 Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=1,  # Regression (single output)
        problem_type="regression"
    )
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
    output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
    logging_dir=str(config.OUTPUT_DIR / 'logs'),
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=config.DEVICE == 'cuda',
        report_to='tensorboard',
        dataloader_num_workers=0,  # Set to 0 for Windows
        disable_tqdm=False
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        smape = calculate_smape(labels, predictions)
        mse = np.mean((predictions - labels) ** 2)
        return {'smape': smape, 'mse': mse}
    
    # Initialize trainer
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
    
    trainer.train()
    
    print("\n✓ Training complete!")
    
    return model, tokenizer, trainer

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, tokenizer, test_df):
    """Evaluate on test set"""
    
    print("\n📊 Evaluating on test set...")
    
    test_dataset = PriceDataset(
        test_df['catalog_content'].values,
        test_df['price'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    
    y_pred = predictions.predictions.flatten()
    y_true = test_df['price'].values
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 0.01)
    
    smape = calculate_smape(y_true, y_pred)
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    
    print(f"\n📈 Test Results:")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"  MSE:   {mse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"\n  Prediction range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    print(f"  Actual range:     ${y_true.min():.2f} - ${y_true.max():.2f}")
    
    return smape

# ============================================================================
# Generate Predictions
# ============================================================================

def generate_predictions(model, tokenizer, test_df):
    """Generate predictions for submission"""
    
    print("\n🔮 Generating test predictions...")
    
    # Create dummy labels
    dummy_prices = np.zeros(len(test_df))
    
    test_dataset = PriceDataset(
        test_df['catalog_content'].values,
        dummy_prices,
        tokenizer,
        config.MAX_LENGTH
    )
    
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    
    y_pred = predictions.predictions.flatten()
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 0.01)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred
    })
    
    modeling_dir = config.BASE_DIR.parent / 'modeling'
    modeling_dir.mkdir(parents=True, exist_ok=True)
    output_path = modeling_dir / 'test_out_distilbert_simple.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved: {output_path}")
    print(f"  Samples: {len(submission):,}")
    print(f"  Price range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    
    return submission

# ============================================================================
# Main
# ============================================================================

def main():
    """Main pipeline"""
    
    print("\n" + "="*80)
    print("STARTING DISTILBERT FINE-TUNING")
    print("="*80)
    
    # Load data
    train_df, val_df, test_df, final_test_df = load_and_prepare_data()
    
    # Train
    model, tokenizer, trainer = train_model(train_df, val_df)
    
    # Evaluate
    smape = evaluate(model, tokenizer, test_df)
    
    # Generate predictions
    submission = generate_predictions(model, tokenizer, final_test_df)
    
    # Save model
    print("\n💾 Saving model...")
    model_path = config.OUTPUT_DIR / 'final_model'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Model saved: {model_path}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n🎯 Test SMAPE:     {smape:.2f}%")
    print(f"📊 Baseline SMAPE: 63.28%")
    
    if smape < 63.28:
        print(f"🚀 Improvement:    {63.28 - smape:.2f}% ✓")
    else:
        print(f"⚠️  Change:         +{smape - 63.28:.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
