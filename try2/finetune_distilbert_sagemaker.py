"""
DistilBERT Fine-Tuning for Product Price Prediction - AWS SageMaker Optimized
Amazon ML Hackathon - Smart Product Pricing Challenge

Optimized for AWS SageMaker Code Editor with:
- Multi-GPU support
- Checkpointing and resumption
- Mixed precision training (FP16)
- Gradient accumulation for large effective batch sizes
- SageMaker-specific optimizations

Expected improvement: 63.28% → 50-55% SMAPE

Author: Generated for Amazon ML Challenge 2025
Date: October 12, 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
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
print("🚀 DistilBERT Fine-Tuning - AWS SageMaker Optimized")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'modeling', 'distilbert_sagemaker')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    
    # Model
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128  # Good balance for GPU memory
    
    # Training - Optimized for ml.g4dn.xlarge (16GB VRAM)
    BATCH_SIZE = 32  # Per device
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32 * 2 = 64
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # Data split
    VAL_SIZE = 0.15
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    
    # Performance optimizations
    FP16 = True  # Mixed precision training
    DATALOADER_NUM_WORKERS = 4  # Parallel data loading
    DATALOADER_PIN_MEMORY = True  # Faster GPU transfers
    
    # Checkpointing
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 3  # Keep only 3 best checkpoints
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_GPU = torch.cuda.device_count()

config = Config()

print(f"📍 Device: {config.DEVICE}")
print(f"📍 GPUs Available: {config.N_GPU}")
print(f"📍 Model: {config.MODEL_NAME}")
print(f"📍 Batch Size: {config.BATCH_SIZE} x {config.GRADIENT_ACCUMULATION_STEPS} = {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS} (effective)")
print(f"📍 Mixed Precision: {config.FP16}")
print("="*80)

# ============================================================================
# GPU & Memory Utilities
# ============================================================================

def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")

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
    """Optimized PyTorch Dataset for text→price regression"""
    
    def __init__(self, texts, prices, tokenizer, max_length):
        # Pre-tokenize all texts at initialization for faster training
        print(f"  Tokenizing {len(texts):,} samples...")
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(prices, dtype=torch.float)
        print(f"  ✓ Tokenization complete")
    
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
    train1 = pd.read_csv(os.path.join(config.DATASET_DIR, 'train1.csv'))
    train2 = pd.read_csv(os.path.join(config.DATASET_DIR, 'train2.csv'))
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    print(f"✓ Training samples: {len(train_df):,}")
    
    # Load test data
    test1 = pd.read_csv(os.path.join(config.DATASET_DIR, 'test1.csv'))
    test2 = pd.read_csv(os.path.join(config.DATASET_DIR, 'test2.csv'))
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Test samples:     {len(test_df):,}")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    train_df['catalog_content'] = train_df['catalog_content'].fillna('').astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].fillna('').astype(str)
    
    # Remove line breaks and extra whitespace
    train_df['catalog_content'] = train_df['catalog_content'].str.replace(r'[\n\r]+', ' ', regex=True).str.strip()
    test_df['catalog_content'] = test_df['catalog_content'].str.replace(r'[\n\r]+', ' ', regex=True).str.strip()
    
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
    
    # Price statistics
    print(f"\n📊 Price Statistics:")
    print(f"  Train - Mean: ${train_split['price'].mean():.2f}, Median: ${train_split['price'].median():.2f}")
    print(f"  Val   - Mean: ${val_split['price'].mean():.2f}, Median: ${val_split['price'].median():.2f}")
    print(f"  Test  - Mean: ${test_split['price'].mean():.2f}, Median: ${test_split['price'].median():.2f}")
    
    return train_split, val_split, test_split, test_df

# ============================================================================
# Training
# ============================================================================

def train_model(train_df, val_df):
    """Fine-tune DistilBERT with SageMaker optimizations"""
    
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print("\n📦 Creating datasets...")
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
    
    # Print initial GPU usage
    if config.DEVICE == 'cuda':
        print("\n💾 Initial GPU Memory:")
        print_gpu_utilization()
    
    # SageMaker-optimized training arguments
    training_args = TrainingArguments(
        # Output & Logging
        output_dir=config.OUTPUT_DIR,
        logging_dir=os.path.join(config.OUTPUT_DIR, 'logs'),
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        
        # Training
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,  # Larger for inference
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        
        # Evaluation & Saving
        eval_strategy='steps',
        eval_steps=config.EVAL_STEPS,
        save_strategy='steps',
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        # Performance Optimizations
        fp16=config.FP16 and config.DEVICE == 'cuda',  # Mixed precision
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        
        # Reporting
        report_to='tensorboard',
        disable_tqdm=False,
        
        # Reproducibility
        seed=config.RANDOM_STATE,
        data_seed=config.RANDOM_STATE,
        
        # Distributed training (if multiple GPUs)
        ddp_find_unused_parameters=False if config.N_GPU > 1 else None,
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.01)
        
        smape = calculate_smape(labels, predictions)
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(mse)
        
        return {
            'smape': smape,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
    )
    
    # Check for existing checkpoint to resume
    checkpoint = None
    if os.path.exists(config.CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(config.CHECKPOINT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint = os.path.join(config.CHECKPOINT_DIR, latest_checkpoint)
            print(f"\n🔄 Resuming from checkpoint: {checkpoint}")
    
    print("\n🚀 Starting training...")
    print("="*80)
    
    # Train model
    trainer.train(resume_from_checkpoint=checkpoint)
    
    print("\n✓ Training complete!")
    
    # Print final GPU usage
    if config.DEVICE == 'cuda':
        print("\n💾 Final GPU Memory:")
        print_gpu_utilization()
    
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
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
            dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
            fp16=config.FP16 and config.DEVICE == 'cuda',
        )
    )
    
    predictions = trainer.predict(test_dataset)
    
    y_pred = predictions.predictions.flatten()
    y_true = test_df['price'].values
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 0.01)
    
    smape = calculate_smape(y_true, y_pred)
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    print(f"\n📈 Test Results:")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"  RMSE:  ${rmse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"  MSE:   {mse:.2f}")
    print(f"\n  Prediction range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    print(f"  Actual range:     ${y_true.min():.2f} - ${y_true.max():.2f}")
    
    # Price band analysis
    print(f"\n📊 Price Band Performance:")
    bands = [0, 50, 100, 200, 500, np.inf]
    band_labels = ['$0-50', '$50-100', '$100-200', '$200-500', '$500+']
    
    for i, (low, high) in enumerate(zip(bands[:-1], bands[1:])):
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            band_smape = calculate_smape(y_true[mask], y_pred[mask])
            print(f"  {band_labels[i]:10s}: {band_smape:.2f}% (n={mask.sum():,})")
    
    return smape, predictions

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
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
            dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
            fp16=config.FP16 and config.DEVICE == 'cuda',
        )
    )
    
    predictions = trainer.predict(test_dataset)
    
    y_pred = predictions.predictions.flatten()
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 0.01)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred
    })
    
    output_path = os.path.join(config.BASE_DIR, 'modeling', 'test_out_distilbert_sagemaker.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved: {output_path}")
    print(f"  Samples: {len(submission):,}")
    print(f"  Price range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    print(f"  Mean price: ${y_pred.mean():.2f}")
    print(f"  Median price: ${np.median(y_pred):.2f}")
    
    return submission

# ============================================================================
# Save Results
# ============================================================================

def save_results(smape, training_history):
    """Save training results and metrics"""
    
    results = {
        'test_smape': float(smape),
        'baseline_smape': 63.28,
        'improvement': float(63.28 - smape),
        'config': {
            'model': config.MODEL_NAME,
            'max_length': config.MAX_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'gradient_accumulation_steps': config.GRADIENT_ACCUMULATION_STEPS,
            'effective_batch_size': config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'fp16': config.FP16,
        },
        'hardware': {
            'device': config.DEVICE,
            'n_gpu': config.N_GPU,
        }
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main pipeline"""
    
    print("\n" + "="*80)
    print("STARTING DISTILBERT FINE-TUNING - SAGEMAKER OPTIMIZED")
    print("="*80)
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df, final_test_df = load_and_prepare_data()
    
    # Train
    model, tokenizer, trainer = train_model(train_df, val_df)
    
    # Clear GPU cache before evaluation
    if config.DEVICE == 'cuda':
        clear_gpu_cache()
    
    # Evaluate
    smape, test_predictions = evaluate(model, tokenizer, test_df)
    
    # Generate predictions
    submission = generate_predictions(model, tokenizer, final_test_df)
    
    # Save model
    print("\n💾 Saving model...")
    model_path = os.path.join(config.OUTPUT_DIR, 'final_model')
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save results
    save_results(smape, trainer.state.log_history)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n🎯 Test SMAPE:     {smape:.2f}%")
    print(f"📊 Baseline SMAPE: 63.28%")
    
    if smape < 63.28:
        print(f"🚀 Improvement:    {63.28 - smape:.2f}% ✓")
    else:
        print(f"⚠️  Change:         +{smape - 63.28:.2f}%")
    
    print("\n📁 Output files:")
    print(f"  - Model: {model_path}")
    print(f"  - Predictions: {os.path.join(config.BASE_DIR, 'modeling', 'test_out_distilbert_sagemaker.csv')}")
    print(f"  - Results: {os.path.join(config.OUTPUT_DIR, 'training_results.json')}")
    print(f"  - TensorBoard logs: {os.path.join(config.OUTPUT_DIR, 'logs')}")
    
    print("\n💡 To view training logs, run:")
    print(f"   tensorboard --logdir {os.path.join(config.OUTPUT_DIR, 'logs')}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
