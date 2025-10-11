# 🤖 DistilBERT Fine-Tuning Guide

## Overview

This approach fine-tunes DistilBERT (a lightweight BERT variant) on product catalog text to predict prices directly.

**Expected Performance:** 50-55% SMAPE (vs. 63.28% baseline)

---

## Why DistilBERT?

### Advantages
✅ **Smaller & Faster:** 66M parameters vs. BERT's 110M  
✅ **40% faster** training/inference  
✅ **End-to-End:** No manual feature engineering needed  
✅ **Context-Aware:** Understands product descriptions semantically  
✅ **Transfer Learning:** Leverages pre-trained language understanding

### Comparison with Current Approach

| Method | Features | SMAPE | Training Time |
|--------|----------|-------|---------------|
| V3 Clean (Current) | 47 hand-crafted | 63.28% | ~10 min |
| DistilBERT | None (text only) | 50-55% (est.) | 2-3 hours |
| DistilBERT + V3 | 47 + text embeddings | <50% (est.) | 3-4 hours |

---

## Quick Start

### 1. Setup Virtual Environment (Already Done!)

```powershell
# Environment already created: venv_distilbert
# Location: C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert
```

### 2. Activate Environment

```powershell
.\venv_distilbert\Scripts\Activate.ps1
```

### 3. Verify Installation

```powershell
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 4. Run Training (Simple Version - Recommended)

```powershell
cd try2
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe finetune_distilbert_simple.py
```

**Expected Runtime:**
- CPU: 4-6 hours
- GPU: 1-2 hours

### 5. Monitor Training (Optional)

Open another terminal:
```powershell
cd try2\modeling\distilbert_simple\logs
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe -m tensorboard.main --logdir=.
```

Then open: http://localhost:6006

---

## Files Created

### Scripts
- `try2/finetune_distilbert_simple.py` - **Recommended:** Simplified version
- `try2/finetune_distilbert.py` - Advanced version with custom model

### Outputs
- `try2/modeling/distilbert_simple/` - Model checkpoints
- `try2/modeling/distilbert_simple/final_model/` - Best model
- `try2/modeling/test_out_distilbert_simple.csv` - Predictions

---

## How It Works

### 1. Data Preparation
- Loads 75K training samples
- Cleans `catalog_content` (removes newlines, HTML)
- Splits: 60% train / 15% val / 25% test (stratified by price)

### 2. Tokenization
```python
# Example
Input:  "Item Name: Pillsbury Brownie Mix, 15.5oz (Pack of 12)"
Tokens: [CLS] item name pillsbury brownie mix 15.5 oz pack of 12 [SEP]
IDs:    [101, 3174, 2171, 8069, ...]
```

### 3. Model Architecture
```
Input Text → DistilBERT Encoder → [CLS] Token → Linear Layer → Price
              (6 transformer layers)   (768 dims)     (1 dim)
```

### 4. Training
- **Loss:** MSE (Mean Squared Error)
- **Optimizer:** AdamW with weight decay
- **Learning Rate:** 2e-5 with warmup
- **Batch Size:** 32
- **Epochs:** 3 (with early stopping)

### 5. Evaluation
- **Metric:** SMAPE (matches competition metric)
- **Validation:** Continuous monitoring on 15% val set
- **Early Stopping:** Stops if val loss doesn't improve for 3 checkpoints

---

## Configuration Options

Edit `finetune_distilbert_simple.py` to customize:

```python
class Config:
    # Model
    MODEL_NAME = 'distilbert-base-uncased'  # Try: 'microsoft/mpnet-base', 'roberta-base'
    MAX_LENGTH = 128  # Increase to 256 for longer descriptions
    
    # Training
    BATCH_SIZE = 32  # Reduce to 16 if out of memory
    LEARNING_RATE = 2e-5  # Try: 1e-5, 3e-5
    NUM_EPOCHS = 3  # Increase to 5 for better performance
```

---

## Alternative Models

### If DistilBERT doesn't work well:

1. **MPNet (Better Context Understanding)**
   ```python
   MODEL_NAME = 'microsoft/mpnet-base'
   ```
   - Best for semantic similarity
   - Slightly slower than DistilBERT

2. **RoBERTa (More Robust)**
   ```python
   MODEL_NAME = 'roberta-base'
   ```
   - Better on complex text
   - 125M parameters

3. **DeBERTa (State-of-the-art)**
   ```python
   MODEL_NAME = 'microsoft/deberta-v3-base'
   ```
   - Best performance
   - Slowest training

---

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Reduce sequence length
MAX_LENGTH = 64  # or 96
```

### Training Too Slow
```python
# Use smaller model
MODEL_NAME = 'distilbert-base-uncased'  # Already smallest

# Reduce max length
MAX_LENGTH = 64

# Reduce training data (for testing)
train_split = train_split.sample(frac=0.1)  # Use 10% of data
```

### Poor Performance
```python
# Increase epochs
NUM_EPOCHS = 5

# Increase max length (capture more context)
MAX_LENGTH = 256

# Try different model
MODEL_NAME = 'microsoft/mpnet-base'

# Add learning rate scheduler
# (already included in TrainingArguments)
```

---

## Expected Results

### Baseline Comparison
| Model | Features | SMAPE | Notes |
|-------|----------|-------|-------|
| V3 Clean | 47 hand-crafted | 63.28% | Current best |
| DistilBERT | Text only | 50-55% | Expected |
| DistilBERT + V3 | Combined | <50% | Next step |

### Price Distribution Validation
After training, check if predictions match training distribution:
- Training range: $0.13 - $2,796
- Predictions should be similar
- No extreme outliers (<$0 or >$10,000)

---

## Next Steps After DistilBERT

### 1. Ensemble DistilBERT + V3 Features
Combine transformer predictions with hand-crafted features:
```python
final_pred = 0.6 * distilbert_pred + 0.4 * xgboost_pred
```

### 2. Add Image Features
Extract ResNet/EfficientNet embeddings from product images:
- Expected: Additional 5-10% improvement

### 3. Multi-Task Learning
Train on related tasks:
- Brand classification
- Category prediction
- Price range classification

---

## Performance Monitoring

During training, watch for:

✅ **Good Signs:**
- Val loss decreasing steadily
- SMAPE decreasing on validation set
- Predictions in reasonable range ($0.10 - $3,000)

⚠️ **Warning Signs:**
- Val loss increasing (overfitting)
- SMAPE stuck or increasing
- Predictions all similar (underfitting)

---

## Files & Paths

```
Smart-Product-Pricing/
├── venv_distilbert/                    ← Virtual environment
├── requirements_distilbert.txt         ← Dependencies
├── DISTILBERT_GUIDE.md                 ← This file
│
└── try2/
    ├── dataset/                         ← Raw data
    │   ├── train1.csv, train2.csv
    │   └── test1.csv, test2.csv
    │
    ├── finetune_distilbert_simple.py   ← Main script
    ├── finetune_distilbert.py          ← Advanced version
    │
    └── modeling/
        ├── distilbert_simple/           ← Outputs
        │   ├── checkpoint-XXX/          ← Training checkpoints
        │   ├── final_model/             ← Best model
        │   └── logs/                    ← TensorBoard logs
        │
        └── test_out_distilbert_simple.csv  ← Predictions
```

---

## FAQ

**Q: How long will training take?**  
A: 1-2 hours with GPU, 4-6 hours with CPU

**Q: Do I need a GPU?**  
A: No, but highly recommended. CPU training is 3-5x slower.

**Q: Can I stop and resume training?**  
A: Yes! Checkpoints are saved every 200 steps. Use `resume_from_checkpoint` parameter.

**Q: What if I run out of disk space?**  
A: Model checkpoints are ~250MB each. Set `save_total_limit=2` to keep only 2 checkpoints.

**Q: How do I know if it's working?**  
A: Check validation SMAPE in terminal output. Should decrease from ~80% → 50-55%.

---

## Contact & Support

For issues:
1. Check error messages in terminal
2. Verify all paths are correct
3. Ensure data files exist in `try2/dataset/`
4. Check Python version: `python --version` (should be 3.11.x)

---

**Good luck! 🚀**

Expected improvement: **63.28% → 50-55% SMAPE**  
Training time: **1-2 hours (GPU) / 4-6 hours (CPU)**
