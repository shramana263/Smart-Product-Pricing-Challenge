# 🚀 Try4 Quick Start Guide

## Overview

This is an advanced multi-modal fusion system combining:
- **DeBERTa-v3-large** (1024-dim text embeddings)
- **CLIP ViT-Large** (768-dim image embeddings)
- **Engineered Features** (~40 handcrafted features)

Total: ~1850 features with graceful outlier handling!

---

## ⚡ Quick Start (3 Options)

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd try4
python main_pipeline.py
```

This runs all 6 stages:
1. DeBERTa embedding extraction
2. CLIP embedding extraction
3. Tabular feature engineering
4. Outlier detection
5. Outlier treatment
6. Fusion model training

**Time:** ~2-4 hours on GPU

---

### Option 2: Run Individual Stages

```bash
# Stage 1: Text embeddings (DeBERTa)
python feature_extraction/deberta_embeddings.py

# Stage 2: Image embeddings (CLIP)
python feature_extraction/clip_embeddings.py

# Stage 3: Tabular features
python feature_extraction/tabular_features.py

# Stage 4: Detect outliers
python preprocessing/outlier_detection.py

# Stage 5: Treat outliers
python preprocessing/outlier_treatment.py

# Stage 6: Train model
python modeling/fusion_model.py
```

**Benefit:** Can resume from any stage if interrupted.

---

### Option 3: Feature Analysis First

Analyze your data before training:

```bash
python analyze_features.py
```

This shows:
- Text sufficiency scores
- Image information quality
- Cross-modal consistency
- Adaptive feature weights

---

## 📋 Prerequisites

### Install Dependencies
```bash
pip install -r requirements.txt
```

### GPU Recommended
- DeBERTa-v3-large: ~3.5GB VRAM
- CLIP ViT-Large: ~2GB VRAM
- Training: ~8GB VRAM recommended

**CPU Mode:** Will work but ~10x slower.

---

## 📊 Expected Results

| Metric | Baseline (try3) | Target (try4) |
|--------|-----------------|---------------|
| SMAPE  | ~42%           | **<35%**      |
| Outlier SMAPE | ~140% | **<70%**      |
| Budget ($0-10) | 122% | **<40%**      |

---

## 🔍 Key Features

### 1. Graceful Outlier Handling ⭐⭐⭐
- **4 Detection Methods:** Isolation Forest, IQR, Z-Score, DBSCAN
- **Ensemble Voting:** Sample is outlier if 2+ methods agree
- **5 Treatment Strategies:**
  - Winsorization (cap extremes)
  - Robust scaling (quantile-based)
  - Log transform (compress values)
  - Separate modeling (dedicated models)
  - Confidence weighting (weighted predictions)

### 2. Intelligent Feature Analysis
- **Text Sufficiency:** Does text have enough info?
- **Image Quality:** Visual signal strength
- **Cross-Modal Consistency:** Do text/image agree?
- **Adaptive Weights:** Dynamic feature importance

### 3. Multi-Modal Fusion
- **DeBERTa:** Semantic understanding from text
- **CLIP:** Visual patterns from images
- **Tabular:** Domain knowledge features

---

## 📁 Output Structure

```
try4/outputs/
├── embeddings/
│   ├── deberta_train_embeddings.csv  # 1024 dims
│   ├── deberta_test_embeddings.csv
│   ├── clip_train_embeddings.csv     # 768 dims
│   └── clip_test_embeddings.csv
├── features/
│   ├── tabular_train_features.csv    # ~40 features
│   ├── tabular_test_features.csv
│   └── train_features_treated.csv    # After outlier treatment
├── analysis/
│   ├── outlier_detection_results.csv # Outlier flags & scores
│   └── feature_analysis_results.csv  # Quality scores
├── models/
│   ├── lightgbm_fold1.txt           # Saved models
│   ├── lightgbm_fold2.txt
│   └── ...
└── predictions/
    ├── oof_predictions.csv          # Out-of-fold
    ├── test_predictions.csv         # Final submission
    └── feature_importance.csv       # Feature analysis
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size in `config/config.py`:
```python
TEXT_MODEL = {
    'batch_size': 8,  # Reduce from 16
    ...
}
```

**Solution 2:** Use CPU:
```python
DEVICE = 'cpu'
USE_FP16 = False
```

### Issue: Download Timeout for Images

**Solution:** Increase timeout in `config/config.py`:
```python
IMAGE_MODEL = {
    'download_timeout': 20,  # Increase from 10
    'max_retries': 5,        # Increase from 3
    ...
}
```

### Issue: Embeddings Not Found

**Solution:** Run feature extraction first:
```bash
python feature_extraction/deberta_embeddings.py
python feature_extraction/clip_embeddings.py
python feature_extraction/tabular_features.py
```

---

## 🎯 Performance Tips

### 1. Skip Completed Stages
The pipeline automatically skips stages with existing outputs:
```bash
# Will skip stages 1-5 if outputs exist
python main_pipeline.py
```

### 2. Use Multiple GPUs (Advanced)
Edit config to distribute work:
```python
DEVICE = 'cuda:0'  # For stage 1
# Then manually change to 'cuda:1' for stage 2
```

### 3. Reduce Features for Speed
In `modeling/fusion_model.py`, use PCA:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=500)  # Reduce from ~1850
X_train = pca.fit_transform(X_train)
```

---

## 📈 Monitoring Progress

### View Training Logs
```bash
# Watch pipeline log
tail -f outputs/pipeline.log
```

### Check Intermediate Results
```python
import pandas as pd

# Check embeddings
deberta = pd.read_csv('outputs/embeddings/deberta_train_embeddings.csv')
print(f"DeBERTa shape: {deberta.shape}")

# Check outliers
outliers = pd.read_csv('outputs/analysis/outlier_detection_results.csv')
print(f"Outliers detected: {outliers['is_outlier'].sum()}")

# Check predictions
preds = pd.read_csv('outputs/predictions/oof_predictions.csv')
print(f"OOF SMAPE: {calculate_smape(preds['price_true'], preds['price_pred']):.3f}%")
```

---

## 🔄 Re-running After Changes

### Rerun Single Stage
```bash
# Force rerun by deleting output
rm outputs/embeddings/deberta_train_embeddings.csv
python feature_extraction/deberta_embeddings.py
```

### Rerun from Stage N
```bash
# Delete all outputs after stage N
rm outputs/analysis/*
rm outputs/models/*
rm outputs/predictions/*
python main_pipeline.py  # Will skip stages 1-3
```

---

## 🎓 Advanced Usage

### Custom Outlier Detection
Edit `preprocessing/outlier_detection.py`:
```python
OUTLIER_CONFIG = {
    'ensemble_threshold': 3,  # Require 3 methods (stricter)
    'isolation_forest': {
        'contamination': 0.03,  # Reduce from 0.05
    },
}
```

### Custom Feature Engineering
Edit `feature_extraction/tabular_features.py`:
```python
# Add your own features
df['custom_feature'] = df['price'] / df['quantity']
feature_cols.append('custom_feature')
```

### Try CatBoost Instead of LightGBM
```bash
python modeling/catboost_trainer.py  # If you create this
```

---

## 📚 Next Steps

After successful training:

1. **Analyze Results:**
   ```bash
   python analyze_features.py
   ```

2. **Check Feature Importance:**
   ```python
   import pandas as pd
   imp = pd.read_csv('outputs/predictions/feature_importance.csv')
   print(imp.head(20))
   ```

3. **Submit Predictions:**
   ```bash
   # Your test_predictions.csv is ready!
   ls outputs/predictions/test_predictions.csv
   ```

4. **Experiment:**
   - Try different outlier thresholds
   - Add more engineered features
   - Ensemble with other models
   - Fine-tune hyperparameters

---

## ❓ FAQ

**Q: How long does it take?**
A: ~2-4 hours on GPU, ~20-30 hours on CPU.

**Q: Can I use fewer features?**
A: Yes! Edit config to skip CLIP or use PCA reduction.

**Q: What if I don't have images?**
A: The model will use zero vectors for CLIP, still works!

**Q: How to improve further?**
A: Try ensemble with XGBoost, add more text preprocessing, use larger models.

**Q: Can I deploy this?**
A: Yes! Save models with joblib, load for inference.

---

## 🤝 Support

Check these files for details:
- `README.md` - Project overview
- `config/config.py` - All settings
- `*.py` - Each script has detailed docstrings

---

**Good luck! 🎉**
