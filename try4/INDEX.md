# 📚 Try4 Documentation Index

Welcome to Try4 - Advanced Multi-Modal Fusion for Price Prediction!

---

## 🚀 Quick Navigation

### Getting Started
1. **[README.md](README.md)** - Start here! Project overview and features
2. **[QUICK_START.md](QUICK_START.md)** - Quick start guide with 3 options
3. **[check_system.py](check_system.py)** - Run this first to validate setup

### Understanding the System
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual architecture diagrams and data flow
5. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation details
6. **[COMPARISON.md](COMPARISON.md)** - How Try4 compares to previous approaches

### Running the Pipeline
7. **[main_pipeline.py](main_pipeline.py)** - Main orchestrator (run this!)
8. **[analyze_features.py](analyze_features.py)** - Feature quality analysis tool

---

## 📖 Documentation by Topic

### Configuration
- **[config/config.py](config/config.py)** - Central configuration file
  - Model settings (DeBERTa, CLIP)
  - Outlier detection parameters
  - Training hyperparameters
  - All paths and constants

### Feature Extraction
- **[feature_extraction/deberta_embeddings.py](feature_extraction/deberta_embeddings.py)**
  - DeBERTa-v3-large text embeddings (1024-dim)
  - Fine-tuning with regression head
  - 5-fold cross-validation
  
- **[feature_extraction/clip_embeddings.py](feature_extraction/clip_embeddings.py)**
  - CLIP ViT-Large image embeddings (768-dim)
  - Robust image downloading
  - Missing image handling
  
- **[feature_extraction/tabular_features.py](feature_extraction/tabular_features.py)**
  - Text statistics, brand/category encoding
  - Quantity extraction and normalization
  - Safe target encoding with K-Fold CV

### Outlier Handling ⭐ (Priority)
- **[preprocessing/outlier_detection.py](preprocessing/outlier_detection.py)**
  - Multi-strategy detection (4 methods)
  - Ensemble voting
  - Confidence scoring
  
- **[preprocessing/outlier_treatment.py](preprocessing/outlier_treatment.py)**
  - 5 treatment strategies
  - Winsorization, robust scaling
  - Separate modeling, confidence weighting

### Modeling
- **[modeling/fusion_model.py](modeling/fusion_model.py)**
  - Multi-modal feature fusion
  - LightGBM training with outlier awareness
  - 5-fold cross-validation
  - Feature importance analysis

---

## 🎯 By Use Case

### "I want to run the full pipeline"
1. Run `python check_system.py` to validate setup
2. Run `python main_pipeline.py` to execute all stages
3. Check `outputs/predictions/test_predictions.csv` for results

### "I want to understand outlier handling"
1. Read **[ARCHITECTURE.md](ARCHITECTURE.md)** - Section on outlier handling
2. Review **[preprocessing/outlier_detection.py](preprocessing/outlier_detection.py)**
3. Review **[preprocessing/outlier_treatment.py](preprocessing/outlier_treatment.py)**
4. See **[COMPARISON.md](COMPARISON.md)** for performance impact

### "I want to analyze features"
1. Run `python analyze_features.py`
2. Check `outputs/analysis/feature_analysis_results.csv`
3. See text sufficiency, image quality, cross-modal consistency

### "I want to customize the pipeline"
1. Edit **[config/config.py](config/config.py)** for parameters
2. Modify individual stage scripts for custom logic
3. Run stages independently or full pipeline

### "I need to troubleshoot issues"
1. Check **[QUICK_START.md](QUICK_START.md)** - Troubleshooting section
2. Review `outputs/pipeline.log` for errors
3. Run `python check_system.py` to diagnose

---

## 📊 By Component

### Text Processing Pipeline
```
catalog_content 
  → deberta_embeddings.py 
  → deberta_train_embeddings.csv (1024 cols)
```

### Image Processing Pipeline
```
image_link 
  → clip_embeddings.py 
  → clip_train_embeddings.csv (768 cols)
```

### Tabular Feature Pipeline
```
raw data 
  → tabular_features.py 
  → tabular_train_features.csv (~40 cols)
```

### Outlier Pipeline
```
features 
  → outlier_detection.py 
  → outlier_detection_results.csv
  → outlier_treatment.py 
  → train_features_treated.csv
```

### Training Pipeline
```
all features 
  → fusion_model.py 
  → models (5 folds) 
  → test_predictions.csv
```

---

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `config/config.py` | Central configuration |
| `requirements.txt` | Python dependencies |

---

## 📁 Output Files

| Directory | Contents |
|-----------|----------|
| `outputs/embeddings/` | DeBERTa and CLIP embeddings |
| `outputs/features/` | Tabular and treated features |
| `outputs/analysis/` | Outlier detection and feature analysis |
| `outputs/models/` | Trained LightGBM models |
| `outputs/predictions/` | OOF and test predictions |

---

## 🎓 Learning Path

### Beginner
1. Start with **[README.md](README.md)**
2. Read **[QUICK_START.md](QUICK_START.md)**
3. Run `python check_system.py`
4. Run `python main_pipeline.py`

### Intermediate
1. Read **[ARCHITECTURE.md](ARCHITECTURE.md)**
2. Review individual stage scripts
3. Experiment with `config/config.py`
4. Run `python analyze_features.py`

### Advanced
1. Read **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
2. Study outlier handling implementation
3. Customize feature extraction
4. Implement custom models (CatBoost, MLP)
5. Read **[COMPARISON.md](COMPARISON.md)** for insights

---

## 🔍 Search Guide

### Find Code Examples
- **Text embeddings:** `feature_extraction/deberta_embeddings.py`
- **Image embeddings:** `feature_extraction/clip_embeddings.py`
- **Feature engineering:** `feature_extraction/tabular_features.py`
- **Outlier detection:** `preprocessing/outlier_detection.py`
- **Outlier treatment:** `preprocessing/outlier_treatment.py`
- **Model training:** `modeling/fusion_model.py`

### Find Configuration
- **All settings:** `config/config.py`
- **Model configs:** `TEXT_MODEL`, `IMAGE_MODEL` in config
- **Outlier configs:** `OUTLIER_CONFIG`, `TREATMENT_CONFIG` in config
- **Training configs:** `LIGHTGBM_CONFIG`, `CV_CONFIG` in config

### Find Documentation
- **Architecture:** `ARCHITECTURE.md`
- **Implementation:** `IMPLEMENTATION_SUMMARY.md`
- **Comparison:** `COMPARISON.md`
- **Quick start:** `QUICK_START.md`

---

## ❓ FAQ Quick Links

**Q: How do I run the pipeline?**
→ See [QUICK_START.md](QUICK_START.md) - Section "Quick Start"

**Q: What are the system requirements?**
→ See [QUICK_START.md](QUICK_START.md) - Section "Prerequisites"

**Q: How does outlier handling work?**
→ See [ARCHITECTURE.md](ARCHITECTURE.md) - Section "Outlier Handling Pipeline"

**Q: How to customize features?**
→ Edit [feature_extraction/tabular_features.py](feature_extraction/tabular_features.py)

**Q: How to change model parameters?**
→ Edit [config/config.py](config/config.py)

**Q: What's the expected performance?**
→ See [COMPARISON.md](COMPARISON.md) - Section "Performance Progression"

**Q: How does Try4 compare to Try3?**
→ See [COMPARISON.md](COMPARISON.md)

**Q: What if I encounter errors?**
→ See [QUICK_START.md](QUICK_START.md) - Section "Troubleshooting"

---

## 📞 Support Resources

1. **System Check:** Run `python check_system.py`
2. **Logs:** Check `outputs/pipeline.log`
3. **Documentation:** This index + all linked files
4. **Code Comments:** All scripts have detailed docstrings

---

## 🎯 Common Workflows

### First-Time Setup
```bash
python check_system.py       # Validate system
python main_pipeline.py      # Run full pipeline
```

### Resume After Interruption
```bash
python main_pipeline.py      # Automatically skips completed stages
```

### Analyze Existing Results
```bash
python analyze_features.py   # Feature quality analysis
```

### Custom Feature Engineering
```bash
# Edit feature_extraction/tabular_features.py
python feature_extraction/tabular_features.py
python modeling/fusion_model.py
```

### Experiment with Outlier Thresholds
```bash
# Edit config/config.py - OUTLIER_CONFIG section
python preprocessing/outlier_detection.py
python preprocessing/outlier_treatment.py
python modeling/fusion_model.py
```

---

## 📈 Performance Monitoring

**Check OOF SMAPE:**
```python
import pandas as pd
oof = pd.read_csv('outputs/predictions/oof_predictions.csv')
# Calculate SMAPE from oof['price_true'] and oof['price_pred']
```

**Check Feature Importance:**
```python
import pandas as pd
imp = pd.read_csv('outputs/predictions/feature_importance.csv')
print(imp.head(20))
```

**Check Outlier Statistics:**
```python
import pandas as pd
outliers = pd.read_csv('outputs/analysis/outlier_detection_results.csv')
print(f"Outliers: {outliers['is_outlier'].sum()}")
```

---

## 🚀 Next Steps After Reading

1. ✅ Run `python check_system.py`
2. ✅ Read [QUICK_START.md](QUICK_START.md)
3. ✅ Run `python main_pipeline.py`
4. ✅ Analyze results
5. ✅ Submit predictions!

---

**Happy Modeling! 🎉**

For any questions, refer to the specific documentation files linked above.
