# 🚀 Try4: DeBERTa-v3-large + CLIP + Engineered Features

## 📋 Overview

**Advanced Hybrid Multi-Modal Fusion for Price Prediction**

This implementation combines three powerful feature sources:
1. **🔤 DeBERTa-v3-large**: State-of-the-art text understanding (1024-dim)
2. **🖼️ CLIP ViT-Large**: Visual semantic embeddings (768-dim)
3. **⚙️ Engineered Features**: 30-50 handcrafted features

**Total Feature Space**: ~1850-2000 dimensions

---

## 🎯 Key Features

### ✅ 1. Graceful Outlier Handling (Priority++)
- **Multi-Stage Detection**:
  - Isolation Forest for anomaly detection
  - IQR filtering on price distribution
  - Z-score based statistical outliers
  - DBSCAN for cluster-based outliers
- **Smart Treatment**:
  - Robust scaling (not affected by outliers)
  - Winsorization (cap extreme values)
  - Separate modeling for outlier segments
  - Confidence-weighted predictions

### ✅ 2. Intelligent Feature Analysis
- **Text Sufficiency Analysis**: Detect when text has enough information
- **Image Information Extraction**: Extract complementary visual signals
- **Cross-Modal Validation**: Test text predictions with visual cues
- **Adaptive Weighting**: Dynamic feature importance based on data quality

### ✅ 3. Advanced Model Architecture
- **DeBERTa-v3-large**: Fine-tuned on product descriptions
- **CLIP ViT-Large**: Pre-trained vision encoder
- **Fusion Options**:
  - LightGBM (fastest, robust)
  - CatBoost (handles categoricals well)
  - 2-3 layer MLP (deep fusion)

### ✅ 4. Comprehensive Pipeline
```
Stage 1: Fine-tune DeBERTa → Save embeddings
Stage 2: Extract CLIP features → Image embeddings
Stage 3: Engineer tabular features → 30-50 features
Stage 4: Outlier detection & treatment → Clean data
Stage 5: Feature fusion & training → Final model
Stage 6: Evaluation with SMAPE → Performance metrics
```

---

## 📁 Project Structure

```
try4/
├── config/
│   ├── config.py              # Central configuration
│   └── paths.py               # Path management
├── feature_extraction/
│   ├── deberta_embeddings.py  # DeBERTa text embeddings
│   ├── clip_embeddings.py     # CLIP image embeddings
│   └── tabular_features.py    # Engineered features
├── preprocessing/
│   ├── outlier_detection.py   # Multi-strategy outlier detection
│   ├── outlier_treatment.py   # Graceful outlier handling
│   └── data_cleaning.py       # Data validation & cleaning
├── modeling/
│   ├── fusion_model.py        # Multi-modal fusion
│   ├── lightgbm_trainer.py    # LightGBM training
│   ├── catboost_trainer.py    # CatBoost training
│   └── mlp_trainer.py         # Deep learning fusion
├── outputs/
│   ├── embeddings/            # Saved embeddings
│   ├── models/                # Trained models
│   └── predictions/           # Final predictions
├── main_pipeline.py           # Main execution script
├── analyze_features.py        # Feature analysis & validation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python main_pipeline.py
```

### 3. Run Individual Stages
```bash
# Stage 1: Extract DeBERTa embeddings
python feature_extraction/deberta_embeddings.py

# Stage 2: Extract CLIP embeddings
python feature_extraction/clip_embeddings.py

# Stage 3: Engineer features
python feature_extraction/tabular_features.py

# Stage 4: Detect & treat outliers
python preprocessing/outlier_detection.py
python preprocessing/outlier_treatment.py

# Stage 5: Train fusion model
python modeling/fusion_model.py
```

---

## 🔧 Configuration

Edit `config/config.py` to customize:
- Model paths and versions
- Feature dimensions
- Outlier detection thresholds
- Training hyperparameters
- Evaluation metrics

---

## 📊 Expected Performance

| Metric | Baseline (try3) | Target (try4) | Improvement |
|--------|-----------------|---------------|-------------|
| SMAPE  | ~42%           | **<35%**      | ~7-10%      |
| Outlier Handling | Basic | **Advanced** | Robust |
| Feature Fusion | Text only | **Multi-modal** | Rich |

---

## 🎓 Advanced Features

### Outlier Detection Strategies
1. **Isolation Forest**: Detects anomalies in high-dimensional space
2. **IQR Method**: Statistical outlier detection (1.5×IQR)
3. **Z-Score**: Standard deviation based (|z| > 3)
4. **DBSCAN**: Density-based clustering outliers

### Outlier Treatment Methods
1. **Winsorization**: Cap at 1st/99th percentile
2. **Robust Scaling**: Min-max with quantile ranges
3. **Log Transform**: Compress extreme values
4. **Separate Modeling**: Train dedicated models for outliers

### Feature Analysis
- **Text Sufficiency Score**: How much info is in text
- **Image Information Score**: Visual signal strength
- **Cross-Modal Consistency**: Agreement between modalities
- **Adaptive Weights**: Dynamic feature importance

---

## 📝 Notes

- **GPU Required**: DeBERTa-v3-large and CLIP require GPU
- **Memory**: ~16GB RAM recommended
- **Time**: Full pipeline ~2-4 hours on GPU
- **Incremental**: Can run stages separately and resume

---

## 🤝 Contributing

This is an experimental research implementation for the Amazon ML Challenge.
Feel free to modify and improve!

---

## 📄 License

MIT License - Free to use and modify

---

**Happy Modeling! 🎉**
