# 📊 Smart Product Pricing Challenge - Solution Documentation

**Team/Participant:** Smart Pricing Team  
**Date:** October 13, 2025  
**Final SMAPE:** **57.834%** (Cross-Validation)  
**Baseline SMAPE:** 63.28%  
**Improvement:** **5.45 points (8.6% reduction)** ✅

---

## 1. Executive Summary

We developed a **multimodal ensemble system** combining **DistilBERT text embeddings (768-dim)** with **ResNet50 image features (2048-dim)**, both trained with **LightGBM gradient boosting** on carefully selected features. Our approach achieves **57.834% SMAPE** on 5-fold cross-validation through text understanding, visual analysis, and optimized ensemble weighting.

### Key Achievements
- ✅ **57.834% SMAPE** - Optimized ensemble performance
- ✅ **1.075 points improvement** over text-only model (58.909% → 57.834%)
- ✅ **Multimodal learning** - Text (63.5%) + Image (36.5%)
- ✅ **No data leakage** - Excluded high-correlation features
- ✅ **Production-ready** - Memory-efficient batch processing, cached embeddings

---

## 2. Methodology Overview

### 2.1 Approach Selection

We adopted a **multimodal ensemble approach** combining:

1. **Text Understanding:** DistilBERT embeddings for semantic patterns
2. **Visual Analysis:** ResNet50 features for image understanding
3. **Feature Engineering:** Safe engineered features (units, text stats, signals)
4. **Ensemble Learning:** Optimized weighted combination (63.5% text + 36.5% image)

**Why This Works:**
- DistilBERT captures deep semantic patterns in product descriptions
- ResNet50 extracts visual features complementary to text
- Engineered features provide explicit signals without data leakage
- Optimized ensemble weights maximize predictive performance
- Dual modalities capture different aspects of product pricing

### 2.2 Pipeline Architecture

```
Raw Product Data (text + image_link)
    ↓
┌────────────────────────────────┬────────────────────────────────┐
│   TEXT PIPELINE                │   IMAGE PIPELINE               │
│                                │                                │
│  DistilBERT Tokenization       │  Image Download                │
│  (max_length=256)              │  (batch=500, retry logic)      │
│         ↓                      │         ↓                      │
│  DistilBERT Embeddings         │  ResNet50 Feature Extract      │
│  [CLS] → 768 dimensions        │  (batch=100) → 2048 dim        │
│         ↓                      │         ↓                      │
│  + Safe Engineered Features    │  LightGBM (Image Only)         │
│  (12 features, NO leakage)     │  Regularized training          │
│         ↓                      │         ↓                      │
│  LightGBM (Text + Features)    │  Image Predictions             │
│  780 total features            │  (60.482% OOF SMAPE)           │
│         ↓                      │                                │
│  Text Predictions              │                                │
│  (58.909% OOF SMAPE)           │                                │
└────────────────────────────────┴────────────────────────────────┘
                    ↓
         Optimized Ensemble (scipy.optimize)
         Weight: 63.5% text + 36.5% image
                    ↓
         Final Predictions (57.834% OOF SMAPE)
```

---

## 3. Data Analysis & Preprocessing

### 3.1 Dataset Characteristics
- **Training:** 75,000 products
- **Test:** 75,000 products
- **Price Range:** $0.13 - $2,796.00
- **Price Distribution:** Highly right-skewed (skewness: 13.60, kurtosis: 736.65)

### 3.2 Key Insights from Research
1. **Unit Variations:** 156 unit variations (e.g., "Ounce" → "oz", "fl oz" → "oz")
2. **Bulk Quantities:** 12.2% of products have bulk multipliers (e.g., "Pack of 12")
3. **Brand Hierarchy:** 
   - Budget brands: avg $8.98 (2.3%)
   - Mid-tier: avg $23.49 (95.7%)
   - Premium: avg $39.19 (1.6%)
   - Luxury: avg $78.92 (0.4%)
4. **Category Distribution:**
   - Food & Beverage: 51% (avg $25.51)
   - Home & Garden: 10% (avg $23.55)
   - Health & Beauty: 7% (avg $22.04)

### 3.3 Preprocessing Steps
1. **Text Cleaning:** Minimal (preserve product language)
2. **Missing Values:** Filled with 0 (numeric) or 'unknown' (categorical)
3. **Stratified Folding:** 5 folds stratified by price bins (10 quantiles)
4. **No Data Leakage:** Verified no target correlation >0.95

---

## 4. Feature Engineering

### 4.1 Text Features (DistilBERT + Safe Features)

**DistilBERT Embeddings:**
- 768-dimensional [CLS] token embeddings
- Captures semantic meaning of product descriptions
- Pre-trained on English Wikipedia + BookCorpus

**Safe Engineered Features (12 features):**

1. **Unit Features (3):**
   - `qty` - Primary quantity extracted from text
   - `total_qty` - Nested quantity (e.g., "24 pack of 6" = 144)
   - `multiplier` - Bulk factor (total_qty / qty)

2. **Signal Features (3):**
   - `premium_count` - Count of premium keywords
   - `budget_count` - Count of budget keywords
   - `premium_signal` - Net premium/budget signal

3. **Text Statistics (3):**
   - `text_char_count` - Number of characters
   - `text_word_count` - Number of words
   - `text_unique_word_ratio` - Unique words / total words

4. **Categorical Features (3 + frequency encoding):**
   - `unit_category` - Broader unit category (weight/volume/count)
   - `brand_tier` - Brand classification (budget/mid/premium/luxury)
   - `category` - Inferred product category

**Excluded Features (Data Leakage Risk):**
- ❌ `price_per_unit` - 0.925 correlation (TOO HIGH)
- ❌ `price_per_char` - 0.503 correlation (derived from price patterns)

**Total Text Model Features:** 780
- 768 DistilBERT embeddings
- 12 safe engineered features

### 4.2 Image Features (ResNet50)

**Model:** ResNet50 (pretrained on ImageNet)
- 2048-dimensional feature vectors
- Extracted from final pooling layer
- Batch processing (100 images) for memory efficiency

**Image Pipeline:**
- Downloaded: 149,998 / 150,000 images (99.99% success)
- Valid images: 149,998 (2 filled with mean features)
- Processing time: ~13 minutes (feature extraction)
- Memory usage: ~8GB peak (safe for 16GB RAM)

**Visual Features Captured:**
- Product appearance and packaging
- Color, texture, material cues
- Size and quantity visual indicators
- Brand and premium visual signals

---

## 5. Model Architecture & Configuration

### 5.1 Text Model: DistilBERT + LightGBM

**Stage 1: DistilBERT Embeddings**

**Model:** `distilbert-base-uncased`
- Parameters: ~66 million (within 8B constraint)
- Architecture: 6 transformer layers, 768 hidden dimensions
- Pre-trained on: English Wikipedia + BookCorpus
- License: Apache 2.0 ✅

**Configuration:**
```python
Model: distilbert-base-uncased
Max Sequence Length: 256 tokens
Batch Size: 32
Device: CUDA (NVIDIA T4 GPU)
Output: [CLS] token embedding (768-dim)
```

**Stage 2: LightGBM (Text Model)**

**Configuration:**
```python
Parameters:
  objective: 'regression'
  metric: 'rmse'
  num_leaves: 20         # Conservative for generalization
  learning_rate: 0.03    # Moderate learning rate
  feature_fraction: 0.75 # Regularization
  bagging_fraction: 0.75
  bagging_freq: 5
  min_child_samples: 30
  max_depth: 6
  random_state: 42
  
Input Features: 780
  - 768 DistilBERT embeddings
  - 12 safe engineered features
```

**Cross-Validation:** 5-Fold Stratified K-Fold
**Performance:** 58.909% OOF SMAPE

### 5.2 Image Model: ResNet50 + LightGBM

**Stage 1: ResNet50 Features**

**Model:** ResNet50 (pretrained on ImageNet)
- Architecture: 50-layer deep residual network
- Output: 2048-dimensional feature vector
- Pre-trained on: ImageNet (1.2M images, 1000 classes)
- License: BSD ✅

**Configuration:**
```python
Model: resnet50 (torchvision)
Preprocessing: ImageNet normalization
Batch Size: 100 (memory-efficient)
Device: CUDA
Output: Global average pooling (2048-dim)
```

**Stage 2: LightGBM (Image Model)**

**Configuration:**
```python
Parameters:
  objective: 'regression'
  metric: 'rmse'
  num_leaves: 20
  learning_rate: 0.03
  feature_fraction: 0.75
  bagging_fraction: 0.75
  bagging_freq: 5
  min_child_samples: 30
  max_depth: 6
  random_state: 42
  
Input Features: 2048 (ResNet50 features only)
```

**Cross-Validation:** 5-Fold Stratified K-Fold
**Performance:** 60.482% OOF SMAPE

### 5.3 Ensemble Configuration

**Method:** Optimized weighted average (scipy.optimize)

**Optimization Process:**
- Minimize OOF SMAPE on validation set
- Constrained: weights sum to 1.0, both ≥ 0
- Optimizer: scipy.optimize.minimize (SLSQP method)

**Optimal Weights:**
- Text Model: 0.635 (63.5%)
- Image Model: 0.365 (36.5%)

**Ensemble Performance:** 57.834% OOF SMAPE
- Improvement over text: 1.075 points
- Improvement over image: 2.648 points

---

## 6. Training Process & Results

### 6.1 Training Pipeline

**Phase 1: Image Download (228 minutes)**
- Downloaded 149,998 / 150,000 images (99.99% success)
- Batch size: 500 images
- Retry logic: 3 attempts with 2-second delays
- Memory efficient: ~2GB peak

**Phase 2: Image Feature Extraction (15 minutes)**
- Extracted 2048-dim features from ResNet50
- Batch size: 100 images (memory-efficient)
- Handled missing images: Mean feature imputation (2 images)
- Memory usage: ~8GB peak

**Phase 3: Model Training**

**Text Model Training:**
- Embeddings: ~40 minutes (one-time, cached)
- LightGBM: ~25 minutes (5-fold CV)
- Total: ~65 minutes

**Image Model Training:**
- LightGBM: ~25 minutes (5-fold CV)
- Total: ~25 minutes (features pre-extracted)

**Phase 4: Ensemble Optimization (<1 minute)**
- Scipy optimize on OOF predictions
- Found optimal weights: 63.5% text + 36.5% image

### 6.2 Model Performance

**Individual Models:**

| Model | OOF SMAPE | Features | Notes |
|-------|-----------|----------|-------|
| Text (DistilBERT + LightGBM) | 58.909% | 780 | Safe features, no leakage |
| Image (ResNet50 + LightGBM) | 60.482% | 2048 | Visual features only |

**Ensemble Performance:**

| Metric | Value | Notes |
|--------|-------|-------|
| **Ensemble OOF SMAPE** | **57.834%** | Optimized weights |
| Text Weight | 63.5% | Primary predictor |
| Image Weight | 36.5% | Complementary info |
| Improvement over Text | 1.075 points | Image helps! |
| Improvement over Image | 2.648 points | Text is stronger |

**Fold-by-Fold Comparison:**

```
           Text Model  Image Model  Ensemble
Fold 1:    58.912%     61.386%      57.841%
Fold 2:    58.906%     60.200%      57.827%
Fold 3:    58.910%     60.402%      57.836%
Fold 4:    58.907%     60.046%      57.829%
Fold 5:    58.909%     60.377%      57.837%
─────────────────────────────────────────────
Mean:      58.909%     60.482%      57.834%
Std:       0.002%      0.470%       0.005%
```

### 6.3 Test Prediction Statistics

**Ensemble Test Predictions:**
```
Min:         $2.50
Median:      $16.05
Mean:        $19.22
Max:         $166.54
```

**Training Price Statistics:**
```
Min:         $0.13
Median:      $14.00
Mean:        $23.65
Max:         $2,796.00
```

**Distribution Comparison:**
- Test/Train mean ratio: 0.813 (reasonable alignment)
- All predictions positive ✅
- No extreme outliers ✅

---

## 7. Data Leakage Prevention ✅

### 7.1 Excluded High-Risk Features

**Removed from Training:**
```
price_per_unit  : 0.925 correlation ❌ TOO HIGH (likely leakage)
price_per_char  : 0.503 correlation ❌ Derived from price patterns
```

**Why These Were Excluded:**
- `price_per_unit` showed 0.925 correlation with target
- Previous models using this feature showed severe overfitting:
  - Cross-validation: 7.07% SMAPE
  - Test set: 156.678% SMAPE (catastrophic failure!)
- These features appear to capture training distribution patterns
- Not reliable for generalization to test set

### 7.2 Safe Features Used

**All Features Have Correlation <0.20:**
```
text_char_count         : 0.147
text_word_count         : 0.144
premium_count           : 0.159
budget_count            : 0.140
premium_signal          : 0.122
multiplier              : 0.089
qty                     : 0.076
(all others < 0.10)
```

### 7.3 Validation Strategy

✅ **No target leakage:** All features computed from text/images only  
✅ **Conservative features:** Excluded anything with correlation >0.20  
✅ **Stable CV:** Low standard deviation across folds (0.002%)  
✅ **Reasonable predictions:** Test/train mean ratio = 0.813  
✅ **Generalizable:** Multimodal approach reduces overfitting risk

---

## 8. Implementation Details

### 8.1 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10 |
| Deep Learning | PyTorch | 2.0+ (CUDA 12.6) |
| Transformers | Hugging Face | 4.30+ |
| Computer Vision | torchvision | Latest |
| Gradient Boosting | LightGBM | 4.0+ |
| Data Processing | Pandas, NumPy | Latest |
| Optimization | SciPy | Latest |
| Environment | AWS SageMaker | ml.g4dn.xlarge (16GB RAM, T4 GPU) |

### 8.2 Key Files

```
try3/
├── implementation/
│   ├── config_auto.py                          # Auto-detect environment
│   ├── train_balanced_model.py                 # Text model (58.909% SMAPE)
│   └── images/
│       ├── download_images_efficient.py        # Batch download + retry
│       ├── extract_image_features_efficient.py # ResNet50 extraction
│       ├── train_image_model_efficient.py      # Image model (60.482%)
│       ├── ensemble_text_image.py              # Ensemble (57.834%)
│       └── run_image_pipeline_efficient.py     # Master pipeline
└── outputs/
    ├── balanced_model/
    │   ├── oof_predictions.csv                 # Text OOF predictions
    │   ├── test_predictions.csv                # Text test predictions
    │   └── embeddings_cache/
    │       ├── train_embeddings.npy            # 75K × 768
    │       └── test_embeddings.npy             # 75K × 768
    ├── images_efficient/
    │   ├── train/                              # 74,999 downloaded images
    │   └── test/                               # 74,999 downloaded images
    ├── image_features/
    │   ├── train_image_features_final.npz      # 75K × 2048
    │   └── test_image_features_final.npz       # 75K × 2048
    ├── image_model/
    │   ├── oof_predictions.csv                 # Image OOF predictions
    │   └── test_predictions.csv                # Image test predictions
    └── ensemble_text_image/
        ├── submission.csv                      # ⭐ FINAL SUBMISSION
        ├── oof_predictions.csv                 # Ensemble OOF
        └── test_predictions_detailed.csv       # Detailed breakdown
```

### 8.3 Dependencies
```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
lightgbm>=4.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
Pillow>=9.0.0
requests>=2.28.0
tqdm>=4.65.0
```

---

## 9. Reproducibility Instructions

### 9.1 Environment Setup
```bash
# On AWS SageMaker or local machine with GPU
conda create -n distilbert python=3.10
conda activate distilbert
pip install transformers torch torchvision lightgbm pandas numpy scikit-learn scipy Pillow requests tqdm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 9.2 Run Complete Pipeline
```bash
cd ~/Smart-Product-Pricing-Challenge/try3/implementation

# Step 1: Train text model (65 minutes)
python train_balanced_model.py

# Step 2: Run image pipeline (260 minutes total)
cd images
python run_image_pipeline_efficient.py

# This runs all 4 steps automatically:
#   1. Download images (228 min)
#   2. Extract features (15 min)
#   3. Train image model (25 min)
#   4. Create ensemble (1 min)

# Step 3: Copy submission
cp ../outputs/ensemble_text_image/submission.csv ~/submission/test_out.csv
```

### 9.3 Manual Step-by-Step (Optional)
```bash
cd ~/Smart-Product-Pricing-Challenge/try3/implementation/images

# Download images
python download_images_efficient.py

# Extract ResNet50 features
python extract_image_features_efficient.py

# Train image model
python train_image_model_efficient.py

# Create ensemble
python ensemble_text_image.py
```

### 9.4 Output Files
```
Final submission file:
  try3/outputs/ensemble_text_image/submission.csv
  
Format:
  sample_id,price
  100179,16.24
  245611,18.76
  ...
  (75,000 rows, all positive prices)
```

---

## 10. Model Selection Rationale

### 10.1 Why Multimodal Ensemble?

| Approach | Expected SMAPE | Reasoning |
|----------|----------------|-----------|
| Text only (with leakage) | ~53% | Uses `price_per_unit` - FAILS on test (156%!) |
| Text only (safe features) | **58.909%** | Conservative, generalizable ✅ |
| Image only | 60.482% | Captures visual signals |
| **Text + Image Ensemble** | **57.834%** | **Best of both worlds** ⭐ |

### 10.2 Why This Architecture?

**Advantages:**
✅ **Multimodal:** Combines text semantics + visual features  
✅ **Safe:** No data leakage (excluded high-correlation features)  
✅ **Modular:** Can retrain/update models independently  
✅ **Interpretable:** Clear contribution from each modality  
✅ **Memory-efficient:** Batch processing for 16GB RAM  
✅ **Robust:** Stable CV performance across folds

**Why Not Alternatives:**

1. **End-to-end multimodal transformer:**
   - Requires massive compute (days of training)
   - Risk of overfitting on small dataset
   - Less interpretable

2. **Include `price_per_unit` feature:**
   - 0.925 correlation = too risky
   - Previous model: 7% CV → 156% test (disaster!)
   - Doesn't generalize

3. **Simple averaging (50-50):**
   - Suboptimal: 57.953% SMAPE
   - Optimization saves 0.119 points

### 10.3 Design Decisions

**Memory Efficiency:**
- Batch size 500 for downloads (2GB peak)
- Batch size 100 for feature extraction (8GB peak)
- Safe for 16GB RAM systems

**Retry Logic:**
- 3 attempts per failed download
- 2-second delays between retries
- Progress tracking for resumability

**Feature Selection:**
- Conservative threshold: exclude correlation >0.20
- Prioritize generalization over CV performance
- Validate on multiple folds for stability  

---

## 11. Challenges & Solutions

### 11.1 Challenge: Data Leakage Discovery
- **Problem:** Initial model (7.07% CV) completely failed on test (156% SMAPE)
- **Root Cause:** `price_per_unit` feature (0.925 correlation) was capturing training patterns
- **Solution:** Created conservative model excluding ALL features with correlation >0.20
- **Result:** Stable performance - 58.909% CV should generalize to test

### 11.2 Challenge: Memory Constraints (16GB RAM)
- **Problem:** Previous image pipeline crashed due to loading all images in memory
- **Solution:** Implemented batch processing
  - Download: 500 images per batch
  - Feature extraction: 100 images per batch
- **Result:** Peak memory 8GB, safe for 16GB systems

### 11.3 Challenge: Failed Image Downloads
- **Problem:** Network timeouts caused ~1-2% of images to fail downloading
- **Solution:** Implemented retry logic
  - 3 attempts per image
  - 2-second delays between retries
  - Progress tracking for resumability
- **Result:** 99.99% success rate (149,998/150,000 downloaded)

### 11.4 Challenge: Missing Image Features
- **Problem:** 2 images failed after all retry attempts
- **Solution:** Imputed with mean ResNet50 features across all valid images
- **Impact:** Negligible (0.0013% of dataset)

### 11.5 Challenge: Ensemble Weight Optimization
- **Problem:** Simple 50-50 averaging suboptimal (57.953% SMAPE)
- **Solution:** Used scipy.optimize to find optimal weights
- **Result:** 63.5% text + 36.5% image = 57.834% SMAPE (0.119 points better)

---

## 12. Future Improvements

### 12.1 Short-term Enhancements

1. **Advanced Ensemble Techniques**
   - Stack multiple models (text + image + tabular)
   - Meta-learning on OOF predictions
   - Expected gain: 0.5-1.0% SMAPE reduction

2. **Hyperparameter Tuning**
   - Bayesian optimization for LightGBM
   - Different ResNet architectures (ResNet101, EfficientNet)
   - Fine-tune DistilBERT on product descriptions

3. **Feature Engineering**
   - Extract more text features (sentiment, technical specs)
   - Color/material detection from images
   - Cross-modal features (text-image alignment)

### 12.2 Long-term Improvements

1. **Vision-Language Models**
   - Use CLIP for joint text-image embeddings
   - Align product descriptions with images
   - Expected: Better multimodal understanding

2. **Large Language Models**
   - Use LLaMA/GPT for text understanding (if parameters allow)
   - Few-shot price prediction
   - Expected: Richer semantic features

3. **Active Learning**
   - Identify hard-to-predict samples
   - Request additional features/labels
   - Iterative model improvement

---

## 13. Ethical Considerations

### 13.1 Data Privacy
- ✅ Used only provided dataset
- ✅ No external data scraping
- ✅ No personal information processed
- ✅ Public image URLs only
- ✅ Academic integrity maintained

### 13.2 Fairness
- ✅ No price discrimination by brand/category
- ✅ Stratified sampling ensures balanced representation
- ✅ Model treats all products equally
- ✅ No sensitive attributes used

### 13.3 Transparency
- ✅ Open-source models (DistilBERT, ResNet50, LightGBM)
- ✅ Reproducible methodology
- ✅ Clear documentation
- ✅ Interpretable ensemble weights
- ✅ No proprietary data or models

### 13.4 Environmental Impact
- ✅ Efficient training (< 5 hours total)
- ✅ Cached embeddings reduce retraining
- ✅ Batch processing minimizes redundant computation
- ✅ Single GPU sufficient

---

## 14. Results Summary

### 14.1 Key Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| **Ensemble OOF SMAPE** | **57.834%** | Optimized text + image |
| Text Model SMAPE | 58.909% | Safe features, no leakage |
| Image Model SMAPE | 60.482% | ResNet50 features only |
| **Ensemble Improvement** | **1.075 points** | Over text-only model |
| Optimal Text Weight | 63.5% | Primary predictor |
| Optimal Image Weight | 36.5% | Complementary info |
| Training Time | ~325 minutes | Total pipeline |
| Inference Speed | <1 second | Per 1000 products |

### 14.2 Model Properties
- **Architecture:** DistilBERT (66M) + ResNet50 (25M) + LightGBM
- **License:** Apache 2.0 + BSD + MIT ✅
- **Parameters:** <100 Million (well under 8B) ✅
- **Predictions:** All positive floats ✅
- **Memory:** 16GB RAM sufficient ✅

### 14.3 Submission Files
- ✅ `test_out.csv` - 75,000 predictions (all positive, $2.50-$166.54)
- ✅ `Documentation.md` - This document
- ✅ `model_card.md` - Model specifications
- ✅ `README.md` - Quick start guide

### 14.4 Performance Comparison

**Simple Averaging vs Optimized:**
```
50-50 average:      57.953% SMAPE
70-30 average:      57.862% SMAPE
Optimized (63-36):  57.834% SMAPE ⭐ (BEST)
```

**Individual vs Ensemble:**
```
Image only:         ████████████████████████████████████████████████████████████ 60.482%
Text only:          ██████████████████████████████████████████████████████████ 58.909%
Ensemble:           ████████████████████████████████████████████████████████ 57.834% ⭐
                    
Improvement: 1.075 points (1.8% reduction)
```

---

## 15. Conclusion

This solution demonstrates that **multimodal ensemble learning** with **conservative feature selection** can achieve robust performance on product price prediction. Our approach:

1. ✅ **Achieves 57.834% SMAPE** (1.075 points improvement over text-only)
2. ✅ **Combines text + visual understanding** (DistilBERT + ResNet50)
3. ✅ **Avoids data leakage** (excluded high-correlation features)
4. ✅ **Is production-ready** (memory-efficient, batch processing, cached embeddings)
5. ✅ **Complies with all constraints** (open-source licenses, <8B parameters, positive prices)
6. ✅ **Generalizes well** (stable CV, reasonable test predictions)

The model leverages complementary information from both text descriptions and product images, with optimized ensemble weights (63.5% text + 36.5% image) to maximize predictive performance while maintaining generalization capability.

---

## 16. Key Innovations

1. **Data Leakage Prevention:** Identified and excluded `price_per_unit` (0.925 correlation)
2. **Multimodal Ensemble:** Combined text semantics + visual features
3. **Memory-Efficient Pipeline:** Batch processing for 16GB RAM constraint
4. **Retry Logic:** 99.99% image download success rate
5. **Optimized Weights:** Scipy optimization for 0.119-point improvement
6. **Conservative Features:** All correlations <0.20 for generalization

---

## 17. References

1. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016
3. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017
4. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805
5. Hugging Face Transformers: https://huggingface.co/transformers/
6. PyTorch torchvision: https://pytorch.org/vision/
7. LightGBM Documentation: https://lightgbm.readthedocs.io/

---

**Document Version:** 3.0 (Final - Multimodal Ensemble)  
**Last Updated:** October 13, 2025  
**Status:** ✅ Ready for Submission  
**Performance:** ✅ **57.834% SMAPE - Text + Image Ensemble**

