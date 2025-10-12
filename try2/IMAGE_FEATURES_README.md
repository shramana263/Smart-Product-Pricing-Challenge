# 🖼️ Image Feature Extraction Pipeline

## Overview

This module extracts comprehensive visual features from product images to improve price prediction accuracy. It combines deep learning embeddings (ResNet50) with hand-crafted color and quality features.

## Current Status

- **Version:** V5 (Text + Image Features)
- **Previous Best (V3):** 63.28% SMAPE (text only)
- **Expected V5:** 55-60% SMAPE (5-10% improvement)
- **Status:** Ready to run

---

## Features Extracted

### 1. Deep Learning Embeddings (128 features)
- **Model:** ResNet50 (pre-trained on ImageNet)
- **Original dimension:** 2048
- **Reduced via PCA:** 128 dimensions
- **Captures:** High-level visual patterns, textures, shapes

### 2. Color Features (21 features)
- **RGB Statistics:** Mean and std for R, G, B channels
- **HSV Statistics:** Mean and std for Hue, Saturation, Value
- **Dominant Colors:** Top 3 most frequent colors (R, G, B for each)
- **Captures:** Color schemes, packaging appearance

### 3. Quality Features (7 features)
- **Image Dimensions:** Width, height, aspect ratio, area
- **Sharpness:** Laplacian variance (edge detection)
- **Contrast:** Pixel intensity standard deviation
- **Brightness:** Mean pixel intensity
- **Captures:** Image quality, product presentation

### 4. Metadata (1 feature)
- **image_available:** Binary flag (1 if downloaded successfully, 0 otherwise)

**Total: 157 image features**

---

## Pipeline Structure

```
try2/
├── image_feature_extraction.py       # Extract features from images
├── combine_features_with_images.py   # Merge with text features
├── train_v5_text_image.py            # Train models with combined features
│
└── preparation/
    ├── image_features_train.csv           # Raw image features (train)
    ├── image_features_test.csv            # Raw image features (test)
    ├── features_v5_text_image_train.csv   # Combined features (train)
    └── features_v5_text_image_test.csv    # Combined features (test)
```

---

## Usage

### Step 1: Extract Image Features

```powershell
cd try2
python image_feature_extraction.py
```

**What it does:**
1. Downloads images from URLs (with retry logic)
2. Extracts ResNet50 embeddings (2048-dim)
3. Computes color statistics (RGB, HSV, dominant colors)
4. Calculates quality metrics (sharpness, contrast, brightness)
5. Fits PCA on training data to reduce embeddings to 128-dim
6. Saves features to `preparation/image_features_*.csv`

**Time estimate:** 30-60 minutes (depends on network speed)

**Output:**
- `preparation/image_features_train.csv` (75,000 × 158)
- `preparation/image_features_test.csv` (75,000 × 158)

### Step 2: Combine with Text Features

```powershell
python combine_features_with_images.py
```

**What it does:**
1. Loads V3 clean text features (47 features)
2. Loads extracted image features (157 features)
3. Merges on `sample_id`
4. Saves combined features

**Output:**
- `preparation/features_v5_text_image_train.csv` (75,000 × 206)
  - 47 text features
  - 157 image features
  - 1 sample_id
  - 1 price (target)
- `preparation/features_v5_text_image_test.csv` (75,000 × 205)

### Step 3: Train Models

```powershell
python train_v5_text_image.py
```

**What it does:**
1. Loads combined features
2. Creates 60/15/25 stratified split
3. Trains XGBoost, LightGBM, CatBoost
4. Compares performance
5. Generates predictions with best model
6. Analyzes feature importance

**Output:**
- `modeling/test_out_v5_text_image.csv` (submission file)
- `modeling/feature_importance_v5_text_image.csv`
- Console: Detailed performance metrics

---

## Technical Details

### Image Processing

**Download with Retry:**
- Max retries: 3
- Timeout: 10 seconds
- User-Agent spoofing to avoid throttling

**ResNet50 Architecture:**
```
Input (224×224×3)
    ↓
ResNet50 (pre-trained)
    ↓
Global Average Pooling
    ↓
2048-dim embeddings
    ↓
PCA reduction
    ↓
128-dim embeddings
```

**Color Analysis:**
- Resize to 100×100 for speed
- Quantize colors (32 bins per channel)
- Extract top 3 dominant colors

**Quality Metrics:**
- Sharpness: `var(Laplacian(grayscale_image))`
- Contrast: `std(pixel_intensities)`
- Brightness: `mean(pixel_intensities)`

### PCA Reduction

- Fitted on training data only
- Applied to both train and test
- Preserves ~85-90% of variance
- Reduces computation and overfitting

### Model Training

**Same strategy as V3:**
- 60% train / 15% validation / 25% test
- Stratified by price quantiles
- Early stopping on validation set
- SMAPE as evaluation metric

---

## Expected Improvements

### Why Images Should Help

1. **Package Size Visual Cues**
   - Single item vs. multipack
   - Travel size vs. family size
   - Visible quantity indicators

2. **Brand Recognition**
   - Premium vs. economy packaging
   - Brand logos and design
   - Packaging quality

3. **Product Category**
   - Food vs. personal care
   - Liquid vs. solid
   - Supplement capsules vs. powder

4. **Quality Indicators**
   - Professional product photography
   - Clear vs. cluttered packaging
   - Premium design elements

### Conservative Estimate

- **Current (V3):** 63.28% SMAPE
- **Target (V5):** 55-60% SMAPE
- **Improvement:** 5-10% reduction

### Optimistic Scenario

If images are highly predictive:
- Could reach 50-55% SMAPE
- Would be 10-13% improvement

---

## Troubleshooting

### Images Not Downloading

**Problem:** Network throttling, timeouts

**Solutions:**
- Increase `DOWNLOAD_TIMEOUT` in `image_feature_extraction.py`
- Increase `MAX_RETRIES`
- Run in batches (modify code to process in chunks)
- Use a VPN if IP is blocked

### Out of Memory (GPU)

**Problem:** ResNet50 + large batch size

**Solutions:**
- Reduce `BATCH_SIZE` in config
- Use CPU instead (set `device='cpu'`)
- Process in smaller batches

### Out of Memory (RAM)

**Problem:** Loading 75K images

**Solutions:**
- Process in batches
- Reduce image processing resolution
- Use chunked DataFrame processing

### Low Improvement

**Problem:** Images not helping score

**Solutions:**
1. Try different CNN architectures:
   - EfficientNet (better accuracy)
   - ViT (Vision Transformer)
2. Fine-tune on product images
3. Extract more color features
4. Add object detection (detect multipacks)

---

## Feature Importance Analysis

After training, check which features matter most:

```python
# Top image features expected:
- img_embed_* (ResNet embeddings)
- dominant_color_* (packaging colors)
- image_width, image_height (product size)
- brightness, contrast (image quality)
```

Compare image vs. text contribution:
```
Image features: ~20-30% of total importance
Text features: ~70-80% of total importance
```

If image contribution is <10%, consider:
- Different image model
- More hand-crafted image features
- Fine-tuning approach

---

## Next Steps After V5

### If V5 Improves Score (55-60% SMAPE)

1. **Add Safe Target Encoding (V4 + V5)**
   - Combine image features with V4
   - Expected: 50-55% SMAPE

2. **Ensemble Models**
   - V3 (text only): 63.28%
   - V5 (text + image): ~57%
   - Weighted average: ~55%

3. **Fine-tune Transformer**
   - Text embeddings from DistilBERT
   - Combine with image features
   - Expected: <50% SMAPE

### If V5 Doesn't Improve (<3% gain)

1. **Try EfficientNet or ViT**
   - More modern architectures
   - Better feature extraction

2. **Fine-tune on Product Images**
   - Train ResNet on this dataset
   - Learn product-specific patterns

3. **Add Object Detection**
   - Detect number of items
   - Identify multipacks
   - Measure package size

4. **Use CLIP Embeddings**
   - Text + image joint embeddings
   - Better text-image alignment

---

## File Sizes (Approximate)

```
image_features_train.csv:       ~150 MB
image_features_test.csv:        ~150 MB
features_v5_text_image_train.csv: ~200 MB
features_v5_text_image_test.csv:  ~200 MB
```

---

## Dependencies

Required packages (add to requirements.txt):
```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scipy>=1.10.0
requests>=2.28.0
```

Already installed:
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- tqdm

---

## Performance Benchmarks

### Image Feature Extraction

| Dataset | Samples | Time (CPU) | Time (GPU) |
|---------|---------|------------|------------|
| Train   | 75,000  | ~45 min    | ~20 min    |
| Test    | 75,000  | ~45 min    | ~20 min    |
| **Total** | **150,000** | **~90 min** | **~40 min** |

### Model Training (V5)

| Model     | Training Time | Memory |
|-----------|---------------|--------|
| XGBoost   | ~5 min        | 2 GB   |
| LightGBM  | ~3 min        | 1.5 GB |
| CatBoost  | ~8 min        | 2 GB   |

---

## References

**ResNet50 Paper:**
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

**PCA for Dimensionality Reduction:**
> Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A.

**Color Features for Product Images:**
> Standard practice in e-commerce image analysis

---

## Contact

For questions about the image pipeline:
- Check `PROJECT_DOCUMENTATION.md` for project overview
- Check `QUICK_START.md` for quick commands
- Review `LEAKAGE_INVESTIGATION_REPORT.md` for feature engineering guidelines

---

**Last Updated:** October 12, 2025  
**Status:** ✅ Ready to run  
**Expected Improvement:** 5-10% SMAPE reduction
