# 🚀 Quick Start - Image Features Pipeline

## TL;DR - Run Everything

```powershell
cd try2
python run_image_pipeline.py
```

This will:
1. Extract image features (~40-90 minutes)
2. Combine with text features (~2 minutes)
3. Train models (~15 minutes)
4. Generate submission file

**Total time:** ~1-2 hours  
**Expected result:** 55-60% SMAPE (improvement from 63.28%)

---

## Step-by-Step (Manual Execution)

### Prerequisites

Ensure you have V3 clean features:
- `preparation/features_v3_clean_train.csv` ✓
- `preparation/features_v3_clean_test.csv` ✓

If missing, run:
```powershell
python advanced_feature_engineering_clean.py
```

### Step 1: Extract Image Features (40-90 minutes)

```powershell
cd try2
python image_feature_extraction.py
```

**Output:**
- `preparation/image_features_train.csv` (75K samples × 158 features)
- `preparation/image_features_test.csv` (75K samples × 158 features)

**Features:**
- 128 ResNet50 embeddings (via PCA)
- 21 color features (RGB, HSV, dominant colors)
- 7 quality features (sharpness, contrast, brightness)
- 1 metadata (image_available)
- 1 sample_id

**Note:** GPU recommended but not required (CPU works, just slower)

### Step 2: Combine Features (~2 minutes)

```powershell
python combine_features_with_images.py
```

**Output:**
- `preparation/features_v5_text_image_train.csv` (75K × 206)
- `preparation/features_v5_text_image_test.csv` (75K × 205)

**Features:**
- 47 text features (from V3 clean)
- 157 image features
- Total: 204 features

### Step 3: Train Models (~15 minutes)

```powershell
python train_v5_text_image.py
```

**Output:**
- `modeling/test_out_v5_text_image.csv` (submission file)
- `modeling/feature_importance_v5_text_image.csv`
- Console output with SMAPE scores

**Models trained:**
- XGBoost
- LightGBM
- CatBoost

**Best model automatically selected for submission**

---

## Expected Results

### Performance Comparison

| Version | Features | Test SMAPE | Status |
|---------|----------|------------|--------|
| V3 Clean | 47 (text only) | 63.28% | ✅ Baseline |
| **V5 (Target)** | **204 (text + image)** | **55-60%** | **🎯 Expected** |

### Improvement Breakdown

**Conservative estimate:**
- 5-10% SMAPE reduction
- Final score: ~57-60%

**Optimistic estimate:**
- 10-15% SMAPE reduction
- Final score: ~50-55%

---

## File Outputs

```
try2/
└── preparation/
    ├── image_features_train.csv          (~150 MB)
    ├── image_features_test.csv           (~150 MB)
    ├── features_v5_text_image_train.csv  (~200 MB)
    └── features_v5_text_image_test.csv   (~200 MB)

└── modeling/
    ├── test_out_v5_text_image.csv        (submission file)
    └── feature_importance_v5_text_image.csv
```

---

## Troubleshooting

### Images not downloading

**Symptom:** Many "Failed to download" warnings

**Solutions:**
1. Check internet connection
2. Increase timeout in `image_feature_extraction.py`:
   ```python
   DOWNLOAD_TIMEOUT = 20  # increase from 10
   MAX_RETRIES = 5        # increase from 3
   ```
3. Run again - script skips already processed images

### Out of memory (GPU)

**Symptom:** CUDA out of memory error

**Solution:** Force CPU mode
```python
# In image_feature_extraction.py, line ~490
device = 'cpu'  # Force CPU
```

### Out of memory (RAM)

**Symptom:** Process killed or memory error

**Solution:** Process in smaller batches (requires code modification)

### No improvement in score

**Symptom:** V5 score ≈ V3 score (63%)

**Solutions:**
1. Try EfficientNet instead of ResNet50
2. Fine-tune image model on this dataset
3. Add object detection features
4. Use CLIP embeddings

---

## Performance Tips

### Faster Execution

1. **Use GPU for image extraction:**
   - ~40 minutes vs ~90 minutes on CPU
   - Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

2. **Skip re-extraction:**
   - If `image_features_*.csv` exist, use them
   - Only re-run if you change extraction logic

3. **Parallel processing:**
   - Already implemented in download function
   - 100 parallel workers by default

### Lower Memory Usage

1. **Reduce PCA dimensions:**
   ```python
   EMBEDDING_DIM = 64  # instead of 128
   ```

2. **Process in batches:**
   - Modify code to save intermediate results
   - Process 10K samples at a time

---

## Validation

### Check Image Features Quality

```python
import pandas as pd

# Load image features
img_df = pd.read_csv('./preparation/image_features_train.csv')

# Check download success rate
success_rate = img_df['image_available'].mean() * 100
print(f"Images downloaded: {success_rate:.1f}%")

# Should be >80%
# If <80%, check network/timeout settings
```

### Check Combined Features

```python
import pandas as pd

# Load combined features
df = pd.read_csv('./preparation/features_v5_text_image_train.csv')

print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

# Should have:
# - 75,000 rows
# - 206 columns (204 features + sample_id + price)
# - 0 missing values
```

---

## Next Steps After V5

### If Score Improved (55-60% SMAPE)

1. **Combine with V4 Safe Target Encoding:**
   ```powershell
   # Modify combine script to use V4 instead of V3
   # Expected: 50-55% SMAPE
   ```

2. **Ensemble Models:**
   ```python
   # Weighted average:
   # 0.4 * V3_predictions + 0.6 * V5_predictions
   ```

3. **Fine-tune Transformer:**
   ```powershell
   python finetune_distilbert.py  # If available
   ```

### If Score Didn't Improve (<3% gain)

1. **Try EfficientNet:**
   - Better accuracy than ResNet50
   - Modify `image_feature_extraction.py`

2. **Use Vision Transformer (ViT):**
   - State-of-the-art for image classification
   - Better for product images

3. **Fine-tune on Dataset:**
   - Train image model specifically for this task
   - Learn product-specific patterns

4. **Add Object Detection:**
   - Detect number of items (multipacks)
   - Measure package size
   - Identify product category

---

## Command Reference

```powershell
# Full pipeline (automated)
cd try2
python run_image_pipeline.py

# Individual steps
python image_feature_extraction.py         # Step 1: Extract
python combine_features_with_images.py     # Step 2: Combine
python train_v5_text_image.py             # Step 3: Train

# Check current best model (V3)
python train_v3_fair_comparison.py

# Compare results
# V3: modeling/test_out_v3_fair.csv
# V5: modeling/test_out_v5_text_image.csv
```

---

## FAQ

**Q: How long does image extraction take?**  
A: 40-90 minutes depending on GPU/CPU and network speed.

**Q: Can I use my existing image embeddings?**  
A: Yes! Just ensure format matches and run Step 2 onwards.

**Q: What if some images fail to download?**  
A: Features will be zeros for those samples. Models handle this fine.

**Q: Should I use GPU?**  
A: Recommended for 2x speedup, but CPU works fine.

**Q: Can I run this overnight?**  
A: Yes! Use `run_image_pipeline.py` for unattended execution.

**Q: How do I know if images helped?**  
A: Compare Test SMAPE in training output:
- V3: 63.28%
- V5: Should be 55-60%

**Q: What if I want to try different image models?**  
A: Modify `image_feature_extraction.py` to use EfficientNet, ViT, or CLIP.

---

## Resource Requirements

**Disk Space:**
- Image features: ~300 MB
- Combined features: ~400 MB
- Models: ~100 MB
- **Total:** ~800 MB

**RAM:**
- Minimum: 8 GB
- Recommended: 16 GB

**GPU (optional):**
- Any CUDA-capable GPU with 4+ GB VRAM
- GTX 1060 or better recommended

**Internet:**
- Need to download 150K images
- Total data: ~2-3 GB
- Broadband recommended

---

## Version History

**V5 (Current):**
- Added image features (ResNet50 + color + quality)
- 204 total features
- Expected: 55-60% SMAPE

**V3 Clean:**
- Text features only (47 features)
- Current best: 63.28% SMAPE
- No target leakage ✓

**V4 Safe Target:**
- V3 + CV-based target encoding
- Not yet tested with images

---

**Last Updated:** October 12, 2025  
**Status:** ✅ Ready to run  
**Priority:** ⭐⭐⭐ HIGH (Major improvement expected)
