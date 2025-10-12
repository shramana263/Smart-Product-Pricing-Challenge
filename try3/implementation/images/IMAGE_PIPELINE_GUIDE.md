# Memory-Efficient Image Pipeline Guide

## Overview
Complete pipeline for adding image features to the model, optimized for **16GB RAM** on SageMaker.

---

## 🎯 Key Features

### 1. **Memory-Efficient Processing**
- Batch processing (500 images for download, 100 for feature extraction)
- Incremental saving to prevent memory overflow
- Automatic garbage collection

### 2. **Robust Image Download**
- Retry logic with configurable attempts (default: 3)
- Progress saving after each batch
- Resume capability if interrupted
- Failed images tracked and retried at the end

### 3. **Feature Extraction**
- ResNet50 pretrained on ImageNet
- 2048-dimensional features per image
- Missing images filled with mean features
- GPU acceleration when available

### 4. **Model Training**
- Image-only LightGBM model
- 5-fold cross-validation
- Predictions saved for ensembling

### 5. **Ensemble**
- Optimal weight optimization
- Combines DistilBERT (text) + ResNet50 (image)
- Better performance than individual models

---

## 📋 Pipeline Steps

### **Option A: Run Complete Pipeline** (Recommended)
```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation/images
python run_image_pipeline_efficient.py
```

**Estimated Time:** 60-100 minutes total

---

### **Option B: Run Steps Individually**

#### **Step 1: Download Images** (~30-60 min)
```bash
python download_images_efficient.py
```

**What it does:**
- Downloads 75k training + 75k test images
- Processes in batches of 500
- Retries failed downloads at the end
- Saves progress after each batch

**Resume if interrupted:**
- Script automatically resumes from last completed batch
- Progress tracked in `train_progress.json` and `test_progress.json`

**Output:**
- `try3/outputs/images_efficient/train/` - Training images (224x224 JPG)
- `try3/outputs/images_efficient/test/` - Test images (224x224 JPG)

---

#### **Step 2: Extract Features** (~20-30 min)
```bash
python extract_image_features_efficient.py
```

**What it does:**
- Loads ResNet50 pretrained model
- Extracts 2048-dim features per image
- Processes in batches of 100
- Fills missing images with mean features

**Output:**
- `try3/outputs/image_features/train_image_features_final.npz`
- `try3/outputs/image_features/test_image_features_final.npz`

**Memory Usage:**
- Peak: ~8-10 GB
- Safe for 16GB RAM instances

---

#### **Step 3: Train Image Model** (~5-10 min)
```bash
python train_image_model_efficient.py
```

**What it does:**
- Trains LightGBM on 2048 image features
- 5-fold cross-validation
- Saves OOF and test predictions

**Output:**
- `try3/outputs/image_model/oof_predictions.csv`
- `try3/outputs/image_model/test_predictions.csv`

**Expected Performance:**
- OOF SMAPE: 65-75% (images alone)

---

#### **Step 4: Ensemble Models** (~1-2 min)
```bash
python ensemble_text_image.py
```

**What it does:**
- Loads text model predictions (DistilBERT)
- Loads image model predictions (ResNet50)
- Finds optimal ensemble weights
- Creates final submission

**Output:**
- `try3/outputs/ensemble_text_image/submission.csv` ⭐
- `try3/outputs/ensemble_text_image/oof_predictions.csv`

**Expected Performance:**
- Text alone: ~58.9% SMAPE
- Image alone: ~65-75% SMAPE
- **Ensemble: 55-58% SMAPE** (better than text alone!)

---

## 🔧 Configuration

All scripts use `config_auto.py` for path management:
- Automatically detects SageMaker/Windows/Linux
- No manual path configuration needed

### Key Settings:

**Download Settings:**
```python
batch_size = 500        # Images per batch
max_retries = 3         # Retry attempts
timeout = 10            # Seconds per image
image_size = (224, 224) # Resize for consistency
```

**Feature Extraction:**
```python
batch_size = 100        # Images per feature extraction batch
feature_dim = 2048      # ResNet50 output dimension
device = 'cuda'         # Auto-detect GPU
```

**Model Training:**
```python
n_folds = 5            # Cross-validation folds
learning_rate = 0.05   # LightGBM learning rate
max_depth = 8          # Tree depth
```

---

## 📊 Expected Timeline

| Step | Time | Memory Peak | Resumable |
|------|------|-------------|-----------|
| 1. Download | 30-60 min | ~2 GB | ✅ Yes |
| 2. Extract | 20-30 min | ~8 GB | ✅ Yes |
| 3. Train | 5-10 min | ~4 GB | ❌ No |
| 4. Ensemble | 1-2 min | ~1 GB | ❌ No |
| **Total** | **60-100 min** | **~8 GB** | - |

---

## 🚨 Troubleshooting

### **Problem: Download Fails**
**Symptoms:** Many "Request error" or "Timeout" messages

**Solutions:**
1. Check internet connection
2. Increase timeout: Edit `download_images_efficient.py`
   ```python
   'timeout': 15,  # Increase from 10
   'max_retries': 5,  # Increase from 3
   ```
3. Resume from progress file (automatic)

---

### **Problem: Out of Memory**
**Symptoms:** "CUDA out of memory" or "Killed"

**Solutions:**
1. Reduce batch size in feature extraction:
   ```python
   'batch_size': 50,  # Reduce from 100
   ```
2. Use CPU instead of GPU:
   ```python
   'device': 'cpu',
   ```
3. Close other processes

---

### **Problem: Missing Images**
**Symptoms:** "Valid images: 70,000 / 75,000"

**Don't worry!** The pipeline handles this:
- Missing images filled with mean features
- Model still trains successfully
- Minimal impact on performance

---

### **Problem: Feature Extraction Takes Too Long**
**Symptoms:** Stuck on "Extracting" for > 1 hour

**Solutions:**
1. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
2. Verify images downloaded correctly:
   ```bash
   ls -l try3/outputs/images_efficient/train/ | wc -l
   ```
3. Kill and restart (script caches completed work)

---

## 📈 Performance Expectations

### Individual Models:
- **Text Model (DistilBERT):** 58.9% SMAPE
- **Image Model (ResNet50):** 65-75% SMAPE

### Ensemble:
- **Expected:** 55-58% SMAPE
- **Improvement:** 1-4 percentage points
- **Weights:** Typically 70% text, 30% image

### Why Ensemble?
- Text captures semantic meaning from descriptions
- Images capture visual features (color, shape, texture)
- Together they complement each other
- More robust predictions

---

## 🎯 Next Steps After Pipeline

### 1. **Copy Submission**
```bash
cp try3/outputs/ensemble_text_image/submission.csv \
   submission/test_out.csv
```

### 2. **Verify Submission**
```bash
cd submission
python verify_submission.py  # If you have this script
```

### 3. **Submit to Competition**
- Upload `submission/test_out.csv`
- Expected score: 55-58% SMAPE

---

## 💡 Tips for Best Results

### **Tip 1: Run During Off-Peak Hours**
- Image download is network-intensive
- Run overnight if possible

### **Tip 2: Monitor Progress**
```bash
# Check download progress
tail -f try3/outputs/images_efficient/train_progress.json

# Check feature extraction
watch -n 10 ls -lh try3/outputs/image_features/
```

### **Tip 3: Save Intermediate Results**
All scripts save intermediate results:
- Download: Progress files + images
- Extraction: .npz feature files
- Training: Model predictions

**You can stop and resume anytime!**

### **Tip 4: Experiment with Ensemble Weights**
If you want to manually adjust:
```python
# In ensemble_text_image.py, try different weights:
text_weight = 0.7
image_weight = 0.3
# Or let optimizer find best weights (default)
```

---

## 📁 Output Structure

```
try3/outputs/
├── images_efficient/
│   ├── train/              # 75k training images
│   ├── test/               # 75k test images
│   ├── train_progress.json # Download progress
│   └── test_progress.json  # Download progress
│
├── image_features/
│   ├── train_image_features_final.npz  # 75k x 2048
│   └── test_image_features_final.npz   # 75k x 2048
│
├── image_model/
│   ├── oof_predictions.csv    # Training predictions
│   ├── test_predictions.csv   # Test predictions
│   └── predictions.npz        # Numpy arrays
│
└── ensemble_text_image/
    ├── submission.csv ⭐        # FINAL SUBMISSION
    ├── oof_predictions.csv     # Training predictions
    └── test_predictions_detailed.csv
```

---

## 🔬 Technical Details

### **ResNet50 Architecture**
- Pretrained on ImageNet (1.2M images, 1000 classes)
- 50 layers deep
- Outputs 2048-dim feature vector per image
- Captures: edges, textures, shapes, objects

### **Why ResNet50?**
- Proven performance on product images
- Fast inference (~10 ms per image)
- Small features (2048-dim vs 224x224x3 = 150k pixels)
- Pretrained = no need to train from scratch

### **Image Preprocessing**
1. Resize to 256x256
2. Center crop to 224x224
3. Convert to tensor
4. Normalize with ImageNet mean/std

### **Missing Image Handling**
- ~5-10% images may fail to download
- Options:
  1. Use mean features (default)
  2. Use zeros (worse)
  3. Retry download (already done)

---

## 🎓 Learning Resources

### **Understanding the Pipeline:**
1. **Image Features:** What ResNet50 extracts
   - Early layers: Edges, textures
   - Middle layers: Patterns, shapes
   - Late layers: Objects, concepts

2. **Ensemble Methods:** Why combining models works
   - Diversity: Text ≠ Image information
   - Complementarity: Text for semantics, images for visuals
   - Robustness: Less overfitting

3. **Transfer Learning:** Why pretrained models
   - ImageNet features generalize well
   - Faster than training from scratch
   - Better than random initialization

---

## ✅ Success Checklist

Before submitting:
- [ ] All 4 pipeline steps completed
- [ ] Ensemble SMAPE < 58%
- [ ] Submission file has 75,000 rows
- [ ] All prices > $0
- [ ] File format: `sample_id,price`
- [ ] Copied to `submission/test_out.csv`

---

## 🚀 Quick Start Commands

```bash
# Navigate to image pipeline
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation/images

# Run complete pipeline (60-100 min)
python run_image_pipeline_efficient.py

# Or run steps individually:
python download_images_efficient.py      # Step 1: 30-60 min
python extract_image_features_efficient.py  # Step 2: 20-30 min
python train_image_model_efficient.py    # Step 3: 5-10 min
python ensemble_text_image.py            # Step 4: 1-2 min

# Copy final submission
cp ../../outputs/ensemble_text_image/submission.csv \
   ../../../../submission/test_out.csv
```

---

## 📞 Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review progress files for error messages
3. Ensure 16GB RAM available
4. Verify GPU is available (optional but faster)

---

**Good luck! 🎯**
