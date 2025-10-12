# IMAGE PIPELINE - QUICK REFERENCE

## 🚀 ONE-COMMAND RUN

```bash
cd ~/Smart-Product-Pricing-Challenge/try3/implementation/images
python run_image_pipeline_efficient.py
```

**Time:** 60-100 minutes | **Memory:** ~8GB peak | **Resumable:** Yes

---

## 📋 INDIVIDUAL STEPS

### 1️⃣ Download Images (30-60 min)
```bash
python download_images_efficient.py
```
- Batch size: 500 images
- Retry: 3 attempts per image
- Output: `outputs/images_efficient/train/` + `test/`

### 2️⃣ Extract Features (20-30 min)
```bash
python extract_image_features_efficient.py
```
- Model: ResNet50 (pretrained)
- Features: 2048-dim per image
- Output: `outputs/image_features/*.npz`

### 3️⃣ Train Model (5-10 min)
```bash
python train_image_model_efficient.py
```
- Model: LightGBM
- CV: 5-fold
- Expected: 65-75% SMAPE

### 4️⃣ Ensemble (1-2 min)
```bash
python ensemble_text_image.py
```
- Combines: Text (58.9%) + Image (65-75%)
- **Expected: 55-58% SMAPE** ⭐

---

## 📊 EXPECTED PERFORMANCE

| Model | OOF SMAPE | Features |
|-------|-----------|----------|
| Text (DistilBERT) | 58.9% | 768 embeddings |
| Image (ResNet50) | 65-75% | 2048 features |
| **Ensemble** | **55-58%** | Both |

**Improvement:** 1-4 percentage points

---

## 🔧 KEY CONFIGURATIONS

### Memory Issues?
Edit batch sizes:
```python
# download_images_efficient.py
'batch_size': 500  # Reduce to 250

# extract_image_features_efficient.py
'batch_size': 100  # Reduce to 50
```

### Timeout Issues?
```python
# download_images_efficient.py
'timeout': 10       # Increase to 15
'max_retries': 3    # Increase to 5
```

### Use CPU Instead of GPU?
```python
# extract_image_features_efficient.py
'device': 'cpu'     # Instead of 'cuda'
```

---

## 🚨 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Download fails | Check internet, increase timeout/retries |
| Out of memory | Reduce batch size, close other processes |
| Missing images | Normal! Pipeline handles automatically |
| GPU not found | Use CPU mode (slower but works) |
| Script stuck | Kill and restart (progress is saved) |

---

## 📁 OUTPUT FILES

```
outputs/
├── images_efficient/          # Downloaded images
├── image_features/            # ResNet50 features
├── image_model/               # Image-only predictions
└── ensemble_text_image/
    └── submission.csv ⭐      # FINAL SUBMISSION
```

---

## ✅ SUBMISSION CHECKLIST

```bash
# 1. Check ensemble performance
cat outputs/ensemble_text_image/oof_predictions.csv

# 2. Copy to submission folder
cp outputs/ensemble_text_image/submission.csv \
   ../../../../submission/test_out.csv

# 3. Verify format
head ../../../../submission/test_out.csv

# 4. Submit to competition!
```

---

## ⏱️ RESUME CAPABILITY

All scripts support resuming:
- ✅ **Download:** Auto-resumes from last batch
- ✅ **Extract:** Uses cached features if available
- ❌ **Train:** Must complete in one run (fast anyway)
- ❌ **Ensemble:** Must complete in one run (very fast)

---

## 💡 PRO TIPS

1. **Run overnight** - Download is slow
2. **Check progress** - Monitor `.json` files
3. **GPU recommended** - 3x faster feature extraction
4. **Keep intermediate files** - Speed up reruns
5. **Experiment with weights** - Try different ensemble ratios

---

## 📈 WHY IT WORKS

- **Text:** Captures product descriptions, semantics
- **Images:** Captures visual features, quality cues
- **Together:** Complementary information = better predictions

---

## 🎯 SUCCESS METRICS

- ✅ Download: >95% success rate
- ✅ Features: 75k x 2048 arrays
- ✅ Model: <75% SMAPE
- ✅ Ensemble: <58% SMAPE
- ✅ Submission: 75k rows, valid format

---

**Questions? Check IMAGE_PIPELINE_GUIDE.md for details!**
