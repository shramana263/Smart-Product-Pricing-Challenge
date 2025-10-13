# 🎯 IMMEDIATE ACTION REQUIRED - Try4 Setup

## ❌ Issues Detected

Your `python main_pipeline.py` run failed due to:
1. ❌ Missing `__init__.py` files (ModuleNotFoundError: config)
2. ❌ Missing tokenizer dependencies (tiktoken, protobuf, sentencepiece)

## ✅ ONE-COMMAND FIX

Run this **RIGHT NOW** on your SageMaker instance:

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
chmod +x fix_all.sh
./fix_all.sh
```

This will:
1. Create all missing `__init__.py` files
2. Install tiktoken, protobuf, sentencepiece
3. Verify all imports work
4. Confirm setup is ready

**Time:** ~2 minutes

---

## 🚀 After Fix Complete

Then simply run:
```bash
python main_pipeline.py
```

The pipeline will:
- ✅ Download DeBERTa-v3-large (~1.5GB)
- ✅ Download CLIP ViT-Large (~890MB)
- ✅ Extract embeddings from 150K samples
- ✅ Engineer 40 tabular features
- ✅ Detect & treat outliers
- ✅ Train fusion model
- ✅ Generate predictions

**Total Time:** ~3-4 hours (hands-off)

---

## 📊 What You'll Get

After successful completion:

```
try4/outputs/
├── embeddings/
│   ├── deberta_train_embeddings.csv   # 1024-dim text embeddings
│   ├── deberta_test_embeddings.csv
│   ├── clip_train_embeddings.csv      # 768-dim image embeddings
│   └── clip_test_embeddings.csv
├── features/
│   ├── tabular_train_features.csv     # 40 engineered features
│   ├── tabular_test_features.csv
│   └── train_features_treated.csv     # After outlier treatment
├── models/
│   └── lightgbm_fold*.txt             # 5 trained models
└── predictions/
    ├── test_predictions.csv           # 🎯 YOUR SUBMISSION FILE
    └── oof_predictions.csv            # Validation results
```

**Expected SMAPE:** <35% (vs Try3's ~42%)

---

## 🔍 Monitor Progress

### In Another Terminal:
```bash
# GPU usage
watch -n 2 nvidia-smi

# Output files
watch -n 5 'ls -lh outputs/embeddings/ outputs/predictions/'
```

### Current Stage:
```bash
# Check which stage is running
ps aux | grep python | grep -v grep
```

---

## ⚠️ Pipeline Stages & Times

| Stage | Description | Time | Output |
|-------|-------------|------|--------|
| 1️⃣ | DeBERTa embeddings | 1-1.5h | deberta_*_embeddings.csv |
| 2️⃣ | CLIP embeddings | 1-1.5h | clip_*_embeddings.csv |
| 3️⃣ | Tabular features | 5-10m | tabular_*_features.csv |
| 4️⃣ | Outlier detection | 5-10m | outlier_detection_results.csv |
| 5️⃣ | Outlier treatment | 5-10m | train_features_treated.csv |
| 6️⃣ | Train fusion model | 30-60m | test_predictions.csv |

---

## 🆘 If You See Errors

### "No module named 'config'"
```bash
python fix_imports.py
```

### "No module named 'tiktoken'"
```bash
pip install tiktoken protobuf sentencepiece
```

### CLIP download interrupted (Ctrl+C)
Just run `python main_pipeline.py` again - it will resume

### Out of memory
Reduce batch size in `config/config.py`:
```python
TEXT_MODEL = {'batch_size': 8}  # Default is 16
```

---

## ✅ Verification

Before running pipeline, verify everything is ready:
```bash
python check_system.py
```

Should show:
- ✅ GPU: NVIDIA A10G (or similar)
- ✅ CUDA: Available
- ✅ Packages: All installed
- ✅ Images: ~75K train, ~75K test
- ✅ Datasets: 4 CSV files
- ✅ Disk space: >50GB free

---

## 🎯 Next Steps After Pipeline Completes

1. **Check Results:**
   ```bash
   python analyze_features.py
   ```

2. **View Predictions:**
   ```bash
   head outputs/predictions/test_predictions.csv
   ```

3. **Check SMAPE:**
   ```python
   import pandas as pd
   oof = pd.read_csv('outputs/predictions/oof_predictions.csv')
   # SMAPE calculation will be in output
   ```

4. **Submit:**
   ```bash
   # Your submission file is ready!
   ls -lh outputs/predictions/test_predictions.csv
   ```

---

## 🚨 CRITICAL: Run Fix First!

**DO NOT** skip the fix_all.sh step - it's essential!

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
chmod +x fix_all.sh
./fix_all.sh
python main_pipeline.py
```

**Good luck! 🎉**

---

**Questions?** Check `TROUBLESHOOTING.md` for detailed solutions.
