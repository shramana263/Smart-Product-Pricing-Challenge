# 🚨 Try4 Setup Issues - COMPLETE FIX GUIDE

## ⚡ QUICK FIX (Run This First!)

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
chmod +x fix_all.sh
./fix_all.sh
python main_pipeline.py
```

---

## 🐛 Common Errors & Solutions

### Error 1: `ModuleNotFoundError: No module named 'config'`

**Symptom:**
```
Traceback (most recent call last):
  File "feature_extraction/deberta_embeddings.py", line 34
    from config.config import ...
ModuleNotFoundError: No module named 'config'
```

**Cause:** Missing `__init__.py` files in Python packages

**Fix:**
```bash
python fix_imports.py
```

**Manual Fix:**
```bash
touch config/__init__.py
touch feature_extraction/__init__.py
touch preprocessing/__init__.py
touch modeling/__init__.py
```

---

### Error 2: `ModuleNotFoundError: No module named 'tiktoken'`

**Symptom:**
```
ModuleNotFoundError: No module named 'tiktoken'
```

**Cause:** Missing tokenizer dependency for DeBERTa

**Fix:**
```bash
pip install tiktoken
```

---

### Error 3: `ImportError: requires the protobuf library`

**Symptom:**
```
ImportError: requires the protobuf library but it was not found in your environment
```

**Cause:** Missing protobuf dependency

**Fix:**
```bash
pip install protobuf
```

---

### Error 4: SentencePiece Tokenizer Error

**Symptom:**
```
ValueError: Converting from SentencePiece and Tiktoken failed
```

**Cause:** Missing sentencepiece dependency

**Fix:**
```bash
pip install sentencepiece
```

---

### Error 5: CLIP Model Download Interrupted

**Symptom:**
```
KeyboardInterrupt during CLIP model download
```

**Cause:** User interrupted download (Ctrl+C) or slow connection

**Fix:**
The download will resume automatically on next run. If it keeps failing:
```bash
# Pre-download CLIP manually
python -c "
import clip
import torch
model, preprocess = clip.load('ViT-L/14', device='cpu')
print('✅ CLIP downloaded successfully')
"
```

---

## 📋 Complete Setup Checklist

### Step 1: Pull Latest Code
```bash
cd ~/Smart-Product-Pricing-Challenge
git pull origin feature_engineering
```

### Step 2: Navigate to Try4
```bash
cd try4
```

### Step 3: Install All Dependencies
```bash
pip install -r requirements.txt
pip install tiktoken protobuf sentencepiece
pip install git+https://github.com/openai/CLIP.git
```

### Step 4: Fix Python Imports
```bash
python fix_imports.py
```

### Step 5: Verify Setup
```bash
python check_system.py
```

Should show:
- ✅ GPU detected
- ✅ All packages installed
- ✅ Images found
- ✅ Datasets found
- ✅ Config imports working

### Step 6: Run Pipeline
```bash
python main_pipeline.py
```

---

## 🔍 Verification Commands

### Test Config Import
```bash
python -c "from config.config import DATA_DIR; print('✅ Config OK')"
```

### Test All Tokenizer Dependencies
```bash
python -c "
import tiktoken
import google.protobuf
import sentencepiece
from transformers import AutoTokenizer
print('✅ All tokenizer deps OK')
"
```

### Test DeBERTa Tokenizer
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
print('✅ DeBERTa tokenizer OK')
"
```

### Test CLIP
```bash
python -c "
import clip
import torch
print('✅ CLIP installed')
"
```

### Test Full Pipeline
```bash
python check_system.py
```

---

## 📦 Required Packages Summary

### Core ML
- torch>=2.0.0
- transformers>=4.35.0
- scikit-learn>=1.3.0

### Tokenizers (DeBERTa)
- tiktoken>=0.5.0
- protobuf>=3.20.0
- sentencepiece>=0.1.99

### Vision (CLIP)
- clip (from GitHub)
- ftfy>=6.0.0
- regex>=2023.0.0
- Pillow>=10.0.0

### Boosting
- lightgbm>=4.0.0
- catboost>=1.2.0

### Outlier Detection
- pyod>=1.1.0

---

## 🎯 Files Changed in Fix

1. **config/__init__.py** - New (makes config a package)
2. **feature_extraction/__init__.py** - New
3. **preprocessing/__init__.py** - New
4. **modeling/__init__.py** - New
5. **main_pipeline.py** - Updated (sets PYTHONPATH)
6. **requirements.txt** - Updated (added tokenizer deps)
7. **fix_imports.py** - New (automated fix)
8. **fix_all.sh** - New (complete fix script)

---

## 🚀 After Successful Setup

Pipeline will run these stages:
1. **1️⃣ DeBERTa Embeddings** (~1-1.5 hours, 1024-dim)
2. **2️⃣ CLIP Embeddings** (~1-1.5 hours, 768-dim)
3. **3️⃣ Tabular Features** (~5-10 minutes, 40 features)
4. **4️⃣ Outlier Detection** (~5-10 minutes)
5. **5️⃣ Outlier Treatment** (~5-10 minutes)
6. **6️⃣ Train Fusion Model** (~30-60 minutes, LightGBM)

**Total Time:** ~3-4 hours on ml.g5.xlarge

---

## 💡 Tips

### Resume After Interruption
Pipeline automatically skips completed stages. If interrupted:
```bash
python main_pipeline.py  # Will skip stages with existing outputs
```

### Force Rerun Stage
```bash
# Delete output to force rerun
rm outputs/embeddings/deberta_train_embeddings.csv
python main_pipeline.py
```

### Run Single Stage
```bash
python feature_extraction/deberta_embeddings.py
```

### Monitor Progress
```bash
# In another terminal
watch -n 5 'ls -lh outputs/embeddings/'
watch -n 5 'nvidia-smi'
```

---

## 📞 Still Having Issues?

### Check Logs
```bash
# Pipeline logs
tail -f outputs/pipeline.log

# System info
python check_system.py
```

### Verify Environment
```bash
# Python version
python --version  # Should be 3.10+

# CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Disk space
df -h ~

# Memory
free -h
```

### Clean Reinstall
```bash
# Remove cached packages
pip cache purge

# Reinstall everything
pip uninstall -y torch transformers
pip install -r requirements.txt
pip install tiktoken protobuf sentencepiece
pip install git+https://github.com/openai/CLIP.git

# Fix imports
python fix_imports.py

# Verify
python check_system.py
```

---

## ✅ Success Indicators

You'll know setup is complete when:
- `python check_system.py` shows all ✅
- `python main_pipeline.py` starts running
- No `ModuleNotFoundError` or `ImportError`
- Stage 1 (DeBERTa) downloads model and starts processing
- GPU utilization shows in `nvidia-smi`

---

## 🎉 Ready to Run!

Once all fixes applied:
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py
```

Monitor with:
```bash
# GPU usage
watch -n 2 nvidia-smi

# Pipeline progress
tail -f outputs/pipeline.log
```

Expected results after ~3-4 hours:
- `outputs/predictions/test_predictions.csv` - Final submission
- `outputs/predictions/oof_predictions.csv` - Validation results
- Target SMAPE: **<35%** (vs Try3's ~42%)
