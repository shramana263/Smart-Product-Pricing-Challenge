# 🔧 Quick Fix for FP16 Dtype Issue

## Problem
```
RuntimeError: Index put requires the source and destination dtypes match, 
got Float for the destination and Half for the source.
```

## Solution

The issue is in `train_two_stage_model.py` at **line 281** and **line 294**.

### Fix #1 (Line 281):
**Change:**
```python
batch_preds = torch.zeros(len(price_class), device=cls_output.device)
```

**To:**
```python
batch_preds = torch.zeros(len(price_class), device=cls_output.device, dtype=cls_output.dtype)
```

### Fix #2 (Line 294):
**Change:**
```python
batch_preds = torch.zeros(len(pred_class), device=cls_output.device)
```

**To:**
```python
batch_preds = torch.zeros(len(pred_class), device=cls_output.device, dtype=cls_output.dtype)
```

---

## 🚀 Quick Fix on SageMaker

### Option 1: Edit File Directly
```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation

# Open in editor
nano train_two_stage_model.py

# Find line 281 (around the middle of the TwoStageModel class)
# Add: dtype=cls_output.dtype to both torch.zeros() calls
```

### Option 2: Use sed (Automated)
```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation

# Backup original
cp train_two_stage_model.py train_two_stage_model.py.backup

# Fix line 281
sed -i 's/batch_preds = torch.zeros(len(price_class), device=cls_output.device)/batch_preds = torch.zeros(len(price_class), device=cls_output.device, dtype=cls_output.dtype)/' train_two_stage_model.py

# Fix line 294
sed -i 's/batch_preds = torch.zeros(len(pred_class), device=cls_output.device)/batch_preds = torch.zeros(len(pred_class), device=cls_output.device, dtype=cls_output.dtype)/' train_two_stage_model.py

# Verify fix
grep -n "batch_preds = torch.zeros" train_two_stage_model.py
```

### Option 3: Re-upload from Local
```bash
# On your local machine (Windows):
# 1. File is already fixed locally
# 2. Upload to SageMaker via Jupyter interface or git

# Or use git:
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge
git pull origin feature_engineering
```

---

## ✅ Verify Fix

After applying fix, check the lines:

```bash
# Should see dtype=cls_output.dtype in both lines
sed -n '281p;294p' train_two_stage_model.py
```

Expected output:
```python
            batch_preds = torch.zeros(len(price_class), device=cls_output.device, dtype=cls_output.dtype)
...
            batch_preds = torch.zeros(len(pred_class), device=cls_output.device, dtype=cls_output.dtype)
```

---

## 🚀 Run Training

```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation
python train_two_stage_model.py
```

**Expected:** Should now train successfully for ~2.5 hours! 🎯
