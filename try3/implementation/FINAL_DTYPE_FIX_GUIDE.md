# 🔧 FINAL FIX for FP16 Dtype Issue

## ⚠️ Problem
Even with `dtype=cls_output.dtype`, the `pred` tensor from regression heads is still in **Float32**, but `batch_preds` is in **Float16** (FP16 mode).

## ✅ Solution
Cast `pred` to match `batch_preds.dtype` before assignment.

---

## 🚀 Quick Fix (Copy-Paste into SageMaker Terminal)

```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation

# Backup
cp train_two_stage_model.py train_two_stage_model.py.backup_final

# Fix 1: Line 285 - Cast pred to correct dtype
sed -i '285s/batch_preds\[mask\] = pred$/batch_preds[mask] = pred.to(dtype=batch_preds.dtype)/' train_two_stage_model.py

# Fix 2: Lines 296-297 - Store output and cast dtype
python3 << 'EOF'
with open('train_two_stage_model.py', 'r') as f:
    lines = f.readlines()

# Find and replace line 296 (0-indexed = 295)
for i, line in enumerate(lines):
    if i == 295 and 'batch_preds[mask] = self.regression_heads' in line:
        indent = '                    '
        lines[i] = f'{indent}head_output = self.regression_heads[i](cls_output[mask]).squeeze(-1)\n'
        lines.insert(i+1, f'{indent}batch_preds[mask] = head_output.to(dtype=batch_preds.dtype)\n')
        break

with open('train_two_stage_model.py', 'w') as f:
    f.writelines(lines)
print("✓ Fix applied!")
EOF

# Verify
echo ""
echo "🔍 Verifying fixes..."
grep -n "pred.to(dtype" train_two_stage_model.py
grep -n "head_output" train_two_stage_model.py

echo ""
echo "✅ Ready to train!"
```

---

## 📋 What Changed

### Change 1 (Line 285):
**Before:**
```python
batch_preds[mask] = pred
```

**After:**
```python
batch_preds[mask] = pred.to(dtype=batch_preds.dtype)
```

### Change 2 (Lines 296-297):
**Before:**
```python
batch_preds[mask] = self.regression_heads[i](cls_output[mask]).squeeze(-1)
```

**After:**
```python
head_output = self.regression_heads[i](cls_output[mask]).squeeze(-1)
batch_preds[mask] = head_output.to(dtype=batch_preds.dtype)
```

---

## 🎯 Why This Works

1. **Mixed Precision (FP16)** converts model parameters to Float16
2. **Regression heads output Float32** by default
3. **batch_preds is Float16** (inherits from cls_output)
4. **Assignment fails** due to dtype mismatch
5. **Solution:** Explicitly cast to matching dtype

---

## ✅ Verify Fix Worked

After running the fix, check:
```bash
python3 << 'EOF'
with open('train_two_stage_model.py', 'r') as f:
    lines = f.readlines()
    print("Line 285:", lines[284].strip())
    print("Line 296:", lines[295].strip())
    print("Line 297:", lines[296].strip())
EOF
```

**Expected output:**
```
Line 285: batch_preds[mask] = pred.to(dtype=batch_preds.dtype)
Line 296: head_output = self.regression_heads[i](cls_output[mask]).squeeze(-1)
Line 297: batch_preds[mask] = head_output.to(dtype=batch_preds.dtype)
```

---

## 🚀 Run Training

```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation
python train_two_stage_model.py
```

**This time it WILL work!** 🎯

---

## 💡 Alternative: Use Git (Easier!)

```bash
cd /home/sagemaker-user/Smart-Product-Pricing-Challenge

# Commit and push from local first, then:
git pull origin feature_engineering

cd try3/implementation
python train_two_stage_model.py
```

The fix is already in your local repo, just needs to be synced to SageMaker!
