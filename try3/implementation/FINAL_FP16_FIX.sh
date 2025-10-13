#!/bin/bash
# FINAL FIX for FP16 dtype issue
# This ensures pred tensors are cast to the correct dtype

echo "🔧 Applying FINAL FP16 dtype fix..."

cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation

# Backup
cp train_two_stage_model.py train_two_stage_model.py.backup_final
echo "✓ Backup created"

# Fix 1: Line 285 - Add .to(dtype=batch_preds.dtype) when assigning pred
sed -i '285s/batch_preds\[mask\] = pred$/batch_preds[mask] = pred.to(dtype=batch_preds.dtype)/' train_two_stage_model.py

# Fix 2: Line 296-297 - Store head output and cast dtype
sed -i '296s/batch_preds\[mask\] = self\.regression_heads\[i\](cls_output\[mask\])\.squeeze(-1)$/head_output = self.regression_heads[i](cls_output[mask]).squeeze(-1)\n                    batch_preds[mask] = head_output.to(dtype=batch_preds.dtype)/' train_two_stage_model.py

echo "✓ Fixes applied"
echo ""
echo "🔍 Verifying changes..."
grep -A1 -n "batch_preds\[mask\]" train_two_stage_model.py | head -20
echo ""
echo "✅ Ready to train!"
echo ""
echo "Run: python train_two_stage_model.py"
