#!/bin/bash
# Quick fix for dtype issue on SageMaker

echo "🔧 Applying FP16 dtype fix to train_two_stage_model.py..."

cd /home/sagemaker-user/Smart-Product-Pricing-Challenge/try3/implementation

# Backup original
cp train_two_stage_model.py train_two_stage_model.py.backup
echo "✓ Backup created"

# Fix line 282 (around line 281-282)
sed -i '282s/batch_preds = torch.zeros(len(price_class), device=cls_output.device)/batch_preds = torch.zeros(len(price_class), device=cls_output.device, dtype=cls_output.dtype)/' train_two_stage_model.py

# Fix line 293 (around line 293-294)  
sed -i '293s/batch_preds = torch.zeros(len(pred_class), device=cls_output.device)/batch_preds = torch.zeros(len(pred_class), device=cls_output.device, dtype=cls_output.dtype)/' train_two_stage_model.py

echo "✓ Fix applied"

# Verify
echo ""
echo "🔍 Verifying fixes..."
grep -n "batch_preds = torch.zeros.*dtype=cls_output.dtype" train_two_stage_model.py

echo ""
echo "✅ Fix complete! Ready to train."
echo ""
echo "Run: python train_two_stage_model.py"
