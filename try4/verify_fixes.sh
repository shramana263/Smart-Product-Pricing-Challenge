#!/bin/bash

# ✅ Verification Script - Check if fixes are applied

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     🔍 Verifying Bug Fixes                                ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd ~/Smart-Product-Pricing-Challenge/try4

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Checking DeBERTa Multi-GPU Fix"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if grep -q "DataParallel" feature_extraction/deberta_embeddings.py; then
    echo "✅ Multi-GPU (DataParallel) code found!"
    echo "   Lines containing DataParallel:"
    grep -n "DataParallel" feature_extraction/deberta_embeddings.py
else
    echo "❌ Multi-GPU fix NOT applied!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Checking Tabular Features Float Conversion Fix"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if grep -q "astype('float32')" feature_extraction/tabular_features.py; then
    echo "✅ Float32 conversion found!"
    echo "   Count of fixes:"
    grep -c "astype('float32')" feature_extraction/tabular_features.py
    echo "   (Should show 2 - one for train, one for test)"
else
    echo "❌ Float conversion fix NOT applied!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. GPU Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo "  GPU $line"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Quick Python Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python << 'PYEOF'
import torch

print(f"  PyTorch version:     {torch.__version__}")
print(f"  CUDA available:      {torch.cuda.is_available()}")
print(f"  CUDA version:        {torch.version.cuda}")
print(f"  Number of GPUs:      {torch.cuda.device_count()}")

if torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"         {mem_total:.1f} GB VRAM")
PYEOF

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     ✅ Verification Complete!                             ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "  Test DeBERTa stage:"
echo "    python feature_extraction/deberta_embeddings.py"
echo ""
echo "  Test Tabular stage:"
echo "    python feature_extraction/tabular_features.py"
echo ""
echo "  Run full pipeline:"
echo "    python main_pipeline.py"
echo ""
