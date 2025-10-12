#!/bin/bash
# Fix PyTorch GPU Installation on SageMaker
# Run this if you accidentally installed CPU-only PyTorch

echo "======================================================================"
echo "🔧 Fixing PyTorch Installation - Installing GPU Version"
echo "======================================================================"

# Make sure we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "distilbert" ]]; then
    echo "⚠️  Please activate the distilbert environment first:"
    echo "   conda activate distilbert"
    exit 1
fi

echo ""
echo "📦 Step 1: Uninstalling CPU-only PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "📥 Step 2: Installing GPU-enabled PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "✅ Step 3: Verifying GPU installation..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo ""
echo "======================================================================"
echo "✅ Installation Complete!"
echo "======================================================================"
echo ""
echo "If CUDA is still not available, check:"
echo "  1. Your SageMaker instance type (must be ml.g4dn.* or ml.g5.*)"
echo "  2. Run: nvidia-smi (to verify GPU is available)"
echo ""
