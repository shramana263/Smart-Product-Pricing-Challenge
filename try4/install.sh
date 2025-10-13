#!/bin/bash
# Quick installation script for Try4 on SageMaker

echo "🚀 Installing Try4 Requirements..."
echo ""

# Step 1: Install main requirements
echo "📦 Step 1/3: Installing main requirements..."
pip install -r requirements.txt

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ Main requirements installed successfully"
else
    echo "❌ Error installing main requirements"
    exit 1
fi

echo ""

# Step 2: Install CLIP from official repo
echo "📦 Step 2/3: Installing CLIP from official OpenAI repository..."
pip install git+https://github.com/openai/CLIP.git

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ CLIP installed successfully"
else
    echo "⚠️ Failed to install from git, trying alternative..."
    pip install clip-openai
    if [ $? -eq 0 ]; then
        echo "✅ CLIP installed successfully (alternative method)"
    else
        echo "❌ Error installing CLIP"
        exit 1
    fi
fi

echo ""

# Step 3: Verify installation
echo "📦 Step 3/3: Verifying installation..."
python -c "
import torch
import clip
import transformers
import lightgbm
print('✅ All key packages installed successfully!')
print(f'   - PyTorch: {torch.__version__}')
print(f'   - Transformers: {transformers.__version__}')
print(f'   - CLIP: Available')
print(f'   - LightGBM: {lightgbm.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run system check: python check_system.py"
    echo "  2. Run pipeline: python main_pipeline.py"
else
    echo "❌ Verification failed"
    exit 1
fi
