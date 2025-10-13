#!/bin/bash
# Complete Fix for Try4 Setup Issues

set -e

echo "🔧 TRY4 COMPLETE FIX SCRIPT"
echo "="*60
echo ""

cd ~/Smart-Product-Pricing-Challenge/try4

echo "Step 1: Fixing Python imports..."
python fix_imports.py
echo ""

echo "Step 2: Installing missing dependencies..."
pip install tiktoken protobuf sentencepiece --quiet
echo "✅ Installed: tiktoken, protobuf, sentencepiece"
echo ""

echo "Step 3: Verifying installations..."
python -c "
import tiktoken
import google.protobuf
import sentencepiece
print('✅ All tokenizer dependencies installed')
"
echo ""

echo "Step 4: Testing imports..."
python -c "
from config.config import DATA_DIR, TEXT_MODEL, IMAGE_MODEL
print('✅ Config imports working')
print(f'  DATA_DIR: {DATA_DIR}')
print(f'  TEXT_MODEL: {TEXT_MODEL[\"name\"]}')
print(f'  IMAGE_MODEL: {IMAGE_MODEL[\"name\"]}')
"
echo ""

echo "="*60
echo "✅ ALL FIXES COMPLETE!"
echo "="*60
echo ""
echo "You can now run:"
echo "  python main_pipeline.py"
echo ""
