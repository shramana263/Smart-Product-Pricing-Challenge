"""
Quick verification script for DistilBERT setup
Tests that all libraries are working correctly
"""

import sys
print("="*80)
print("🔍 DistilBERT Environment Verification")
print("="*80)

# Python version
print(f"\n✓ Python: {sys.version}")

# Core libraries
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - GPU Device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch Error: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers Error: {e}")

try:
    import datasets
    print(f"✓ Datasets: {datasets.__version__}")
except Exception as e:
    print(f"✗ Datasets Error: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas Error: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy Error: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ Scikit-learn Error: {e}")

# Test tokenizer loading
print("\n" + "="*80)
print("🧪 Testing DistilBERT Tokenizer...")
print("="*80)

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    test_text = "Item Name: Pillsbury Brownie Mix, 15.5oz (Pack of 12)"
    tokens = tokenizer(test_text, return_tensors='pt')
    
    print(f"✓ Tokenizer loaded successfully!")
    print(f"  - Input text: {test_text}")
    print(f"  - Token count: {tokens['input_ids'].shape[1]}")
    print(f"  - First 10 token IDs: {tokens['input_ids'][0][:10].tolist()}")
    
except Exception as e:
    print(f"✗ Tokenizer Error: {e}")

# Test dataset existence
print("\n" + "="*80)
print("📂 Checking Dataset Files...")
print("="*80)

import os

dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
required_files = ['train1.csv', 'train2.csv', 'test1.csv', 'test2.csv']

for file in required_files:
    filepath = os.path.join(dataset_dir, file)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ {file:15} ({size:.1f} MB)")
    else:
        print(f"✗ {file:15} NOT FOUND!")

print("\n" + "="*80)
print("✅ Verification Complete!")
print("="*80)

print("\n📋 Next Steps:")
print("1. Review DISTILBERT_GUIDE.md for full instructions")
print("2. Run: finetune_distilbert_simple.py")
print("3. Expected training time: 1-2 hours (GPU) or 4-6 hours (CPU)")
print("4. Expected SMAPE improvement: 63.28% → 50-55%")

print("\n" + "="*80)
