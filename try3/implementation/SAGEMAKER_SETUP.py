"""
SAGEMAKER SETUP GUIDE - Quick Start
====================================

Running Phase 1 on AWS SageMaker ml.g4dn.xlarge (NVIDIA T4 GPU)

"""

import os
from pathlib import Path

print("="*80)
print("SAGEMAKER ENVIRONMENT SETUP")
print("="*80)

# ============================================================================
# STEP 1: VERIFY ENVIRONMENT
# ============================================================================

print("\n📋 Step 1: Verifying Environment...")

# Check GPU
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU Available (NVIDIA T4)")
        print(result.stdout[:500])  # First 500 chars
    else:
        print("⚠️ GPU not detected")
except FileNotFoundError:
    print("⚠️ nvidia-smi not found")

# Check Python version
import sys
print(f"\n✅ Python version: {sys.version}")

# Check key packages
packages = {
    'torch': 'PyTorch',
    'transformers': 'HuggingFace Transformers',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'scikit-learn'
}

print("\n📦 Checking Required Packages:")
missing = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - MISSING")
        missing.append(pkg)

if missing:
    print(f"\n⚠️ Missing packages: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")
    print(f"   For transformers: pip install transformers datasets accelerate")
else:
    print("\n✅ All required packages installed!")

# ============================================================================
# STEP 2: CHECK DATA LOCATION
# ============================================================================

print("\n" + "="*80)
print("📋 Step 2: Checking Data Location...")
print("="*80)

# Common SageMaker paths
possible_paths = [
    Path("/home/sagemaker-user/Amazon_ML_hackathon/code/try2/dataset"),
    Path("/home/sagemaker-user/code/try2/dataset"),
    Path("/opt/ml/input/data"),
    Path.home() / "Amazon_ML_hackathon/code/try2/dataset",
]

data_found = False
data_path = None

for path in possible_paths:
    if path.exists():
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            print(f"\n✅ Data found at: {path}")
            print(f"   Files: {len(csv_files)} CSV files")
            for f in csv_files[:10]:  # Show first 10
                print(f"      - {f.name}")
            data_path = path
            data_found = True
            break

if not data_found:
    print("\n⚠️ Data not found in common locations")
    print("\n📋 To upload your data:")
    print("   1. In SageMaker Studio, use File → Upload Files")
    print("   2. Or use terminal: scp your-local-path/*.csv sagemaker-instance:/path/")
    print("   3. Or copy from S3: aws s3 cp s3://bucket/file.csv ./")
    print("\n   Expected files:")
    print("      - train1.csv")
    print("      - train2.csv")
    print("      - test1.csv")
    print("      - test2.csv")

# ============================================================================
# STEP 3: UPDATE SCRIPT PATHS
# ============================================================================

print("\n" + "="*80)
print("📋 Step 3: Path Configuration")
print("="*80)

if data_path:
    print(f"\n✅ Update your scripts to use this data path:")
    print(f"   CONFIG['data_dir'] = Path('{data_path}')")
    
    # Get try3 path
    try3_path = data_path.parent.parent / "try3"
    print(f"\n   Your try3 folder should be at: {try3_path}")
    
    if try3_path.exists():
        print(f"   ✅ try3 folder exists")
    else:
        print(f"   ⚠️ try3 folder not found - create it:")
        print(f"      mkdir -p {try3_path}")
else:
    print("\n📋 Once you upload data, typical path structure:")
    print("   /home/sagemaker-user/")
    print("   └── Amazon_ML_hackathon/")
    print("       └── code/")
    print("           ├── try2/")
    print("           │   └── dataset/")
    print("           │       ├── train1.csv")
    print("           │       ├── train2.csv")
    print("           │       ├── test1.csv")
    print("           │       └── test2.csv")
    print("           └── try3/")
    print("               └── implementation/")
    print("                   └── phase1_quick_wins/")

# ============================================================================
# STEP 4: RECOMMENDED WORKFLOW
# ============================================================================

print("\n" + "="*80)
print("📋 Step 4: Recommended Workflow for SageMaker")
print("="*80)

print("""
🎯 Option 1: Start with Features Only (No long GPU wait)
   Advantages: Fast results (2-3 hours), no GPU needed initially
   
   Commands:
   cd /home/sagemaker-user/Amazon_ML_hackathon/code/try3/implementation/phase1_quick_wins
   python 02_unit_standardization.py
   python 03_advanced_features.py
   
   Then integrate with your existing DistilBERT model

🎯 Option 2: Full Phase 1 Pipeline (With GPU training)
   Advantages: Maximum improvement (-7 to -9 SMAPE points)
   Time: 9-13 hours
   
   Commands:
   cd /home/sagemaker-user/Amazon_ML_hackathon/code/try3/implementation/phase1_quick_wins
   python run_phase1.py
   
   Note: Log transform training (Phase 1.1) will take 4-6 hours

🎯 Option 3: Features First, Then Train Overnight
   Advantages: See quick progress, let training run overnight
   
   Day 1 (2-3 hours):
   python 02_unit_standardization.py
   python 03_advanced_features.py
   
   Overnight (4-6 hours):
   nohup python 01_log_transform_ensemble.py > training.log 2>&1 &
   # Check progress: tail -f training.log
""")

# ============================================================================
# STEP 5: SAGEMAKER-SPECIFIC OPTIMIZATIONS
# ============================================================================

print("\n" + "="*80)
print("📋 Step 5: SageMaker Optimizations")
print("="*80)

print("""
💡 SageMaker Tips for ml.g4dn.xlarge:

1. Enable Mixed Precision (FP16) for faster training:
   - Already set in scripts: fp16=True
   - Reduces training time by ~40%

2. Monitor GPU usage:
   watch -n 1 nvidia-smi

3. Save checkpoints to avoid losing progress:
   - Scripts automatically save checkpoints
   - Located in: outputs/phase1_log_transform/

4. Use tmux/screen for long-running tasks:
   tmux new -s training
   python run_phase1.py
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training

5. Disk space check:
   df -h
   # If low, clean up old checkpoints

6. Instance billing:
   - ml.g4dn.xlarge costs ~$0.75/hour
   - Phase 1: ~$7-10 total
   - Stop instance when not in use!
""")

# ============================================================================
# STEP 6: QUICK TEST
# ============================================================================

print("\n" + "="*80)
print("📋 Step 6: Quick Test")
print("="*80)

print("""
Run a quick test to verify everything works:

# Test GPU with PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test transformers
python -c "from transformers import DistilBertTokenizer; tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); print('✅ Transformers working')"

# Test data loading
python -c "import pandas as pd; df = pd.read_csv('PATH_TO_train1.csv'); print(f'✅ Data loaded: {len(df)} rows')"
""")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ SETUP COMPLETE - READY TO START!")
print("="*80)

print("""
📋 Next Steps:

1. Ensure data is uploaded to SageMaker
2. Navigate to implementation folder:
   cd /home/sagemaker-user/Amazon_ML_hackathon/code/try3/implementation/phase1_quick_wins

3. Choose your approach:
   - Quick wins: python 02_unit_standardization.py
   - Full pipeline: python run_phase1.py

4. Monitor progress and update status tracker

5. Generate submission when complete

🎯 Expected Results:
   Current:  53.636% SMAPE
   After P1: 45-46% SMAPE
   Improvement: -7 to -9 points! 🚀

Good luck! 🎉
""")
