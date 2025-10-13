"""
System Check & Configuration Validator

Checks if all prerequisites are met before running the pipeline:
- Python packages installed
- Data files available
- GPU availability
- Disk space
- Memory requirements
"""

import sys
import subprocess
from pathlib import Path

print("="*80)
print("🔍 TRY4 SYSTEM CHECK")
print("="*80)

# ============================================================================
# 1. Python Version
# ============================================================================

print("\n1️⃣ Python Version")
print("-"*80)
print(f"Version: {sys.version}")

if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required!")
    sys.exit(1)
else:
    print("✅ Python version OK")

# ============================================================================
# 2. Required Packages
# ============================================================================

print("\n2️⃣ Required Packages")
print("-"*80)

required_packages = {
    'torch': 'torch',
    'transformers': 'transformers',
    'scikit-learn': 'sklearn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'lightgbm': 'lightgbm',
    'PIL': 'PIL',
    'requests': 'requests',
}

missing_packages = []

for display_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✅ {display_name}")
    except ImportError:
        print(f"❌ {display_name} - NOT INSTALLED")
        missing_packages.append(display_name)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\nInstall with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# 3. GPU Availability
# ============================================================================

print("\n3️⃣ GPU Availability")
print("-"*80)

try:
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"     Memory: {props.total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - will use CPU (much slower)")
        print("   Consider using Google Colab or AWS for GPU access")
except Exception as e:
    print(f"⚠️ Could not check GPU: {e}")

# ============================================================================
# 4. Data Files
# ============================================================================

print("\n4️⃣ Data Files")
print("-"*80)

base_dir = Path(__file__).parent
data_dir = base_dir.parent / "try2" / "dataset"

required_files = [
    'train1.csv',
    'train2.csv',
    'test1.csv',
    'test2.csv'
]

missing_files = []

for file in required_files:
    file_path = data_dir / file
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1e6
        print(f"✅ {file} ({size_mb:.1f} MB)")
    else:
        print(f"❌ {file} - NOT FOUND")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Missing data files: {', '.join(missing_files)}")
    print(f"\nExpected location: {data_dir}")
    sys.exit(1)

# ============================================================================
# 5. Disk Space
# ============================================================================

print("\n5️⃣ Disk Space")
print("-"*80)

try:
    import shutil
    
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stat = shutil.disk_usage(output_dir)
    free_gb = stat.free / 1e9
    
    print(f"Available: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("⚠️ Low disk space! Recommend 10+ GB for outputs")
    else:
        print("✅ Sufficient disk space")
except Exception as e:
    print(f"⚠️ Could not check disk space: {e}")

# ============================================================================
# 6. Memory (RAM)
# ============================================================================

print("\n6️⃣ System Memory")
print("-"*80)

try:
    import psutil
    
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    available_gb = mem.available / 1e9
    
    print(f"Total:     {total_gb:.1f} GB")
    print(f"Available: {available_gb:.1f} GB")
    
    if available_gb < 8:
        print("⚠️ Low memory! Recommend 16+ GB RAM")
        print("   Consider reducing batch sizes in config")
    else:
        print("✅ Sufficient memory")
except ImportError:
    print("⚠️ psutil not installed, skipping memory check")
except Exception as e:
    print(f"⚠️ Could not check memory: {e}")

# ============================================================================
# 7. Configuration
# ============================================================================

print("\n7️⃣ Configuration")
print("-"*80)

sys.path.insert(0, str(base_dir))

try:
    from config.config import (
        TEXT_MODEL, IMAGE_MODEL, DEVICE, USE_FP16,
        DATA_DIR, OUTPUT_DIR
    )
    
    print(f"Text Model:   {TEXT_MODEL['name']}")
    print(f"Image Model:  {IMAGE_MODEL['name']}")
    print(f"Device:       {DEVICE}")
    print(f"FP16:         {USE_FP16}")
    print(f"Data Dir:     {DATA_DIR}")
    print(f"Output Dir:   {OUTPUT_DIR}")
    
    print("✅ Configuration loaded")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

# ============================================================================
# 8. Output Directories
# ============================================================================

print("\n8️⃣ Output Directories")
print("-"*80)

output_dirs = [
    'outputs/embeddings',
    'outputs/features',
    'outputs/analysis',
    'outputs/models',
    'outputs/predictions'
]

for dir_path in output_dirs:
    full_path = base_dir / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ {dir_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("📊 SYSTEM CHECK SUMMARY")
print("="*80)

print("\n✅ All checks passed!")
print("\nYou're ready to run the pipeline:")
print("  python main_pipeline.py")

print("\nOr run individual stages:")
print("  python feature_extraction/deberta_embeddings.py")
print("  python feature_extraction/clip_embeddings.py")
print("  python feature_extraction/tabular_features.py")
print("  python preprocessing/outlier_detection.py")
print("  python preprocessing/outlier_treatment.py")
print("  python modeling/fusion_model.py")

print("\nFor help:")
print("  See QUICK_START.md")
print("  See README.md")

print("\n" + "="*80)
