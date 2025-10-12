"""
AUTO-CONFIGURATION FOR SAGEMAKER / LOCAL ENVIRONMENTS
======================================================

This script automatically detects whether you're running on:
- AWS SageMaker
- Local Windows
- Local Linux/Mac

And sets the correct paths accordingly.
"""

from pathlib import Path
import os

def detect_environment():
    """Detect if running on SageMaker or local"""
    # Check for SageMaker indicators
    if os.path.exists('/opt/ml') or 'sagemaker' in os.getcwd().lower():
        return 'sagemaker'
    elif os.name == 'nt':  # Windows
        return 'windows'
    else:  # Linux/Mac
        return 'linux'

def get_data_path(env):
    """Get data directory path based on environment"""
    if env == 'sagemaker':
        # Common SageMaker paths
        possible_paths = [
            Path("/home/sagemaker-user/Amazon_ML_hackathon/code/try2/dataset"),
            Path("/home/sagemaker-user/code/try2/dataset"),
            Path("/opt/ml/input/data"),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        # Default if none exist yet
        return Path("/home/sagemaker-user/Amazon_ML_hackathon/code/try2/dataset")
    
    elif env == 'windows':
        # Windows path (your current setup)
        return Path("C:/Users/param/Core/Code/Hackathon/Amazon_ML_hackathon/code/try2/dataset")
    
    else:  # linux
        # Linux/Mac local path
        return Path.home() / "Amazon_ML_hackathon/code/try2/dataset"

def get_output_path(env):
    """Get output directory path"""
    data_path = get_data_path(env)
    # Go up 2 levels (dataset -> try2 -> code) then into try3
    try3_path = data_path.parent.parent / "try3"
    return try3_path / "outputs"

# ============================================================================
# AUTO-DETECT AND CONFIGURE
# ============================================================================

ENV = detect_environment()
DATA_DIR = get_data_path(ENV)
OUTPUT_DIR = get_output_path(ENV)

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Print configuration
print("="*80)
print("AUTO-CONFIGURATION")
print("="*80)
print(f"\n✅ Environment detected: {ENV.upper()}")
print(f"✅ Data directory: {DATA_DIR}")
print(f"✅ Output directory: {OUTPUT_DIR}")

# Verify data exists
if DATA_DIR.exists():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV files")
    else:
        print(f"⚠️ No CSV files found in {DATA_DIR}")
else:
    print(f"⚠️ Data directory does not exist: {DATA_DIR}")
    print(f"   Create it with: mkdir -p {DATA_DIR}")

print("="*80)

# Export for use in other scripts
__all__ = ['ENV', 'DATA_DIR', 'OUTPUT_DIR', 'detect_environment', 'get_data_path', 'get_output_path']

# ============================================================================
# USAGE IN OTHER SCRIPTS
# ============================================================================
"""
To use this in your Phase 1 scripts, add at the top:

from config_auto import DATA_DIR, OUTPUT_DIR

CONFIG = {
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'phase1_log_transform',
    # ... rest of config
}

This will work on both SageMaker and local without any changes!
"""
