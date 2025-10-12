"""
Download Model Helper for Local Machine
Run this on your Windows/Mac/Linux machine after pulling from Git
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("📥 Download DistilBERT Model to Local Machine")
print("=" * 80)
print()

# Get base directory
base_dir = Path(__file__).parent
model_dir = base_dir / "try2" / "modeling" / "distilbert_sagemaker" / "final_model"

print(f"📂 Will download to: {model_dir}")
print()

# Check if model already exists
if model_dir.exists() and any(model_dir.iterdir()):
    print("⚠️  Model directory already exists and is not empty!")
    response = input("   Overwrite? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cancelled")
        sys.exit(0)

print("Choose download method:")
print("  1) Hugging Face Hub (Recommended)")
print("  2) AWS S3")
print("  3) Manual (I'll download archive myself)")
print()

choice = input("Enter choice (1-3): ").strip()

if choice == "1":
    print()
    print("📦 Downloading from Hugging Face Hub...")
    print()
    
    username = input("Enter Hugging Face username: ").strip()
    repo_name = input("Enter repo name (e.g., amazon-ml-distilbert): ").strip()
    
    try:
        from huggingface_hub import snapshot_download
        print()
        print(f"📥 Downloading {username}/{repo_name}...")
        
        snapshot_download(
            repo_id=f"{username}/{repo_name}",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
        
        print()
        print("✅ Model downloaded successfully!")
        print(f"📍 Location: {model_dir}")
        
    except ImportError:
        print()
        print("❌ huggingface_hub not installed!")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)
        
    except Exception as e:
        print()
        print(f"❌ Error downloading: {e}")
        print()
        print("💡 Try manually:")
        print(f"   huggingface-cli download {username}/{repo_name} --local-dir {model_dir}")
        sys.exit(1)

elif choice == "2":
    print()
    print("📦 Downloading from AWS S3...")
    print()
    
    bucket = input("Enter S3 bucket name: ").strip()
    s3_path = input("Enter S3 path (e.g., models/distilbert): ").strip()
    
    import subprocess
    
    print()
    print(f"📥 Downloading from s3://{bucket}/{s3_path}/...")
    
    # Create directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download with aws cli
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{s3_path}/",
        str(model_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print()
        print("✅ Model downloaded successfully!")
        print(f"📍 Location: {model_dir}")
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Error downloading: {e}")
        print(e.stderr)
        print()
        print("💡 Make sure AWS CLI is installed and configured")
        print("   aws configure")
        sys.exit(1)
        
    except FileNotFoundError:
        print()
        print("❌ AWS CLI not found!")
        print("   Install from: https://aws.amazon.com/cli/")
        sys.exit(1)

elif choice == "3":
    print()
    print("📦 Manual Download Instructions:")
    print()
    print("1. Download the model archive from your cloud storage")
    print("   (Google Drive, Dropbox, OneDrive, etc.)")
    print()
    print("2. Extract the archive:")
    if sys.platform == "win32":
        print("   # PowerShell:")
        print(f"   tar -xzf distilbert_model.tar.gz")
    else:
        print("   # Terminal:")
        print(f"   tar -xzf distilbert_model.tar.gz")
    print()
    print("3. Move extracted files to:")
    print(f"   {model_dir}")
    print()
    sys.exit(0)

else:
    print("❌ Invalid choice")
    sys.exit(1)

# Verify download
print()
print("🔍 Verifying download...")

expected_files = [
    "model.safetensors",
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "tokenizer.json"
]

missing = []
for fname in expected_files:
    if not (model_dir / fname).exists():
        missing.append(fname)

if missing:
    print()
    print("⚠️  Warning: Some files are missing:")
    for fname in missing:
        print(f"   - {fname}")
    print()
    print("Model may not work properly!")
else:
    print("✅ All files present!")

# Show model size
print()
print("📊 Model Size:")
total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
size_mb = total_size / (1024 * 1024)
print(f"   {size_mb:.1f} MB")

print()
print("=" * 80)
print("✅ Done!")
print("=" * 80)
print()
print("🚀 Ready to use! Load with:")
print()
print("   from transformers import AutoModelForSequenceClassification, AutoTokenizer")
print(f"   model = AutoModelForSequenceClassification.from_pretrained('{model_dir}')")
print(f"   tokenizer = AutoTokenizer.from_pretrained('{model_dir}')")
print()
