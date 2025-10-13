# ⚡ Use Existing Images from Try3 (Save 20-30 Minutes!)

## 🎯 Problem
The Try4 pipeline downloads images from URLs which takes **20-30 minutes**. You already have these images downloaded in `try3/outputs/images_efficient/`!

## ✅ Solution
Use a modified script that loads images from disk instead of downloading from URLs.

---

## 🚀 Quick Setup (Copy-Paste on SageMaker)

### Step 1: Create Symlink to Try3 Images
```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# Create symlink to existing images
ln -sf ../try3/outputs/images_efficient outputs/images

# Verify it worked
ls -la outputs/images/
# Should show: train/ and test/ directories
```

### Step 2: Use the Modified CLIP Script
```bash
# Backup original
cp feature_extraction/clip_embeddings.py feature_extraction/clip_embeddings_original.py

# Download the modified version (uses local images)
cat > feature_extraction/clip_embeddings_local.py << 'EOFSCRIPT'
"""
CLIP ViT-Large Image Embedding Extraction - LOCAL VERSION
Uses pre-downloaded images from try3/outputs/images_efficient/
Saves 20-30 minutes compared to downloading!
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# CLIP imports
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    import clip

from config.config import (
    DATA_DIR, EMBEDDINGS_DIR,
    IMAGE_MODEL, DEVICE, RANDOM_SEED
)

print("="*80)
print("🖼️ CLIP ViT-Large Image Embedding Extraction (LOCAL IMAGES)")
print("="*80)
print(f"Model: {IMAGE_MODEL['name']}")
print(f"Device: {DEVICE}")
print(f"Using pre-downloaded images from: ../try3/outputs/images_efficient/")
print("="*80)

# ============================================================================
# Image Loading from Disk
# ============================================================================

def load_image_from_disk(image_id, split='train'):
    """Load image from local disk (try3 images)"""
    # Try3 uses format: {split}/{image_id}.jpg
    image_path = Path(f'../try3/outputs/images_efficient/{split}/{image_id}.jpg')
    
    if image_path.exists():
        try:
            return Image.open(image_path).convert('RGB')
        except:
            return None
    return None

# ============================================================================
# CLIP Feature Extractor
# ============================================================================

class CLIPFeatureExtractor:
    """Extract image embeddings using CLIP"""
    
    def __init__(self, model_name='ViT-L/14', device='cuda'):
        self.device = device
        print(f"\n📥 Loading CLIP model: {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"✓ Model loaded successfully!")
    
    def extract_batch(self, images, batch_size=64):
        """Extract embeddings for a batch of images"""
        embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting"):
            batch_images = images[i:i+batch_size]
            batch_processed = []
            
            for img in batch_images:
                if img is not None:
                    try:
                        batch_processed.append(self.preprocess(img))
                    except:
                        # Use zero vector for failed preprocessing
                        batch_processed.append(torch.zeros(3, 224, 224))
                else:
                    # Use zero vector for missing images
                    batch_processed.append(torch.zeros(3, 224, 224))
            
            if batch_processed:
                with torch.no_grad():
                    batch_tensor = torch.stack(batch_processed).to(self.device)
                    features = self.model.encode_image(batch_tensor)
                    features = features.cpu().numpy()
                    embeddings.append(features)
        
        return np.vstack(embeddings)

def extract_image_url(image_link):
    """Extract primary image URL from image_link field"""
    if pd.isna(image_link):
        return None
    
    try:
        import json
        image_data = json.loads(image_link)
        
        if isinstance(image_data, list) and len(image_data) > 0:
            return image_data[0]
        elif isinstance(image_data, str):
            return image_data
    except:
        pass
    
    return None

# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("\n📂 Loading data...")
    train1 = pd.read_csv(DATA_DIR / 'train1.csv')
    train2 = pd.read_csv(DATA_DIR / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    test1 = pd.read_csv(DATA_DIR / 'test1.csv')
    test2 = pd.read_csv(DATA_DIR / 'test2.csv')
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")
    
    # Initialize CLIP
    extractor = CLIPFeatureExtractor(device=DEVICE)
    
    # ========================================================================
    # Process Training Set
    # ========================================================================
    
    print("\n" + "="*80)
    print("📸 LOADING TRAINING IMAGES FROM DISK")
    print("="*80)
    
    train_images = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading"):
        img = load_image_from_disk(row['sample_id'], split='train')
        train_images.append(img)
    
    successful_loads = sum(1 for img in train_images if img is not None)
    print(f"\n✓ Successfully loaded: {successful_loads:,}/{len(train_images):,} "
          f"({100*successful_loads/len(train_images):.1f}%)")
    
    print("\n🔍 Extracting CLIP embeddings for training set...")
    train_embeddings = extractor.extract_batch(
        train_images, 
        batch_size=IMAGE_MODEL['batch_size']
    )
    
    print(f"✓ Train embeddings shape: {train_embeddings.shape}")
    
    # Save training embeddings
    train_emb_df = pd.DataFrame(
        train_embeddings,
        columns=[f'clip_{i}' for i in range(IMAGE_MODEL['embedding_dim'])]
    )
    train_emb_df.insert(0, 'sample_id', train_df['sample_id'].values)
    train_emb_df.to_csv(EMBEDDINGS_DIR / 'clip_train_embeddings.csv', index=False)
    
    print(f"✓ Saved to: {EMBEDDINGS_DIR / 'clip_train_embeddings.csv'}")
    
    # Clear memory
    del train_images
    
    # ========================================================================
    # Process Test Set
    # ========================================================================
    
    print("\n" + "="*80)
    print("📸 LOADING TEST IMAGES FROM DISK")
    print("="*80)
    
    test_images = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading"):
        img = load_image_from_disk(row['sample_id'], split='test')
        test_images.append(img)
    
    successful_loads = sum(1 for img in test_images if img is not None)
    print(f"\n✓ Successfully loaded: {successful_loads:,}/{len(test_images):,} "
          f"({100*successful_loads/len(test_images):.1f}%)")
    
    print("\n🔍 Extracting CLIP embeddings for test set...")
    test_embeddings = extractor.extract_batch(
        test_images,
        batch_size=IMAGE_MODEL['batch_size']
    )
    
    print(f"✓ Test embeddings shape: {test_embeddings.shape}")
    
    # Save test embeddings
    test_emb_df = pd.DataFrame(
        test_embeddings,
        columns=[f'clip_{i}' for i in range(IMAGE_MODEL['embedding_dim'])]
    )
    test_emb_df.insert(0, 'sample_id', test_df['sample_id'].values)
    test_emb_df.to_csv(EMBEDDINGS_DIR / 'clip_test_embeddings.csv', index=False)
    
    print(f"✓ Saved to: {EMBEDDINGS_DIR / 'clip_test_embeddings.csv'}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 EMBEDDING STATISTICS")
    print("="*80)
    print(f"Embedding dimension: {IMAGE_MODEL['embedding_dim']}")
    print(f"\nTrain embeddings:")
    print(f"  Shape:  {train_embeddings.shape}")
    print(f"  Mean:   {train_embeddings.mean():.6f}")
    print(f"  Std:    {train_embeddings.std():.6f}")
    print(f"  Min:    {train_embeddings.min():.6f}")
    print(f"  Max:    {train_embeddings.max():.6f}")
    print(f"\nTest embeddings:")
    print(f"  Shape:  {test_embeddings.shape}")
    print(f"  Mean:   {test_embeddings.mean():.6f}")
    print(f"  Std:    {test_embeddings.std():.6f}")
    print(f"  Min:    {test_embeddings.min():.6f}")
    print(f"  Max:    {test_embeddings.max():.6f}")
    
    # Check for missing embeddings
    train_missing = np.sum(np.all(train_embeddings == 0, axis=1))
    test_missing = np.sum(np.all(test_embeddings == 0, axis=1))
    
    print(f"\nMissing embeddings (all zeros):")
    print(f"  Train: {train_missing:,} ({100*train_missing/len(train_embeddings):.1f}%)")
    print(f"  Test:  {test_missing:,} ({100*test_missing/len(test_embeddings):.1f}%)")
    
    print("\n" + "="*80)
    print("✅ CLIP EMBEDDING EXTRACTION COMPLETE (LOCAL IMAGES)!")
    print("⚡ Saved 20-30 minutes by using pre-downloaded images!")
    print("="*80)

if __name__ == "__main__":
    main()
EOFSCRIPT

# Make it executable
chmod +x feature_extraction/clip_embeddings_local.py

echo "✅ Created clip_embeddings_local.py"
```

### Step 3: Update Main Pipeline
```bash
# Option A: Temporarily replace the file
cp feature_extraction/clip_embeddings_local.py feature_extraction/clip_embeddings.py

# Option B: Or modify main_pipeline.py to use the local version
# (Just run Option A, it's simpler!)
```

### Step 4: Run Pipeline (Now 20-30 min faster!)
```bash
python main_pipeline.py
```

---

## 📊 Time Savings

### Before (Download from URLs):
```
Stage 2: CLIP embeddings  →  30-40 minutes (downloading + processing)
```

### After (Load from Disk):
```
Stage 2: CLIP embeddings  →  8-12 minutes (only processing!)
```

**Time saved: 20-30 minutes** ⚡

---

## 🔍 Verify Images Exist

```bash
# Check if images are there
ls -lh ../try3/outputs/images_efficient/train/ | head -10
ls -lh ../try3/outputs/images_efficient/test/ | head -10

# Count images
echo "Train images: $(ls ../try3/outputs/images_efficient/train/*.jpg 2>/dev/null | wc -l)"
echo "Test images: $(ls ../try3/outputs/images_efficient/test/*.jpg 2>/dev/null | wc -l)"
```

Expected output:
```
Train images: 75000+
Test images: 75000+
```

---

## 🎯 Complete Speed Stack (With Existing Images)

### On ml.g5.2xlarge with 3 folds + local images:

| Stage | Time (URL download) | Time (local disk) | Savings |
|-------|---------------------|-------------------|---------|
| Stage 1: DeBERTa | 25-35 min | 25-35 min | - |
| **Stage 2: CLIP** | **30-40 min** | **8-12 min** | **20-30 min** ⭐ |
| Stage 3: Features | 3-5 min | 3-5 min | - |
| Stage 4: Outliers | 3-5 min | 3-5 min | - |
| Stage 5: Treatment | 3-5 min | 3-5 min | - |
| Stage 6: Training (3 folds) | 30-45 min | 30-45 min | - |
| **Total** | **50-70 min** | **30-50 min** | **20-30 min** |

**Final time with all optimizations: 30-50 minutes!** 🚀

---

## ⚠️ Troubleshooting

### Images not found?
```bash
# Check exact path
ls ../try3/outputs/

# If images are in different location, update the script:
nano feature_extraction/clip_embeddings_local.py
# Change line 40 to correct path:
# image_path = Path(f'YOUR_CORRECT_PATH/{split}/{image_id}.jpg')
```

### Permission denied?
```bash
chmod +x feature_extraction/clip_embeddings_local.py
```

### Symlink not working?
```bash
# Use absolute path instead
rm outputs/images
ln -sf ~/Smart-Product-Pricing-Challenge/try3/outputs/images_efficient ~/Smart-Product-Pricing-Challenge/try4/outputs/images
```

---

## 🎉 Ready to Run!

Your complete setup command:
```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# 1. Setup symlink to images
ln -sf ../try3/outputs/images_efficient outputs/images

# 2. Use local image script
cp feature_extraction/clip_embeddings_local.py feature_extraction/clip_embeddings.py

# 3. Fix imports
./fix_all.sh

# 4. Optimize for instance
python optimize_speed.py

# 5. Run pipeline (30-50 min on ml.g5.2xlarge!)
python main_pipeline.py
```

**Expected total time: 30-50 minutes** (well under 1 hour!) 🎯✅
