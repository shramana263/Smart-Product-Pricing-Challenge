#!/bin/bash

# ⚡ Use Existing Images from Try3 - Auto Setup
# This script configures Try4 to use pre-downloaded images from Try3
# Saves 20-30 minutes of downloading time!

set -e  # Exit on error

echo "========================================"
echo "⚡ Try4 - Use Existing Images Setup"
echo "========================================"
echo ""

# Check if we're in try4 directory
if [ ! -f "config/config.py" ]; then
    echo "❌ Error: Must run from try4/ directory"
    echo "Usage: cd ~/Smart-Product-Pricing-Challenge/try4 && ./use_existing_images.sh"
    exit 1
fi

echo "Step 1: Checking for existing images in try3..."
if [ -d "../try3/outputs/images_efficient/train" ] && [ -d "../try3/outputs/images_efficient/test" ]; then
    TRAIN_COUNT=$(ls -1 ../try3/outputs/images_efficient/train/*.jpg 2>/dev/null | wc -l)
    TEST_COUNT=$(ls -1 ../try3/outputs/images_efficient/test/*.jpg 2>/dev/null | wc -l)
    echo "✅ Found images!"
    echo "   Train images: $TRAIN_COUNT"
    echo "   Test images:  $TEST_COUNT"
else
    echo "❌ Error: Images not found at ../try3/outputs/images_efficient/"
    echo "Please check the path or download images first."
    exit 1
fi

echo ""
echo "Step 2: Creating symlink to existing images..."
# Create outputs directory if it doesn't exist
mkdir -p outputs

# Remove old symlink if exists
if [ -L "outputs/images" ]; then
    rm outputs/images
    echo "   Removed old symlink"
fi

# Create new symlink
ln -sf ../../try3/outputs/images_efficient outputs/images
echo "✅ Symlink created: outputs/images -> ../try3/outputs/images_efficient"

# Verify symlink works
if [ -d "outputs/images/train" ]; then
    echo "✅ Symlink verified - images accessible"
else
    echo "⚠️  Warning: Symlink may not be working correctly"
fi

echo ""
echo "Step 3: Creating local image loading script..."

# Backup original clip_embeddings.py if not already backed up
if [ ! -f "feature_extraction/clip_embeddings_original.py" ]; then
    cp feature_extraction/clip_embeddings.py feature_extraction/clip_embeddings_original.py
    echo "✅ Original script backed up to clip_embeddings_original.py"
fi

# Create the local version
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
print(f"Using pre-downloaded images from: outputs/images/")
print("⚡ Saving 20-30 minutes by loading from disk!")
print("="*80)

# ============================================================================
# Image Loading from Disk
# ============================================================================

def load_image_from_disk(image_id, split='train'):
    """Load image from local disk (try3 images via symlink)"""
    # Images accessible via: outputs/images/{split}/{image_id}.jpg
    base_dir = Path(__file__).parent.parent
    image_path = base_dir / 'outputs' / 'images' / split / f'{image_id}.jpg'
    
    if image_path.exists():
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
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
    print("📸 LOADING TRAINING IMAGES FROM DISK (Try3 Cache)")
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
    print("📸 LOADING TEST IMAGES FROM DISK (Try3 Cache)")
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
    print("⚡ Saved 20-30 minutes by using pre-downloaded images from Try3!")
    print("="*80)

if __name__ == "__main__":
    main()
EOFSCRIPT

chmod +x feature_extraction/clip_embeddings_local.py
echo "✅ Created clip_embeddings_local.py"

echo ""
echo "Step 4: Activating local image loading..."
cp feature_extraction/clip_embeddings_local.py feature_extraction/clip_embeddings.py
echo "✅ Replaced clip_embeddings.py with local version"

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Your Try4 pipeline will now:"
echo "  ✅ Load images from disk (not download)"
echo "  ⚡ Save 20-30 minutes on Stage 2"
echo "  🚀 Complete in ~30-50 minutes (ml.g5.2xlarge + 3 folds)"
echo ""
echo "Next steps:"
echo "  1. ./fix_all.sh          # Fix imports & install dependencies"
echo "  2. python optimize_speed.py   # Optimize for your instance"
echo "  3. python main_pipeline.py    # Run pipeline (30-50 min!)"
echo ""
echo "To restore original (download) version:"
echo "  cp feature_extraction/clip_embeddings_original.py feature_extraction/clip_embeddings.py"
echo ""
