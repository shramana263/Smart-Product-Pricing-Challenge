"""
MEMORY-EFFICIENT IMAGE FEATURE EXTRACTION

Strategy:
1. Process images in small batches (100 at a time)
2. Use ResNet50 pretrained on ImageNet
3. Extract 2048-dim features per image
4. Save features incrementally
5. Handle missing images gracefully
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import json
import gc

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("MEMORY-EFFICIENT IMAGE FEATURE EXTRACTION")
print("="*80)
print()

# Configuration
CONFIG = {
    'data_dir': DATA_DIR,
    'image_dir': OUTPUT_DIR / "images_efficient",
    'output_dir': OUTPUT_DIR / "image_features",
    'batch_size': 100,  # Process 100 images at a time
    'feature_dim': 2048,  # ResNet50 output
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"✓ Device: {CONFIG['device']}")
print(f"✓ Batch size: {CONFIG['batch_size']}")
print()

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================
print("="*80)
print("STEP 1: LOADING RESNET50")
print("="*80)

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
# Remove the final classification layer
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(CONFIG['device'])
model.eval()

print(f"✓ ResNet50 loaded")
print(f"✓ Output dimension: {CONFIG['feature_dim']}")
print()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 2: LOADING DATA")
print("="*80)

# Load CSV files
train_files = [CONFIG['data_dir'] / 'train1.csv', CONFIG['data_dir'] / 'train2.csv']
test_files = [CONFIG['data_dir'] / 'test1.csv', CONFIG['data_dir'] / 'test2.csv']

train_dfs = []
for f in train_files:
    if f.exists():
        df = pd.read_csv(f)
        train_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

test_dfs = []
for f in test_files:
    if f.exists():
        df = pd.read_csv(f)
        test_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\n✓ Total training samples: {len(train_df):,}")
print(f"✓ Total test samples: {len(test_df):,}")
print()

# ============================================================================
# STEP 3: EXTRACT FEATURES FUNCTION
# ============================================================================
def extract_features_batch(sample_ids, image_dir, model, transform, device, batch_size=100):
    """
    Extract features for a batch of images
    
    Returns:
        features: (n_samples, feature_dim) array
        valid_mask: (n_samples,) boolean array indicating which images were processed
    """
    features = np.zeros((len(sample_ids), CONFIG['feature_dim']), dtype=np.float32)
    valid_mask = np.zeros(len(sample_ids), dtype=bool)
    
    # Process in mini-batches
    for i in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        # Load images
        for local_idx, sample_id in enumerate(batch_ids):
            img_path = image_dir / f"{sample_id}.jpg"
            
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                    batch_indices.append(i + local_idx)
                except Exception as e:
                    pass  # Skip corrupted images
        
        # Extract features if we have any valid images
        if len(batch_images) > 0:
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_features = model(batch_tensor)
                batch_features = batch_features.squeeze()
                batch_features = batch_features.cpu().numpy()
            
            # Handle single image case
            if len(batch_images) == 1:
                batch_features = batch_features.reshape(1, -1)
            
            # Store features
            for feat_idx, orig_idx in enumerate(batch_indices):
                features[orig_idx] = batch_features[feat_idx]
                valid_mask[orig_idx] = True
            
            # Clean up
            del batch_tensor, batch_features
            torch.cuda.empty_cache()
    
    return features, valid_mask

# ============================================================================
# STEP 4: EXTRACT TRAINING FEATURES
# ============================================================================
print("="*80)
print("STEP 4: EXTRACTING TRAINING FEATURES")
print("="*80)

train_image_dir = CONFIG['image_dir'] / 'train'
train_output_file = CONFIG['output_dir'] / 'train_image_features.npz'

# Check if already extracted
if train_output_file.exists():
    print(f"✓ Loading cached features from {train_output_file}")
    data = np.load(train_output_file)
    train_features = data['features']
    train_valid = data['valid_mask']
    train_sample_ids = data['sample_ids']
    print(f"✓ Loaded {len(train_features):,} features")
    print(f"✓ Valid images: {train_valid.sum():,} / {len(train_valid):,}")
else:
    print(f"✓ Processing {len(train_df):,} images...")
    
    train_sample_ids = train_df['sample_id'].values
    
    # Extract features in batches
    all_features = []
    all_valid = []
    
    n_chunks = (len(train_sample_ids) + 1000 - 1) // 1000  # Process 1000 at a time
    
    for chunk_idx in tqdm(range(n_chunks), desc="Extracting"):
        start_idx = chunk_idx * 1000
        end_idx = min((chunk_idx + 1) * 1000, len(train_sample_ids))
        chunk_ids = train_sample_ids[start_idx:end_idx]
        
        chunk_features, chunk_valid = extract_features_batch(
            chunk_ids,
            train_image_dir,
            model,
            transform,
            CONFIG['device'],
            batch_size=CONFIG['batch_size']
        )
        
        all_features.append(chunk_features)
        all_valid.append(chunk_valid)
        
        # Clean up
        gc.collect()
        if CONFIG['device'] == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate
    train_features = np.vstack(all_features)
    train_valid = np.concatenate(all_valid)
    
    # Save
    np.savez_compressed(
        train_output_file,
        features=train_features,
        valid_mask=train_valid,
        sample_ids=train_sample_ids
    )
    
    print(f"✓ Extracted {len(train_features):,} features")
    print(f"✓ Valid images: {train_valid.sum():,} / {len(train_valid):,}")
    print(f"✓ Saved to {train_output_file}")

print()

# ============================================================================
# STEP 5: EXTRACT TEST FEATURES
# ============================================================================
print("="*80)
print("STEP 5: EXTRACTING TEST FEATURES")
print("="*80)

test_image_dir = CONFIG['image_dir'] / 'test'
test_output_file = CONFIG['output_dir'] / 'test_image_features.npz'

# Check if already extracted
if test_output_file.exists():
    print(f"✓ Loading cached features from {test_output_file}")
    data = np.load(test_output_file)
    test_features = data['features']
    test_valid = data['valid_mask']
    test_sample_ids = data['sample_ids']
    print(f"✓ Loaded {len(test_features):,} features")
    print(f"✓ Valid images: {test_valid.sum():,} / {len(test_valid):,}")
else:
    print(f"✓ Processing {len(test_df):,} images...")
    
    test_sample_ids = test_df['sample_id'].values
    
    # Extract features in batches
    all_features = []
    all_valid = []
    
    n_chunks = (len(test_sample_ids) + 1000 - 1) // 1000
    
    for chunk_idx in tqdm(range(n_chunks), desc="Extracting"):
        start_idx = chunk_idx * 1000
        end_idx = min((chunk_idx + 1) * 1000, len(test_sample_ids))
        chunk_ids = test_sample_ids[start_idx:end_idx]
        
        chunk_features, chunk_valid = extract_features_batch(
            chunk_ids,
            test_image_dir,
            model,
            transform,
            CONFIG['device'],
            batch_size=CONFIG['batch_size']
        )
        
        all_features.append(chunk_features)
        all_valid.append(chunk_valid)
        
        # Clean up
        gc.collect()
        if CONFIG['device'] == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate
    test_features = np.vstack(all_features)
    test_valid = np.concatenate(all_valid)
    
    # Save
    np.savez_compressed(
        test_output_file,
        features=test_features,
        valid_mask=test_valid,
        sample_ids=test_sample_ids
    )
    
    print(f"✓ Extracted {len(test_features):,} features")
    print(f"✓ Valid images: {test_valid.sum():,} / {len(test_valid):,}")
    print(f"✓ Saved to {test_output_file}")

print()

# ============================================================================
# STEP 6: HANDLE MISSING IMAGES
# ============================================================================
print("="*80)
print("STEP 6: HANDLING MISSING IMAGES")
print("="*80)

# For missing images, use mean features
train_mean_feature = train_features[train_valid].mean(axis=0)
test_mean_feature = test_features[test_valid].mean(axis=0)

# Fill missing values
train_features[~train_valid] = train_mean_feature
test_features[~test_valid] = test_mean_feature

print(f"✓ Filled {(~train_valid).sum():,} missing training images with mean features")
print(f"✓ Filled {(~test_valid).sum():,} missing test images with mean features")
print()

# Save final features
train_final_file = CONFIG['output_dir'] / 'train_image_features_final.npz'
test_final_file = CONFIG['output_dir'] / 'test_image_features_final.npz'

np.savez_compressed(
    train_final_file,
    features=train_features,
    sample_ids=train_sample_ids
)

np.savez_compressed(
    test_final_file,
    features=test_features,
    sample_ids=test_sample_ids
)

print(f"✓ Saved final features:")
print(f"   Train: {train_final_file}")
print(f"   Test: {test_final_file}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ FEATURE EXTRACTION COMPLETE!")
print("="*80)
print()
print(f"📊 Statistics:")
print(f"   Training:")
print(f"      Total samples: {len(train_features):,}")
print(f"      Valid images: {train_valid.sum():,} ({train_valid.sum()/len(train_valid)*100:.2f}%)")
print(f"      Feature dimension: {train_features.shape[1]}")
print(f"   ")
print(f"   Test:")
print(f"      Total samples: {len(test_features):,}")
print(f"      Valid images: {test_valid.sum():,} ({test_valid.sum()/len(test_valid)*100:.2f}%)")
print(f"      Feature dimension: {test_features.shape[1]}")
print()
print(f"📁 Output files:")
print(f"   {train_final_file}")
print(f"   {test_final_file}")
print()
print(f"🚀 Next step: Train image-only model")
print(f"   python train_image_model_efficient.py")
print()
