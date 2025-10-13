"""
CLIP ViT-Large Image Embedding Extraction

Extracts 768-dimensional visual embeddings from product images using
OpenAI's CLIP ViT-Large-Patch14 model.

Key Features:
- Robust image downloading with retries
- Batch processing for efficiency
- Graceful handling of missing/corrupted images
- Feature caching for reusability
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import time
import warnings
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
print("🖼️ CLIP ViT-Large Image Embedding Extraction")
print("="*80)
print(f"Model: {IMAGE_MODEL['name']}")
print(f"Device: {DEVICE}")
print("="*80)

# ============================================================================
# Image Download Utilities
# ============================================================================

def download_image_with_retry(url, max_retries=3, timeout=10):
    """Download image from URL with retry logic"""
    if not isinstance(url, str) or pd.isna(url):
        return None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
        except Exception as e:
            if attempt == max_retries - 1:
                pass  # Silent failure
            time.sleep(0.5)
    
    return None

# ============================================================================
# CLIP Feature Extractor
# ============================================================================

class CLIPFeatureExtractor:
    """Extract image embeddings using CLIP"""
    
    def __init__(self, model_name='ViT-L/14', device='cpu'):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"\n🔄 Loading CLIP {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print("✓ CLIP model loaded")
    
    def extract_single(self, image):
        """
        Extract embedding from single image
        
        Args:
            image: PIL Image or None
            
        Returns:
            numpy array of shape (768,) or zeros if image is None
        """
        if image is None:
            # Return zero vector for missing images
            return np.zeros(IMAGE_MODEL['embedding_dim'])
        
        try:
            # Preprocess and extract
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            # Return zero vector on error
            return np.zeros(IMAGE_MODEL['embedding_dim'])
    
    def extract_batch(self, images, batch_size=32):
        """
        Extract embeddings from batch of images
        
        Args:
            images: List of PIL Images (can contain None)
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (n_images, 768)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP features"):
            batch = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            batch_indices = []
            
            for j, img in enumerate(batch):
                if img is not None:
                    try:
                        img_tensor = self.preprocess(img)
                        batch_tensors.append(img_tensor)
                        batch_indices.append(j)
                    except:
                        pass
            
            # Extract features for valid images
            batch_embeddings = np.zeros((len(batch), IMAGE_MODEL['embedding_dim']))
            
            if len(batch_tensors) > 0:
                batch_tensors = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.model.encode_image(batch_tensors)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy()
                
                for idx, feat in zip(batch_indices, features):
                    batch_embeddings[idx] = feat
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

# ============================================================================
# Image URL Processing
# ============================================================================

def extract_image_url(image_link):
    """
    Extract primary image URL from image_link string
    
    Args:
        image_link: String containing image URLs separated by ;
        
    Returns:
        Primary image URL or None
    """
    if pd.isna(image_link) or not isinstance(image_link, str):
        return None
    
    # Split by ; and take first URL
    urls = [url.strip() for url in str(image_link).split(';') if url.strip()]
    return urls[0] if urls else None

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main extraction pipeline"""
    
    # Set seed
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
    
    # Extract primary image URLs
    print("\n🔗 Extracting image URLs...")
    train_df['image_url'] = train_df['image_link'].apply(extract_image_url)
    test_df['image_url'] = test_df['image_link'].apply(extract_image_url)
    
    train_available = train_df['image_url'].notna().sum()
    test_available = test_df['image_url'].notna().sum()
    
    print(f"✓ Train images available: {train_available:,} ({100*train_available/len(train_df):.1f}%)")
    print(f"✓ Test images available:  {test_available:,} ({100*test_available/len(test_df):.1f}%)")
    
    # Initialize CLIP
    extractor = CLIPFeatureExtractor(device=DEVICE)
    
    # ========================================================================
    # Process Training Set
    # ========================================================================
    
    print("\n" + "="*80)
    print("📸 DOWNLOADING & PROCESSING TRAINING IMAGES")
    print("="*80)
    
    train_images = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Downloading"):
        img = download_image_with_retry(
            row['image_url'],
            max_retries=IMAGE_MODEL['max_retries'],
            timeout=IMAGE_MODEL['download_timeout']
        )
        train_images.append(img)
    
    successful_downloads = sum(1 for img in train_images if img is not None)
    print(f"\n✓ Successfully downloaded: {successful_downloads:,}/{len(train_images):,} "
          f"({100*successful_downloads/len(train_images):.1f}%)")
    
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
    print("📸 DOWNLOADING & PROCESSING TEST IMAGES")
    print("="*80)
    
    test_images = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Downloading"):
        img = download_image_with_retry(
            row['image_url'],
            max_retries=IMAGE_MODEL['max_retries'],
            timeout=IMAGE_MODEL['download_timeout']
        )
        test_images.append(img)
    
    successful_downloads = sum(1 for img in test_images if img is not None)
    print(f"\n✓ Successfully downloaded: {successful_downloads:,}/{len(test_images):,} "
          f"({100*successful_downloads/len(test_images):.1f}%)")
    
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
    print("✅ CLIP EMBEDDING EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
