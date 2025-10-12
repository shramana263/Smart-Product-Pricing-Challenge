"""
Image Feature Extraction Pipeline
==================================
Extracts visual features from product images using pre-trained CNN models.

Features extracted:
1. ResNet50 embeddings (2048-dim) → Reduced to 128-dim via PCA
2. Image statistics (color, brightness, saturation)
3. Image quality metrics (sharpness, contrast)
4. Dominant colors (top 3)
5. Image dimensions and aspect ratio

Author: ML Challenge Team
Date: October 12, 2025
"""

import pandas as pd
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

IMAGE_SIZE = 224  # Standard size for ResNet
EMBEDDING_DIM = 128  # Reduced dimension via PCA
BATCH_SIZE = 32
NUM_WORKERS = 4
DOWNLOAD_TIMEOUT = 10
MAX_RETRIES = 3

# ============================================================================
# Image Download Utilities
# ============================================================================

def download_image_with_retry(image_url, max_retries=MAX_RETRIES, timeout=DOWNLOAD_TIMEOUT):
    """
    Download image from URL with retry logic
    
    Args:
        image_url: URL of the image
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        
    Returns:
        PIL Image object or None if failed
    """
    if not isinstance(image_url, str) or pd.isna(image_url):
        return None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
        except Exception as e:
            if attempt == max_retries - 1:
                # print(f"Failed to download {image_url}: {e}")
                pass
            time.sleep(0.5)  # Brief pause before retry
    return None

# ============================================================================
# Feature Extraction Class
# ============================================================================

class ImageFeatureExtractor:
    """Extract comprehensive features from product images"""
    
    def __init__(self, device='cpu'):
        """
        Initialize the feature extractor
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet50
        print("Loading ResNet50 model...")
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.pca = None  # Will be fitted on training data
        
    def extract_cnn_features(self, image):
        """
        Extract ResNet50 features from image
        
        Args:
            image: PIL Image
            
        Returns:
            numpy array of features (2048-dim)
        """
        if image is None:
            return np.zeros(2048)
        
        try:
            # Transform image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten
            features = features.squeeze().cpu().numpy()
            return features
        except Exception as e:
            # print(f"Error extracting CNN features: {e}")
            return np.zeros(2048)
    
    def extract_color_features(self, image):
        """
        Extract color-based features
        
        Args:
            image: PIL Image
            
        Returns:
            dict of color features
        """
        if image is None:
            return self._default_color_features()
        
        try:
            # Resize for faster processing
            img_small = image.resize((100, 100))
            img_array = np.array(img_small)
            
            # Basic color statistics
            mean_rgb = img_array.mean(axis=(0, 1))
            std_rgb = img_array.std(axis=(0, 1))
            
            # Convert to HSV for more features
            img_hsv = image.convert('HSV')
            hsv_array = np.array(img_hsv.resize((100, 100)))
            mean_hsv = hsv_array.mean(axis=(0, 1))
            std_hsv = hsv_array.std(axis=(0, 1))
            
            # Dominant colors (simplified - top 3 colors by frequency)
            pixels = img_array.reshape(-1, 3)
            # Quantize colors to reduce complexity
            quantized = (pixels // 32) * 32
            unique, counts = np.unique(quantized, axis=0, return_counts=True)
            top_colors = unique[np.argsort(-counts)[:3]]
            
            features = {
                'mean_r': mean_rgb[0],
                'mean_g': mean_rgb[1],
                'mean_b': mean_rgb[2],
                'std_r': std_rgb[0],
                'std_g': std_rgb[1],
                'std_b': std_rgb[2],
                'mean_hue': mean_hsv[0],
                'mean_saturation': mean_hsv[1],
                'mean_value': mean_hsv[2],
                'std_hue': std_hsv[0],
                'std_saturation': std_hsv[1],
                'std_value': std_hsv[2],
                'dominant_color_1_r': top_colors[0][0] if len(top_colors) > 0 else 0,
                'dominant_color_1_g': top_colors[0][1] if len(top_colors) > 0 else 0,
                'dominant_color_1_b': top_colors[0][2] if len(top_colors) > 0 else 0,
                'dominant_color_2_r': top_colors[1][0] if len(top_colors) > 1 else 0,
                'dominant_color_2_g': top_colors[1][1] if len(top_colors) > 1 else 0,
                'dominant_color_2_b': top_colors[1][2] if len(top_colors) > 1 else 0,
                'dominant_color_3_r': top_colors[2][0] if len(top_colors) > 2 else 0,
                'dominant_color_3_g': top_colors[2][1] if len(top_colors) > 2 else 0,
                'dominant_color_3_b': top_colors[2][2] if len(top_colors) > 2 else 0,
            }
            
            return features
        except Exception as e:
            # print(f"Error extracting color features: {e}")
            return self._default_color_features()
    
    def extract_quality_features(self, image):
        """
        Extract image quality metrics
        
        Args:
            image: PIL Image
            
        Returns:
            dict of quality features
        """
        if image is None:
            return self._default_quality_features()
        
        try:
            img_array = np.array(image.convert('L'))  # Convert to grayscale
            
            # Sharpness (Laplacian variance)
            from scipy import ndimage
            laplacian = ndimage.laplace(img_array.astype(float))
            sharpness = laplacian.var()
            
            # Contrast (standard deviation)
            contrast = img_array.std()
            
            # Brightness (mean intensity)
            brightness = img_array.mean()
            
            # Image dimensions
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 1.0
            
            features = {
                'image_width': width,
                'image_height': height,
                'aspect_ratio': aspect_ratio,
                'image_area': width * height,
                'sharpness': sharpness,
                'contrast': contrast,
                'brightness': brightness,
            }
            
            return features
        except Exception as e:
            # print(f"Error extracting quality features: {e}")
            return self._default_quality_features()
    
    def _default_color_features(self):
        """Return default color features when image is unavailable"""
        return {f: 0.0 for f in [
            'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
            'mean_hue', 'mean_saturation', 'mean_value',
            'std_hue', 'std_saturation', 'std_value',
            'dominant_color_1_r', 'dominant_color_1_g', 'dominant_color_1_b',
            'dominant_color_2_r', 'dominant_color_2_g', 'dominant_color_2_b',
            'dominant_color_3_r', 'dominant_color_3_g', 'dominant_color_3_b',
        ]}
    
    def _default_quality_features(self):
        """Return default quality features when image is unavailable"""
        return {
            'image_width': 0,
            'image_height': 0,
            'aspect_ratio': 1.0,
            'image_area': 0,
            'sharpness': 0.0,
            'contrast': 0.0,
            'brightness': 0.0,
        }
    
    def extract_all_features(self, image_url, sample_id):
        """
        Extract all features from an image URL
        
        Args:
            image_url: URL of the image
            sample_id: Sample identifier
            
        Returns:
            dict of all features
        """
        # Download image
        image = download_image_with_retry(image_url)
        
        # Extract CNN features
        cnn_features = self.extract_cnn_features(image)
        
        # Extract color features
        color_features = self.extract_color_features(image)
        
        # Extract quality features
        quality_features = self.extract_quality_features(image)
        
        # Combine all features
        all_features = {
            'sample_id': sample_id,
            'image_available': 1 if image is not None else 0,
        }
        
        # Add CNN features (will be reduced via PCA later)
        for i, val in enumerate(cnn_features):
            all_features[f'resnet_feat_{i}'] = val
        
        # Add color and quality features
        all_features.update(color_features)
        all_features.update(quality_features)
        
        return all_features
    
    def fit_pca(self, features_df):
        """
        Fit PCA on ResNet features from training data
        
        Args:
            features_df: DataFrame with ResNet features
        """
        print(f"\nFitting PCA to reduce {2048} dims to {EMBEDDING_DIM} dims...")
        
        # Extract ResNet features
        resnet_cols = [f'resnet_feat_{i}' for i in range(2048)]
        resnet_features = features_df[resnet_cols].values
        
        # Fit PCA
        self.pca = PCA(n_components=EMBEDDING_DIM, random_state=42)
        self.pca.fit(resnet_features)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"✓ PCA fitted: {explained_var*100:.2f}% variance explained")
    
    def transform_with_pca(self, features_df):
        """
        Transform ResNet features using fitted PCA
        
        Args:
            features_df: DataFrame with ResNet features
            
        Returns:
            DataFrame with reduced features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted! Call fit_pca first.")
        
        # Extract ResNet features
        resnet_cols = [f'resnet_feat_{i}' for i in range(2048)]
        resnet_features = features_df[resnet_cols].values
        
        # Transform
        reduced_features = self.pca.transform(resnet_features)
        
        # Create new DataFrame
        result_df = features_df.drop(columns=resnet_cols).copy()
        
        # Add reduced features
        for i in range(EMBEDDING_DIM):
            result_df[f'img_embed_{i}'] = reduced_features[:, i]
        
        return result_df

# ============================================================================
# Main Processing Functions
# ============================================================================

def process_dataset(df, extractor, desc="Processing"):
    """
    Process entire dataset and extract image features
    
    Args:
        df: DataFrame with sample_id and image_link columns
        extractor: ImageFeatureExtractor instance
        desc: Description for progress bar
        
    Returns:
        DataFrame with all extracted features
    """
    print(f"\n{desc}...")
    print(f"Total samples: {len(df)}")
    
    all_features = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        features = extractor.extract_all_features(
            row['image_link'], 
            row['sample_id']
        )
        all_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Report statistics
    available = features_df['image_available'].sum()
    print(f"✓ Images successfully processed: {available}/{len(df)} ({available/len(df)*100:.1f}%)")
    
    return features_df

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("IMAGE FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (this will be slower)")
    
    # Initialize extractor
    extractor = ImageFeatureExtractor(device=device)
    
    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n" + "-"*70)
    print("Loading datasets...")
    print("-"*70)
    
    train1 = pd.read_csv('./dataset/train1.csv')
    train2 = pd.read_csv('./dataset/train2.csv')
    test1 = pd.read_csv('./dataset/test1.csv')
    test2 = pd.read_csv('./dataset/test2.csv')
    
    train_df = pd.concat([train1, train2], ignore_index=True)
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Train samples: {len(train_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    
    # ========================================================================
    # Extract features from training data
    # ========================================================================
    train_features = process_dataset(
        train_df[['sample_id', 'image_link']], 
        extractor, 
        "Extracting training image features"
    )
    
    # ========================================================================
    # Fit PCA on training features
    # ========================================================================
    extractor.fit_pca(train_features)
    
    # ========================================================================
    # Transform training features
    # ========================================================================
    print("\nTransforming training features with PCA...")
    train_features_reduced = extractor.transform_with_pca(train_features)
    
    # ========================================================================
    # Extract and transform test features
    # ========================================================================
    test_features = process_dataset(
        test_df[['sample_id', 'image_link']], 
        extractor, 
        "Extracting test image features"
    )
    
    print("\nTransforming test features with PCA...")
    test_features_reduced = extractor.transform_with_pca(test_features)
    
    # ========================================================================
    # Save features
    # ========================================================================
    print("\n" + "-"*70)
    print("Saving features...")
    print("-"*70)
    
    output_dir = './preparation'
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, 'image_features_train.csv')
    test_output = os.path.join(output_dir, 'image_features_test.csv')
    
    train_features_reduced.to_csv(train_output, index=False)
    test_features_reduced.to_csv(test_output, index=False)
    
    print(f"✓ Training features saved: {train_output}")
    print(f"  Shape: {train_features_reduced.shape}")
    print(f"✓ Test features saved: {test_output}")
    print(f"  Shape: {test_features_reduced.shape}")
    
    # ========================================================================
    # Feature summary
    # ========================================================================
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    
    feature_cols = [c for c in train_features_reduced.columns if c != 'sample_id']
    print(f"\nTotal features extracted: {len(feature_cols)}")
    print(f"  - Image embeddings (PCA): {EMBEDDING_DIM}")
    print(f"  - Color features: 21")
    print(f"  - Quality features: 7")
    print(f"  - Metadata: 1 (image_available)")
    
    print("\nNext steps:")
    print("1. Combine with text features (features_v3_clean)")
    print("2. Train models with combined features")
    print("3. Expected improvement: 5-10% SMAPE reduction")

if __name__ == "__main__":
    main()
