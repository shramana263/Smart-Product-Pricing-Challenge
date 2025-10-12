"""
Combine Text Features (V3 Clean) with Image Features
====================================================
Merges the clean hand-crafted text features with extracted image features
to create a comprehensive feature set.

Features combined:
- 47 text features (from features_v3_clean)
- 157 image features (128 embeddings + 21 color + 7 quality + 1 metadata)
- Total: 204 features

Author: ML Challenge Team
Date: October 12, 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def combine_features(text_features_path, image_features_path, output_path, dataset_name):
    """
    Combine text and image features
    
    Args:
        text_features_path: Path to text features CSV
        image_features_path: Path to image features CSV
        output_path: Path to save combined features
        dataset_name: Name of dataset (for logging)
    """
    print(f"\nCombining {dataset_name} features...")
    
    # Load features
    text_df = pd.read_csv(text_features_path)
    image_df = pd.read_csv(image_features_path)
    
    print(f"  Text features: {text_df.shape}")
    print(f"  Image features: {image_df.shape}")
    
    # Merge on sample_id
    combined_df = text_df.merge(image_df, on='sample_id', how='left')
    
    # Fill any missing image features with 0 (shouldn't happen, but just in case)
    image_cols = [c for c in image_df.columns if c != 'sample_id']
    combined_df[image_cols] = combined_df[image_cols].fillna(0)
    
    # Save
    combined_df.to_csv(output_path, index=False)
    
    print(f"✓ Combined features saved: {output_path}")
    print(f"  Shape: {combined_df.shape}")
    
    # Feature breakdown
    text_feat_count = len([c for c in text_df.columns if c not in ['sample_id', 'price']])
    image_feat_count = len(image_cols)
    
    return combined_df, text_feat_count, image_feat_count

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("COMBINING TEXT AND IMAGE FEATURES")
    print("="*70)
    
    # Paths
    prep_dir = './preparation'
    
    # Text features (V3 Clean - NO LEAKAGE)
    text_train = os.path.join(prep_dir, 'features_v3_clean_train.csv')
    text_test = os.path.join(prep_dir, 'features_v3_clean_test.csv')
    
    # Image features
    image_train = os.path.join(prep_dir, 'image_features_train.csv')
    image_test = os.path.join(prep_dir, 'image_features_test.csv')
    
    # Check if image features exist
    if not os.path.exists(image_train) or not os.path.exists(image_test):
        print("\n❌ ERROR: Image features not found!")
        print("Please run 'image_feature_extraction.py' first.")
        return
    
    # Output paths
    combined_train = os.path.join(prep_dir, 'features_v5_text_image_train.csv')
    combined_test = os.path.join(prep_dir, 'features_v5_text_image_test.csv')
    
    # Combine training features
    train_df, text_count, image_count = combine_features(
        text_train, image_train, combined_train, "Training"
    )
    
    # Combine test features
    test_df, _, _ = combine_features(
        text_test, image_test, combined_test, "Test"
    )
    
    # Summary
    print("\n" + "="*70)
    print("FEATURE COMBINATION COMPLETE!")
    print("="*70)
    
    print(f"\nFeature breakdown:")
    print(f"  Text features (V3 Clean): {text_count}")
    print(f"  Image features: {image_count}")
    print(f"    - Image embeddings (ResNet + PCA): 128")
    print(f"    - Color features: 21")
    print(f"    - Quality features: 7")
    print(f"    - Metadata: 1")
    print(f"  Total features: {text_count + image_count}")
    
    print(f"\nOutput files:")
    print(f"  Training: {combined_train}")
    print(f"  Test: {combined_test}")
    
    # Check for missing values
    print(f"\nData quality check:")
    print(f"  Training missing values: {train_df.isnull().sum().sum()}")
    print(f"  Test missing values: {test_df.isnull().sum().sum()}")
    
    # Price statistics (training only)
    if 'price' in train_df.columns:
        print(f"\nPrice statistics (training):")
        print(f"  Min: ${train_df['price'].min():.2f}")
        print(f"  Max: ${train_df['price'].max():.2f}")
        print(f"  Mean: ${train_df['price'].mean():.2f}")
        print(f"  Median: ${train_df['price'].median():.2f}")
    
    print("\nNext step:")
    print("  Run: python train_v5_text_image.py")
    print("  Expected: 55-60% SMAPE (5-10% improvement over V3)")

if __name__ == "__main__":
    main()
