"""
Feature Analysis & Validation Tool

Analyzes the quality and information content of different feature sources:
- Text sufficiency: Does text have enough information?
- Image information: What visual signals are present?
- Cross-modal consistency: Do text and image agree?
- Adaptive weighting: Which features are most reliable?

This implements the intelligent feature analysis requirement.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, EMBEDDINGS_DIR, FEATURES_DIR, ANALYSIS_DIR,
    ANALYSIS_CONFIG
)

print("="*80)
print("🔍 FEATURE ANALYSIS & VALIDATION")
print("="*80)

# ============================================================================
# Text Sufficiency Analysis
# ============================================================================

def analyze_text_sufficiency(df):
    """
    Analyze if text contains sufficient information for price prediction
    
    Returns:
        DataFrame with text sufficiency scores
    """
    print("\n📝 Analyzing Text Sufficiency...")
    print("-"*80)
    
    results = pd.DataFrame()
    results['sample_id'] = df['sample_id']
    
    text = df['catalog_content'].fillna('')
    
    # 1. Text length score (longer = more info)
    text_lengths = text.str.len()
    results['text_length_score'] = np.clip(
        text_lengths / ANALYSIS_CONFIG['text_sufficiency']['min_text_length'],
        0, 1
    )
    
    # 2. Key information presence
    required_fields = ANALYSIS_CONFIG['text_sufficiency']['required_fields']
    
    for field in required_fields:
        if field == 'price':
            # Check for price mentions (even if not the target)
            results[f'has_{field}'] = text.str.contains(
                r'\$\d+|\d+\.\d+|\d+ oz|\d+ lb',
                case=False, regex=True
            ).astype(int)
        elif field == 'quantity':
            results[f'has_{field}'] = text.str.contains(
                r'\d+\s*(oz|ounce|lb|pound|kg|gram|g|ml|liter|l|count|pack)',
                case=False, regex=True
            ).astype(int)
        elif field == 'brand':
            results[f'has_{field}'] = text.str.contains('Item Name:', case=False).astype(int)
    
    # 3. Information density (unique words / total words)
    word_counts = text.str.split().str.len()
    unique_words = text.apply(lambda x: len(set(str(x).lower().split())))
    results['info_density_score'] = unique_words / (word_counts + 1)
    
    # 4. Overall text sufficiency score
    field_cols = [f'has_{f}' for f in required_fields]
    results['text_sufficiency_score'] = (
        0.3 * results['text_length_score'] +
        0.5 * results[field_cols].mean(axis=1) +
        0.2 * results['info_density_score']
    )
    
    # Statistics
    print(f"Text Length Score:")
    print(f"  Mean:   {results['text_length_score'].mean():.3f}")
    print(f"  Median: {results['text_length_score'].median():.3f}")
    
    print(f"\nKey Information Presence:")
    for field in required_fields:
        col = f'has_{field}'
        if col in results.columns:
            pct = 100 * results[col].mean()
            print(f"  {field.capitalize():12s}: {pct:>5.1f}%")
    
    print(f"\nOverall Text Sufficiency:")
    print(f"  Mean:   {results['text_sufficiency_score'].mean():.3f}")
    print(f"  Median: {results['text_sufficiency_score'].median():.3f}")
    
    # Categorize
    results['text_quality'] = pd.cut(
        results['text_sufficiency_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['low', 'medium', 'high']
    )
    
    print(f"\nText Quality Distribution:")
    for quality in ['low', 'medium', 'high']:
        count = (results['text_quality'] == quality).sum()
        pct = 100 * count / len(results)
        print(f"  {quality.capitalize():8s}: {count:>6,} ({pct:>5.1f}%)")
    
    return results

# ============================================================================
# Image Information Analysis
# ============================================================================

def analyze_image_information(df):
    """
    Analyze visual information content in images
    
    Returns:
        DataFrame with image information scores
    """
    print("\n🖼️ Analyzing Image Information...")
    print("-"*80)
    
    results = pd.DataFrame()
    results['sample_id'] = df['sample_id']
    
    # 1. Image availability
    has_image = df['image_link'].notna()
    results['has_image'] = has_image.astype(int)
    
    # 2. Image count (more images = more info)
    results['image_count'] = df['image_link'].fillna('').str.count(';') + 1
    results['image_count'] = results['image_count'].where(has_image, 0)
    
    # 3. Load CLIP embeddings if available
    clip_train_path = EMBEDDINGS_DIR / 'clip_train_embeddings.csv'
    
    if clip_train_path.exists():
        clip_df = pd.read_csv(clip_train_path)
        clip_features = [col for col in clip_df.columns if col.startswith('clip_')]
        
        # Image information score based on CLIP embedding variance
        # Higher variance = more visual information
        results = results.merge(clip_df[['sample_id'] + clip_features], on='sample_id', how='left')
        
        clip_variance = results[clip_features].var(axis=1)
        results['image_info_score'] = np.clip(clip_variance / clip_variance.quantile(0.95), 0, 1)
        
        # Check if image is all zeros (missing/failed download)
        results['image_missing'] = (results[clip_features].sum(axis=1) == 0).astype(int)
        results['image_info_score'] = results['image_info_score'].where(
            results['image_missing'] == 0,
            0
        )
        
        results.drop(columns=clip_features, inplace=True)
    else:
        print("  ⚠️ CLIP embeddings not found, using basic metrics only")
        results['image_info_score'] = results['has_image'] * 0.5
        results['image_missing'] = (~has_image).astype(int)
    
    # Statistics
    print(f"Image Availability:")
    print(f"  Available: {results['has_image'].sum():>6,} ({100*results['has_image'].mean():>5.1f}%)")
    print(f"  Missing:   {(~results['has_image'].astype(bool)).sum():>6,} ({100*(1-results['has_image'].mean()):>5.1f}%)")
    
    if 'image_info_score' in results.columns:
        print(f"\nImage Information Score:")
        print(f"  Mean:   {results['image_info_score'].mean():.3f}")
        print(f"  Median: {results['image_info_score'].median():.3f}")
        
        # Categorize
        results['image_quality'] = pd.cut(
            results['image_info_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        print(f"\nImage Quality Distribution:")
        for quality in ['low', 'medium', 'high']:
            count = (results['image_quality'] == quality).sum()
            pct = 100 * count / len(results)
            print(f"  {quality.capitalize():8s}: {count:>6,} ({pct:>5.1f}%)")
    
    return results

# ============================================================================
# Cross-Modal Consistency Analysis
# ============================================================================

def analyze_cross_modal_consistency(text_results, image_results):
    """
    Analyze consistency between text and image signals
    
    Returns:
        DataFrame with consistency scores
    """
    print("\n🔄 Analyzing Cross-Modal Consistency...")
    print("-"*80)
    
    results = text_results[['sample_id']].copy()
    
    # Merge text and image scores
    results = results.merge(
        text_results[['sample_id', 'text_sufficiency_score']],
        on='sample_id'
    )
    results = results.merge(
        image_results[['sample_id', 'image_info_score']],
        on='sample_id'
    )
    
    # Calculate consistency
    # High consistency = both high or both low
    # Low consistency = one high, one low (conflicting signals)
    
    text_high = results['text_sufficiency_score'] > 0.6
    image_high = results['image_info_score'] > 0.6
    
    text_low = results['text_sufficiency_score'] < 0.3
    image_low = results['image_info_score'] < 0.3
    
    results['consistency_score'] = 0.5  # Default: medium
    
    # Both high or both low = high consistency
    results.loc[text_high & image_high, 'consistency_score'] = 1.0
    results.loc[text_low & image_low, 'consistency_score'] = 0.8
    
    # One high, one low = low consistency
    results.loc[text_high & image_low, 'consistency_score'] = 0.3
    results.loc[text_low & image_high, 'consistency_score'] = 0.3
    
    # Statistics
    print(f"Consistency Score:")
    print(f"  Mean:   {results['consistency_score'].mean():.3f}")
    print(f"  Median: {results['consistency_score'].median():.3f}")
    
    # Categories
    results['consistency'] = pd.cut(
        results['consistency_score'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['low', 'medium', 'high']
    )
    
    print(f"\nConsistency Distribution:")
    for level in ['low', 'medium', 'high']:
        count = (results['consistency'] == level).sum()
        pct = 100 * count / len(results)
        print(f"  {level.capitalize():8s}: {count:>6,} ({pct:>5.1f}%)")
    
    return results

# ============================================================================
# Adaptive Feature Weighting
# ============================================================================

def calculate_adaptive_weights(text_results, image_results, consistency_results):
    """
    Calculate adaptive feature weights based on quality scores
    
    Returns:
        DataFrame with recommended weights for each modality
    """
    print("\n⚖️ Calculating Adaptive Feature Weights...")
    print("-"*80)
    
    results = pd.DataFrame()
    results['sample_id'] = text_results['sample_id']
    
    # Base weights from config
    base_text_weight = ANALYSIS_CONFIG['cross_modal']['weight_text']
    base_image_weight = ANALYSIS_CONFIG['cross_modal']['weight_image']
    
    # Adjust based on quality scores
    text_score = text_results['text_sufficiency_score'].values
    image_score = image_results['image_info_score'].values
    consistency_score = consistency_results['consistency_score'].values
    
    # Normalize scores
    total_score = text_score + image_score + 1e-8
    
    results['text_weight'] = base_text_weight * (text_score / total_score)
    results['image_weight'] = base_image_weight * (image_score / total_score)
    
    # Boost weights for high consistency
    results['text_weight'] *= (0.5 + 0.5 * consistency_score)
    results['image_weight'] *= (0.5 + 0.5 * consistency_score)
    
    # Normalize to sum to 1
    total_weight = results['text_weight'] + results['image_weight']
    results['text_weight'] /= total_weight
    results['image_weight'] /= total_weight
    
    # Statistics
    print(f"Text Weight:")
    print(f"  Mean:   {results['text_weight'].mean():.3f}")
    print(f"  Median: {results['text_weight'].median():.3f}")
    print(f"  Range:  [{results['text_weight'].min():.3f}, {results['text_weight'].max():.3f}]")
    
    print(f"\nImage Weight:")
    print(f"  Mean:   {results['image_weight'].mean():.3f}")
    print(f"  Median: {results['image_weight'].median():.3f}")
    print(f"  Range:  [{results['image_weight'].min():.3f}, {results['image_weight'].max():.3f}]")
    
    return results

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main analysis pipeline"""
    
    print("\n📂 Loading data...")
    train1 = pd.read_csv(DATA_DIR / 'train1.csv')
    train2 = pd.read_csv(DATA_DIR / 'train2.csv')
    df = pd.concat([train1, train2], ignore_index=True)
    
    print(f"✓ Loaded {len(df):,} samples")
    
    # Run analyses
    text_results = analyze_text_sufficiency(df)
    image_results = analyze_image_information(df)
    consistency_results = analyze_cross_modal_consistency(text_results, image_results)
    weight_results = calculate_adaptive_weights(text_results, image_results, consistency_results)
    
    # Merge all results
    final_results = text_results.merge(image_results, on='sample_id')
    final_results = final_results.merge(consistency_results, on='sample_id')
    final_results = final_results.merge(weight_results, on='sample_id')
    
    # Save
    output_path = ANALYSIS_DIR / 'feature_analysis_results.csv'
    final_results.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    print(f"✓ Saved to: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nSamples by modality quality:")
    print(f"  High text, high image:   {((text_results['text_quality']=='high') & (image_results['image_quality']=='high')).sum():>6,}")
    print(f"  High text, low image:    {((text_results['text_quality']=='high') & (image_results['image_quality']=='low')).sum():>6,}")
    print(f"  Low text, high image:    {((text_results['text_quality']=='low') & (image_results['image_quality']=='high')).sum():>6,}")
    print(f"  Low text, low image:     {((text_results['text_quality']=='low') & (image_results['image_quality']=='low')).sum():>6,}")
    
    print("\n✅ FEATURE ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
