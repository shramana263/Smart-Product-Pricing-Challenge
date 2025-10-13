"""
Graceful Outlier Treatment

Implements multiple strategies for handling detected outliers:
1. Winsorization - Cap extreme values at percentiles
2. Robust Scaling - Scale using quantile ranges
3. Log Transform - Compress extreme values
4. Separate Modeling - Train dedicated models for outliers
5. Confidence Weighting - Weight predictions by confidence

Priority: HIGHEST - Critical for improving SMAPE on extreme values
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    ANALYSIS_DIR, FEATURES_DIR,
    TREATMENT_CONFIG, RANDOM_SEED
)

print("="*80)
print("🔧 GRACEFUL OUTLIER TREATMENT")
print("="*80)
print("Strategies: Winsorization, Robust Scaling, Log Transform, Separate Modeling")
print("="*80)

# ============================================================================
# Treatment Strategies
# ============================================================================

class WinsorizerOutlierTreatment:
    """Winsorization - Cap extreme values"""
    
    def __init__(self, limits=(0.01, 0.99)):
        """
        Args:
            limits: (lower, upper) percentiles to cap at
        """
        self.limits = limits
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, prices):
        """Fit winsorization bounds"""
        self.lower_bound = np.percentile(prices, self.limits[0] * 100)
        self.upper_bound = np.percentile(prices, self.limits[1] * 100)
        return self
    
    def transform(self, prices):
        """Apply winsorization"""
        return np.clip(prices, self.lower_bound, self.upper_bound)
    
    def fit_transform(self, prices):
        """Fit and transform"""
        return self.fit(prices).transform(prices)

class RobustOutlierScaler:
    """Robust scaling using quantile ranges"""
    
    def __init__(self, quantile_range=(5, 95)):
        self.scaler = RobustScaler(quantile_range=quantile_range)
    
    def fit(self, X):
        """Fit scaler"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        """Transform features"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform"""
        return self.scaler.fit_transform(X)

class LogTransformTreatment:
    """Log transformation to compress extreme values"""
    
    def __init__(self, offset=1):
        self.offset = offset
    
    def transform(self, prices):
        """Apply log1p transform"""
        return np.log1p(prices)
    
    def inverse_transform(self, log_prices):
        """Inverse transform"""
        return np.expm1(log_prices)

# ============================================================================
# Outlier Segmentation
# ============================================================================

def segment_by_outlier_status(df, outlier_results):
    """
    Segment data into outlier and normal groups
    
    Args:
        df: DataFrame with features
        outlier_results: DataFrame with outlier detection results
        
    Returns:
        normal_df, outlier_df, normal_indices, outlier_indices
    """
    # Merge outlier flags
    df_merged = df.merge(
        outlier_results[['sample_id', 'is_outlier']],
        on='sample_id',
        how='left'
    )
    
    # Split
    normal_mask = ~df_merged['is_outlier'].fillna(False)
    outlier_mask = df_merged['is_outlier'].fillna(False)
    
    normal_df = df_merged[normal_mask].copy()
    outlier_df = df_merged[outlier_mask].copy()
    
    normal_indices = normal_mask.values
    outlier_indices = outlier_mask.values
    
    return normal_df, outlier_df, normal_indices, outlier_indices

# ============================================================================
# Price Range Segmentation
# ============================================================================

def segment_by_price_range(df, price_col='price'):
    """
    Segment data by price ranges for targeted treatment
    
    Returns:
        Dictionary of {range_name: (df, indices)}
    """
    segments = {}
    
    # Define price ranges
    ranges = [
        ('budget', 0, 10),
        ('economy', 10, 20),
        ('mid_range', 20, 50),
        ('premium', 50, 100),
        ('luxury', 100, np.inf)
    ]
    
    for name, lower, upper in ranges:
        mask = (df[price_col] >= lower) & (df[price_col] < upper)
        segments[name] = (df[mask].copy(), mask.values)
    
    return segments

# ============================================================================
# Main Treatment Pipeline
# ============================================================================

def apply_comprehensive_treatment(df, outlier_results):
    """
    Apply comprehensive outlier treatment
    
    Args:
        df: DataFrame with features and prices
        outlier_results: Outlier detection results
        
    Returns:
        Treated DataFrame with additional metadata
    """
    print("\n" + "="*80)
    print("APPLYING TREATMENT STRATEGIES")
    print("="*80)
    
    df_treated = df.copy()
    
    # ========================================================================
    # Strategy 1: Winsorization
    # ========================================================================
    
    print("\n📌 Strategy 1: Winsorization")
    print("-"*40)
    
    winsorizer = WinsorizerOutlierTreatment(
        limits=TREATMENT_CONFIG['winsorize_limits']
    )
    
    df_treated['price_winsorized'] = winsorizer.fit_transform(df['price'].values)
    
    n_capped = np.sum(df_treated['price_winsorized'] != df['price'])
    print(f"✓ Capped {n_capped:,} values ({100*n_capped/len(df):.2f}%)")
    print(f"  Lower bound: ${winsorizer.lower_bound:.2f}")
    print(f"  Upper bound: ${winsorizer.upper_bound:.2f}")
    
    # ========================================================================
    # Strategy 2: Log Transform
    # ========================================================================
    
    print("\n📐 Strategy 2: Log Transform")
    print("-"*40)
    
    log_transformer = LogTransformTreatment()
    df_treated['price_log'] = log_transformer.transform(df['price'].values)
    df_treated['price_winsorized_log'] = log_transformer.transform(df_treated['price_winsorized'].values)
    
    print(f"✓ Applied log1p transform")
    print(f"  Original range:    ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Log range:         {df_treated['price_log'].min():.3f} - {df_treated['price_log'].max():.3f}")
    print(f"  Winsorized range:  ${df_treated['price_winsorized'].min():.2f} - ${df_treated['price_winsorized'].max():.2f}")
    
    # ========================================================================
    # Strategy 3: Outlier Segmentation
    # ========================================================================
    
    print("\n🎯 Strategy 3: Outlier Segmentation")
    print("-"*40)
    
    normal_df, outlier_df, normal_idx, outlier_idx = segment_by_outlier_status(
        df_treated, outlier_results
    )
    
    df_treated['outlier_segment'] = 'normal'
    df_treated.loc[outlier_idx, 'outlier_segment'] = 'outlier'
    
    print(f"✓ Normal samples:  {len(normal_df):,} ({100*len(normal_df)/len(df):.2f}%)")
    print(f"✓ Outlier samples: {len(outlier_df):,} ({100*len(outlier_df)/len(df):.2f}%)")
    
    # ========================================================================
    # Strategy 4: Price Range Segmentation
    # ========================================================================
    
    print("\n💰 Strategy 4: Price Range Segmentation")
    print("-"*40)
    
    segments = segment_by_price_range(df_treated, price_col='price')
    
    df_treated['price_segment'] = 'unknown'
    for name, (seg_df, seg_idx) in segments.items():
        df_treated.loc[seg_idx, 'price_segment'] = name
        if len(seg_df) > 0:
            print(f"  {name:12s}: {len(seg_df):>6,} ({100*len(seg_df)/len(df):>5.2f}%) "
                  f"- Mean: ${seg_df['price'].mean():>7.2f}")
    
    # ========================================================================
    # Strategy 5: Confidence Scores
    # ========================================================================
    
    print("\n🎲 Strategy 5: Confidence Scoring")
    print("-"*40)
    
    # Confidence based on outlier detection count
    df_treated = df_treated.merge(
        outlier_results[['sample_id', 'outlier_count']],
        on='sample_id',
        how='left'
    )
    
    # Confidence score: 1.0 for normal, decreasing with outlier_count
    df_treated['confidence_score'] = 1.0 - (df_treated['outlier_count'].fillna(0) * 0.15)
    df_treated['confidence_score'] = df_treated['confidence_score'].clip(0.1, 1.0)
    
    print(f"✓ Confidence scores assigned")
    print(f"  Mean:   {df_treated['confidence_score'].mean():.3f}")
    print(f"  Median: {df_treated['confidence_score'].median():.3f}")
    print(f"  Min:    {df_treated['confidence_score'].min():.3f}")
    print(f"  Max:    {df_treated['confidence_score'].max():.3f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 TREATMENT SUMMARY")
    print("="*80)
    print(f"Original features:    {len(df.columns)}")
    print(f"Treated features:     {len(df_treated.columns)}")
    print(f"Added features:       {len(df_treated.columns) - len(df.columns)}")
    print(f"\nNew features:")
    new_features = [col for col in df_treated.columns if col not in df.columns]
    for feat in new_features:
        print(f"  - {feat}")
    
    return df_treated

# ============================================================================
# Robust Scaling of Features
# ============================================================================

def apply_robust_scaling(df, feature_cols):
    """
    Apply robust scaling to features
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to scale
        
    Returns:
        DataFrame with scaled features
    """
    print("\n" + "="*80)
    print("🔄 APPLYING ROBUST SCALING TO FEATURES")
    print("="*80)
    
    df_scaled = df.copy()
    
    scaler = RobustOutlierScaler(
        quantile_range=TREATMENT_CONFIG['robust_scaling']['quantile_range']
    )
    
    # Scale only numeric features
    numeric_features = [col for col in feature_cols 
                       if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Scaling {len(numeric_features)} numeric features...")
    
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    print(f"✓ Features scaled")
    print(f"  Original range: [{df[numeric_features].min().min():.3f}, {df[numeric_features].max().max():.3f}]")
    print(f"  Scaled range:   [{df_scaled[numeric_features].min().min():.3f}, {df_scaled[numeric_features].max().max():.3f}]")
    
    return df_scaled

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main treatment pipeline"""
    
    np.random.seed(RANDOM_SEED)
    
    print("\n📂 Loading data...")
    
    # Load outlier detection results
    outlier_results = pd.read_csv(ANALYSIS_DIR / 'outlier_detection_results.csv')
    print(f"✓ Loaded outlier results: {len(outlier_results):,} samples")
    
    # Load features (try embeddings first, fallback to basic)
    from pathlib import Path
    embeddings_dir = Path(__file__).parent.parent / 'outputs' / 'embeddings'
    
    if (embeddings_dir / 'deberta_train_embeddings.csv').exists():
        print("✓ Loading embeddings...")
        
        deberta_df = pd.read_csv(embeddings_dir / 'deberta_train_embeddings.csv')
        clip_df = pd.read_csv(embeddings_dir / 'clip_train_embeddings.csv')
        
        # Merge
        df = deberta_df.merge(clip_df, on='sample_id')
        
        # Add prices
        from config.config import DATA_DIR
        train1 = pd.read_csv(DATA_DIR / 'train1.csv')
        train2 = pd.read_csv(DATA_DIR / 'train2.csv')
        train_df = pd.concat([train1, train2], ignore_index=True)[['sample_id', 'price']]
        
        df = df.merge(train_df, on='sample_id')
        
        feature_cols = [col for col in df.columns if col not in ['sample_id', 'price']]
        
    else:
        print("⚠️ Embeddings not found")
        print("Please run feature extraction first:")
        print("  1. python feature_extraction/deberta_embeddings.py")
        print("  2. python feature_extraction/clip_embeddings.py")
        return
    
    print(f"✓ Loaded {len(df):,} samples with {len(feature_cols)} features")
    
    # Apply treatment
    df_treated = apply_comprehensive_treatment(df, outlier_results)
    
    # Apply robust scaling to embeddings
    df_scaled = apply_robust_scaling(df_treated, feature_cols)
    
    # Save treated data
    output_path = FEATURES_DIR / 'train_features_treated.csv'
    df_scaled.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {df_scaled.shape}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 FINAL SUMMARY")
    print("="*80)
    print(f"Total samples:        {len(df_scaled):,}")
    print(f"Total features:       {len(df_scaled.columns)}")
    print(f"Embedding features:   {len(feature_cols)}")
    print(f"Treatment features:   {len(df_scaled.columns) - len(feature_cols) - 2}")
    
    print(f"\nPrice statistics:")
    print(f"  Original:")
    print(f"    Mean:   ${df_scaled['price'].mean():.2f}")
    print(f"    Median: ${df_scaled['price'].median():.2f}")
    print(f"    Std:    ${df_scaled['price'].std():.2f}")
    
    print(f"  Winsorized:")
    print(f"    Mean:   ${df_scaled['price_winsorized'].mean():.2f}")
    print(f"    Median: ${df_scaled['price_winsorized'].median():.2f}")
    print(f"    Std:    ${df_scaled['price_winsorized'].std():.2f}")
    
    print("\n✅ OUTLIER TREATMENT COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Train fusion model: python modeling/fusion_model.py")
    print("  2. Or run full pipeline: python main_pipeline.py")

if __name__ == "__main__":
    main()
