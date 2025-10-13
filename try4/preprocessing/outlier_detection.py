"""
Advanced Multi-Strategy Outlier Detection

Implements multiple complementary outlier detection methods:
1. Isolation Forest - Detects anomalies in high-dimensional space
2. IQR Method - Classical statistical approach
3. Z-Score - Standard deviation based detection
4. DBSCAN - Density-based clustering

Uses ensemble voting: A sample is flagged as outlier if detected by
multiple methods (configurable threshold).

Priority: HIGHEST - Graceful outlier handling is critical for SMAPE improvement
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, ANALYSIS_DIR,
    OUTLIER_CONFIG, RANDOM_SEED
)

print("="*80)
print("🔍 ADVANCED MULTI-STRATEGY OUTLIER DETECTION")
print("="*80)
print("Methods: Isolation Forest, IQR, Z-Score, DBSCAN")
print(f"Ensemble Threshold: {OUTLIER_CONFIG['ensemble_threshold']} methods")
print("="*80)

# ============================================================================
# Individual Detection Methods
# ============================================================================

class IsolationForestDetector:
    """Isolation Forest - Effective for high-dimensional data"""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit_predict(self, X):
        """
        Fit and predict outliers
        
        Returns:
            Boolean array (True = outlier)
        """
        predictions = self.model.fit_predict(X)
        return predictions == -1  # -1 indicates outlier

class IQRDetector:
    """Interquartile Range - Robust statistical method"""
    
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
    
    def fit_predict(self, prices):
        """
        Detect outliers using IQR method
        
        Args:
            prices: 1D array of prices
            
        Returns:
            Boolean array (True = outlier)
        """
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.multiplier * IQR
        upper_bound = Q3 + self.multiplier * IQR
        
        return (prices < lower_bound) | (prices > upper_bound)

class ZScoreDetector:
    """Z-Score - Standard deviation based detection"""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def fit_predict(self, prices):
        """
        Detect outliers using Z-score
        
        Args:
            prices: 1D array of prices
            
        Returns:
            Boolean array (True = outlier)
        """
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return np.zeros(len(prices), dtype=bool)
        
        z_scores = np.abs((prices - mean) / std)
        return z_scores > self.threshold

class DBSCANDetector:
    """DBSCAN - Density-based clustering"""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    
    def fit_predict(self, X):
        """
        Detect outliers using DBSCAN
        
        Returns:
            Boolean array (True = outlier)
        """
        # Normalize data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        clusters = self.model.fit_predict(X_scaled)
        return clusters == -1  # -1 indicates noise/outlier

# ============================================================================
# Ensemble Outlier Detector
# ============================================================================

class EnsembleOutlierDetector:
    """
    Ensemble outlier detector using multiple methods
    
    A sample is flagged as outlier if detected by at least N methods
    (where N = ensemble_threshold)
    """
    
    def __init__(self, ensemble_threshold=2):
        self.ensemble_threshold = ensemble_threshold
        
        # Initialize detectors
        self.isolation_forest = IsolationForestDetector(
            **OUTLIER_CONFIG['isolation_forest']
        )
        self.iqr = IQRDetector(
            **OUTLIER_CONFIG['iqr']
        )
        self.zscore = ZScoreDetector(
            **OUTLIER_CONFIG['zscore']
        )
        self.dbscan = DBSCANDetector(
            **OUTLIER_CONFIG['dbscan']
        )
    
    def detect(self, df, feature_cols, price_col='price'):
        """
        Detect outliers using ensemble of methods
        
        Args:
            df: DataFrame with features and prices
            feature_cols: List of feature column names
            price_col: Name of price column
            
        Returns:
            DataFrame with outlier detection results
        """
        print("\n" + "="*80)
        print("RUNNING OUTLIER DETECTION")
        print("="*80)
        
        results = pd.DataFrame()
        results['sample_id'] = df['sample_id'] if 'sample_id' in df.columns else df.index
        results['price'] = df[price_col]
        
        # Prepare feature matrix
        X = df[feature_cols].values
        prices = df[price_col].values
        
        # Method 1: Isolation Forest (on features)
        print("\n🌲 Method 1: Isolation Forest")
        try:
            results['outlier_if'] = self.isolation_forest.fit_predict(X)
            print(f"   Detected: {results['outlier_if'].sum():,} outliers "
                  f"({100*results['outlier_if'].mean():.2f}%)")
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            results['outlier_if'] = False
        
        # Method 2: IQR (on prices)
        print("\n📊 Method 2: IQR")
        try:
            results['outlier_iqr'] = self.iqr.fit_predict(prices)
            print(f"   Detected: {results['outlier_iqr'].sum():,} outliers "
                  f"({100*results['outlier_iqr'].mean():.2f}%)")
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            results['outlier_iqr'] = False
        
        # Method 3: Z-Score (on prices)
        print("\n📈 Method 3: Z-Score")
        try:
            results['outlier_zscore'] = self.zscore.fit_predict(prices)
            print(f"   Detected: {results['outlier_zscore'].sum():,} outliers "
                  f"({100*results['outlier_zscore'].mean():.2f}%)")
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            results['outlier_zscore'] = False
        
        # Method 4: DBSCAN (on features)
        print("\n🔍 Method 4: DBSCAN")
        try:
            results['outlier_dbscan'] = self.dbscan.fit_predict(X)
            print(f"   Detected: {results['outlier_dbscan'].sum():,} outliers "
                  f"({100*results['outlier_dbscan'].mean():.2f}%)")
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            results['outlier_dbscan'] = False
        
        # Ensemble voting
        print("\n" + "="*80)
        print(f"🗳️ ENSEMBLE VOTING (Threshold: {self.ensemble_threshold} methods)")
        print("="*80)
        
        outlier_cols = ['outlier_if', 'outlier_iqr', 'outlier_zscore', 'outlier_dbscan']
        results['outlier_count'] = results[outlier_cols].sum(axis=1)
        results['is_outlier'] = results['outlier_count'] >= self.ensemble_threshold
        
        print(f"\n✓ Total outliers detected: {results['is_outlier'].sum():,} "
              f"({100*results['is_outlier'].mean():.2f}%)")
        
        # Statistics by detection count
        print("\nDetection Count Distribution:")
        print("-"*40)
        for count in range(5):
            n = (results['outlier_count'] == count).sum()
            pct = 100 * n / len(results)
            print(f"  {count} methods: {n:>6,} ({pct:>5.2f}%)")
        
        # Price statistics for outliers
        if results['is_outlier'].any():
            print("\nOutlier Price Statistics:")
            print("-"*40)
            outlier_prices = results[results['is_outlier']]['price']
            normal_prices = results[~results['is_outlier']]['price']
            
            print(f"  Outliers:")
            print(f"    Count:  {len(outlier_prices):,}")
            print(f"    Mean:   ${outlier_prices.mean():.2f}")
            print(f"    Median: ${outlier_prices.median():.2f}")
            print(f"    Min:    ${outlier_prices.min():.2f}")
            print(f"    Max:    ${outlier_prices.max():.2f}")
            
            print(f"  Normal:")
            print(f"    Count:  {len(normal_prices):,}")
            print(f"    Mean:   ${normal_prices.mean():.2f}")
            print(f"    Median: ${normal_prices.median():.2f}")
            print(f"    Min:    ${normal_prices.min():.2f}")
            print(f"    Max:    ${normal_prices.max():.2f}")
        
        return results

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_outlier_patterns(results_df):
    """Analyze patterns in detected outliers"""
    
    print("\n" + "="*80)
    print("📊 OUTLIER PATTERN ANALYSIS")
    print("="*80)
    
    outliers = results_df[results_df['is_outlier']]
    
    if len(outliers) == 0:
        print("No outliers detected!")
        return
    
    # Price range analysis
    print("\nPrice Range Distribution:")
    print("-"*40)
    
    bins = [0, 10, 20, 50, 100, np.inf]
    labels = ['$0-10', '$10-20', '$20-50', '$50-100', '$100+']
    
    outliers['price_range'] = pd.cut(outliers['price'], bins=bins, labels=labels)
    
    for label in labels:
        count = (outliers['price_range'] == label).sum()
        pct = 100 * count / len(outliers) if len(outliers) > 0 else 0
        print(f"  {label:10s}: {count:>5,} ({pct:>5.1f}%)")
    
    # Detection method overlap
    print("\nMethod Overlap:")
    print("-"*40)
    
    methods = ['outlier_if', 'outlier_iqr', 'outlier_zscore', 'outlier_dbscan']
    method_names = ['Isolation Forest', 'IQR', 'Z-Score', 'DBSCAN']
    
    for method, name in zip(methods, method_names):
        if method in outliers.columns:
            count = outliers[method].sum()
            pct = 100 * count / len(outliers)
            print(f"  {name:20s}: {count:>5,} ({pct:>5.1f}%)")

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main outlier detection pipeline"""
    
    # Set seed
    np.random.seed(RANDOM_SEED)
    
    print("\n📂 Loading data...")
    
    # Check if embeddings exist
    embeddings_exist = (
        (DATA_DIR.parent / 'try4' / 'outputs' / 'embeddings' / 'deberta_train_embeddings.csv').exists() and
        (DATA_DIR.parent / 'try4' / 'outputs' / 'embeddings' / 'clip_train_embeddings.csv').exists()
    )
    
    if embeddings_exist:
        print("✓ Loading embeddings...")
        deberta_df = pd.read_csv(DATA_DIR.parent / 'try4' / 'outputs' / 'embeddings' / 'deberta_train_embeddings.csv')
        clip_df = pd.read_csv(DATA_DIR.parent / 'try4' / 'outputs' / 'embeddings' / 'clip_train_embeddings.csv')
        
        # Load prices
        train1 = pd.read_csv(DATA_DIR / 'train1.csv')
        train2 = pd.read_csv(DATA_DIR / 'train2.csv')
        train_df = pd.concat([train1, train2], ignore_index=True)[['sample_id', 'price']]
        
        # Merge
        df = train_df.merge(deberta_df, on='sample_id').merge(clip_df, on='sample_id')
        
        # Feature columns (all except sample_id and price)
        feature_cols = [col for col in df.columns if col not in ['sample_id', 'price']]
        
        print(f"✓ Loaded embeddings: {len(feature_cols)} features")
    else:
        print("⚠️ Embeddings not found, using basic features...")
        
        # Load raw data and create basic features
        train1 = pd.read_csv(DATA_DIR / 'train1.csv')
        train2 = pd.read_csv(DATA_DIR / 'train2.csv')
        df = pd.concat([train1, train2], ignore_index=True)
        
        # Extract simple numeric features
        df['text_length'] = df['catalog_content'].fillna('').str.len()
        df['has_image'] = df['image_link'].notna().astype(int)
        df['price_log'] = np.log1p(df['price'])
        
        feature_cols = ['text_length', 'has_image', 'price_log']
        
        print(f"✓ Created {len(feature_cols)} basic features")
    
    print(f"✓ Total samples: {len(df):,}")
    
    # Initialize detector
    detector = EnsembleOutlierDetector(
        ensemble_threshold=OUTLIER_CONFIG['ensemble_threshold']
    )
    
    # Detect outliers
    results = detector.detect(df, feature_cols, price_col='price')
    
    # Analyze patterns
    analyze_outlier_patterns(results)
    
    # Save results
    output_path = ANALYSIS_DIR / 'outlier_detection_results.csv'
    results.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    print(f"✓ Saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    print(f"Total samples:       {len(results):,}")
    print(f"Outliers detected:   {results['is_outlier'].sum():,}")
    print(f"Outlier percentage:  {100*results['is_outlier'].mean():.2f}%")
    print(f"Normal samples:      {(~results['is_outlier']).sum():,}")
    
    print("\n✅ OUTLIER DETECTION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run outlier treatment: python preprocessing/outlier_treatment.py")
    print("  2. Train fusion model with treated data")

if __name__ == "__main__":
    main()
