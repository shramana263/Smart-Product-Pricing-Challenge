"""
Safe Target Encoding Implementation
Uses K-Fold Cross-Validation to prevent target leakage

Instead of using the full dataset's price averages (leakage),
we calculate target encoding separately for each fold using
only that fold's training data.

Expected Improvement: 5-10% SMAPE reduction from 63.28%
Target: <55% SMAPE
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

class SafeTargetEncoder:
    """
    Safe target encoder using K-Fold cross-validation
    Prevents target leakage by computing statistics only from training folds
    """
    
    def __init__(self, n_splits=5, smoothing=10, min_samples=10):
        """
        Args:
            n_splits: Number of folds for cross-validation
            smoothing: Smoothing factor (higher = more regularization toward global mean)
            min_samples: Minimum samples required to trust the encoding
        """
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.global_mean = None
        self.encodings = {}  # For test data
        
    def fit_transform(self, X, y, categorical_features):
        """
        Fit and transform using K-Fold cross-validation
        
        Args:
            X: Training features DataFrame
            y: Target variable (prices)
            categorical_features: List of categorical column names to encode
            
        Returns:
            X_encoded: DataFrame with added target encoding columns
        """
        X_encoded = X.copy()
        self.global_mean = y.mean()
        
        print(f"\n{'='*70}")
        print("SAFE TARGET ENCODING (K-Fold CV)")
        print(f"{'='*70}")
        print(f"Global mean price: ${self.global_mean:.2f}")
        print(f"Folds: {self.n_splits}")
        print(f"Smoothing: {self.smoothing}")
        print(f"Min samples: {self.min_samples}")
        
        for col in categorical_features:
            if col not in X.columns:
                continue
                
            print(f"\nEncoding: {col}")
            
            # Initialize target encoding column
            X_encoded[f'{col}_target_enc'] = np.nan
            
            # K-Fold cross-validation
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Get fold data
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                
                # Calculate statistics from training fold only
                encoding_map = self._calculate_encoding(
                    X_train_fold[col], 
                    y_train_fold
                )
                
                # Apply to validation fold
                X_encoded.loc[val_idx, f'{col}_target_enc'] = \
                    X.iloc[val_idx][col].map(encoding_map).fillna(self.global_mean)
            
            # Calculate final encoding for test data (from full training set)
            self.encodings[col] = self._calculate_encoding(X[col], y)
            
            unique_vals = X[col].nunique()
            print(f"  ✓ Encoded {unique_vals} unique values")
            print(f"  ✓ Range: ${X_encoded[f'{col}_target_enc'].min():.2f} - ${X_encoded[f'{col}_target_enc'].max():.2f}")
        
        return X_encoded
    
    def transform(self, X, categorical_features):
        """
        Transform test data using fitted encodings
        
        Args:
            X: Test features DataFrame
            categorical_features: List of categorical column names to encode
            
        Returns:
            X_encoded: DataFrame with added target encoding columns
        """
        X_encoded = X.copy()
        
        for col in categorical_features:
            if col not in X.columns or col not in self.encodings:
                continue
            
            X_encoded[f'{col}_target_enc'] = \
                X[col].map(self.encodings[col]).fillna(self.global_mean)
        
        return X_encoded
    
    def _calculate_encoding(self, categories, target):
        """
        Calculate target encoding with smoothing
        
        Formula: (n * mean + smoothing * global_mean) / (n + smoothing)
        Where n = count of category
        
        This provides:
        - Regularization toward global mean for rare categories
        - More weight to category mean for frequent categories
        """
        stats = pd.DataFrame({
            'mean': target.groupby(categories).mean(),
            'count': target.groupby(categories).count()
        })
        
        # Apply smoothing
        stats['encoding'] = (
            (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / 
            (stats['count'] + self.smoothing)
        )
        
        return stats['encoding'].to_dict()


def create_safe_target_encoded_features():
    """Create features with safe target encoding"""
    
    print("\n" + "="*70)
    print("CREATING FEATURES WITH SAFE TARGET ENCODING")
    print("="*70)
    
    # Load clean features
    print("\nLoading clean features...")
    train_df = pd.read_csv('./preparation/features_v3_clean_train.csv')
    test_df = pd.read_csv('./preparation/features_v3_clean_test.csv')
    
    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")
    
    # Load raw data to get brand and category strings
    print("\nLoading raw data for brand/category extraction...")
    train_raw1 = pd.read_csv('./dataset/train1.csv')
    train_raw2 = pd.read_csv('./dataset/train2.csv')
    train_raw = pd.concat([train_raw1, train_raw2], ignore_index=True)
    
    test_raw1 = pd.read_csv('./dataset/test1.csv')
    test_raw2 = pd.read_csv('./dataset/test2.csv')
    test_raw = pd.concat([test_raw1, test_raw2], ignore_index=True)
    
    # Extract brand and category from catalog_content
    def extract_brand(content):
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('Item Name:'):
                title = line.replace('Item Name:', '').strip()
                # Get first word as brand
                brand_match = re.search(r'^(\w+)', title)
                return brand_match.group(1).lower() if brand_match else 'unknown'
        return 'unknown'
    
    def detect_category(content):
        title_lower = content.lower()
        categories = {
            'supplement': ['vitamin', 'supplement', 'protein', 'omega'],
            'beverage': ['juice', 'drink', 'tea', 'coffee', 'water'],
            'snack': ['chip', 'cracker', 'cookie', 'candy', 'snack'],
            'condiment': ['sauce', 'dressing', 'ketchup', 'mustard'],
            'personal_care': ['shampoo', 'soap', 'lotion', 'cream'],
            'grain': ['rice', 'pasta', 'cereal', 'oat'],
            'canned': ['canned', 'jar'],
            'frozen': ['frozen'],
            'dairy': ['milk', 'cheese', 'yogurt']
        }
        
        for category, keywords in categories.items():
            if any(kw in title_lower for kw in keywords):
                return category
        return 'other'
    
    train_raw['brand'] = train_raw['catalog_content'].apply(extract_brand)
    train_raw['category'] = train_raw['catalog_content'].apply(detect_category)
    test_raw['brand'] = test_raw['catalog_content'].apply(extract_brand)
    test_raw['category'] = test_raw['catalog_content'].apply(detect_category)
    
    # Merge with features
    train_df = train_df.merge(
        train_raw[['sample_id', 'brand', 'category']], 
        on='sample_id', 
        how='left'
    )
    test_df = test_df.merge(
        test_raw[['sample_id', 'brand', 'category']], 
        on='sample_id', 
        how='left'
    )
    
    print(f"✓ Added brand and category columns")
    print(f"  Train brands: {train_df['brand'].nunique()}")
    print(f"  Train categories: {train_df['category'].nunique()}")
    
    # Separate features and target
    X_train = train_df.drop(['sample_id', 'price'], axis=1)
    y_train = train_df['price']
    X_test = test_df.drop(['sample_id'], axis=1)
    
    # Categorical features to encode
    categorical_features = ['brand', 'category']
    
    # Initialize safe encoder
    encoder = SafeTargetEncoder(
        n_splits=5,
        smoothing=10,  # Balance between category mean and global mean
        min_samples=10
    )
    
    # Fit and transform training data
    X_train_encoded = encoder.fit_transform(X_train, y_train, categorical_features)
    
    # Transform test data
    X_test_encoded = encoder.transform(X_test, categorical_features)
    
    # Add back sample_id and price
    train_encoded_df = pd.concat([
        train_df[['sample_id', 'price']],
        X_train_encoded
    ], axis=1)
    
    test_encoded_df = pd.concat([
        test_df[['sample_id']],
        X_test_encoded
    ], axis=1)
    
    # Save encoded features
    train_path = './preparation/features_v4_safe_target_train.csv'
    test_path = './preparation/features_v4_safe_target_test.csv'
    
    train_encoded_df.to_csv(train_path, index=False)
    test_encoded_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*70}")
    print("FEATURES SAVED")
    print(f"{'='*70}")
    print(f"✓ Train: {train_path}")
    print(f"  Shape: {train_encoded_df.shape}")
    print(f"✓ Test: {test_path}")
    print(f"  Shape: {test_encoded_df.shape}")
    
    # Feature summary
    new_features = [col for col in X_train_encoded.columns if '_target_enc' in col]
    print(f"\n✓ Added {len(new_features)} safe target encoding features:")
    for feat in new_features:
        print(f"  - {feat}")
    
    print(f"\n✓ Total features: {len(X_train_encoded.columns)}")
    print(f"✅ NO TARGET LEAKAGE (CV-based encoding)")
    
    return train_encoded_df, test_encoded_df


if __name__ == "__main__":
    train_df, test_df = create_safe_target_encoded_features()
    
    print("\n" + "="*70)
    print("✓ SAFE TARGET ENCODING COMPLETED!")
    print("="*70)
    print("\nNext step: Train models with new features!")
    print("Expected: 5-10% SMAPE improvement from 63.28%")
    print("Target: <55% SMAPE")
