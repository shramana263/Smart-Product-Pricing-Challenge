"""
Engineered Tabular Features

Extracts handcrafted features from product data:
- Text statistics (length, word count, etc.)
- Brand and category encoding
- Price-related features
- Quantity and unit extraction
- Target encoding with CV

These features complement deep embeddings with domain knowledge.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from config.config import (
    DATA_DIR, FEATURES_DIR,
    TABULAR_FEATURES, RANDOM_SEED
)

print("="*80)
print("⚙️ ENGINEERED TABULAR FEATURES")
print("="*80)

# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_text_features(df):
    """Extract text-based features"""
    print("\n📝 Extracting text features...")
    
    df['text'] = df['catalog_content'].fillna('')
    
    # Basic statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).lower().split())))
    df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
    
    # Special characters
    df['digit_count'] = df['text'].str.count(r'\d')
    df['upper_count'] = df['text'].str.count(r'[A-Z]')
    df['special_char_count'] = df['text'].str.count(r'[^a-zA-Z0-9\s]')
    
    # Line counts
    df['line_count'] = df['text'].str.count('\n') + 1
    
    print(f"  ✓ Created {8} text features")
    
    return df

def extract_brand_category(df):
    """Extract brand and category from text"""
    print("\n🏷️ Extracting brand and category...")
    
    def extract_brand(content):
        if pd.isna(content):
            return 'unknown'
        lines = str(content).split('\n')
        for line in lines:
            if 'Item Name:' in line:
                title = line.replace('Item Name:', '').strip()
                brand_match = re.search(r'^(\w+)', title)
                return brand_match.group(1).lower() if brand_match else 'unknown'
        return 'unknown'
    
    def detect_category(content):
        if pd.isna(content):
            return 'other'
        
        content_lower = str(content).lower()
        categories = {
            'supplement': ['vitamin', 'supplement', 'protein', 'omega', 'probiotic'],
            'beverage': ['juice', 'drink', 'tea', 'coffee', 'water', 'soda'],
            'snack': ['chip', 'cracker', 'cookie', 'candy', 'snack', 'bar'],
            'condiment': ['sauce', 'dressing', 'ketchup', 'mustard', 'mayo'],
            'personal_care': ['shampoo', 'soap', 'lotion', 'cream', 'wash'],
            'grain': ['rice', 'pasta', 'cereal', 'oat', 'wheat'],
            'canned': ['canned', 'jar'],
            'frozen': ['frozen'],
            'dairy': ['milk', 'cheese', 'yogurt'],
            'meat': ['chicken', 'beef', 'pork', 'turkey']
        }
        
        for category, keywords in categories.items():
            if any(kw in content_lower for kw in keywords):
                return category
        return 'other'
    
    df['brand'] = df['catalog_content'].apply(extract_brand)
    df['category'] = df['catalog_content'].apply(detect_category)
    
    print(f"  ✓ Extracted {df['brand'].nunique()} unique brands")
    print(f"  ✓ Extracted {df['category'].nunique()} unique categories")
    
    return df

def extract_quantity_features(df):
    """Extract quantity and unit information"""
    print("\n📦 Extracting quantity features...")
    
    def extract_quantity(content):
        if pd.isna(content):
            return np.nan
        
        # Look for patterns like "24 oz", "1.5 kg", "500ml"
        patterns = [
            r'(\d+\.?\d*)\s*(oz|ounce|pound|lb|kg|gram|g|ml|liter|l|count|pack)',
            r'(\d+\.?\d*)(oz|lb|kg|g|ml|l)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(content).lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return np.nan
    
    def extract_unit(content):
        if pd.isna(content):
            return 'unknown'
        
        units = ['oz', 'ounce', 'pound', 'lb', 'kg', 'gram', 'g', 'ml', 'liter', 'l', 'count', 'pack']
        content_lower = str(content).lower()
        
        for unit in units:
            if unit in content_lower:
                return unit
        return 'unknown'
    
    df['quantity'] = df['catalog_content'].apply(extract_quantity)
    df['unit'] = df['catalog_content'].apply(extract_unit)
    df['has_quantity'] = df['quantity'].notna().astype(int)
    
    # Normalized quantity (rough conversion to oz)
    unit_to_oz = {
        'oz': 1, 'ounce': 1,
        'pound': 16, 'lb': 16,
        'kg': 35.27, 'gram': 0.035, 'g': 0.035,
        'ml': 0.034, 'liter': 33.8, 'l': 33.8,
        'count': 1, 'pack': 1, 'unknown': 1
    }
    
    df['normalized_quantity'] = df.apply(
        lambda row: row['quantity'] * unit_to_oz.get(row['unit'], 1) 
        if pd.notna(row['quantity']) else np.nan,
        axis=1
    )
    
    print(f"  ✓ Quantity available: {df['has_quantity'].sum()} samples")
    print(f"  ✓ Created normalized quantity feature")
    
    return df

def extract_price_features(df):
    """Extract price-derived features"""
    print("\n💰 Extracting price features...")
    
    if 'price' not in df.columns:
        print("  ⚠️ Price column not found, skipping")
        return df
    
    df['price_log'] = np.log1p(df['price'])
    df['price_sqrt'] = np.sqrt(df['price'])
    df['price_squared'] = df['price'] ** 2
    
    # Price per unit
    df['price_per_unit'] = df['price'] / (df['normalized_quantity'] + 1)
    
    # Price bins
    df['price_bin'] = pd.cut(
        df['price'],
        bins=[0, 10, 20, 50, 100, np.inf],
        labels=['budget', 'economy', 'mid', 'premium', 'luxury']
    )
    
    print(f"  ✓ Created {5} price-related features")
    
    return df

def extract_image_features(df):
    """Extract image-related features"""
    print("\n🖼️ Extracting image features...")
    
    df['has_image'] = df['image_link'].notna().astype(int)
    df['image_count'] = df['image_link'].fillna('').str.count(';') + 1
    df['image_count'] = df['image_count'].where(df['has_image'] == 1, 0)
    
    print(f"  ✓ Image availability: {df['has_image'].sum()} samples")
    
    return df

def safe_target_encode(train_df, test_df, categorical_features, target_col='price'):
    """Apply safe target encoding using K-Fold CV"""
    print("\n🎯 Applying safe target encoding...")
    
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    global_mean = train_df[target_col].mean()
    smoothing = TABULAR_FEATURES['smoothing']
    n_folds = TABULAR_FEATURES['target_encoding_folds']
    
    print(f"  Global mean: ${global_mean:.2f}")
    print(f"  Smoothing:   {smoothing}")
    print(f"  Folds:       {n_folds}")
    
    for col in categorical_features:
        if col not in train_df.columns:
            continue
        
        print(f"  Encoding: {col}")
        
        # Initialize encoding column
        train_encoded[f'{col}_target_enc'] = np.nan
        
        # K-Fold CV for training
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        for train_idx, val_idx in kf.split(train_df):
            X_train_fold = train_df.iloc[train_idx]
            y_train_fold = train_df.iloc[train_idx][target_col]
            
            # Calculate encoding
            stats = pd.DataFrame({
                'mean': y_train_fold.groupby(X_train_fold[col]).mean(),
                'count': y_train_fold.groupby(X_train_fold[col]).count()
            })
            
            stats['encoding'] = (
                (stats['count'] * stats['mean'] + smoothing * global_mean) /
                (stats['count'] + smoothing)
            )
            
            encoding_map = stats['encoding'].to_dict()
            
            # Apply to validation fold
            train_encoded.loc[val_idx, f'{col}_target_enc'] = \
                train_df.iloc[val_idx][col].map(encoding_map).fillna(global_mean)
        
        # For test: use full training set
        stats = pd.DataFrame({
            'mean': train_df[target_col].groupby(train_df[col]).mean(),
            'count': train_df[target_col].groupby(train_df[col]).count()
        })
        
        stats['encoding'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )
        
        encoding_map = stats['encoding'].to_dict()
        test_encoded[f'{col}_target_enc'] = \
            test_df[col].map(encoding_map).fillna(global_mean)
    
    return train_encoded, test_encoded

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main feature engineering pipeline"""
    
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
    
    # Extract features
    print("\n" + "="*80)
    print("FEATURE EXTRACTION")
    print("="*80)
    
    for df in [train_df, test_df]:
        df = extract_text_features(df)
        df = extract_brand_category(df)
        df = extract_quantity_features(df)
        df = extract_image_features(df)
        
        if 'price' in df.columns:
            df = extract_price_features(df)
    
    # Target encoding (only for categorical features with manageable cardinality)
    categorical_features = ['brand', 'category', 'unit', 'price_bin']
    
    train_df, test_df = safe_target_encode(
        train_df, test_df, 
        categorical_features,
        target_col='price'
    )
    
    # Select final features
    feature_cols = [
        'text_length', 'word_count', 'char_count', 'unique_word_count',
        'avg_word_length', 'digit_count', 'upper_count', 'special_char_count',
        'line_count', 'has_quantity', 'normalized_quantity', 'has_image',
        'image_count', 'price_log', 'price_sqrt', 'price_squared',
        'price_per_unit', 'brand_target_enc', 'category_target_enc',
        'unit_target_enc', 'price_bin_target_enc'
    ]
    
    # Remove features not in test set
    feature_cols = [col for col in feature_cols if col in test_df.columns]
    
    # Prepare final DataFrames
    train_features = train_df[['sample_id'] + feature_cols + ['price']].copy()
    test_features = test_df[['sample_id'] + feature_cols].copy()
    
    # Fill NaN values
    for col in feature_cols:
        if train_features[col].dtype in ['float64', 'int64']:
            train_features[col].fillna(train_features[col].median(), inplace=True)
            test_features[col].fillna(train_features[col].median(), inplace=True)
    
    # Save
    train_path = FEATURES_DIR / 'tabular_train_features.csv'
    test_path = FEATURES_DIR / 'tabular_test_features.csv'
    
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print("\n" + "="*80)
    print("💾 SAVING FEATURES")
    print("="*80)
    print(f"✓ Train: {train_path}")
    print(f"  Shape: {train_features.shape}")
    print(f"✓ Test:  {test_path}")
    print(f"  Shape: {test_features.shape}")
    
    print(f"\n✓ Total engineered features: {len(feature_cols)}")
    print("\nFeature list:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    print("\n✅ TABULAR FEATURE ENGINEERING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
