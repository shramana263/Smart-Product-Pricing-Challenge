"""
Advanced Feature Engineering for Product Price Prediction (CLEAN - NO LEAKAGE)
Goal: Improve SMAPE from 47.52% to <45%

REMOVED LEAKAGE FEATURES:
- estimated_unit_price (uses target price)
- brand_avg_price, brand_median_price, brand_price_std (use target prices)
- category_avg_price, category_median_price (use target prices)

Key Improvements:
1. Interaction features (brand × category, value × unit)
2. Better quantity extraction (multi-pack, size indicators)
3. Advanced text features (brand reputation, size categories)
4. Ratio features
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for product pricing (NO TARGET LEAKAGE)
    """
    
    def __init__(self):
        self.brand_frequencies = {}
        self.category_sizes = {}
        self.brand_category_freq = {}
        
    def fit(self, df):
        """
        Fit the feature engineer on training data
        
        Args:
            df: Training DataFrame with columns including 'catalog_content'
        """
        print("\n" + "="*70)
        print("FITTING ADVANCED FEATURE ENGINEER (CLEAN VERSION)")
        print("="*70)
        
        # Parse catalog content
        df = self._parse_catalog_content(df)
        
        # Calculate brand frequencies
        self._fit_brand_stats(df)
        
        # Calculate category sizes
        self._fit_category_stats(df)
        
        print("\n✓ Feature engineer fitted successfully (NO LEAKAGE)")
        
        return self
    
    def transform(self, df):
        """
        Transform data with advanced features
        
        Args:
            df: DataFrame with 'catalog_content' column
            
        Returns:
            DataFrame with additional features
        """
        print("\n" + "="*70)
        print("TRANSFORMING DATA WITH ADVANCED FEATURES (CLEAN)")
        print("="*70)
        
        df = df.copy()
        
        # Parse catalog content
        df = self._parse_catalog_content(df)
        
        # 1. Basic text features
        df = self._add_basic_text_features(df)
        
        # 2. Advanced quantity features
        df = self._add_advanced_quantity_features(df)
        
        # 3. Brand features (NO PRICE LEAKAGE)
        df = self._add_brand_features(df)
        
        # 4. Category features (NO PRICE LEAKAGE)
        df = self._add_category_features(df)
        
        # 5. Interaction features
        df = self._add_interaction_features(df)
        
        # 6. Text quality features
        df = self._add_text_quality_features(df)
        
        # 7. Size category features
        df = self._add_size_category_features(df)
        
        # 8. Advanced text patterns
        df = self._add_advanced_text_patterns(df)
        
        print(f"\n✓ Transform completed. Shape: {df.shape}")
        
        return df
    
    def _parse_catalog_content(self, df):
        """Parse catalog_content into title, description, brand, and IPQ"""
        if 'title' not in df.columns:
            def parse_content(content):
                lines = content.split('\n')
                title = ''
                description = ''
                ipq = 1
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith('Item Name:'):
                        title = line.replace('Item Name:', '').strip()
                    elif line.startswith('Product Description:'):
                        # Description is everything after this line
                        description = '\n'.join(lines[i+1:]).strip()
                        break
                    elif 'Item Pack Quantity' in line:
                        # Extract IPQ
                        ipq_match = re.search(r'Item Pack Quantity\s*:\s*(\d+)', line, re.IGNORECASE)
                        ipq = int(ipq_match.group(1)) if ipq_match else 1
                
                return title, description, ipq
            
            parsed = df['catalog_content'].apply(parse_content)
            df['title'] = parsed.apply(lambda x: x[0])
            df['description'] = parsed.apply(lambda x: x[1])
            df['ipq'] = parsed.apply(lambda x: x[2])
        
        # Extract brand (first word of title)
        if 'brand' not in df.columns:
            def extract_brand(title):
                brand_match = re.search(r'^(\w+)', title)
                return brand_match.group(1).lower() if brand_match else 'unknown'
            
            df['brand'] = df['title'].apply(extract_brand)
        
        return df
    
    def _fit_brand_stats(self, df):
        """Fit brand statistics (frequencies only - NO PRICES)"""
        print("\nFitting brand statistics (frequency only)...")
        
        # Brand frequency
        brand_counts = df['brand'].value_counts().to_dict()
        self.brand_frequencies = brand_counts
        
        print(f"  ✓ Brands detected: {len(brand_counts)}")
    
    def _detect_category(self, title):
        """Detect product category from title"""
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
        
        title_lower = title.lower()
        for category, keywords in categories.items():
            if any(kw in title_lower for kw in keywords):
                return category
        return 'other'
    
    def _fit_category_stats(self, df):
        """Fit category statistics (sizes only - NO PRICES)"""
        print("\nFitting category statistics (size only)...")
        
        # Detect categories
        df['category'] = df['title'].apply(self._detect_category)
        
        # Category sizes
        category_counts = df['category'].value_counts().to_dict()
        self.category_sizes = category_counts
        
        # Brand-category co-occurrence
        brand_cat = df.groupby(['brand', 'category']).size().to_dict()
        self.brand_category_freq = brand_cat
        
        print(f"  ✓ Categories detected: {len(category_counts)}")
    
    def _add_basic_text_features(self, df):
        """Add basic text features"""
        print("\n1. Adding basic text features...")
        
        # Title features
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        # Description features
        df['desc_length'] = df['description'].str.len()
        df['desc_word_count'] = df['description'].str.split().str.len()
        df['has_description'] = (df['desc_length'] > 0).astype(int)
        
        # Capital letters (brand emphasis)
        df['capital_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        
        # Number count
        df['number_count'] = df['title'].str.count(r'\d+')
        
        print(f"  ✓ Added 7 basic text features")
        return df
    
    def _extract_quantity_features(self, title):
        """Extract all quantity-related features from title"""
        features = {
            'multipack_size': 1,
            'has_multipack': False,
            'weight_value': 0,
            'has_weight': False,
            'volume_value': 0,
            'has_volume': False,
            'total_numbers': 0
        }
        
        title_lower = title.lower()
        
        # Multipack patterns
        multipack_patterns = [
            r'(\d+)\s*pack',
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*count',
            r'(\d+)\s*pieces?',
            r'(\d+)\s*units?'
        ]
        
        for pattern in multipack_patterns:
            match = re.search(pattern, title_lower)
            if match:
                features['multipack_size'] = max(features['multipack_size'], int(match.group(1)))
                features['has_multipack'] = True
        
        # Weight patterns (g, kg, lb, oz, mg)
        weight_patterns = [
            r'(\d+\.?\d*)\s*(?:kg|kilogram)',
            r'(\d+\.?\d*)\s*(?:g|gram)',
            r'(\d+\.?\d*)\s*(?:lb|pound)',
            r'(\d+\.?\d*)\s*(?:oz|ounce)',
            r'(\d+\.?\d*)\s*mg'
        ]
        
        for pattern in weight_patterns:
            match = re.search(pattern, title_lower)
            if match:
                features['weight_value'] = max(features['weight_value'], float(match.group(1)))
                features['has_weight'] = True
        
        # Volume patterns (l, ml, fl oz)
        volume_patterns = [
            r'(\d+\.?\d*)\s*(?:l|liter|litre)',
            r'(\d+\.?\d*)\s*(?:ml|milliliter)',
            r'(\d+\.?\d*)\s*(?:fl\s*oz)',
            r'(\d+\.?\d*)\s*gallon'
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, title_lower)
            if match:
                features['volume_value'] = max(features['volume_value'], float(match.group(1)))
                features['has_volume'] = True
        
        # Total numbers in title
        features['total_numbers'] = len(re.findall(r'\d+\.?\d*', title))
        
        return features
    
    def _add_advanced_quantity_features(self, df):
        """Add advanced quantity extraction features"""
        print("\n2. Adding advanced quantity features...")
        
        # Extract quantity features
        qty_features = df['title'].apply(self._extract_quantity_features)
        
        df['multipack_size'] = qty_features.apply(lambda x: x['multipack_size'])
        df['has_multipack'] = qty_features.apply(lambda x: x['has_multipack']).astype(int)
        df['weight_value'] = qty_features.apply(lambda x: x['weight_value'])
        df['has_weight'] = qty_features.apply(lambda x: x['has_weight']).astype(int)
        df['volume_value'] = qty_features.apply(lambda x: x['volume_value'])
        df['has_volume'] = qty_features.apply(lambda x: x['has_volume']).astype(int)
        df['total_numbers_in_title'] = qty_features.apply(lambda x: x['total_numbers'])
        
        # Total quantity (IPQ × multipack)
        df['total_quantity'] = df['ipq'] * df['multipack_size']
        
        # Log transforms (better for modeling)
        df['log_total_quantity'] = np.log1p(df['total_quantity'])
        df['log_weight'] = np.log1p(df['weight_value'])
        df['log_volume'] = np.log1p(df['volume_value'])
        
        print(f"  ✓ Added 11 advanced quantity features")
        return df
    
    def _add_brand_features(self, df):
        """Add brand-based features (NO PRICE LEAKAGE)"""
        print("\n3. Adding brand features (no price leakage)...")
        
        # Brand frequency
        df['brand_frequency'] = df['brand'].map(self.brand_frequencies).fillna(0)
        
        # Brand popularity tiers (based on frequency, not price)
        brand_freq_median = df['brand_frequency'].median()
        brand_freq_q75 = df['brand_frequency'].quantile(0.75)
        
        df['brand_tier'] = 1  # Default: niche brand
        df.loc[df['brand_frequency'] >= brand_freq_median, 'brand_tier'] = 2  # Mid-tier
        df.loc[df['brand_frequency'] >= brand_freq_q75, 'brand_tier'] = 3  # Popular
        
        # Is popular brand
        df['is_popular_brand'] = (df['brand_frequency'] >= brand_freq_q75).astype(int)
        
        # Brand name length (indicator of complexity)
        df['brand_name_length'] = df['brand'].str.len()
        
        # Brand in title
        df['brand_in_title'] = df.apply(
            lambda row: int(row['brand'] in row['title'].lower()),
            axis=1
        )
        
        print(f"  ✓ Added 5 brand features (no leakage)")
        return df
    
    def _add_category_features(self, df):
        """Add category features (NO PRICE LEAKAGE)"""
        print("\n4. Adding category features (no price leakage)...")
        
        # Detect category (if not already done)
        if 'category' not in df.columns:
            df['category'] = df['title'].apply(self._detect_category)
        
        # Category size (number of products)
        df['category_size'] = df['category'].map(self.category_sizes).fillna(0)
        
        # Category diversity (is it a large/diverse category?)
        df['is_large_category'] = (df['category_size'] >= df['category_size'].quantile(0.5)).astype(int)
        
        print(f"  ✓ Added 3 category features (no leakage)")
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features"""
        print("\n5. Adding interaction features...")
        
        # Brand-category frequency
        df['brand_category_key'] = df['brand'] + '_' + df['category']
        df['brand_category_frequency'] = df['brand_category_key'].map(
            lambda x: self.brand_category_freq.get(tuple(x.split('_')), 0)
        )
        df.drop('brand_category_key', axis=1, inplace=True)
        
        # Quantity-category ratio
        category_mean_qty = df.groupby('category')['total_quantity'].transform('mean')
        df['quantity_category_ratio'] = df['total_quantity'] / (category_mean_qty + 1)
        
        # Tier-multipack interaction
        df['tier_multipack_interaction'] = df['brand_tier'] * df['has_multipack']
        
        # Brand tier × quantity
        df['brand_tier_quantity'] = df['brand_tier'] * df['log_total_quantity']
        
        # Weight × brand tier
        df['weight_brand_interaction'] = df['log_weight'] * df['brand_tier']
        
        print(f"  ✓ Added 5 interaction features")
        return df
    
    def _add_text_quality_features(self, df):
        """Add text quality and keyword features"""
        print("\n6. Adding text quality features...")
        
        # Premium keywords
        premium_keywords = ['organic', 'premium', 'artisan', 'gourmet', 'natural', 'pure']
        df['premium_keyword_count'] = df['title'].str.lower().apply(
            lambda x: sum(kw in x for kw in premium_keywords)
        )
        df['has_premium_keywords'] = (df['premium_keyword_count'] > 0).astype(int)
        
        # Value keywords
        value_keywords = ['value', 'economy', 'bulk', 'family', 'saver']
        df['value_keyword_count'] = df['title'].str.lower().apply(
            lambda x: sum(kw in x for kw in value_keywords)
        )
        
        # Quality keywords
        quality_keywords = ['fresh', 'quality', 'authentic', 'real', 'genuine']
        df['quality_keyword_count'] = df['title'].str.lower().apply(
            lambda x: sum(kw in x for kw in quality_keywords)
        )
        
        # Composite scores
        df['premium_score'] = (
            df['premium_keyword_count'] + 
            df['brand_tier'] + 
            df['quality_keyword_count']
        )
        
        df['value_score'] = df['value_keyword_count'] + df['has_multipack']
        
        print(f"  ✓ Added 7 text quality features")
        return df
    
    def _add_size_category_features(self, df):
        """Add size category indicators"""
        print("\n7. Adding size category features...")
        
        # Size categories
        df['is_travel_size'] = df['title'].str.lower().str.contains('travel|mini|sample|trial', regex=True).astype(int)
        df['is_family_size'] = df['title'].str.lower().str.contains('family|xl|large|jumbo|giant', regex=True).astype(int)
        df['is_multipack'] = df['title'].str.lower().str.contains('pack|count|piece|unit', regex=True).astype(int)
        df['is_standard_size'] = ((df['is_travel_size'] == 0) & (df['is_family_size'] == 0)).astype(int)
        
        print(f"  ✓ Added 4 size category features")
        return df
    
    def _add_advanced_text_patterns(self, df):
        """Add advanced text pattern features"""
        print("\n8. Adding advanced text patterns...")
        
        # Special characters count
        df['special_char_count'] = df['title'].apply(
            lambda x: sum(1 for c in x if c in '&-+()[]')
        )
        
        # Has parentheses (often indicates additional info)
        df['has_parentheses'] = df['title'].str.contains(r'\(', regex=True).astype(int)
        
        # Has ampersand (often brand combos or flavor descriptions)
        df['has_ampersand'] = df['title'].str.contains('&', regex=False).astype(int)
        
        # Title complexity (words per sentence)
        df['title_complexity'] = df['title_word_count'] / (df['title'].str.count(r'[,;]') + 1)
        
        # Numeric density
        df['numeric_density'] = df['number_count'] / (df['title_length'] + 1)
        
        # Category relevance (is category word in title?)
        df['category_in_title'] = df.apply(
            lambda row: int(row['category'] in row['title'].lower()),
            axis=1
        )
        
        print(f"  ✓ Added 6 advanced text pattern features")
        return df


def create_final_features_clean(train_df, test_df):
    """
    Create final feature set for modeling (NO TARGET LEAKAGE)
    
    Args:
        train_df: Training DataFrame (price not used for features)
        test_df: Test DataFrame
        
    Returns:
        train_features, test_features DataFrames
    """
    print("\n" + "="*70)
    print("CREATING FINAL FEATURE SET (CLEAN - NO LEAKAGE)")
    print("="*70)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Fit on training data (NO PRICE USED)
    engineer.fit(train_df)
    
    # Transform both datasets
    train_features = engineer.transform(train_df)
    test_features = engineer.transform(test_df)
    
    # Select numeric features
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove price and sample_id
    feature_cols = [c for c in numeric_cols if c not in ['price', 'sample_id']]
    
    print(f"\n✓ Features selected: {len(feature_cols)}")
    print(f"✓ NO TARGET LEAKAGE (price never used for features)")
    
    # Keep sample_id and price (if exists)
    if 'price' in train_features.columns:
        train_final = train_features[['sample_id', 'price'] + feature_cols]
    else:
        train_final = train_features[['sample_id'] + feature_cols]
    
    test_final = test_features[['sample_id'] + feature_cols]
    
    print(f"\n✓ Train shape: {train_final.shape}")
    print(f"✓ Test shape: {test_final.shape}")
    
    return train_final, test_final


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ADVANCED FEATURE ENGINEERING - CLEAN VERSION")
    print("NO TARGET LEAKAGE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir.parent / 'dataset'
    prep_dir = base_dir / 'preparation'

    train1 = pd.read_csv(dataset_dir / 'train1.csv')
    train2 = pd.read_csv(dataset_dir / 'train2.csv')
    test1 = pd.read_csv(dataset_dir / 'test1.csv')
    test2 = pd.read_csv(dataset_dir / 'test2.csv')
    
    train_df = pd.concat([train1, train2], ignore_index=True)
    test_df = pd.concat([test1, test2], ignore_index=True)
    
    print(f"✓ Train: {train_df.shape}")
    print(f"✓ Test: {test_df.shape}")
    
    # Create features
    train_features, test_features = create_final_features_clean(train_df, test_df)
    
    # Save features
    train_path = prep_dir / 'features_v3_clean_train.csv'
    test_path = prep_dir / 'features_v3_clean_test.csv'
    
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"\n✓ Features saved:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    # Display sample
    print(f"\n{'='*70}")
    print("FEATURE PREVIEW")
    print(f"{'='*70}")
    print("\nFirst 3 rows of training features:")
    print(train_features.head(3))
    
    print(f"\n{'='*70}")
    print("FEATURE LIST")
    print(f"{'='*70}")
    feature_list = [c for c in train_features.columns if c not in ['sample_id', 'price']]
    print(f"\nTotal features: {len(feature_list)}")
    for i, feat in enumerate(feature_list, 1):
        print(f"{i:2d}. {feat}")
    
    # Save feature list
    with open(prep_dir / 'features_v3_clean_list.txt', 'w') as f:
        f.write(f"Advanced Features V3 (CLEAN - NO LEAKAGE)\n")
        f.write(f"Total: {len(feature_list)} features\n")
        f.write("="*70 + "\n\n")
        for i, feat in enumerate(feature_list, 1):
            f.write(f"{i:2d}. {feat}\n")
    
    print(f"\n✓ Feature list saved: features_v3_clean_list.txt")
    print(f"\n{'='*70}")
    print("✓ FEATURE ENGINEERING COMPLETED (NO LEAKAGE!)")
    print(f"{'='*70}\n")
