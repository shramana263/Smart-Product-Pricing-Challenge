"""
Advanced Feature Engineering for Product Price Prediction
Goal: Improve SMAPE from 47.52% to <45%

Key Improvements:
1. Interaction features (brand × category, value × unit)
2. Better quantity extraction (multi-pack, size indicators)
3. Price tier features (brand clustering by price)
4. Advanced text features (brand reputation, size categories)
5. Ratio features
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
    Advanced feature engineering for product pricing
    """
    
    def __init__(self):
        self.brand_price_stats = {}
        self.brand_clusters = {}
        self.category_stats = {}
        
    def fit(self, df):
        """
        Fit the feature engineer on training data
        
        Args:
            df: Training DataFrame with columns including 'catalog_content', 'price'
        """
        print("\n" + "="*70)
        print("FITTING ADVANCED FEATURE ENGINEER")
        print("="*70)
        
        # Parse catalog content
        df = self._parse_catalog_content(df)
        
        # Calculate brand price statistics
        self._fit_brand_stats(df)
        
        # Calculate category statistics
        self._fit_category_stats(df)
        
        # Cluster brands by price
        self._fit_brand_clusters(df)
        
        print("\n✓ Feature engineer fitted successfully")
        
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
        print("TRANSFORMING DATA WITH ADVANCED FEATURES")
        print("="*70)
        
        df = df.copy()
        
        # Parse catalog content
        df = self._parse_catalog_content(df)
        
        # 1. Basic text features
        df = self._add_basic_text_features(df)
        
        # 2. Advanced quantity features
        df = self._add_advanced_quantity_features(df)
        
        # 3. Brand features
        df = self._add_brand_features(df)
        
        # 4. Category features
        df = self._add_category_features(df)
        
        # 5. Interaction features
        df = self._add_interaction_features(df)
        
        # 6. Text quality features
        df = self._add_text_quality_features(df)
        
        # 7. Size category features
        df = self._add_size_category_features(df)
        
        # 8. Price indicator features
        df = self._add_price_indicator_features(df)
        
        print(f"\n✓ Transform completed. Shape: {df.shape}")
        
        return df
    
    def _parse_catalog_content(self, df):
        """Parse catalog_content into components"""
        
        if 'title' not in df.columns:
            print("\nParsing catalog_content...")
            
            def parse_content(content):
                try:
                    if pd.isna(content):
                        return '', '', '', ''
                    
                    content = str(content)
                    
                    # Extract title (first part before description)
                    title_match = re.search(r'^(.*?)(?:\s*Description\s*:|$)', content, re.IGNORECASE)
                    title = title_match.group(1).strip() if title_match else content[:100]
                    
                    # Extract description
                    desc_match = re.search(r'Description\s*:\s*(.*?)(?:\s*Item Pack Quantity|$)', content, re.IGNORECASE)
                    description = desc_match.group(1).strip() if desc_match else ''
                    
                    # Extract IPQ
                    ipq_match = re.search(r'Item Pack Quantity\s*:\s*(\d+)', content, re.IGNORECASE)
                    ipq = int(ipq_match.group(1)) if ipq_match else 1
                    
                    # Extract brand (first word, even if lowercase)
                    brand_match = re.search(r'^(\w+)', title)
                    brand = brand_match.group(1).lower() if brand_match else 'unknown'
                    
                    # Clean brand
                    brand = re.sub(r'[^\w]', '', brand)[:20]  # Max 20 chars
                    
                    return title, description, brand, ipq
                    
                except:
                    return '', '', 'unknown', 1
            
            parsed = df['catalog_content'].apply(parse_content)
            df['title'] = parsed.apply(lambda x: x[0])
            df['description'] = parsed.apply(lambda x: x[1])
            df['brand'] = parsed.apply(lambda x: x[2])
            df['ipq'] = parsed.apply(lambda x: x[3])
            
            print(f"  ✓ Parsed catalog content")
        
        return df
    
    def _fit_brand_stats(self, df):
        """Calculate brand price statistics"""
        if 'price' in df.columns:
            print("\nCalculating brand price statistics...")
            brand_stats = df.groupby('brand')['price'].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).to_dict('index')
            
            self.brand_price_stats = brand_stats
            print(f"  ✓ Calculated stats for {len(brand_stats)} brands")
    
    def _fit_category_stats(self, df):
        """Calculate category price statistics"""
        if 'price' in df.columns:
            print("\nCalculating category statistics...")
            
            # Detect category from text
            df['category_detected'] = df['title'].apply(self._detect_category)
            
            category_stats = df.groupby('category_detected')['price'].agg([
                'mean', 'median', 'std', 'count'
            ]).to_dict('index')
            
            self.category_stats = category_stats
            print(f"  ✓ Calculated stats for {len(category_stats)} categories")
    
    def _fit_brand_clusters(self, df):
        """Cluster brands into price tiers"""
        if 'price' in df.columns:
            print("\nClustering brands by price tier...")
            
            brand_avg_price = df.groupby('brand')['price'].mean()
            
            # Only cluster if we have enough brands
            n_brands = len(brand_avg_price)
            n_clusters = min(5, max(2, n_brands // 10))  # Adaptive clustering
            
            if n_brands >= n_clusters:
                X = brand_avg_price.values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                self.brand_clusters = dict(zip(brand_avg_price.index, clusters))
                print(f"  ✓ Clustered {len(self.brand_clusters)} brands into {n_clusters} tiers")
            else:
                # If too few brands, assign all to middle tier
                self.brand_clusters = {brand: 2 for brand in brand_avg_price.index}
                print(f"  ✓ Too few brands ({n_brands}), assigned all to default tier")
    
    def _add_basic_text_features(self, df):
        """Add basic text features"""
        print("\n1. Adding basic text features...")
        
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['desc_length'] = df['description'].str.len()
        df['desc_word_count'] = df['description'].str.split().str.len()
        
        # Has description
        df['has_description'] = (df['desc_length'] > 0).astype(int)
        
        # Capital letters ratio
        df['capital_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        # Number count in title
        df['number_count'] = df['title'].str.count(r'\d+')
        
        print(f"  ✓ Added 7 basic text features")
        return df
    
    def _add_advanced_quantity_features(self, df):
        """Add advanced quantity extraction features"""
        print("\n2. Adding advanced quantity features...")
        
        def extract_all_quantities(text):
            """Extract all numeric values and units"""
            text = str(text).lower()
            
            # Multi-pack patterns
            multipack = re.findall(r'(\d+)\s*(?:pack|pk|count|ct|pcs|pieces)', text)
            
            # Weight/volume patterns
            weights = re.findall(r'(\d+(?:\.\d+)?)\s*(?:oz|ounce|lb|pound|g|gram|kg|kilogram)', text)
            volumes = re.findall(r'(\d+(?:\.\d+)?)\s*(?:fl oz|fluid ounce|ml|l|liter|gallon)', text)
            
            # Count patterns
            counts = re.findall(r'(\d+)\s*(?:count|ct|pcs)', text)
            
            return {
                'multipack': int(multipack[0]) if multipack else 1,
                'has_multipack': len(multipack) > 0,
                'weight_value': float(weights[0]) if weights else 0,
                'has_weight': len(weights) > 0,
                'volume_value': float(volumes[0]) if volumes else 0,
                'has_volume': len(volumes) > 0,
                'count_value': int(counts[0]) if counts else 0,
                'total_numbers': len(re.findall(r'\d+', text))
            }
        
        qty_features = df['title'].apply(extract_all_quantities)
        
        df['multipack_size'] = qty_features.apply(lambda x: x['multipack'])
        df['has_multipack'] = qty_features.apply(lambda x: x['has_multipack']).astype(int)
        df['weight_value'] = qty_features.apply(lambda x: x['weight_value'])
        df['has_weight'] = qty_features.apply(lambda x: x['has_weight']).astype(int)
        df['volume_value'] = qty_features.apply(lambda x: x['volume_value'])
        df['has_volume'] = qty_features.apply(lambda x: x['has_volume']).astype(int)
        df['total_numbers_in_title'] = qty_features.apply(lambda x: x['total_numbers'])
        
        # Total quantity (IPQ × multipack)
        df['total_quantity'] = df['ipq'] * df['multipack_size']
        
        # Unit price estimation
        df['estimated_unit_price'] = 0.0
        mask = df['total_quantity'] > 0
        if 'price' in df.columns:
            df.loc[mask, 'estimated_unit_price'] = df.loc[mask, 'price'] / df.loc[mask, 'total_quantity']
        
        print(f"  ✓ Added 9 advanced quantity features")
        return df
    
    def _add_brand_features(self, df):
        """Add brand-based features"""
        print("\n3. Adding brand features...")
        
        # Brand frequency
        brand_counts = df['brand'].value_counts()
        df['brand_frequency'] = df['brand'].map(brand_counts)
        
        # Brand price statistics
        df['brand_avg_price'] = df['brand'].map(
            lambda x: self.brand_price_stats.get(x, {}).get('mean', 0)
        )
        df['brand_median_price'] = df['brand'].map(
            lambda x: self.brand_price_stats.get(x, {}).get('median', 0)
        )
        df['brand_price_std'] = df['brand'].map(
            lambda x: self.brand_price_stats.get(x, {}).get('std', 0)
        )
        df['brand_product_count'] = df['brand'].map(
            lambda x: self.brand_price_stats.get(x, {}).get('count', 0)
        )
        
        # Brand tier (cluster)
        df['brand_tier'] = df['brand'].map(self.brand_clusters).fillna(2)
        
        # Is popular brand
        df['is_popular_brand'] = (df['brand_frequency'] > df['brand_frequency'].quantile(0.75)).astype(int)
        
        print(f"  ✓ Added 7 brand features")
        return df
    
    def _add_category_features(self, df):
        """Add category features"""
        print("\n4. Adding category features...")
        
        # Detect category
        df['category'] = df['title'].apply(self._detect_category)
        
        # Category price stats
        df['category_avg_price'] = df['category'].map(
            lambda x: self.category_stats.get(x, {}).get('mean', 0)
        )
        df['category_median_price'] = df['category'].map(
            lambda x: self.category_stats.get(x, {}).get('median', 0)
        )
        
        # Category size
        category_counts = df['category'].value_counts()
        df['category_size'] = df['category'].map(category_counts)
        
        print(f"  ✓ Added 4 category features")
        return df
    
    def _detect_category(self, text):
        """Detect product category from text"""
        text = str(text).lower()
        
        if any(word in text for word in ['coffee', 'tea', 'juice', 'soda', 'water', 'drink', 'beverage']):
            return 'beverage'
        elif any(word in text for word in ['sauce', 'ketchup', 'mustard', 'mayo', 'dressing', 'vinegar']):
            return 'condiment'
        elif any(word in text for word in ['chip', 'cookie', 'candy', 'chocolate', 'snack', 'cracker']):
            return 'snack'
        elif any(word in text for word in ['rice', 'pasta', 'bread', 'cereal', 'grain', 'flour']):
            return 'grain'
        elif any(word in text for word in ['vitamin', 'supplement', 'protein', 'capsule', 'tablet']):
            return 'supplement'
        elif any(word in text for word in ['soap', 'shampoo', 'lotion', 'cream', 'care']):
            return 'personal_care'
        else:
            return 'other'
    
    def _add_interaction_features(self, df):
        """Add interaction features"""
        print("\n5. Adding interaction features...")
        
        # Brand × Category
        df['brand_category_interaction'] = df['brand'].astype(str) + '_' + df['category'].astype(str)
        
        # Brand-category frequency
        brand_cat_counts = df['brand_category_interaction'].value_counts()
        df['brand_category_frequency'] = df['brand_category_interaction'].map(brand_cat_counts)
        
        # Quantity × Category (different categories have different price/quantity relationships)
        df['quantity_category_ratio'] = df['total_quantity'] / (df['category_size'] + 1)
        
        # Brand tier × has multipack
        df['tier_multipack_interaction'] = df['brand_tier'] * df['has_multipack']
        
        print(f"  ✓ Added 4 interaction features")
        return df
    
    def _add_text_quality_features(self, df):
        """Add text quality and marketing features"""
        print("\n6. Adding text quality features...")
        
        # Premium keywords
        premium_keywords = ['premium', 'gourmet', 'organic', 'artisan', 'luxury', 'deluxe', 
                          'imported', 'authentic', 'craft', 'signature', 'select', 'choice']
        
        df['premium_keyword_count'] = df['title'].apply(
            lambda x: sum(1 for word in premium_keywords if word in str(x).lower())
        )
        df['has_premium_keywords'] = (df['premium_keyword_count'] > 0).astype(int)
        
        # Value keywords
        value_keywords = ['value', 'economy', 'budget', 'pack', 'bulk', 'family size']
        df['value_keyword_count'] = df['title'].apply(
            lambda x: sum(1 for word in value_keywords if word in str(x).lower())
        )
        
        # Quality indicators
        quality_keywords = ['fresh', 'natural', 'pure', 'real', 'whole', 'best']
        df['quality_keyword_count'] = df['title'].apply(
            lambda x: sum(1 for word in quality_keywords if word in str(x).lower())
        )
        
        # Has brand in title (often indicates authenticity)
        df['brand_in_title'] = df.apply(
            lambda row: int(row['brand'] in str(row['title']).lower()), axis=1
        )
        
        print(f"  ✓ Added 6 text quality features")
        return df
    
    def _add_size_category_features(self, df):
        """Add size category features"""
        print("\n7. Adding size category features...")
        
        def categorize_size(text):
            text = str(text).lower()
            if any(word in text for word in ['travel', 'mini', 'sample', 'trial']):
                return 'travel'
            elif any(word in text for word in ['family', 'bulk', 'jumbo', 'large', 'xl', 'economy']):
                return 'family'
            elif any(word in text for word in ['pack', 'case', 'box of']):
                return 'multipack'
            else:
                return 'standard'
        
        df['size_category'] = df['title'].apply(categorize_size)
        
        # One-hot encode size category
        size_dummies = pd.get_dummies(df['size_category'], prefix='size')
        df = pd.concat([df, size_dummies], axis=1)
        
        print(f"  ✓ Added size category features")
        return df
    
    def _add_price_indicator_features(self, df):
        """Add features that are strong price indicators"""
        print("\n8. Adding price indicator features...")
        
        # Price per unit estimates based on category
        df['is_high_value_category'] = df['category'].isin(['supplement', 'personal_care']).astype(int)
        df['is_low_value_category'] = df['category'].isin(['condiment', 'grain']).astype(int)
        
        # Composite features
        df['brand_tier_quantity'] = df['brand_tier'] * np.log1p(df['total_quantity'])
        df['premium_score'] = (
            df['premium_keyword_count'] + 
            df['brand_tier'] + 
            df['quality_keyword_count']
        )
        
        # Value score (inverse of premium)
        df['value_score'] = df['value_keyword_count'] + df['has_multipack']
        
        print(f"  ✓ Added 6 price indicator features")
        return df


def create_final_features(train_df, test_df):
    """
    Create final feature set for modeling
    
    Args:
        train_df: Training DataFrame with price
        test_df: Test DataFrame without price
        
    Returns:
        train_features, test_features DataFrames
    """
    print("\n" + "="*70)
    print("CREATING FINAL FEATURE SET")
    print("="*70)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Fit on training data
    engineer.fit(train_df)
    
    # Transform both datasets
    train_features = engineer.transform(train_df)
    test_features = engineer.transform(test_df)
    
    # Select numeric features
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-feature columns
    exclude_cols = ['price'] if 'price' in numeric_cols else []
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\n✓ Final feature count: {len(feature_cols)}")
    print(f"✓ Train shape: {train_features[feature_cols].shape}")
    print(f"✓ Test shape: {test_features[feature_cols].shape}")
    
    return train_features, test_features, feature_cols


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED FEATURE ENGINEERING - MAIN SCRIPT")
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
    
    print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Create features
    train_features, test_features, feature_cols = create_final_features(train_df, test_df)
    
    # Save features
    print("\nSaving features...")
    
    # Train features
    train_output = train_features[feature_cols + ['sample_id', 'price']].copy()
    train_output.to_csv(prep_dir / 'features_v2_train.csv', index=False)
    print(f"✓ Train features saved: features_v2_train.csv")
    
    # Test features  
    test_output = test_features[feature_cols + ['sample_id']].copy()
    test_output.to_csv(prep_dir / 'features_v2_test.csv', index=False)
    print(f"✓ Test features saved: features_v2_test.csv")
    
    # Feature list
    with open(prep_dir / 'features_v2_list.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"✓ Feature list saved: features_v2_list.txt")
    
    print("\n" + "="*70)
    print("✓ FEATURE ENGINEERING COMPLETED!")
    print("="*70)
    print(f"\nNew features: {len(feature_cols)}")
    print(f"Files created:")
    print(f"  - features_v2_train.csv ({train_output.shape})")
    print(f"  - features_v2_test.csv ({test_output.shape})")
    print(f"  - features_v2_list.txt")
    print(f"\nNext step: Train models with new features!")
