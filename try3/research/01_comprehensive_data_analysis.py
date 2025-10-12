"""
COMPREHENSIVE DATA RESEARCH & ANALYSIS
======================================
Goal: Deep understanding of data patterns before feature engineering
This will reveal insights to drive our strategy to <41% SMAPE

Research Questions:
1. Price Distribution & Patterns
2. Category Analysis
3. Brand Intelligence
4. Quantity/Size Patterns
5. Text Content Analysis
6. Image-Text Correlation (sample)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Paths
DATA_DIR = Path("../../try2/dataset")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("COMPREHENSIVE DATA RESEARCH - PHASE 1: DATA LOADING & PARSING")
print("="*80)

# Load data (both parts)
print("\n📂 Loading training data...")
train1 = pd.read_csv(DATA_DIR / "train1.csv")
print(f"   ✓ Loaded {len(train1):,} samples from train1.csv")

train2 = pd.read_csv(DATA_DIR / "train2.csv")
print(f"   ✓ Loaded {len(train2):,} samples from train2.csv")

# Combine both parts
train = pd.concat([train1, train2], ignore_index=True)
print(f"   ✓ Combined total: {len(train):,} training samples")

# Parse catalog content
def parse_catalog(text):
    """Parse catalog_content into structured fields"""
    result = {
        'item_name': None,
        'value': None,
        'unit': None,
        'raw_text': text
    }
    
    if pd.isna(text):
        return result
    
    # Extract item name
    item_match = re.search(r'Item Name:\s*(.+?)(?:\n|$)', text)
    if item_match:
        result['item_name'] = item_match.group(1).strip()
    
    # Extract value
    value_match = re.search(r'Value:\s*(\d+\.?\d*)', text)
    if value_match:
        result['value'] = float(value_match.group(1))
    
    # Extract unit
    unit_match = re.search(r'Unit:\s*(.+?)(?:\n|$)', text)
    if unit_match:
        result['unit'] = unit_match.group(1).strip()
    
    return result

print("\n📝 Parsing catalog content...")
parsed = train['catalog_content'].apply(parse_catalog)
train['item_name'] = parsed.apply(lambda x: x['item_name'])
train['value'] = parsed.apply(lambda x: x['value'])
train['unit'] = parsed.apply(lambda x: x['unit'])

print(f"   ✓ Parsed {train['item_name'].notna().sum():,} item names")
print(f"   ✓ Parsed {train['value'].notna().sum():,} values")
print(f"   ✓ Parsed {train['unit'].notna().sum():,} units")

# ============================================================================
# RESEARCH AREA 1: PRICE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 1: PRICE DISTRIBUTION PATTERNS")
print("="*80)

price_stats = {
    'count': len(train),
    'mean': train['price'].mean(),
    'median': train['price'].median(),
    'std': train['price'].std(),
    'min': train['price'].min(),
    'max': train['price'].max(),
    'q25': train['price'].quantile(0.25),
    'q75': train['price'].quantile(0.75),
    'iqr': train['price'].quantile(0.75) - train['price'].quantile(0.25),
    'skewness': train['price'].skew(),
    'kurtosis': train['price'].kurtosis()
}

print("\n📊 Price Statistics:")
for key, value in price_stats.items():
    print(f"   {key:12s}: ${value:>12.2f}" if isinstance(value, float) else f"   {key:12s}: {value:>12,}")

# Price range distribution
print("\n💰 Price Range Distribution:")
bins = [0, 5, 10, 20, 30, 50, 100, 200, float('inf')]
labels = ['$0-5', '$5-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100-200', '$200+']
train['price_range'] = pd.cut(train['price'], bins=bins, labels=labels)
price_range_counts = train['price_range'].value_counts().sort_index()
for range_label, count in price_range_counts.items():
    pct = (count / len(train)) * 100
    print(f"   {range_label:12s}: {count:>8,} samples ({pct:5.2f}%)")

# Visualize price distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram
axes[0, 0].hist(train['price'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Price ($)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Price Distribution (Raw)', fontsize=14, fontweight='bold')
axes[0, 0].axvline(train['price'].median(), color='red', linestyle='--', label=f'Median: ${train["price"].median():.2f}')
axes[0, 0].legend()

# 2. Log-scale histogram
axes[0, 1].hist(np.log1p(train['price']), bins=100, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('log(Price + 1)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')

# 3. Box plot
axes[1, 0].boxplot(train['price'], vert=True)
axes[1, 0].set_ylabel('Price ($)', fontsize=12)
axes[1, 0].set_title('Price Box Plot (Outlier Detection)', fontsize=14, fontweight='bold')

# 4. QQ plot for normality
from scipy import stats
stats.probplot(train['price'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_price_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '01_price_distribution.png'}")

# ============================================================================
# RESEARCH AREA 2: UNIT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 2: UNIT & QUANTITY PATTERNS")
print("="*80)

print("\n📏 Top 20 Units:")
unit_counts = train['unit'].value_counts().head(20)
for unit, count in unit_counts.items():
    pct = (count / len(train)) * 100
    avg_price = train[train['unit'] == unit]['price'].mean()
    print(f"   {str(unit)[:30]:32s}: {count:>8,} ({pct:5.2f}%) | Avg Price: ${avg_price:>8.2f}")

# Value distribution by top units
print("\n📦 Value Statistics by Top Units:")
top_units = train['unit'].value_counts().head(10).index
for unit in top_units:
    unit_data = train[train['unit'] == unit]['value'].dropna()
    if len(unit_data) > 0:
        print(f"\n   {unit}:")
        print(f"      Mean: {unit_data.mean():.2f} | Median: {unit_data.median():.2f} | Std: {unit_data.std():.2f}")
        print(f"      Range: [{unit_data.min():.2f}, {unit_data.max():.2f}]")

# Visualize unit vs price
fig, ax = plt.subplots(figsize=(16, 10))
top_units_for_plot = train['unit'].value_counts().head(15).index
unit_price_data = [train[train['unit'] == unit]['price'].values for unit in top_units_for_plot]
bp = ax.boxplot(unit_price_data, labels=top_units_for_plot, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('Unit Type', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.set_title('Price Distribution by Unit Type (Top 15)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_unit_price_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '02_unit_price_distribution.png'}")

# ============================================================================
# RESEARCH AREA 3: BRAND INTELLIGENCE
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 3: BRAND INTELLIGENCE & PATTERNS")
print("="*80)

# Extract brand (first word/words before comma or hyphen)
def extract_brand(item_name):
    if pd.isna(item_name):
        return 'Unknown'
    
    # Common patterns:
    # "Brand Name Product..."
    # "Brand, Product..."
    
    # Take first 1-3 words
    words = item_name.split()
    if len(words) == 0:
        return 'Unknown'
    
    # Check for common separators
    first_part = item_name.split(',')[0].split('-')[0].split('(')[0].strip()
    
    # Take first 1-2 words
    brand_words = first_part.split()[:2]
    brand = ' '.join(brand_words)
    
    return brand if brand else 'Unknown'

print("\n🏷️ Extracting brands...")
train['brand'] = train['item_name'].apply(extract_brand)
print(f"   ✓ Extracted {train['brand'].nunique():,} unique brands")

print("\n🔝 Top 30 Brands by Frequency:")
brand_counts = train['brand'].value_counts().head(30)
for rank, (brand, count) in enumerate(brand_counts.items(), 1):
    pct = (count / len(train)) * 100
    avg_price = train[train['brand'] == brand]['price'].mean()
    median_price = train[train['brand'] == brand]['price'].median()
    print(f"   {rank:2d}. {brand[:35]:37s}: {count:>6,} ({pct:5.2f}%) | "
          f"Avg: ${avg_price:>7.2f} | Med: ${median_price:>7.2f}")

# Brand price tiers
print("\n💎 Brand Price Tier Analysis:")
brand_stats = train.groupby('brand').agg({
    'price': ['mean', 'median', 'std', 'count']
}).reset_index()
brand_stats.columns = ['brand', 'avg_price', 'median_price', 'std_price', 'count']
brand_stats = brand_stats[brand_stats['count'] >= 10]  # At least 10 samples

# Categorize brands by price tier
brand_stats['price_tier'] = pd.cut(
    brand_stats['avg_price'],
    bins=[0, 10, 20, 30, 50, float('inf')],
    labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
)

tier_distribution = brand_stats['price_tier'].value_counts().sort_index()
print("\n   Brand Distribution by Price Tier:")
for tier, count in tier_distribution.items():
    pct = (count / len(brand_stats)) * 100
    print(f"      {tier:12s}: {count:>5,} brands ({pct:5.2f}%)")

# Visualize top brands
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top 20 brands by count
top_20_brands = train['brand'].value_counts().head(20)
axes[0].barh(range(len(top_20_brands)), top_20_brands.values, color='steelblue')
axes[0].set_yticks(range(len(top_20_brands)))
axes[0].set_yticklabels(top_20_brands.index)
axes[0].set_xlabel('Number of Products', fontsize=12)
axes[0].set_title('Top 20 Brands by Product Count', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Brand price distribution (top 15)
top_15_brands = train['brand'].value_counts().head(15).index
brand_price_data = [train[train['brand'] == brand]['price'].values for brand in top_15_brands]
bp = axes[1].boxplot(brand_price_data, labels=top_15_brands, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
axes[1].set_xlabel('Brand', fontsize=12)
axes[1].set_ylabel('Price ($)', fontsize=12)
axes[1].set_title('Price Distribution by Brand (Top 15)', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_brand_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '03_brand_analysis.png'}")

# ============================================================================
# RESEARCH AREA 4: TEXT CONTENT PATTERNS
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 4: TEXT CONTENT & KEYWORD ANALYSIS")
print("="*80)

# Extract keywords and patterns
def extract_text_features(item_name):
    if pd.isna(item_name):
        return {}
    
    text_lower = item_name.lower()
    
    features = {
        # Size indicators
        'has_pack': bool(re.search(r'pack\s*of\s*\d+|(\d+)\s*pack', text_lower)),
        'pack_size': None,
        
        # Premium indicators
        'is_organic': 'organic' in text_lower,
        'is_premium': any(word in text_lower for word in ['premium', 'deluxe', 'luxury']),
        'is_natural': 'natural' in text_lower,
        
        # Size categories
        'is_travel_size': any(word in text_lower for word in ['travel', 'mini', 'portable']),
        'is_family_size': any(word in text_lower for word in ['family', 'bulk', 'jumbo', 'mega', 'super']),
        
        # Quality indicators
        'has_gourmet': 'gourmet' in text_lower,
        'has_fresh': 'fresh' in text_lower,
        'has_handmade': any(word in text_lower for word in ['handmade', 'artisan', 'craft']),
        
        # Numbers
        'number_count': len(re.findall(r'\d+', item_name)),
        'has_percentage': '%' in text_lower,
        
        # Length metrics
        'text_length': len(item_name),
        'word_count': len(item_name.split()),
    }
    
    # Extract pack size
    pack_match = re.search(r'pack\s*of\s*(\d+)|(\d+)\s*pack', text_lower)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1) or pack_match.group(2))
    
    return features

print("\n🔍 Extracting text features...")
text_features = train['item_name'].apply(extract_text_features)
text_df = pd.DataFrame(text_features.tolist())

print("\n📊 Text Pattern Statistics:")
print(f"   Products with pack info: {text_df['has_pack'].sum():,} ({(text_df['has_pack'].sum()/len(train))*100:.2f}%)")
print(f"   Organic products: {text_df['is_organic'].sum():,} ({(text_df['is_organic'].sum()/len(train))*100:.2f}%)")
print(f"   Premium products: {text_df['is_premium'].sum():,} ({(text_df['is_premium'].sum()/len(train))*100:.2f}%)")
print(f"   Natural products: {text_df['is_natural'].sum():,} ({(text_df['is_natural'].sum()/len(train))*100:.2f}%)")
print(f"   Travel size: {text_df['is_travel_size'].sum():,} ({(text_df['is_travel_size'].sum()/len(train))*100:.2f}%)")
print(f"   Family size: {text_df['is_family_size'].sum():,} ({(text_df['is_family_size'].sum()/len(train))*100:.2f}%)")

# Pack size analysis
pack_data = text_df[text_df['pack_size'].notna()]['pack_size']
if len(pack_data) > 0:
    print(f"\n📦 Pack Size Distribution:")
    print(f"   Mean pack size: {pack_data.mean():.2f}")
    print(f"   Median pack size: {pack_data.median():.0f}")
    print(f"   Most common pack sizes:")
    for size, count in pack_data.value_counts().head(10).items():
        print(f"      Pack of {size}: {count:,} products")

# Price comparison by text features
print("\n💰 Average Price by Text Features:")
for feature in ['is_organic', 'is_premium', 'is_natural', 'has_gourmet', 'is_travel_size', 'is_family_size']:
    mask = text_df[feature].fillna(False).astype(bool)
    has_feature = train[mask]['price'].mean()
    no_feature = train[~mask]['price'].mean()
    diff = has_feature - no_feature
    print(f"   {feature:20s}: WITH: ${has_feature:>7.2f} | WITHOUT: ${no_feature:>7.2f} | Diff: ${diff:>7.2f}")

# ============================================================================
# RESEARCH AREA 5: CATEGORY INFERENCE
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 5: CATEGORY INFERENCE FROM TEXT")
print("="*80)

# Define category keywords
category_keywords = {
    'Food & Beverage': ['coffee', 'tea', 'sauce', 'cookie', 'snack', 'chocolate', 'candy', 'food', 
                        'beverage', 'drink', 'juice', 'water', 'soda', 'chips', 'cereal', 'pasta'],
    'Beauty & Personal Care': ['cream', 'lotion', 'shampoo', 'soap', 'cosmetic', 'makeup', 'perfume',
                               'deodorant', 'toothpaste', 'beauty', 'skin', 'hair', 'nail'],
    'Health & Wellness': ['vitamin', 'supplement', 'protein', 'medicine', 'health', 'wellness',
                          'organic', 'natural', 'essential', 'oil'],
    'Home & Kitchen': ['towel', 'dish', 'kitchen', 'home', 'cleaning', 'paper', 'napkin', 'cup',
                       'plate', 'utensil', 'storage'],
    'Pet Supplies': ['dog', 'cat', 'pet', 'animal', 'puppy', 'kitten'],
    'Baby & Kids': ['baby', 'infant', 'toddler', 'kids', 'children', 'diaper', 'wipe'],
    'Office & School': ['pen', 'pencil', 'paper', 'notebook', 'office', 'school', 'stationery'],
}

def infer_category(item_name):
    if pd.isna(item_name):
        return 'Other'
    
    text_lower = item_name.lower()
    
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return 'Other'

print("\n🏪 Inferring product categories...")
train['inferred_category'] = train['item_name'].apply(infer_category)

print("\n📊 Category Distribution:")
category_counts = train['inferred_category'].value_counts()
for category, count in category_counts.items():
    pct = (count / len(train)) * 100
    avg_price = train[train['inferred_category'] == category]['price'].mean()
    median_price = train[train['inferred_category'] == category]['price'].median()
    print(f"   {category:25s}: {count:>8,} ({pct:5.2f}%) | "
          f"Avg: ${avg_price:>7.2f} | Med: ${median_price:>7.2f}")

# Visualize categories
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Category counts
category_counts.plot(kind='barh', ax=axes[0], color='teal')
axes[0].set_xlabel('Number of Products', fontsize=12)
axes[0].set_title('Product Distribution by Category', fontsize=14, fontweight='bold')

# Category price distribution
categories = train['inferred_category'].value_counts().index
cat_price_data = [train[train['inferred_category'] == cat]['price'].values for cat in categories]
bp = axes[1].boxplot(cat_price_data, labels=categories, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
axes[1].set_xlabel('Category', fontsize=12)
axes[1].set_ylabel('Price ($)', fontsize=12)
axes[1].set_title('Price Distribution by Category', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_category_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '04_category_analysis.png'}")

# ============================================================================
# RESEARCH AREA 6: UNIT-VALUE-PRICE RELATIONSHIP
# ============================================================================
print("\n" + "="*80)
print("RESEARCH AREA 6: UNIT-VALUE-PRICE RELATIONSHIP PATTERNS")
print("="*80)

# Calculate unit price (price per unit value)
train['unit_price'] = train['price'] / train['value']
train['unit_price'] = train['unit_price'].replace([np.inf, -np.inf], np.nan)

print("\n💵 Unit Price Analysis (price per unit value):")
unit_price_data = train[train['unit_price'].notna()]
print(f"   Samples with valid unit price: {len(unit_price_data):,} ({len(unit_price_data)/len(train)*100:.2f}%)")
print(f"   Mean unit price: ${unit_price_data['unit_price'].mean():.4f}")
print(f"   Median unit price: ${unit_price_data['unit_price'].median():.4f}")

# Unit price by unit type
print("\n📊 Average Unit Price by Unit Type (Top 10):")
top_units = train['unit'].value_counts().head(10).index
for unit in top_units:
    unit_data = train[train['unit'] == unit]
    if len(unit_data) > 0:
        avg_unit_price = unit_data['unit_price'].mean()
        median_unit_price = unit_data['unit_price'].median()
        print(f"   {str(unit)[:25]:27s}: Avg: ${avg_unit_price:>8.4f} | Med: ${median_unit_price:>8.4f}")

# Correlation analysis
print("\n🔗 Correlation Analysis:")
numeric_cols = ['price', 'value', 'unit_price']
correlation = train[numeric_cols].corr()
print("\n   Correlation Matrix:")
print(correlation.to_string())

# ============================================================================
# SAVE RESEARCH DATA
# ============================================================================
print("\n" + "="*80)
print("SAVING RESEARCH DATA")
print("="*80)

# Save enriched training data
research_train = train.copy()
research_train = pd.concat([research_train, text_df], axis=1)
research_train.to_csv(OUTPUT_DIR / 'research_train_enriched.csv', index=False)
print(f"\n✓ Saved enriched training data: {OUTPUT_DIR / 'research_train_enriched.csv'}")

# Save summary statistics
with open(OUTPUT_DIR / 'research_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE DATA RESEARCH SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*80 + "\n")
    f.write(f"Total samples: {len(train):,}\n")
    f.write(f"Unique brands: {train['brand'].nunique():,}\n")
    f.write(f"Unique units: {train['unit'].nunique():,}\n\n")
    
    f.write("PRICE STATISTICS\n")
    f.write("-"*80 + "\n")
    for key, value in price_stats.items():
        f.write(f"{key:12s}: ${value:>12.2f}\n" if isinstance(value, float) else f"{key:12s}: {value:>12,}\n")
    
    f.write("\n")
    f.write("TOP 20 BRANDS\n")
    f.write("-"*80 + "\n")
    for rank, (brand, count) in enumerate(brand_counts.items(), 1):
        avg_price = train[train['brand'] == brand]['price'].mean()
        f.write(f"{rank:2d}. {brand[:40]:42s}: {count:>6,} | Avg: ${avg_price:>7.2f}\n")

print(f"✓ Saved research summary: {OUTPUT_DIR / 'research_summary.txt'}")

print("\n" + "="*80)
print("✅ PHASE 1 RESEARCH COMPLETE")
print("="*80)
print("\nKey Outputs:")
print("   1. 01_price_distribution.png - Price distribution analysis")
print("   2. 02_unit_price_distribution.png - Unit type analysis")
print("   3. 03_brand_analysis.png - Brand intelligence")
print("   4. 04_category_analysis.png - Category patterns")
print("   5. research_train_enriched.csv - Enriched dataset")
print("   6. research_summary.txt - Statistical summary")
print("\n📋 Next: Review these insights before Phase 2 (Image Analysis)")
