"""
PHASE 1.2: UNIT STANDARDIZATION
================================
Expected Improvement: 49-50% → 47-48% SMAPE

Research Finding:
- Unit inconsistency causes 3-4 point SMAPE loss
- Example error: Actual=$691, Predicted=$30 (missing "per case")
- 156 unique unit variations (Oz/oz/ounce/Ounce all mean same)
- Nested quantities not extracted: "18 per pack × 16 per case" = 288 total

Strategy:
1. Standardize unit variations (oz/Oz/ounce → oz)
2. Extract nested quantities (pack × case = total_units)
3. Normalize price to per-unit basis
4. Add unit features to model

Time Estimate: 2-3 hours
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import auto-configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': DATA_DIR,
    'unit_dir': OUTPUT_DIR / "phase1_unit_standardization",
    'output_dir': OUTPUT_DIR / "phase1_unit_standardization",
    'random_seed': 42
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1.2: UNIT STANDARDIZATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

train1 = pd.read_csv(CONFIG['data_dir'] / "train1.csv")
train2 = pd.read_csv(CONFIG['data_dir'] / "train2.csv")
train = pd.concat([train1, train2], ignore_index=True)
print(f"✓ Loaded {len(train):,} training samples")

test1 = pd.read_csv(CONFIG['data_dir'] / "test1.csv")
test2 = pd.read_csv(CONFIG['data_dir'] / "test2.csv")
test = pd.concat([test1, test2], ignore_index=True)
print(f"✓ Loaded {len(test):,} test samples")

# ============================================================================
# UNIT MAPPING DICTIONARY
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DEFINE UNIT MAPPINGS")
print("="*80)

# Comprehensive unit standardization mapping
UNIT_MAPPINGS = {
    # Weight units
    'oz': ['oz', 'ounce', 'ounces', 'fl oz', 'fluid ounce', 'fluid ounces'],
    'lb': ['lb', 'lbs', 'pound', 'pounds'],
    'g': ['g', 'gram', 'grams', 'gm'],
    'kg': ['kg', 'kilogram', 'kilograms', 'kilo'],
    'mg': ['mg', 'milligram', 'milligrams'],
    
    # Volume units
    'ml': ['ml', 'milliliter', 'milliliters', 'millilitre', 'millilitres'],
    'l': ['l', 'liter', 'liters', 'litre', 'litres'],
    'gal': ['gal', 'gallon', 'gallons'],
    'qt': ['qt', 'quart', 'quarts'],
    'pt': ['pt', 'pint', 'pints'],
    'cup': ['cup', 'cups'],
    'tbsp': ['tbsp', 'tablespoon', 'tablespoons', 'tbs'],
    'tsp': ['tsp', 'teaspoon', 'teaspoons'],
    
    # Count units
    'count': ['count', 'ct', 'pieces', 'pcs', 'units', 'items'],
    'pack': ['pack', 'packs', 'pk', 'package', 'packages'],
    'box': ['box', 'boxes', 'bx'],
    'case': ['case', 'cases'],
    'bag': ['bag', 'bags'],
    'bottle': ['bottle', 'bottles', 'btl'],
    'can': ['can', 'cans'],
    'jar': ['jar', 'jars'],
    'tube': ['tube', 'tubes'],
    'roll': ['roll', 'rolls'],
    'sheet': ['sheet', 'sheets', 'ply'],
    
    # Length units
    'ft': ['ft', 'foot', 'feet', "'"],
    'in': ['in', 'inch', 'inches', '"'],
    'cm': ['cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres'],
    'm': ['m', 'meter', 'meters', 'metre', 'metres'],
    'yd': ['yd', 'yard', 'yards'],
    
    # Area/Square units
    'sqft': ['sqft', 'sq ft', 'square feet', 'square foot'],
    'sqin': ['sqin', 'sq in', 'square inches', 'square inch'],
    
    # Time units
    'day': ['day', 'days', 'd'],
    'hr': ['hr', 'hour', 'hours', 'hrs'],
    
    # Power/Energy
    'w': ['w', 'watt', 'watts'],
    'kwh': ['kwh', 'kilowatt hour', 'kilowatt-hour'],
    
    # Data
    'gb': ['gb', 'gigabyte', 'gigabytes'],
    'mb': ['mb', 'megabyte', 'megabytes'],
    
    # Serving size
    'serving': ['serving', 'servings', 'srv'],
    'dose': ['dose', 'doses'],
    'capsule': ['capsule', 'capsules', 'cap', 'caps'],
    'tablet': ['tablet', 'tablets', 'tab', 'tabs'],
}

# Create reverse lookup
STANDARD_UNIT = {}
for standard, variations in UNIT_MAPPINGS.items():
    for var in variations:
        STANDARD_UNIT[var.lower()] = standard

print(f"✓ Defined {len(UNIT_MAPPINGS)} standard units")
print(f"✓ Mapped {len(STANDARD_UNIT)} variations")

print("\n📋 Example Mappings:")
examples = [('Ounce', 'oz'), ('fl oz', 'oz'), ('lbs', 'lb'), ('milliliters', 'ml'), 
            ('Count', 'count'), ('Pieces', 'count'), ('sq ft', 'sqft')]
for orig, std in examples:
    print(f"   {orig:15s} → {std}")

# ============================================================================
# UNIT EXTRACTION FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DEFINE EXTRACTION FUNCTIONS")
print("="*80)

def extract_quantity_and_unit(text):
    """
    Extract quantity and unit from text
    
    Examples:
        "16 Ounce" → (16, 'oz')
        "2.5 lbs" → (2.5, 'lb')
        "Pack of 24" → (24, 'pack')
        "500ml" → (500, 'ml')
    """
    if not isinstance(text, str):
        return None, None
    
    text_lower = text.lower()
    
    # Pattern 1: "number unit" (e.g., "16 ounce", "2.5 lbs")
    pattern1 = r'(\d+\.?\d*)\s*(' + '|'.join(re.escape(k) for k in STANDARD_UNIT.keys()) + r')(?:\s|$|,|;)'
    match1 = re.search(pattern1, text_lower)
    if match1:
        qty = float(match1.group(1))
        unit_raw = match1.group(2)
        unit_std = STANDARD_UNIT.get(unit_raw, unit_raw)
        return qty, unit_std
    
    # Pattern 2: "unit of number" (e.g., "Pack of 24", "Box of 12")
    pattern2 = r'(' + '|'.join(re.escape(k) for k in STANDARD_UNIT.keys()) + r')\s*of\s*(\d+\.?\d*)'
    match2 = re.search(pattern2, text_lower)
    if match2:
        unit_raw = match2.group(1)
        qty = float(match2.group(2))
        unit_std = STANDARD_UNIT.get(unit_raw, unit_raw)
        return qty, unit_std
    
    # Pattern 3: "number-unit" (e.g., "500ml", "16oz")
    pattern3 = r'(\d+\.?\d*)(' + '|'.join(re.escape(k) for k in STANDARD_UNIT.keys()) + r')(?:\s|$|,|;)'
    match3 = re.search(pattern3, text_lower)
    if match3:
        qty = float(match3.group(1))
        unit_raw = match3.group(2)
        unit_std = STANDARD_UNIT.get(unit_raw, unit_raw)
        return qty, unit_std
    
    return None, None

def extract_nested_quantities(text):
    """
    Extract nested quantities like "24 per pack, 6 per case" → 144 total
    
    Examples:
        "24 Count (Pack of 6)" → 144
        "18 per pack × 16 per case" → 288
        "12 bottles, 6 packs" → 72
    """
    if not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Pattern 1: "X per Y, Z per W" or "X per Y × Z per W"
    pattern1 = r'(\d+\.?\d*)\s*(?:per|\/)\s*\w+[\s,×x]\s*(\d+\.?\d*)\s*(?:per|\/)\s*\w+'
    match1 = re.search(pattern1, text_lower)
    if match1:
        qty1 = float(match1.group(1))
        qty2 = float(match1.group(2))
        return qty1 * qty2
    
    # Pattern 2: "X (Pack of Y)" or "X (Box of Y)"
    pattern2 = r'(\d+\.?\d*)\s*\w*\s*\((?:pack|box|case|bag)\s+of\s+(\d+\.?\d*)\)'
    match2 = re.search(pattern2, text_lower)
    if match2:
        qty1 = float(match2.group(1))
        qty2 = float(match2.group(2))
        return qty1 * qty2
    
    # Pattern 3: "X, Y packs" or "X × Y" (simple multiplication)
    pattern3 = r'(\d+\.?\d*)\s*[,×x]\s*(\d+\.?\d*)\s*(?:packs|boxes|cases)?'
    match3 = re.search(pattern3, text_lower)
    if match3:
        qty1 = float(match3.group(1))
        qty2 = float(match3.group(2))
        # Only multiply if both are reasonable (< 1000)
        if qty1 < 1000 and qty2 < 1000:
            return qty1 * qty2
    
    return None

def extract_all_quantities(text):
    """
    Extract all quantity indicators from text
    
    Returns:
        - primary_qty: Main quantity (e.g., 16 oz)
        - primary_unit: Standardized unit (e.g., 'oz')
        - total_qty: Total items if nested (e.g., 144 from "24 per pack × 6")
        - multiplier: Factor for bulk quantities
    """
    if not isinstance(text, str):
        return None, None, None, 1.0
    
    # Primary quantity and unit
    primary_qty, primary_unit = extract_quantity_and_unit(text)
    
    # Nested quantity
    total_qty = extract_nested_quantities(text)
    
    # Calculate multiplier (if nested quantity exists)
    multiplier = 1.0
    if total_qty and primary_qty:
        multiplier = total_qty / primary_qty
    elif total_qty:
        multiplier = total_qty
    
    return primary_qty, primary_unit, total_qty, multiplier

print("✓ Extraction functions defined:")
print("   - extract_quantity_and_unit()")
print("   - extract_nested_quantities()")
print("   - extract_all_quantities()")

# ============================================================================
# APPLY UNIT EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: EXTRACT UNITS FROM DATA")
print("="*80)

print("\n🔍 Extracting from training data...")
train[['qty', 'unit', 'total_qty', 'multiplier']] = train['catalog_content'].apply(
    lambda x: pd.Series(extract_all_quantities(x))
)

print("🔍 Extracting from test data...")
test[['qty', 'unit', 'total_qty', 'multiplier']] = test['catalog_content'].apply(
    lambda x: pd.Series(extract_all_quantities(x))
)

# ============================================================================
# ANALYZE EXTRACTION RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: ANALYZE EXTRACTION RESULTS")
print("="*80)

print("\n📊 Extraction Coverage:")
train_with_qty = train['qty'].notna().sum()
train_with_unit = train['unit'].notna().sum()
train_with_nested = train['total_qty'].notna().sum()
train_with_multiplier = (train['multiplier'] > 1).sum()

print(f"   Training Set:")
print(f"      Quantity extracted:     {train_with_qty:>8,} ({train_with_qty/len(train)*100:5.2f}%)")
print(f"      Unit extracted:         {train_with_unit:>8,} ({train_with_unit/len(train)*100:5.2f}%)")
print(f"      Nested qty extracted:   {train_with_nested:>8,} ({train_with_nested/len(train)*100:5.2f}%)")
print(f"      Has multiplier (>1):    {train_with_multiplier:>8,} ({train_with_multiplier/len(train)*100:5.2f}%)")

test_with_qty = test['qty'].notna().sum()
test_with_unit = test['unit'].notna().sum()
test_with_nested = test['total_qty'].notna().sum()
test_with_multiplier = (test['multiplier'] > 1).sum()

print(f"\n   Test Set:")
print(f"      Quantity extracted:     {test_with_qty:>8,} ({test_with_qty/len(test)*100:5.2f}%)")
print(f"      Unit extracted:         {test_with_unit:>8,} ({test_with_unit/len(test)*100:5.2f}%)")
print(f"      Nested qty extracted:   {test_with_nested:>8,} ({test_with_nested/len(test)*100:5.2f}%)")
print(f"      Has multiplier (>1):    {test_with_multiplier:>8,} ({test_with_multiplier/len(test)*100:5.2f}%)")

print("\n📊 Top 20 Standardized Units:")
unit_counts = train['unit'].value_counts().head(20)
for unit, count in unit_counts.items():
    pct = (count / len(train)) * 100
    print(f"   {str(unit):15s}: {count:>8,} ({pct:5.2f}%)")

print("\n📊 Multiplier Distribution:")
multiplier_bins = [1, 2, 5, 10, 20, 50, 100, 500, float('inf')]
multiplier_labels = ['1 (no bulk)', '2-5', '5-10', '10-20', '20-50', '50-100', '100-500', '500+']
train['multiplier_bin'] = pd.cut(train['multiplier'], bins=multiplier_bins, labels=multiplier_labels)
for bin_label, count in train['multiplier_bin'].value_counts().sort_index().items():
    pct = (count / len(train)) * 100
    print(f"   {str(bin_label):15s}: {count:>8,} ({pct:5.2f}%)")

# ============================================================================
# CALCULATE PER-UNIT PRICE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CALCULATE PER-UNIT PRICE")
print("="*80)

# Calculate per-unit price
train['price_per_unit'] = train['price'] / train['multiplier']
train.loc[train['multiplier'] <= 1, 'price_per_unit'] = train.loc[train['multiplier'] <= 1, 'price']

print(f"\n📊 Per-Unit Price Statistics:")
print(f"   Original Price:")
print(f"      Mean:   ${train['price'].mean():.2f}")
print(f"      Median: ${train['price'].median():.2f}")
print(f"      Std:    ${train['price'].std():.2f}")

print(f"\n   Per-Unit Price (multiplier > 1):")
bulk = train[train['multiplier'] > 1]
print(f"      Mean:   ${bulk['price_per_unit'].mean():.2f}")
print(f"      Median: ${bulk['price_per_unit'].median():.2f}")
print(f"      Std:    ${bulk['price_per_unit'].std():.2f}")

# ============================================================================
# UNIT CATEGORY FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CREATE UNIT CATEGORY FEATURES")
print("="*80)

# Group units into broader categories
UNIT_CATEGORIES = {
    'weight': ['oz', 'lb', 'g', 'kg', 'mg'],
    'volume': ['ml', 'l', 'gal', 'qt', 'pt', 'cup', 'tbsp', 'tsp'],
    'count': ['count', 'pack', 'box', 'case', 'bag', 'bottle', 'can', 'jar', 'tube', 'roll', 'sheet'],
    'length': ['ft', 'in', 'cm', 'm', 'yd'],
    'area': ['sqft', 'sqin'],
    'time': ['day', 'hr'],
    'power': ['w', 'kwh'],
    'data': ['gb', 'mb'],
    'medical': ['serving', 'dose', 'capsule', 'tablet'],
}

def categorize_unit(unit):
    """Map unit to broader category"""
    if pd.isna(unit):
        return 'unknown'
    for category, units in UNIT_CATEGORIES.items():
        if unit in units:
            return category
    return 'other'

train['unit_category'] = train['unit'].apply(categorize_unit)
test['unit_category'] = test['unit'].apply(categorize_unit)

print("✓ Unit categories created")
print("\n📊 Unit Category Distribution:")
for category, count in train['unit_category'].value_counts().items():
    pct = (count / len(train)) * 100
    print(f"   {category:12s}: {count:>8,} ({pct:5.2f}%)")

# ============================================================================
# PRICE STATISTICS BY UNIT
# ============================================================================

print("\n" + "="*80)
print("STEP 8: PRICE STATISTICS BY UNIT CATEGORY")
print("="*80)

print("\n📊 Average Price by Unit Category:")
unit_price_stats = train.groupby('unit_category')['price'].agg(['mean', 'median', 'std', 'count'])
unit_price_stats = unit_price_stats.sort_values('mean', ascending=False)
for category, row in unit_price_stats.iterrows():
    print(f"   {category:12s}: Mean=${row['mean']:>7.2f}, Median=${row['median']:>7.2f}, Std=${row['std']:>7.2f} (n={int(row['count']):,})")

# ============================================================================
# EXAMPLE IMPROVEMENTS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: EXAMPLE IMPROVEMENTS")
print("="*80)

# Find examples where multiplier significantly changes interpretation
examples = train[(train['multiplier'] > 10) & (train['price'] > 100)].head(10)

print("\n💡 Examples of Corrected Bulk Quantities:")
print("-" * 100)
print(f"{'Catalog Text (excerpt)':<60} {'Price':<12} {'Multiplier':<12} {'Per-Unit':<12}")
print("-" * 100)

for idx, row in examples.iterrows():
    text_excerpt = row['catalog_content'][:57] + "..." if len(row['catalog_content']) > 60 else row['catalog_content']
    price_str = f"${row['price']:.2f}"
    mult_str = f"{row['multiplier']:.1f}x"
    per_unit_str = f"${row['price_per_unit']:.2f}"
    print(f"{text_excerpt:<60} {price_str:<12} {mult_str:<12} {per_unit_str:<12}")

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 10: SAVE PROCESSED DATA")
print("="*80)

# Save enriched training data
output_train = CONFIG['output_dir'] / 'train_with_units.csv'
train.to_csv(output_train, index=False)
print(f"✓ Saved training data: {output_train}")
print(f"   Columns added: qty, unit, total_qty, multiplier, price_per_unit, multiplier_bin, unit_category")

# Save enriched test data
output_test = CONFIG['output_dir'] / 'test_with_units.csv'
test.to_csv(output_test, index=False)
print(f"✓ Saved test data: {output_test}")

# Save unit mappings for reference
import json
output_mappings = CONFIG['output_dir'] / 'unit_mappings.json'
with open(output_mappings, 'w') as f:
    json.dump({
        'standard_units': list(UNIT_MAPPINGS.keys()),
        'unit_categories': UNIT_CATEGORIES,
        'total_variations': len(STANDARD_UNIT)
    }, f, indent=2)
print(f"✓ Saved unit mappings: {output_mappings}")

# ============================================================================
# INTEGRATION WITH MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 11: MODEL INTEGRATION GUIDE")
print("="*80)

integration_guide = """
# INTEGRATION WITH DISTILBERT MODEL

## Option 1: Add Unit Features to Model Input

Instead of just using `catalog_content` as text input, enrich it:

```python
def enrich_text_with_units(row):
    '''Add unit information to text for better understanding'''
    text = row['catalog_content']
    
    # Add standardized unit info
    if pd.notna(row['unit']):
        text += f" [Unit: {row['unit']}]"
    
    # Add multiplier info
    if row['multiplier'] > 1:
        text += f" [Quantity: {row['multiplier']:.0f}x]"
    
    # Add unit category
    if row['unit_category'] != 'unknown':
        text += f" [Type: {row['unit_category']}]"
    
    return text

# Apply to data
train['enriched_content'] = train.apply(enrich_text_with_units, axis=1)
test['enriched_content'] = test.apply(enrich_text_with_units, axis=1)

# Use enriched_content instead of catalog_content in DistilBERT
```

## Option 2: Train on Per-Unit Price (Then Scale Back)

For bulk items, train model to predict per-unit price, then multiply:

```python
# Training
train['target'] = train['price_per_unit']  # Train on per-unit

# Prediction
pred_per_unit = model.predict(test['enriched_content'])
final_pred = pred_per_unit * test['multiplier']  # Scale back to total
```

## Option 3: Separate Models by Unit Category

Train specialized models for different unit types:

```python
# Weight-based products (oz, lb, g, kg)
model_weight = train_on_subset(train[train['unit_category'] == 'weight'])

# Count-based products (pack, box, case)
model_count = train_on_subset(train[train['unit_category'] == 'count'])

# Volume-based products (ml, l, gal)
model_volume = train_on_subset(train[train['unit_category'] == 'volume'])

# Ensemble predictions
```

## Expected Impact

Based on research findings:

- Multiplier extraction: +2-3 SMAPE points
  (Fixes cases like $691 actual vs $30 predicted)

- Unit standardization: +1-2 SMAPE points
  (Reduces confusion between oz/Oz/ounce)

- Per-unit pricing: +1-2 SMAPE points
  (Better handles bulk quantity variations)

**Total Expected: -3 to -4 SMAPE points**

From 49-50% (after log transform) → 47-48% SMAPE
"""

with open(CONFIG['output_dir'] / 'integration_guide.txt', 'w') as f:
    f.write(integration_guide)

print(f"✓ Integration guide saved: {CONFIG['output_dir'] / 'integration_guide.txt'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ PHASE 1.2 COMPLETE")
print("="*80)

print(f"""
📁 Output Directory: {CONFIG['output_dir']}

📄 Files Created:
   1. train_with_units.csv      - Training data with unit features
   2. test_with_units.csv        - Test data with unit features
   3. unit_mappings.json         - Standardization mappings
   4. integration_guide.txt      - How to use in model

✨ Features Added:
   - qty: Primary quantity (e.g., 16 from "16 oz")
   - unit: Standardized unit (e.g., 'oz' from 'Ounce')
   - total_qty: Nested quantity (e.g., 144 from "24 pack of 6")
   - multiplier: Bulk factor (total_qty / qty)
   - price_per_unit: Price / multiplier
   - unit_category: Broader category (weight/volume/count/etc)
   - multiplier_bin: Binned multiplier ranges

📊 Extraction Success:
   - Quantity: {train_with_qty:,} / {len(train):,} ({train_with_qty/len(train)*100:.1f}%)
   - Unit: {train_with_unit:,} / {len(train):,} ({train_with_unit/len(train)*100:.1f}%)
   - Nested: {train_with_nested:,} / {len(train):,} ({train_with_nested/len(train)*100:.1f}%)
   - Multiplier: {train_with_multiplier:,} / {len(train):,} ({train_with_multiplier/len(train)*100:.1f}%)

🎯 Expected Impact:
   Current SMAPE: 49-50% (after Phase 1.1)
   After Unit Fix: 47-48% SMAPE
   Improvement: -2 to -3 points

💡 Key Insights:
   - {len(UNIT_MAPPINGS)} standard units defined
   - {len(STANDARD_UNIT)} unit variations standardized
   - {train_with_multiplier:,} products have bulk quantities (multiplier > 1)
   - {len(examples)} examples where multiplier significantly changes price interpretation

🚀 Next Steps:
   1. Integrate unit features with DistilBERT model
   2. Try Option 2 (per-unit price prediction)
   3. Validate improvement on holdout set
   4. Move to Phase 1.3 (Advanced Features)

🔗 Integration Options:
   - Option 1: Enrich text with unit tags
   - Option 2: Train on per-unit price
   - Option 3: Separate models by unit category
   
   See integration_guide.txt for details!
""")

print("\n" + "="*80)
print("Ready for model integration! 🚀")
print("="*80)
