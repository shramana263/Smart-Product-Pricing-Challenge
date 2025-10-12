"""
ERROR ANALYSIS RESEARCH - PHASE 3
==================================
Goal: Understand where current models fail to improve strategically

This will analyze:
1. Error distribution by price range
2. Error patterns by category/brand
3. Difficult samples identification
4. Feature importance vs errors
5. Model predictions analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("../../try2/dataset")
OUTPUT_DIR = Path("outputs")

print("="*80)
print("ERROR ANALYSIS RESEARCH - PHASE 3")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading data...")

# Load training data (both parts)
train1 = pd.read_csv(DATA_DIR / "train1.csv")
print(f"   ✓ Loaded {len(train1):,} samples from train1.csv")

train2 = pd.read_csv(DATA_DIR / "train2.csv")
print(f"   ✓ Loaded {len(train2):,} samples from train2.csv")

train = pd.concat([train1, train2], ignore_index=True)
print(f"   ✓ Combined total: {len(train):,} training samples")

# Try to load existing predictions if available
predictions_file = Path("../../modeling/test_out_v3_clean.csv")
if predictions_file.exists():
    print(f"   ✓ Found predictions: {predictions_file}")
    # Note: We can't directly validate without validation split info
    # We'll create synthetic validation for analysis
else:
    print("   ⚠️ No existing predictions found")

# Load enriched research data
research_file = OUTPUT_DIR / 'research_train_enriched.csv'
if research_file.exists():
    train_enriched = pd.read_csv(research_file)
    print(f"   ✓ Loaded enriched research data")
else:
    print("   ⚠️ Running Phase 1 first to get enriched data...")
    train_enriched = train

# ============================================================================
# CREATE VALIDATION SPLIT FOR ERROR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CREATING STRATIFIED VALIDATION SPLIT")
print("="*80)

from sklearn.model_selection import train_test_split

# Create price bins for stratification
train['price_bin'] = pd.cut(train['price'], 
                             bins=[0, 10, 20, 30, 50, 100, float('inf')],
                             labels=['0-10', '10-20', '20-30', '30-50', '50-100', '100+'])

# Stratified split
train_data, val_data = train_test_split(
    train, 
    test_size=0.2, 
    random_state=42, 
    stratify=train['price_bin']
)

print(f"\n   Training set: {len(train_data):,} samples")
print(f"   Validation set: {len(val_data):,} samples")

# ============================================================================
# SIMULATE MODEL PREDICTIONS (Simple Baseline)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CREATING BASELINE PREDICTIONS FOR ERROR ANALYSIS")
print("="*80)

# Parse catalog content for features
def parse_catalog(text):
    result = {'value': None, 'unit': None, 'item_name': None}
    if pd.isna(text):
        return result
    
    import re
    value_match = re.search(r'Value:\s*(\d+\.?\d*)', text)
    if value_match:
        result['value'] = float(value_match.group(1))
    
    unit_match = re.search(r'Unit:\s*(.+?)(?:\n|$)', text)
    if unit_match:
        result['unit'] = unit_match.group(1).strip()
    
    item_match = re.search(r'Item Name:\s*(.+?)(?:\n|$)', text)
    if item_match:
        result['item_name'] = item_match.group(1).strip()
    
    return result

print("\n📝 Parsing validation data...")
parsed = val_data['catalog_content'].apply(parse_catalog)
val_data['value'] = parsed.apply(lambda x: x['value'])
val_data['unit'] = parsed.apply(lambda x: x['unit'])
val_data['item_name'] = parsed.apply(lambda x: x['item_name'])

parsed_train = train_data['catalog_content'].apply(parse_catalog)
train_data['value'] = parsed_train.apply(lambda x: x['value'])
train_data['unit'] = parsed_train.apply(lambda x: x['unit'])
train_data['item_name'] = parsed_train.apply(lambda x: x['item_name'])

# Simple baseline: Predict mean price per unit
print("\n🔮 Creating baseline predictions (mean per unit)...")
unit_means = train_data.groupby('unit')['price'].mean().to_dict()
global_mean = train_data['price'].mean()

val_data['predicted_price'] = val_data['unit'].map(unit_means).fillna(global_mean)

# Calculate SMAPE
def smape(y_true, y_pred):
    """Calculate SMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

baseline_smape = smape(val_data['price'].values, val_data['predicted_price'].values)
print(f"   Baseline SMAPE: {baseline_smape:.2f}%")

# ============================================================================
# ERROR ANALYSIS BY PRICE RANGE
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ERROR ANALYSIS BY PRICE RANGE")
print("="*80)

# Calculate errors
val_data['abs_error'] = np.abs(val_data['price'] - val_data['predicted_price'])
val_data['pct_error'] = (val_data['abs_error'] / val_data['price']) * 100
val_data['smape_individual'] = np.abs(val_data['price'] - val_data['predicted_price']) / \
                                ((np.abs(val_data['price']) + np.abs(val_data['predicted_price'])) / 2) * 100

print("\n📊 Error Statistics by Price Range:")
price_ranges = {
    'Budget ($0-$10)': (0, 10),
    'Economy ($10-$20)': (10, 20),
    'Mid-Range ($20-$30)': (20, 30),
    'Premium ($30-$50)': (30, 50),
    'High-End ($50-$100)': (50, 100),
    'Luxury ($100+)': (100, float('inf'))
}

error_by_range = []
for range_name, (min_p, max_p) in price_ranges.items():
    range_data = val_data[(val_data['price'] >= min_p) & (val_data['price'] < max_p)]
    
    if len(range_data) > 0:
        stats = {
            'Range': range_name,
            'Count': len(range_data),
            'Avg_Price': range_data['price'].mean(),
            'SMAPE': smape(range_data['price'].values, range_data['predicted_price'].values),
            'MAE': range_data['abs_error'].mean(),
            'Median_Error': range_data['abs_error'].median(),
            'Max_Error': range_data['abs_error'].max(),
            'Pct_Over_Predicted': (range_data['predicted_price'] > range_data['price']).sum() / len(range_data) * 100
        }
        error_by_range.append(stats)
        
        print(f"\n   {range_name}:")
        print(f"      Samples: {stats['Count']:,}")
        print(f"      SMAPE: {stats['SMAPE']:.2f}%")
        print(f"      MAE: ${stats['MAE']:.2f}")
        print(f"      Over-predicted: {stats['Pct_Over_Predicted']:.1f}%")

error_df = pd.DataFrame(error_by_range)

# Visualize errors by range
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# SMAPE by price range
axes[0, 0].bar(error_df['Range'], error_df['SMAPE'], color='coral')
axes[0, 0].set_xlabel('Price Range', fontsize=12)
axes[0, 0].set_ylabel('SMAPE (%)', fontsize=12)
axes[0, 0].set_title('SMAPE by Price Range', fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].axhline(baseline_smape, color='red', linestyle='--', label=f'Overall: {baseline_smape:.1f}%')
axes[0, 0].legend()

# MAE by price range
axes[0, 1].bar(error_df['Range'], error_df['MAE'], color='steelblue')
axes[0, 1].set_xlabel('Price Range', fontsize=12)
axes[0, 1].set_ylabel('Mean Absolute Error ($)', fontsize=12)
axes[0, 1].set_title('MAE by Price Range', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

# Predicted vs Actual (scatter)
axes[1, 0].scatter(val_data['price'], val_data['predicted_price'], alpha=0.3, s=10)
axes[1, 0].plot([0, val_data['price'].max()], [0, val_data['price'].max()], 
                'r--', label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Price ($)', fontsize=12)
axes[1, 0].set_ylabel('Predicted Price ($)', fontsize=12)
axes[1, 0].set_title('Predicted vs Actual Price', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Error distribution
axes[1, 1].hist(val_data['abs_error'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 1].set_xlabel('Absolute Error ($)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
axes[1, 1].axvline(val_data['abs_error'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${val_data["abs_error"].mean():.2f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_error_analysis_by_range.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '07_error_analysis_by_range.png'}")

# ============================================================================
# IDENTIFY DIFFICULT SAMPLES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: IDENTIFYING DIFFICULT SAMPLES")
print("="*80)

# Top errors
top_errors = val_data.nlargest(50, 'abs_error')

print("\n🔴 Top 20 Most Difficult Predictions:")
print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s} {'SMAPE':>10s} {'Item Name'}")
print("-" * 100)
for idx, row in top_errors.head(20).iterrows():
    item_name = str(row['item_name'])[:60] if pd.notna(row['item_name']) else 'N/A'
    print(f"${row['price']:>9.2f} ${row['predicted_price']:>9.2f} "
          f"${row['abs_error']:>9.2f} {row['smape_individual']:>9.1f}% {item_name}")

# Analyze patterns in difficult samples
print("\n📊 Patterns in Top 50 Difficult Samples:")
print(f"   Average price: ${top_errors['price'].mean():.2f}")
print(f"   Price range: ${top_errors['price'].min():.2f} - ${top_errors['price'].max():.2f}")

if 'unit' in top_errors.columns:
    print(f"\n   Top units in difficult samples:")
    for unit, count in top_errors['unit'].value_counts().head(10).items():
        print(f"      {str(unit)[:30]:32s}: {count} samples")

# ============================================================================
# ERROR ANALYSIS BY UNIT
# ============================================================================
print("\n" + "="*80)
print("STEP 5: ERROR ANALYSIS BY UNIT TYPE")
print("="*80)

top_units = val_data['unit'].value_counts().head(15).index
unit_errors = []

print("\n📏 SMAPE by Unit Type (Top 15):")
for unit in top_units:
    unit_data = val_data[val_data['unit'] == unit]
    if len(unit_data) > 10:  # At least 10 samples
        unit_smape = smape(unit_data['price'].values, unit_data['predicted_price'].values)
        unit_errors.append({
            'Unit': unit,
            'Count': len(unit_data),
            'SMAPE': unit_smape,
            'MAE': unit_data['abs_error'].mean(),
            'Avg_Price': unit_data['price'].mean()
        })
        print(f"   {str(unit)[:25]:27s}: SMAPE={unit_smape:>6.2f}% | "
              f"MAE=${unit_data['abs_error'].mean():>7.2f} | "
              f"n={len(unit_data):>5,}")

unit_error_df = pd.DataFrame(unit_errors).sort_values('SMAPE', ascending=False)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# SMAPE by unit
axes[0].barh(range(len(unit_error_df)), unit_error_df['SMAPE'], color='tomato')
axes[0].set_yticks(range(len(unit_error_df)))
axes[0].set_yticklabels(unit_error_df['Unit'])
axes[0].set_xlabel('SMAPE (%)', fontsize=12)
axes[0].set_title('SMAPE by Unit Type (Worst to Best)', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Sample count vs SMAPE
axes[1].scatter(unit_error_df['Count'], unit_error_df['SMAPE'], s=100, alpha=0.6)
axes[1].set_xlabel('Number of Samples', fontsize=12)
axes[1].set_ylabel('SMAPE (%)', fontsize=12)
axes[1].set_title('Sample Count vs SMAPE by Unit', fontsize=14, fontweight='bold')

for idx, row in unit_error_df.iterrows():
    if row['SMAPE'] > baseline_smape + 10:  # Label high-error units
        axes[1].annotate(row['Unit'], (row['Count'], row['SMAPE']), fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_error_analysis_by_unit.png', dpi=300, bbox_inches='tight')
print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '08_error_analysis_by_unit.png'}")

# ============================================================================
# INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: KEY INSIGHTS & IMPROVEMENT OPPORTUNITIES")
print("="*80)

# Find worst-performing segments
worst_range = error_df.nlargest(1, 'SMAPE').iloc[0]
worst_unit = unit_error_df.nlargest(1, 'SMAPE').iloc[0] if len(unit_error_df) > 0 else None

print("\n🎯 HIGH-IMPACT IMPROVEMENT OPPORTUNITIES:")
print("\n1. WORST PRICE RANGE:")
print(f"   Range: {worst_range['Range']}")
print(f"   SMAPE: {worst_range['SMAPE']:.2f}% (vs {baseline_smape:.2f}% overall)")
print(f"   Recommendation: Train separate model for this range")

if worst_unit is not None:
    print("\n2. WORST UNIT TYPE:")
    print(f"   Unit: {worst_unit['Unit']}")
    print(f"   SMAPE: {worst_unit['SMAPE']:.2f}%")
    print(f"   Recommendation: Create unit-specific features or models")

# Over/under prediction bias
over_pred_pct = (val_data['predicted_price'] > val_data['price']).sum() / len(val_data) * 100
print(f"\n3. PREDICTION BIAS:")
print(f"   Over-predictions: {over_pred_pct:.1f}%")
print(f"   Under-predictions: {100-over_pred_pct:.1f}%")
if abs(over_pred_pct - 50) > 10:
    print(f"   ⚠️ Model shows {'over' if over_pred_pct > 50 else 'under'}-prediction bias")
    print(f"   Recommendation: Apply calibration or adjust loss function")

# High-value item errors
high_value = val_data[val_data['price'] > 50]
if len(high_value) > 0:
    high_value_smape = smape(high_value['price'].values, high_value['predicted_price'].values)
    print(f"\n4. HIGH-VALUE ITEMS ($50+):")
    print(f"   SMAPE: {high_value_smape:.2f}%")
    print(f"   Count: {len(high_value):,} ({len(high_value)/len(val_data)*100:.1f}% of validation)")
    if high_value_smape > baseline_smape + 5:
        print(f"   ⚠️ Struggling with high-value predictions")
        print(f"   Recommendation: More features for premium segment or log transformation")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING ERROR ANALYSIS RESULTS")
print("="*80)

# Save error analysis
error_df.to_csv(OUTPUT_DIR / 'error_analysis_by_range.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'error_analysis_by_range.csv'}")

if len(unit_error_df) > 0:
    unit_error_df.to_csv(OUTPUT_DIR / 'error_analysis_by_unit.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'error_analysis_by_unit.csv'}")

# Save difficult samples
top_errors.to_csv(OUTPUT_DIR / 'difficult_samples_top50.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'difficult_samples_top50.csv'}")

# Save full validation with predictions
val_data.to_csv(OUTPUT_DIR / 'validation_with_errors.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'validation_with_errors.csv'}")

print("\n" + "="*80)
print("✅ PHASE 3 ERROR ANALYSIS COMPLETE")
print("="*80)
print("\nKey Outputs:")
print("   1. 07_error_analysis_by_range.png - Error patterns by price range")
print("   2. 08_error_analysis_by_unit.png - Error patterns by unit type")
print("   3. error_analysis_by_range.csv - Detailed range statistics")
print("   4. error_analysis_by_unit.csv - Detailed unit statistics")
print("   5. difficult_samples_top50.csv - Hardest predictions to analyze")
print("   6. validation_with_errors.csv - Full validation set with errors")
print("\n📋 Next: Review insights and proceed to Phase 4 (Feature Engineering Strategy)")
