"""
Test DistilBERT Huber Model SMAPE

Quick script to:
1. Load existing Huber model predictions
2. Calculate SMAPE score
3. Show per-fold and overall performance
4. Validate test predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import OUTPUT_DIR

print("="*80)
print("🧪 Testing DistilBERT Huber Model")
print("="*80)
print()

# ============================================================================
# SMAPE Calculation
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape

def calculate_smape_by_range(df):
    """Calculate SMAPE by price range"""
    ranges = [
        (0, 20, 'Budget_Economy'),
        (20, 100, 'Mid_Premium'),
        (100, np.inf, 'Luxury')
    ]
    
    results = []
    for low, high, name in ranges:
        mask = (df['price_true'] >= low) & (df['price_true'] < high)
        if mask.sum() > 0:
            smape = calculate_smape(
                df.loc[mask, 'price_true'].values,
                df.loc[mask, 'price_pred'].values
            )
            results.append({
                'range': name,
                'low': low,
                'high': high,
                'count': mask.sum(),
                'smape': smape
            })
    
    return pd.DataFrame(results)

# ============================================================================
# Load OOF Predictions
# ============================================================================

huber_dir = OUTPUT_DIR / 'distilbert_huber'
oof_file = huber_dir / 'oof_predictions.csv'
test_file = huber_dir / 'test_predictions.csv'

print("📂 Loading predictions...")
print(f"   Directory: {huber_dir}")

if not oof_file.exists():
    print(f"\n❌ ERROR: OOF predictions not found!")
    print(f"   Expected: {oof_file}")
    print(f"\n💡 Make sure you've trained the Huber model first:")
    print(f"   cd try3/implementation")
    print(f"   python train_distilbert_huber.py")
    sys.exit(1)

# Load OOF predictions
oof_df = pd.read_csv(oof_file)
print(f"✓ OOF predictions loaded: {len(oof_df):,} samples")

# Load test predictions if available
if test_file.exists():
    test_df = pd.read_csv(test_file)
    print(f"✓ Test predictions loaded: {len(test_df):,} samples")
else:
    test_df = None
    print("⚠️  Test predictions not found")

print()

# ============================================================================
# Overall SMAPE
# ============================================================================

print("="*80)
print("📊 OVERALL PERFORMANCE")
print("="*80)

y_true = oof_df['price_true'].values
y_pred = oof_df['price_pred'].values

overall_smape = calculate_smape(y_true, y_pred)

print(f"OOF SMAPE: {overall_smape:.3f}%")
print()

# ============================================================================
# Per-Fold SMAPE (if fold column exists)
# ============================================================================

if 'fold' in oof_df.columns:
    print("="*80)
    print("📊 PER-FOLD PERFORMANCE")
    print("="*80)
    
    fold_results = []
    for fold in sorted(oof_df['fold'].unique()):
        fold_mask = oof_df['fold'] == fold
        fold_smape = calculate_smape(
            oof_df.loc[fold_mask, 'price_true'].values,
            oof_df.loc[fold_mask, 'price_pred'].values
        )
        fold_results.append({
            'fold': fold,
            'samples': fold_mask.sum(),
            'smape': fold_smape
        })
        print(f"Fold {fold}: {fold_smape:>7.3f}% ({fold_mask.sum():>6,} samples)")
    
    fold_df = pd.DataFrame(fold_results)
    print(f"\nMean:   {fold_df['smape'].mean():>7.3f}%")
    print(f"Std:    {fold_df['smape'].std():>7.3f}%")
    print()

# ============================================================================
# Per-Range SMAPE
# ============================================================================

print("="*80)
print("📊 PERFORMANCE BY PRICE RANGE")
print("="*80)

range_df = calculate_smape_by_range(oof_df)

for _, row in range_df.iterrows():
    pct = (row['count'] / len(oof_df)) * 100
    print(f"{row['range']:20s} (${row['low']:>4.0f}-${row['high']:>4.0f}): "
          f"{row['smape']:>6.2f}%  ({row['count']:>6,} samples, {pct:>5.1f}%)")

print()

# ============================================================================
# Error Analysis
# ============================================================================

print("="*80)
print("📊 ERROR ANALYSIS")
print("="*80)

# Calculate errors
oof_df['error'] = y_pred - y_true
oof_df['abs_error'] = np.abs(oof_df['error'])
oof_df['pct_error'] = (oof_df['abs_error'] / y_true) * 100

# Statistics
print(f"Mean Absolute Error:      ${oof_df['abs_error'].mean():>8.2f}")
print(f"Median Absolute Error:    ${oof_df['abs_error'].median():>8.2f}")
print(f"Mean Percentage Error:    {oof_df['pct_error'].mean():>8.2f}%")
print(f"Median Percentage Error:  {oof_df['pct_error'].median():>8.2f}%")
print()

# Quantiles
print("Error Distribution (Absolute $):")
for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    val = oof_df['abs_error'].quantile(q)
    print(f"  {int(q*100):>2}th percentile: ${val:>8.2f}")
print()

# Worst predictions
print("Top 10 Worst Predictions:")
worst = oof_df.nlargest(10, 'abs_error')[['sample_id', 'price_true', 'price_pred', 'abs_error']]
print(worst.to_string(index=False))
print()

# ============================================================================
# Test Predictions Analysis
# ============================================================================

if test_df is not None:
    print("="*80)
    print("📊 TEST PREDICTIONS ANALYSIS")
    print("="*80)
    
    print(f"Total predictions: {len(test_df):,}")
    print(f"\nPrice Statistics:")
    print(f"  Min:     ${test_df['price'].min():>10.2f}")
    print(f"  Median:  ${test_df['price'].median():>10.2f}")
    print(f"  Mean:    ${test_df['price'].mean():>10.2f}")
    print(f"  Max:     ${test_df['price'].max():>10.2f}")
    print(f"  Std:     ${test_df['price'].std():>10.2f}")
    
    print(f"\nPrice Distribution:")
    print(f"  < $20:       {(test_df['price'] < 20).sum():>7,} ({(test_df['price'] < 20).sum()/len(test_df)*100:>5.1f}%)")
    print(f"  $20-$100:    {((test_df['price'] >= 20) & (test_df['price'] < 100)).sum():>7,} ({((test_df['price'] >= 20) & (test_df['price'] < 100)).sum()/len(test_df)*100:>5.1f}%)")
    print(f"  $100+:       {(test_df['price'] >= 100).sum():>7,} ({(test_df['price'] >= 100).sum()/len(test_df)*100:>5.1f}%)")
    
    # Check for invalid predictions
    invalid = (test_df['price'] < 0) | (test_df['price'].isna())
    if invalid.sum() > 0:
        print(f"\n⚠️  WARNING: {invalid.sum()} invalid predictions found!")
    else:
        print(f"\n✓ All predictions are valid")
    
    print()

# ============================================================================
# Comparison with Baseline
# ============================================================================

print("="*80)
print("📊 COMPARISON")
print("="*80)

# Simple baseline: always predict median
median_baseline = np.full_like(y_pred, np.median(y_true))
baseline_smape = calculate_smape(y_true, median_baseline)

print(f"Median Baseline:     {baseline_smape:>7.3f}%")
print(f"DistilBERT Huber:    {overall_smape:>7.3f}%")
print(f"Improvement:         {baseline_smape - overall_smape:>7.3f}% ✅")
print()

# ============================================================================
# Summary
# ============================================================================

print("="*80)
print("✅ SUMMARY")
print("="*80)

print(f"✓ Model tested successfully")
print(f"✓ OOF SMAPE: {overall_smape:.3f}%")

if overall_smape < 40:
    print(f"🎉 TARGET ACHIEVED: <40% SMAPE!")
elif overall_smape < 45:
    print(f"📍 Close to target! Gap: {overall_smape - 40:.3f}%")
elif overall_smape < 50:
    print(f"📍 Good progress. Gap to 40%: {overall_smape - 40:.3f}%")
else:
    print(f"📍 Needs improvement. Gap to 40%: {overall_smape - 40:.3f}%")

print()
print("="*80)

# ============================================================================
# Save Analysis Report
# ============================================================================

report_file = huber_dir / 'model_test_report.txt'

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DistilBERT Huber Model - Test Report\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Overall OOF SMAPE: {overall_smape:.3f}%\n\n")
    
    f.write("Per-Range Performance:\n")
    for _, row in range_df.iterrows():
        f.write(f"  {row['range']:20s}: {row['smape']:>6.2f}%  ({row['count']:>6,} samples)\n")
    
    f.write(f"\nMean Absolute Error: ${oof_df['abs_error'].mean():.2f}\n")
    f.write(f"Median Absolute Error: ${oof_df['abs_error'].median():.2f}\n")
    
    if 'fold' in oof_df.columns:
        f.write(f"\nPer-Fold Performance:\n")
        for _, row in fold_df.iterrows():
            f.write(f"  Fold {row['fold']}: {row['smape']:.3f}%\n")
    
    f.write(f"\nBaseline SMAPE: {baseline_smape:.3f}%\n")
    f.write(f"Improvement: {baseline_smape - overall_smape:.3f}%\n")

print(f"✓ Report saved: {report_file}")
print()
