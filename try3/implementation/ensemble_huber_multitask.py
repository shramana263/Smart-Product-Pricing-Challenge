"""
Ensemble: Huber + Multi-Task Models

Combines:
1. DistilBERT with Huber Loss (47.378%)
2. Multi-Task DistilBERT (expected 43-44%)

Expected: <40% SMAPE with optimal weighting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import OUTPUT_DIR

print("="*80)
print("🎯 Ensemble: Huber + Multi-Task")
print("="*80)
print()

# ============================================================================
# Load OOF Predictions
# ============================================================================

print("📂 Loading OOF predictions...")

# Model 1: Huber Loss
huber_oof = pd.read_csv(OUTPUT_DIR / 'distilbert_huber' / 'oof_predictions.csv')
print(f"✓ Huber OOF loaded: {len(huber_oof)} samples")

# Model 2: Multi-Task
multitask_oof = pd.read_csv(OUTPUT_DIR / 'multitask_huber' / 'oof_predictions.csv')
print(f"✓ Multi-Task OOF loaded: {len(multitask_oof)} samples")

# ============================================================================
# Calculate SMAPE
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(numerator / denominator) * 100

# ============================================================================
# Optimize Ensemble Weights
# ============================================================================

print("\n🔍 Optimizing ensemble weights...")

y_true = huber_oof['price_true'].values
huber_pred = huber_oof['price_pred'].values
multitask_pred = multitask_oof['price_pred'].values

best_smape = float('inf')
best_weight = 0.5

# Grid search for optimal weight
for weight in np.arange(0.0, 1.01, 0.05):
    ensemble_pred = weight * huber_pred + (1 - weight) * multitask_pred
    smape = calculate_smape(y_true, ensemble_pred)
    
    if smape < best_smape:
        best_smape = smape
        best_weight = weight

print(f"✓ Optimal weight found: {best_weight:.2f}")
print(f"  Huber weight:      {best_weight:.2f}")
print(f"  Multi-Task weight: {1-best_weight:.2f}")

# ============================================================================
# Ensemble Results
# ============================================================================

ensemble_oof = best_weight * huber_pred + (1 - best_weight) * multitask_pred

huber_smape = calculate_smape(y_true, huber_pred)
multitask_smape = calculate_smape(y_true, multitask_pred)
ensemble_smape = calculate_smape(y_true, ensemble_oof)

print("\n" + "="*80)
print("📊 ENSEMBLE RESULTS")
print("="*80)
print(f"Huber Loss:       {huber_smape:>7.3f}%")
print(f"Multi-Task:       {multitask_smape:>7.3f}%")
print(f"Ensemble:         {ensemble_smape:>7.3f}%")
print("-"*30)
print(f"Improvement:      {huber_smape - ensemble_smape:>7.3f}% ✅")

if ensemble_smape < 40.0:
    print("\n🎉 TARGET ACHIEVED: <40% SMAPE!")
else:
    print(f"\n📍 Gap to 40%: {ensemble_smape - 40.0:.3f}%")

# Save ensemble OOF
ensemble_oof_df = pd.DataFrame({
    'sample_id': huber_oof['sample_id'],
    'price_true': y_true,
    'price_pred': ensemble_oof
})

ensemble_dir = OUTPUT_DIR / 'ensemble_huber_multitask'
ensemble_dir.mkdir(parents=True, exist_ok=True)
ensemble_oof_df.to_csv(ensemble_dir / 'oof_predictions.csv', index=False)

print(f"\n✓ Ensemble OOF saved: {ensemble_dir / 'oof_predictions.csv'}")

# ============================================================================
# Ensemble Test Predictions
# ============================================================================

print("\n🔮 Creating ensemble test predictions...")

# Load test predictions
huber_test = pd.read_csv(OUTPUT_DIR / 'distilbert_huber' / 'test_predictions.csv')
multitask_test = pd.read_csv(OUTPUT_DIR / 'multitask_huber' / 'test_predictions.csv')

# Apply optimal weights
ensemble_test = best_weight * huber_test['price'].values + \
                (1 - best_weight) * multitask_test['price'].values

# Create submission
submission = pd.DataFrame({
    'sample_id': huber_test['sample_id'],
    'price': ensemble_test
})

submission.to_csv(ensemble_dir / 'test_predictions.csv', index=False)

print(f"✓ Ensemble test saved: {ensemble_dir / 'test_predictions.csv'}")
print(f"\n  Min:    ${ensemble_test.min():.2f}")
print(f"  Median: ${np.median(ensemble_test):.2f}")
print(f"  Mean:   ${ensemble_test.mean():.2f}")
print(f"  Max:    ${ensemble_test.max():.2f}")

# ============================================================================
# Detailed Analysis
# ============================================================================

print("\n📊 Per-Range Analysis:")

# Add range labels
def get_range_label(price):
    if price < 20:
        return 'Budget_Economy'
    elif price < 100:
        return 'Mid_Premium'
    else:
        return 'Luxury'

ensemble_oof_df['range'] = ensemble_oof_df['price_true'].apply(get_range_label)

for range_name in ['Budget_Economy', 'Mid_Premium', 'Luxury']:
    mask = ensemble_oof_df['range'] == range_name
    if mask.sum() > 0:
        range_smape = calculate_smape(
            ensemble_oof_df.loc[mask, 'price_true'].values,
            ensemble_oof_df.loc[mask, 'price_pred'].values
        )
        count = mask.sum()
        print(f"  {range_name:20s} ({count:>6,} samples): {range_smape:>6.2f}%")

print("\n" + "="*80)
print("✅ ENSEMBLE COMPLETE!")
print("="*80)

# ============================================================================
# Recommendations
# ============================================================================

print("\n💡 NEXT STEPS:")
print("-"*80)

if ensemble_smape < 40.0:
    print("✅ You've achieved <40% SMAPE!")
    print("🎯 Submit the ensemble predictions:")
    print(f"   {ensemble_dir / 'test_predictions.csv'}")
elif ensemble_smape < 42.0:
    print("📍 Very close to <40%!")
    print("💡 Consider:")
    print("   1. Add image features (expected -2 to -3 points)")
    print("   2. Test-time augmentation (expected -1 point)")
    print("   3. Hyperparameter tuning (expected -1 point)")
elif ensemble_smape < 45.0:
    print("📍 Good progress, but need more improvement")
    print("💡 Try:")
    print("   1. Add price-aware attention layer")
    print("   2. Use larger model (distilbert-large)")
    print("   3. More training epochs with lower learning rate")
else:
    print("⚠️ Ensemble didn't improve much")
    print("💡 Check:")
    print("   1. Multi-task model trained properly?")
    print("   2. Models too similar (high correlation)?")
    print("   3. Try different architectures")

print("\n" + "="*80)
