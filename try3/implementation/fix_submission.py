"""
Fix negative predictions in submission file
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import OUTPUT_DIR

print("="*80)
print("FIXING SUBMISSION - REMOVING NEGATIVE PRICES")
print("="*80)
print()

# Load submission
submission_path = OUTPUT_DIR / "final_model" / "submission.csv"
submission = pd.read_csv(submission_path)

print(f"📊 Original Submission Stats:")
print(f"   Total predictions: {len(submission):,}")
print(f"   Negative prices: {(submission.PREDICTED_PRICE < 0).sum():,}")
print(f"   Min price: ${submission.PREDICTED_PRICE.min():.2f}")
print(f"   Max price: ${submission.PREDICTED_PRICE.max():.2f}")
print(f"   Mean price: ${submission.PREDICTED_PRICE.mean():.2f}")
print()

# FIX 1: Clip to minimum $0.01 (prices can't be negative)
submission['PREDICTED_PRICE_FIXED'] = submission['PREDICTED_PRICE'].clip(lower=0.01)

print(f"✅ After Fix (clip to $0.01):")
print(f"   Negative prices: {(submission.PREDICTED_PRICE_FIXED < 0).sum():,}")
print(f"   Min price: ${submission.PREDICTED_PRICE_FIXED.min():.2f}")
print(f"   Max price: ${submission.PREDICTED_PRICE_FIXED.max():.2f}")
print(f"   Mean price: ${submission.PREDICTED_PRICE_FIXED.mean():.2f}")
print()

# Save fixed submission
fixed_submission = submission[['ITEM_ID', 'PREDICTED_PRICE_FIXED']].copy()
fixed_submission.columns = ['ITEM_ID', 'PREDICTED_PRICE']

fixed_path = OUTPUT_DIR / "final_model" / "submission_fixed.csv"
fixed_submission.to_csv(fixed_path, index=False)
print(f"✓ Saved fixed submission: {fixed_path}")
print()

# Also fix OOF predictions
oof_path = OUTPUT_DIR / "final_model" / "oof_predictions.csv"
if oof_path.exists():
    oof = pd.read_csv(oof_path)
    
    print(f"📊 OOF Predictions Stats:")
    print(f"   Negative prices: {(oof.PREDICTED_PRICE < 0).sum():,}")
    print(f"   Min price: ${oof.PREDICTED_PRICE.min():.2f}")
    print()
    
    oof['PREDICTED_PRICE'] = oof['PREDICTED_PRICE'].clip(lower=0.01)
    
    oof_fixed_path = OUTPUT_DIR / "final_model" / "oof_predictions_fixed.csv"
    oof.to_csv(oof_fixed_path, index=False)
    print(f"✓ Saved fixed OOF: {oof_fixed_path}")
    
    # Recalculate SMAPE
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        return 100 * np.mean(diff)
    
    oof_smape = smape(oof.PRICE.values, oof.PREDICTED_PRICE.values)
    print(f"   Recalculated OOF SMAPE: {oof_smape:.3f}%")
    print()

print("="*80)
print("✅ SUBMISSION FIXED!")
print("="*80)
print()
print(f"📁 Use this file for submission:")
print(f"   {fixed_path}")
print()
print("💡 The negative predictions were likely due to:")
print("   1. Model predicting in raw space (not log-space)")
print("   2. Some extreme feature values for certain products")
print("   3. LightGBM extrapolating beyond training distribution")
print()
print("🔧 Applied fix: Clipped all predictions to minimum $0.01")
print()
