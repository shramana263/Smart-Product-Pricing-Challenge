"""
ENSEMBLE: TEXT (DISTILBERT) + IMAGE (RESNET50) MODELS

Strategy:
1. Load predictions from both models
2. Find optimal ensemble weights using OOF predictions
3. Apply to test set
4. Simple weighted average for stability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("ENSEMBLE: TEXT + IMAGE MODELS")
print("="*80)
print()

# Configuration
CONFIG = {
    'text_model_dir': OUTPUT_DIR / "balanced_model",
    'image_model_dir': OUTPUT_DIR / "image_model",
    'output_dir': OUTPUT_DIR / "ensemble_text_image",
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load CSV files
train_files = [DATA_DIR / 'train1.csv', DATA_DIR / 'train2.csv']
test_files = [DATA_DIR / 'test1.csv', DATA_DIR / 'test2.csv']

train_dfs = []
for f in train_files:
    if f.exists():
        df = pd.read_csv(f)
        train_dfs.append(df)

test_dfs = []
for f in test_files:
    if f.exists():
        df = pd.read_csv(f)
        test_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

y_true = train_df['price'].values

print(f"✓ Loaded {len(train_df):,} training samples")
print(f"✓ Loaded {len(test_df):,} test samples")
print()

# ============================================================================
# STEP 2: LOAD MODEL PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 2: LOADING MODEL PREDICTIONS")
print("="*80)

# Load text model predictions
text_oof_file = CONFIG['text_model_dir'] / 'oof_predictions.csv'
text_test_file = CONFIG['text_model_dir'] / 'submission.csv'

if not text_oof_file.exists():
    print(f"❌ Text model OOF predictions not found!")
    print(f"   Expected: {text_oof_file}")
    print(f"   Please run train_balanced_model.py first")
    sys.exit(1)

text_oof_df = pd.read_csv(text_oof_file)
text_test_df = pd.read_csv(text_test_file)

# Handle different column names (text model uses 'price', image model uses 'predicted_price')
text_oof_pred = text_oof_df['price'].values if 'predicted_price' not in text_oof_df.columns else text_oof_df['predicted_price'].values
text_test_pred = text_test_df['price'].values

print(f"✓ Text model loaded:")
print(f"   OOF predictions: {len(text_oof_pred):,}")
print(f"   Test predictions: {len(text_test_pred):,}")

# Load image model predictions
image_oof_file = CONFIG['image_model_dir'] / 'oof_predictions.csv'
image_test_file = CONFIG['image_model_dir'] / 'test_predictions.csv'

if not image_oof_file.exists():
    print(f"❌ Image model OOF predictions not found!")
    print(f"   Expected: {image_oof_file}")
    print(f"   Please run train_image_model_efficient.py first")
    sys.exit(1)

image_oof_df = pd.read_csv(image_oof_file)
image_test_df = pd.read_csv(image_test_file)

image_oof_pred = image_oof_df['predicted_price'].values
image_test_pred = image_test_df['price'].values

print(f"✓ Image model loaded:")
print(f"   OOF predictions: {len(image_oof_pred):,}")
print(f"   Test predictions: {len(image_test_pred):,}")
print()

# ============================================================================
# STEP 3: CALCULATE INDIVIDUAL MODEL SMAPE
# ============================================================================
print("="*80)
print("STEP 3: INDIVIDUAL MODEL PERFORMANCE")
print("="*80)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)
    denominator = (y_true + y_pred) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

text_smape = smape(y_true, text_oof_pred)
image_smape = smape(y_true, image_oof_pred)

print(f"📊 Text Model (DistilBERT):")
print(f"   OOF SMAPE: {text_smape:.3f}%")
print()
print(f"📊 Image Model (ResNet50):")
print(f"   OOF SMAPE: {image_smape:.3f}%")
print()

# ============================================================================
# STEP 4: FIND OPTIMAL ENSEMBLE WEIGHTS
# ============================================================================
print("="*80)
print("STEP 4: OPTIMIZING ENSEMBLE WEIGHTS")
print("="*80)

def ensemble_smape(weights):
    """Calculate SMAPE for weighted ensemble"""
    w_text, w_image = weights
    ensemble_pred = w_text * text_oof_pred + w_image * image_oof_pred
    ensemble_pred = np.maximum(ensemble_pred, 0.01)  # Clip to minimum
    return smape(y_true, ensemble_pred)

# Optimize weights (constrained to sum to 1)
initial_weights = [0.6, 0.4]  # Start with text-heavy
bounds = [(0, 1), (0, 1)]
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}

result = minimize(
    ensemble_smape,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
optimal_smape = result.fun

print(f"✓ Optimization complete!")
print()
print(f"📊 Optimal Weights:")
print(f"   Text (DistilBERT): {optimal_weights[0]:.3f}")
print(f"   Image (ResNet50):  {optimal_weights[1]:.3f}")
print()
print(f"📊 Ensemble Performance:")
print(f"   OOF SMAPE: {optimal_smape:.3f}%")
print(f"   Improvement over text: {text_smape - optimal_smape:.3f} points")
print(f"   Improvement over image: {image_smape - optimal_smape:.3f} points")
print()

# ============================================================================
# STEP 5: CREATE ENSEMBLE PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 5: CREATING ENSEMBLE PREDICTIONS")
print("="*80)

# Apply optimal weights to OOF
oof_ensemble = optimal_weights[0] * text_oof_pred + optimal_weights[1] * image_oof_pred
oof_ensemble = np.maximum(oof_ensemble, 0.01)

# Apply optimal weights to test
test_ensemble = optimal_weights[0] * text_test_pred + optimal_weights[1] * image_test_pred
test_ensemble = np.maximum(test_ensemble, 0.01)

print(f"✓ Created ensemble predictions")
print()

# ============================================================================
# STEP 6: SAVE PREDICTIONS
# ============================================================================
print("="*80)
print("STEP 6: SAVING PREDICTIONS")
print("="*80)

# Save OOF predictions
oof_df = train_df[['sample_id', 'price']].copy()
oof_df['predicted_price'] = oof_ensemble
oof_df['text_pred'] = text_oof_pred
oof_df['image_pred'] = image_oof_pred

oof_file = CONFIG['output_dir'] / 'oof_predictions.csv'
oof_df.to_csv(oof_file, index=False)
print(f"✓ Saved OOF predictions: {oof_file}")

# Save test predictions
submission = test_df[['sample_id']].copy()
submission['price'] = test_ensemble

submission_file = CONFIG['output_dir'] / 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"✓ Saved submission: {submission_file} ⭐")
print()

# Save detailed test predictions
test_detailed = test_df[['sample_id']].copy()
test_detailed['price'] = test_ensemble
test_detailed['text_pred'] = text_test_pred
test_detailed['image_pred'] = image_test_pred

detailed_file = CONFIG['output_dir'] / 'test_predictions_detailed.csv'
test_detailed.to_csv(detailed_file, index=False)
print(f"✓ Saved detailed predictions: {detailed_file}")
print()

# ============================================================================
# STEP 7: PREDICTION STATISTICS
# ============================================================================
print("="*80)
print("STEP 7: PREDICTION STATISTICS")
print("="*80)

print(f"📊 Ensemble Test Predictions:")
print(f"   Min:    ${test_ensemble.min():.2f}")
print(f"   Median: ${np.median(test_ensemble):.2f}")
print(f"   Mean:   ${test_ensemble.mean():.2f}")
print(f"   Max:    ${test_ensemble.max():.2f}")
print()

print(f"📊 Training Prices:")
print(f"   Mean:   ${y_true.mean():.2f}")
print()

print(f"📊 Distribution Comparison:")
print(f"   Test/Train mean ratio: {test_ensemble.mean() / y_true.mean():.3f}")
print()

# ============================================================================
# STEP 8: COMPARISON WITH SIMPLE AVERAGES
# ============================================================================
print("="*80)
print("STEP 8: COMPARISON WITH SIMPLE AVERAGES")
print("="*80)

# 50-50 average
avg_50_50 = 0.5 * text_oof_pred + 0.5 * image_oof_pred
avg_50_50 = np.maximum(avg_50_50, 0.01)
smape_50_50 = smape(y_true, avg_50_50)

# 70-30 average (text-heavy)
avg_70_30 = 0.7 * text_oof_pred + 0.3 * image_oof_pred
avg_70_30 = np.maximum(avg_70_30, 0.01)
smape_70_30 = smape(y_true, avg_70_30)

print(f"📊 Simple Averages:")
print(f"   50-50:  {smape_50_50:.3f}%")
print(f"   70-30:  {smape_70_30:.3f}%")
print(f"   Optimal ({optimal_weights[0]:.0%}-{optimal_weights[1]:.0%}): {optimal_smape:.3f}% ⭐")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ ENSEMBLE COMPLETE!")
print("="*80)
print()
print(f"🎯 Final Performance:")
print(f"   Text Model:    {text_smape:.3f}%")
print(f"   Image Model:   {image_smape:.3f}%")
print(f"   Ensemble:      {optimal_smape:.3f}% ⭐")
print(f"   Improvement:   {min(text_smape, image_smape) - optimal_smape:.3f} points")
print()
print(f"⚖️  Ensemble Weights:")
print(f"   Text:  {optimal_weights[0]:.1%}")
print(f"   Image: {optimal_weights[1]:.1%}")
print()
print(f"📁 Recommended submission:")
print(f"   {submission_file}")
print()
print(f"💡 Next step: Submit to competition!")
print()
