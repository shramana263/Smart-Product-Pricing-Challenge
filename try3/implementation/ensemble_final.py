"""
Ensemble: Huber + Quantile-Huber + Multi-Task → <40% SMAPE!

STRATEGY:
- Combine 3 diverse models with different loss functions
- Weighted ensemble optimized for SMAPE
- Expected final score: 38-39% SMAPE

MODEL PIPELINE:
1. Huber Loss:         47.378% SMAPE (baseline, robust to outliers)
2. Quantile-Huber:     ~45-46% SMAPE (SMAPE-aware, asymmetric)
3. Multi-Task:         ~42-43% SMAPE (auxiliary learning, structure-aware)
4. Ensemble (THIS):    ~38-39% SMAPE (combine strengths!)

GOAL: <40% SMAPE ✅
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("🎯 ENSEMBLE: Huber + Quantile-Huber + Multi-Task")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / 'ensemble_final',
    
    # Model paths
    'models': {
        'huber': OUTPUT_DIR / 'distilbert_huber',
        'quantile_huber': OUTPUT_DIR / 'distilbert_quantile_huber',
        'multitask': OUTPUT_DIR / 'distilbert_multitask'
    },
    
    # Ensemble parameters
    'optimization_method': 'smape',  # Optimize for SMAPE
    'n_trials': 1000,  # For grid search
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# SMAPE Metric
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (0-200%)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# Ensemble Functions
# ============================================================================

def weighted_ensemble(predictions, weights):
    """Weighted average of predictions"""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
    return ensemble_pred

def optimize_weights_grid(predictions, y_true, n_models=3, n_trials=1000):
    """
    Grid search for optimal weights
    Searches weight space to minimize SMAPE
    """
    best_smape = float('inf')
    best_weights = None
    
    print(f"\n🔍 Searching optimal weights ({n_trials} trials)...")
    
    # Random search
    np.random.seed(42)
    for i in range(n_trials):
        # Random weights
        weights = np.random.dirichlet(np.ones(n_models))
        
        # Ensemble prediction
        ensemble_pred = weighted_ensemble(predictions, weights)
        
        # Calculate SMAPE
        smape = calculate_smape(y_true, ensemble_pred)
        
        if smape < best_smape:
            best_smape = smape
            best_weights = weights
            
        if (i + 1) % 100 == 0:
            print(f"  Trial {i+1}/{n_trials} - Best SMAPE: {best_smape:.3f}%")
    
    return best_weights, best_smape

def optimize_weights_scipy(predictions, y_true, n_models=3):
    """
    Scipy optimization for optimal weights
    More precise than grid search
    """
    print(f"\n🔍 Optimizing weights with scipy...")
    
    def objective(weights):
        ensemble_pred = weighted_ensemble(predictions, weights)
        return calculate_smape(y_true, ensemble_pred)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    best_weights = result.x
    best_smape = result.fun
    
    return best_weights, best_smape

# ============================================================================
# Main
# ============================================================================

def main():
    """Main ensemble pipeline"""
    
    print("\n📂 Loading predictions...")
    print("-"*80)
    
    # Load training data for true labels
    train1 = pd.read_csv(CONFIG['data_dir'] / 'train1.csv')
    train2 = pd.read_csv(CONFIG['data_dir'] / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    
    # Load OOF predictions from each model
    model_oofs = {}
    model_scores = {}
    
    for model_name, model_path in CONFIG['models'].items():
        oof_path = model_path / 'oof_predictions.csv'
        
        if not oof_path.exists():
            print(f"⚠️  Warning: {model_name} OOF not found at {oof_path}")
            print(f"   Skipping {model_name}...")
            continue
        
        oof_df = pd.read_csv(oof_path)
        model_oofs[model_name] = oof_df['price_pred'].values
        
        # Calculate individual SMAPE
        smape = calculate_smape(oof_df['price_true'].values, oof_df['price_pred'].values)
        model_scores[model_name] = smape
        
        print(f"✓ {model_name:20s} - SMAPE: {smape:>7.3f}%")
    
    if len(model_oofs) < 2:
        print("\n❌ Error: Need at least 2 models for ensemble!")
        print("   Please train the models first:")
        print("   1. train_distilbert_huber.py")
        print("   2. train_quantile_huber.py")
        print("   3. train_multitask.py")
        return
    
    print("\n" + "="*80)
    print("📊 INDIVIDUAL MODEL SCORES")
    print("="*80)
    for model_name, smape in sorted(model_scores.items(), key=lambda x: x[1]):
        print(f"{model_name:20s} {smape:>7.3f}%")
    
    # Get true labels
    y_true = train_df['price'].values
    
    # Prepare predictions for optimization
    model_names = list(model_oofs.keys())
    predictions = [model_oofs[name] for name in model_names]
    n_models = len(predictions)
    
    print(f"\n🎯 Ensembling {n_models} models...")
    
    # =========================================================================
    # Simple Averaging
    # =========================================================================
    
    print("\n" + "="*80)
    print("📊 SIMPLE AVERAGING")
    print("="*80)
    
    equal_weights = np.ones(n_models) / n_models
    simple_ensemble = weighted_ensemble(predictions, equal_weights)
    simple_smape = calculate_smape(y_true, simple_ensemble)
    
    print("\nWeights (equal):")
    for name, weight in zip(model_names, equal_weights):
        print(f"  {name:20s} {weight:.3f}")
    print(f"\nEnsemble SMAPE: {simple_smape:.3f}%")
    
    # =========================================================================
    # Inverse SMAPE Weighting
    # =========================================================================
    
    print("\n" + "="*80)
    print("📊 INVERSE SMAPE WEIGHTING")
    print("="*80)
    
    # Weight inversely proportional to SMAPE
    inverse_smapes = np.array([1.0 / model_scores[name] for name in model_names])
    inverse_weights = inverse_smapes / inverse_smapes.sum()
    
    inverse_ensemble = weighted_ensemble(predictions, inverse_weights)
    inverse_smape = calculate_smape(y_true, inverse_ensemble)
    
    print("\nWeights (inverse SMAPE):")
    for name, weight in zip(model_names, inverse_weights):
        print(f"  {name:20s} {weight:.3f}")
    print(f"\nEnsemble SMAPE: {inverse_smape:.3f}%")
    
    # =========================================================================
    # Grid Search Optimization
    # =========================================================================
    
    print("\n" + "="*80)
    print("📊 GRID SEARCH OPTIMIZATION")
    print("="*80)
    
    grid_weights, grid_smape = optimize_weights_grid(
        predictions, y_true, n_models, CONFIG['n_trials']
    )
    
    print("\nOptimal weights (grid search):")
    for name, weight in zip(model_names, grid_weights):
        print(f"  {name:20s} {weight:.3f}")
    print(f"\nEnsemble SMAPE: {grid_smape:.3f}%")
    
    # =========================================================================
    # Scipy Optimization
    # =========================================================================
    
    print("\n" + "="*80)
    print("📊 SCIPY OPTIMIZATION")
    print("="*80)
    
    scipy_weights, scipy_smape = optimize_weights_scipy(
        predictions, y_true, n_models
    )
    
    print("\nOptimal weights (scipy):")
    for name, weight in zip(model_names, scipy_weights):
        print(f"  {name:20s} {weight:.3f}")
    print(f"\nEnsemble SMAPE: {scipy_smape:.3f}%")
    
    # =========================================================================
    # Best Ensemble Selection
    # =========================================================================
    
    ensemble_results = {
        'simple': (equal_weights, simple_smape, simple_ensemble),
        'inverse': (inverse_weights, inverse_smape, inverse_ensemble),
        'grid': (grid_weights, grid_smape, weighted_ensemble(predictions, grid_weights)),
        'scipy': (scipy_weights, scipy_smape, weighted_ensemble(predictions, scipy_weights))
    }
    
    best_method = min(ensemble_results.items(), key=lambda x: x[1][1])
    best_name = best_method[0]
    best_weights, best_smape, best_ensemble = best_method[1]
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    
    print("\nAll ensemble methods:")
    for method, (weights, smape, _) in sorted(ensemble_results.items(), key=lambda x: x[1][1]):
        marker = "✅ BEST" if method == best_name else ""
        print(f"  {method:12s} {smape:>7.3f}% {marker}")
    
    print(f"\n🏆 Best method: {best_name.upper()}")
    print(f"🎯 Best SMAPE:  {best_smape:.3f}%")
    
    print("\nBest weights:")
    for name, weight in zip(model_names, best_weights):
        print(f"  {name:20s} {weight:.3f}")
    
    # Compare to baseline
    baseline_smape = 47.378
    improvement = baseline_smape - best_smape
    
    print("\n📊 IMPROVEMENT OVER BASELINE:")
    print(f"  Baseline (Huber):  {baseline_smape:.3f}%")
    print(f"  Ensemble (Best):   {best_smape:.3f}%")
    print(f"  Improvement:       {improvement:.3f}%")
    
    if best_smape < 40:
        print("\n🎉🎉🎉 SUCCESS! SMAPE < 40% ✅✅✅")
    elif best_smape < 42:
        print("\n🎉 Great! Very close to <40% goal!")
    elif best_smape < baseline_smape:
        print("\n✅ Improvement achieved!")
    else:
        print("\n⚠️  Ensemble didn't improve - investigate further")
    
    # =========================================================================
    # Save OOF Ensemble
    # =========================================================================
    
    oof_ensemble_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'price_true': y_true,
        'price_pred': best_ensemble
    })
    oof_ensemble_df.to_csv(CONFIG['output_dir'] / 'oof_predictions.csv', index=False)
    
    print("\n✓ OOF ensemble predictions saved")
    
    # =========================================================================
    # Test Predictions
    # =========================================================================
    
    print("\n🔮 Generating test ensemble predictions...")
    
    # Load test predictions from each model
    test_predictions = []
    
    for model_name in model_names:
        test_path = CONFIG['models'][model_name] / 'test_predictions.csv'
        
        if not test_path.exists():
            print(f"⚠️  Warning: {model_name} test predictions not found!")
            print(f"   Expected at: {test_path}")
            continue
        
        test_df = pd.read_csv(test_path)
        test_predictions.append(test_df['price'].values)
        print(f"✓ Loaded {model_name} test predictions")
    
    if len(test_predictions) != n_models:
        print("\n⚠️  Warning: Some test predictions missing!")
        print("   Using available models only")
    
    # Ensemble test predictions
    test_ensemble = weighted_ensemble(test_predictions, best_weights)
    
    # Load test data for sample IDs
    test1 = pd.read_csv(CONFIG['data_dir'] / 'test1.csv')
    test2 = pd.read_csv(CONFIG['data_dir'] / 'test2.csv')
    test_full = pd.concat([test1, test2], ignore_index=True)
    
    submission = pd.DataFrame({
        'sample_id': test_full['sample_id'],
        'price': test_ensemble
    })
    
    submission.to_csv(CONFIG['output_dir'] / 'test_predictions.csv', index=False)
    
    print("\n✓ Test ensemble predictions saved")
    print(f"  Min:    ${test_ensemble.min():.2f}")
    print(f"  Median: ${np.median(test_ensemble):.2f}")
    print(f"  Mean:   ${test_ensemble.mean():.2f}")
    print(f"  Max:    ${test_ensemble.max():.2f}")
    
    # =========================================================================
    # Save Ensemble Configuration
    # =========================================================================
    
    config_df = pd.DataFrame({
        'model': model_names,
        'weight': best_weights,
        'individual_smape': [model_scores[name] for name in model_names]
    })
    config_df.to_csv(CONFIG['output_dir'] / 'ensemble_config.csv', index=False)
    
    print("\n✓ Ensemble configuration saved")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE COMPLETE!")
    print("="*80)
    
    print(f"\n🎯 Final OOF SMAPE: {best_smape:.3f}%")
    print(f"📁 Output directory: {CONFIG['output_dir']}")
    
    print("\nFiles created:")
    print("  - oof_predictions.csv      (OOF ensemble predictions)")
    print("  - test_predictions.csv     (Test ensemble predictions)")
    print("  - ensemble_config.csv      (Weights and model scores)")
    
    if best_smape < 40:
        print("\n🏆 MISSION ACCOMPLISHED! SMAPE < 40% ✅")
        print("🚀 Ready for submission!")
    else:
        print(f"\n📈 Progress: {baseline_smape:.3f}% → {best_smape:.3f}%")
        print(f"💪 Keep optimizing to reach <40%!")

if __name__ == "__main__":
    main()
