"""
Main Pipeline - Try4: DeBERTa + CLIP + Engineered Features

Orchestrates the complete multi-modal fusion pipeline:
1. Extract DeBERTa embeddings (1024-dim)
2. Extract CLIP embeddings (768-dim)  
3. Engineer tabular features (~40)
4. Detect outliers (multi-strategy)
5. Treat outliers gracefully
6. Train fusion model (LightGBM)
7. Generate predictions

Run this script to execute the full pipeline.
"""

import sys
import subprocess
from pathlib import Path
import time

# Add config to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    print_config,
    EMBEDDINGS_DIR, FEATURES_DIR, ANALYSIS_DIR,
    MODELS_DIR, PREDICTIONS_DIR
)

print("="*80)
print("🚀 TRY4: MULTI-MODAL FUSION PIPELINE")
print("="*80)
print("DeBERTa-v3-large + CLIP ViT-Large + Engineered Features")
print("="*80)

# Print configuration
print_config()

# ============================================================================
# Pipeline Stages
# ============================================================================

def run_stage(stage_name, script_path, skip_if_exists=None):
    """
    Run a pipeline stage
    
    Args:
        stage_name: Display name of the stage
        script_path: Path to Python script
        skip_if_exists: Optional path - skip if this file exists
    """
    print("\n" + "="*80)
    print(f"STAGE: {stage_name}")
    print("="*80)
    
    # Check if we can skip
    if skip_if_exists and Path(skip_if_exists).exists():
        print(f"✓ Skipping - output already exists: {skip_if_exists}")
        return True
    
    # Run script
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Stage completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Stage failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Execute full pipeline"""
    
    base_dir = Path(__file__).parent
    
    stages = [
        {
            'name': '1️⃣ Extract DeBERTa Embeddings',
            'script': base_dir / 'feature_extraction' / 'deberta_embeddings.py',
            'skip_if': EMBEDDINGS_DIR / 'deberta_train_embeddings.csv',
            'required': False  # Can skip if already exists
        },
        {
            'name': '2️⃣ Extract CLIP Embeddings',
            'script': base_dir / 'feature_extraction' / 'clip_embeddings.py',
            'skip_if': EMBEDDINGS_DIR / 'clip_train_embeddings.csv',
            'required': False
        },
        {
            'name': '3️⃣ Engineer Tabular Features',
            'script': base_dir / 'feature_extraction' / 'tabular_features.py',
            'skip_if': FEATURES_DIR / 'tabular_train_features.csv',
            'required': False
        },
        {
            'name': '4️⃣ Detect Outliers',
            'script': base_dir / 'preprocessing' / 'outlier_detection.py',
            'skip_if': ANALYSIS_DIR / 'outlier_detection_results.csv',
            'required': False
        },
        {
            'name': '5️⃣ Treat Outliers',
            'script': base_dir / 'preprocessing' / 'outlier_treatment.py',
            'skip_if': FEATURES_DIR / 'train_features_treated.csv',
            'required': False
        },
        {
            'name': '6️⃣ Train Fusion Model',
            'script': base_dir / 'modeling' / 'fusion_model.py',
            'skip_if': None,  # Always run (or check for latest model)
            'required': True
        }
    ]
    
    print("\n" + "="*80)
    print("PIPELINE STAGES")
    print("="*80)
    for i, stage in enumerate(stages, 1):
        skip_text = f" (skip if exists: {stage['skip_if'].name})" if stage['skip_if'] else ""
        print(f"{i}. {stage['name']}{skip_text}")
    
    print("\n" + "="*80)
    print("STARTING PIPELINE EXECUTION")
    print("="*80)
    
    start_time = time.time()
    
    # Execute stages
    for stage in stages:
        success = run_stage(
            stage['name'],
            stage['script'],
            skip_if_exists=stage['skip_if']
        )
        
        if not success and stage['required']:
            print("\n" + "="*80)
            print("❌ PIPELINE FAILED")
            print("="*80)
            print(f"Required stage '{stage['name']}' failed!")
            return False
        
        if not success:
            print(f"\n⚠️ Optional stage '{stage['name']}' failed, continuing...")
    
    # Pipeline completed
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")
    
    print("\n📁 Outputs:")
    print(f"  Embeddings:   {EMBEDDINGS_DIR}")
    print(f"  Features:     {FEATURES_DIR}")
    print(f"  Analysis:     {ANALYSIS_DIR}")
    print(f"  Models:       {MODELS_DIR}")
    print(f"  Predictions:  {PREDICTIONS_DIR}")
    
    # Check for final predictions
    pred_file = PREDICTIONS_DIR / 'test_predictions.csv'
    if pred_file.exists():
        print(f"\n🎯 Final predictions: {pred_file}")
        
        # Load and display stats
        import pandas as pd
        preds = pd.read_csv(pred_file)
        print(f"\n📊 Prediction Statistics:")
        print(f"  Samples: {len(preds):,}")
        print(f"  Min:     ${preds['price'].min():.2f}")
        print(f"  Mean:    ${preds['price'].mean():.2f}")
        print(f"  Median:  ${preds['price'].median():.2f}")
        print(f"  Max:     ${preds['price'].max():.2f}")
    
    # Check OOF results
    oof_file = PREDICTIONS_DIR / 'oof_predictions.csv'
    if oof_file.exists():
        import pandas as pd
        import numpy as np
        
        oof = pd.read_csv(oof_file)
        
        # Calculate SMAPE
        numerator = np.abs(oof['price_pred'] - oof['price_true'])
        denominator = (np.abs(oof['price_true']) + np.abs(oof['price_pred'])) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(numerator / denominator) * 100
        
        print(f"\n🎯 Out-of-Fold SMAPE: {smape:.3f}%")
    
    print("\n" + "="*80)
    print("🎉 Ready for submission!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
