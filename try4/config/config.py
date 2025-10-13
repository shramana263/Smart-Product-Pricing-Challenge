"""
Central Configuration for Try4
All paths, hyperparameters, and settings in one place
"""

from pathlib import Path
import torch

# ============================================================================
# Paths
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "try2" / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

# Output subdirectories
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FEATURES_DIR = OUTPUT_DIR / "features"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Create directories
for dir_path in [EMBEDDINGS_DIR, MODELS_DIR, PREDICTIONS_DIR, FEATURES_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Configuration
# ============================================================================

# Text Model - DeBERTa-v3-large
TEXT_MODEL = {
    'name': 'microsoft/deberta-v3-large',
    'max_length': 256,
    'embedding_dim': 1024,
    'batch_size': 16,  # Smaller due to large model
    'learning_rate': 1e-5,
    'num_epochs': 3,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
}

# Image Model - CLIP ViT-Large
IMAGE_MODEL = {
    'name': 'openai/clip-vit-large-patch14',
    'embedding_dim': 768,
    'batch_size': 32,
    'image_size': 224,
    'download_timeout': 10,
    'max_retries': 3,
}

# Engineered Features
TABULAR_FEATURES = {
    'expected_features': 40,  # Approximate number
    'target_encoding_folds': 5,
    'smoothing': 10,
}

# ============================================================================
# Outlier Detection Configuration
# ============================================================================

OUTLIER_CONFIG = {
    # Isolation Forest
    'isolation_forest': {
        'contamination': 0.05,  # 5% expected outliers
        'n_estimators': 100,
        'random_state': 42,
    },
    
    # IQR Method
    'iqr': {
        'multiplier': 1.5,  # Standard 1.5×IQR
    },
    
    # Z-Score
    'zscore': {
        'threshold': 3.0,  # |z| > 3 is outlier
    },
    
    # DBSCAN
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5,
    },
    
    # Ensemble: A sample is an outlier if detected by N methods
    'ensemble_threshold': 2,  # Detected by at least 2 methods
}

# ============================================================================
# Outlier Treatment Configuration
# ============================================================================

TREATMENT_CONFIG = {
    # Winsorization percentiles
    'winsorize_limits': (0.01, 0.99),  # Cap at 1st and 99th percentile
    
    # Robust scaling
    'robust_scaling': {
        'quantile_range': (5, 95),  # Use 5th-95th percentile for scaling
    },
    
    # Log transform
    'log_transform': True,
    
    # Separate model for outliers
    'separate_outlier_model': True,
    'outlier_threshold_percentile': 95,  # Top 5% and bottom 5%
}

# ============================================================================
# Feature Fusion Configuration
# ============================================================================

FUSION_CONFIG = {
    # Total dimension
    'deberta_dim': TEXT_MODEL['embedding_dim'],  # 1024
    'clip_dim': IMAGE_MODEL['embedding_dim'],    # 768
    'tabular_dim': TABULAR_FEATURES['expected_features'],  # ~40
    'total_dim': 1024 + 768 + 40,  # ~1832
    
    # Feature concatenation order
    'concatenation_order': ['deberta', 'clip', 'tabular'],
    
    # Feature scaling
    'scale_features': True,
    'scaler_type': 'robust',  # 'standard', 'robust', 'minmax'
}

# ============================================================================
# Model Training Configuration
# ============================================================================

# LightGBM (Primary model)
LIGHTGBM_CONFIG = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}

# CatBoost (Alternative model)
CATBOOST_CONFIG = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'early_stopping_rounds': 50,
    'verbose': 100,
    'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
}

# MLP (Deep fusion model)
MLP_CONFIG = {
    'hidden_dims': [1024, 512, 256],
    'dropout': 0.3,
    'activation': 'relu',
    'batch_norm': True,
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
}

# ============================================================================
# Cross-Validation Configuration
# ============================================================================

CV_CONFIG = {
    'n_folds': 5,
    'stratify': True,  # Stratified K-Fold based on price bins
    'n_price_bins': 10,
    'shuffle': True,
    'random_state': 42,
}

# ============================================================================
# Feature Analysis Configuration
# ============================================================================

ANALYSIS_CONFIG = {
    # Text sufficiency analysis
    'text_sufficiency': {
        'min_text_length': 50,  # Characters
        'required_fields': ['price', 'quantity', 'brand'],
        'confidence_threshold': 0.7,
    },
    
    # Image information score
    'image_info': {
        'check_availability': True,
        'min_quality_score': 0.5,
        'expected_categories': ['supplement', 'beverage', 'snack'],
    },
    
    # Cross-modal consistency
    'cross_modal': {
        'consistency_threshold': 0.8,
        'weight_text': 0.6,
        'weight_image': 0.4,
    },
}

# ============================================================================
# Device Configuration
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FP16 = torch.cuda.is_available()  # Mixed precision training

# ============================================================================
# Reproducibility
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# Logging
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': OUTPUT_DIR / 'pipeline.log',
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

METRICS = {
    'primary': 'smape',  # Symmetric Mean Absolute Percentage Error
    'secondary': ['mae', 'rmse', 'r2'],
}

# ============================================================================
# Print Configuration Summary
# ============================================================================

def print_config():
    """Print configuration summary"""
    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\n📁 Paths:")
    print(f"  Data:        {DATA_DIR}")
    print(f"  Output:      {OUTPUT_DIR}")
    
    print(f"\n🔤 Text Model:")
    print(f"  Name:        {TEXT_MODEL['name']}")
    print(f"  Embedding:   {TEXT_MODEL['embedding_dim']}-dim")
    
    print(f"\n🖼️ Image Model:")
    print(f"  Name:        {IMAGE_MODEL['name']}")
    print(f"  Embedding:   {IMAGE_MODEL['embedding_dim']}-dim")
    
    print(f"\n⚙️ Tabular Features:")
    print(f"  Expected:    {TABULAR_FEATURES['expected_features']} features")
    
    print(f"\n🔍 Outlier Detection:")
    print(f"  Methods:     Isolation Forest, IQR, Z-Score, DBSCAN")
    print(f"  Ensemble:    {OUTLIER_CONFIG['ensemble_threshold']} methods")
    
    print(f"\n🚀 Fusion:")
    print(f"  Total dim:   {FUSION_CONFIG['total_dim']}")
    print(f"  DeBERTa:     {FUSION_CONFIG['deberta_dim']}")
    print(f"  CLIP:        {FUSION_CONFIG['clip_dim']}")
    print(f"  Tabular:     {FUSION_CONFIG['tabular_dim']}")
    
    print(f"\n💻 Device:")
    print(f"  Device:      {DEVICE}")
    print(f"  FP16:        {USE_FP16}")
    
    print(f"\n📊 Cross-Validation:")
    print(f"  Folds:       {CV_CONFIG['n_folds']}")
    print(f"  Stratified:  {CV_CONFIG['stratify']}")
    
    print("="*80)

if __name__ == "__main__":
    print_config()
