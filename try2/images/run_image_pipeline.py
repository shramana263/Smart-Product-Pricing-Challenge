"""
Master Pipeline Runner - Image Features Integration
===================================================
Runs the complete pipeline to add image features and improve model performance.

Steps:
1. Extract image features from URLs
2. Combine with existing text features (V3 clean)
3. Train models with combined features
4. Generate submission file

Expected improvement: 5-10% SMAPE reduction
Target: 55-60% SMAPE (down from 63.28%)

Author: ML Challenge Team
Date: October 12, 2025
"""

import os
import sys
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*70}\n")

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ {description}")
        print(f"  Path: {filepath}")
        print(f"  Size: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        print(f"  Expected: {filepath}")
        return False

def run_script(script_name, description):
    """Run a Python script and measure time"""
    print(f"\n{'─'*70}")
    print(f"Running: {script_name}")
    print(f"{'─'*70}\n")
    
    start_time = time.time()
    
    # Run script
    result = os.system(f"python {script_name}")
    
    elapsed_time = time.time() - start_time
    
    if result == 0:
        print(f"\n✓ {description} completed successfully")
        print(f"  Time: {elapsed_time/60:.1f} minutes")
        return True
    else:
        print(f"\n✗ {description} FAILED (exit code: {result})")
        return False

def main():
    """Main execution function"""
    
    print_header("IMAGE FEATURES PIPELINE - MASTER RUNNER")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    total_steps = 3
    pipeline_start_time = time.time()
    
    # ========================================================================
    # Pre-flight Checks
    # ========================================================================
    print_header("PRE-FLIGHT CHECKS")
    
    print("Checking for required input files...")
    
    required_files = [
        ('./dataset/train1.csv', 'Training data 1'),
        ('./dataset/train2.csv', 'Training data 2'),
        ('./dataset/test1.csv', 'Test data 1'),
        ('./dataset/test2.csv', 'Test data 2'),
        ('./preparation/features_v3_clean_train.csv', 'V3 clean text features (train)'),
        ('./preparation/features_v3_clean_test.csv', 'V3 clean text features (test)'),
    ]
    
    all_files_exist = True
    for filepath, desc in required_files:
        if not check_file_exists(filepath, desc):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ CRITICAL ERROR: Missing required files!")
        print("Please ensure V3 clean features exist before running this pipeline.")
        print("Run 'advanced_feature_engineering_clean.py' first if needed.")
        return
    
    print("\n✓ All required files found!")
    
    # Check if image features already exist
    print("\nChecking for existing image features...")
    image_train_exists = os.path.exists('./preparation/image_features_train.csv')
    image_test_exists = os.path.exists('./preparation/image_features_test.csv')
    
    if image_train_exists and image_test_exists:
        print("⚠ Image features already exist!")
        response = input("Do you want to re-extract image features? (y/n): ")
        skip_extraction = response.lower() != 'y'
    else:
        skip_extraction = False
    
    # ========================================================================
    # Step 1: Extract Image Features
    # ========================================================================
    if not skip_extraction:
        print_step(1, total_steps, "Extract Image Features")
        
        print("This step will:")
        print("  - Download 150,000 product images")
        print("  - Extract ResNet50 embeddings")
        print("  - Compute color and quality features")
        print("  - Apply PCA to reduce dimensions")
        print("\n⏱ Estimated time: 40-90 minutes (GPU: 40 min, CPU: 90 min)")
        
        if not run_script('image_feature_extraction.py', 'Image feature extraction'):
            print("\n❌ Pipeline FAILED at Step 1")
            return
    else:
        print_step(1, total_steps, "Extract Image Features [SKIPPED]")
        print("Using existing image features")
    
    # ========================================================================
    # Step 2: Combine Features
    # ========================================================================
    print_step(2, total_steps, "Combine Text and Image Features")
    
    print("This step will:")
    print("  - Load V3 clean text features (47 features)")
    print("  - Load extracted image features (157 features)")
    print("  - Merge into combined feature set (204 features)")
    print("\n⏱ Estimated time: 1-2 minutes")
    
    if not run_script('combine_features_with_images.py', 'Feature combination'):
        print("\n❌ Pipeline FAILED at Step 2")
        return
    
    # ========================================================================
    # Step 3: Train Models
    # ========================================================================
    print_step(3, total_steps, "Train Models with Combined Features")
    
    print("This step will:")
    print("  - Train XGBoost, LightGBM, CatBoost")
    print("  - Use 60/15/25 stratified split")
    print("  - Evaluate with SMAPE metric")
    print("  - Generate submission file")
    print("  - Analyze feature importance")
    print("\n⏱ Estimated time: 10-15 minutes")
    
    if not run_script('train_v5_text_image.py', 'Model training'):
        print("\n❌ Pipeline FAILED at Step 3")
        return
    
    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    pipeline_time = time.time() - pipeline_start_time
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    
    print(f"Total pipeline time: {pipeline_time/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check output files
    print("\n" + "─"*70)
    print("Output Files:")
    print("─"*70)
    
    output_files = [
        ('./preparation/image_features_train.csv', 'Image features (train)'),
        ('./preparation/image_features_test.csv', 'Image features (test)'),
        ('./preparation/features_v5_text_image_train.csv', 'Combined features (train)'),
        ('./preparation/features_v5_text_image_test.csv', 'Combined features (test)'),
        ('./modeling/test_out_v5_text_image.csv', 'Submission file'),
        ('./modeling/feature_importance_v5_text_image.csv', 'Feature importance'),
    ]
    
    for filepath, desc in output_files:
        check_file_exists(filepath, desc)
    
    # Next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("\n1. Check the submission file:")
    print("   ./modeling/test_out_v5_text_image.csv")
    
    print("\n2. Review feature importance:")
    print("   ./modeling/feature_importance_v5_text_image.csv")
    
    print("\n3. Compare with V3 baseline:")
    print("   V3 Clean (text only): 63.28% SMAPE")
    print("   V5 (text + image): Check training output above")
    
    print("\n4. If score improved:")
    print("   - Submit test_out_v5_text_image.csv")
    print("   - Consider ensemble with V3")
    print("   - Try V4 safe target encoding + images")
    
    print("\n5. If score didn't improve enough:")
    print("   - Try different image model (EfficientNet, ViT)")
    print("   - Fine-tune on product images")
    print("   - Add more hand-crafted image features")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Change to try2 directory if not already there
    if not os.path.exists('./dataset'):
        if os.path.exists('./try2/dataset'):
            os.chdir('./try2')
            print("Changed working directory to: try2/")
        else:
            print("❌ ERROR: Cannot find dataset directory!")
            print("Please run this script from the project root or try2 folder.")
            sys.exit(1)
    
    main()
