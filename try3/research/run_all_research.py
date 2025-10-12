"""
MASTER RESEARCH RUNNER
======================
Runs all research phases sequentially to build comprehensive data understanding

Usage:
    python run_all_research.py
"""

import sys
from pathlib import Path
import time

print("="*80)
print("COMPREHENSIVE DATA RESEARCH PIPELINE")
print("="*80)
print("\nThis will run all research phases:")
print("   Phase 1: Data Analysis (price, brands, units, categories)")
print("   Phase 2: Image Analysis (visual patterns, quality)")
print("   Phase 3: Error Analysis (model weaknesses, improvement areas)")
print("\nEstimated time: 10-20 minutes")
print("="*80)

response = input("\nProceed with research? (y/n): ")
if response.lower() != 'y':
    print("Research cancelled.")
    sys.exit(0)

# ============================================================================
# PHASE 1: DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STARTING PHASE 1: COMPREHENSIVE DATA ANALYSIS")
print("="*80)

start_time = time.time()

try:
    print("\n🚀 Running 01_comprehensive_data_analysis.py...")
    exec(open('01_comprehensive_data_analysis.py', encoding='utf-8').read())
    phase1_time = time.time() - start_time
    print(f"\n✅ Phase 1 completed in {phase1_time:.1f} seconds")
except Exception as e:
    print(f"\n❌ Phase 1 failed: {e}")
    print("Please fix errors before proceeding.")
    sys.exit(1)

# ============================================================================
# PHASE 2: IMAGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STARTING PHASE 2: IMAGE ANALYSIS")
print("="*80)

phase2_start = time.time()

try:
    print("\n🚀 Running 02_image_analysis.py...")
    exec(open('02_image_analysis.py', encoding='utf-8').read())
    phase2_time = time.time() - phase2_start
    print(f"\n✅ Phase 2 completed in {phase2_time:.1f} seconds")
except Exception as e:
    print(f"\n❌ Phase 2 failed: {e}")
    print("Continuing to Phase 3 anyway...")

# ============================================================================
# PHASE 3: ERROR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STARTING PHASE 3: ERROR ANALYSIS")
print("="*80)

phase3_start = time.time()

try:
    print("\n🚀 Running 03_error_analysis.py...")
    exec(open('03_error_analysis.py', encoding='utf-8').read())
    phase3_time = time.time() - phase3_start
    print(f"\n✅ Phase 3 completed in {phase3_time:.1f} seconds")
except Exception as e:
    print(f"\n❌ Phase 3 failed: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*80)
print("🎉 ALL RESEARCH PHASES COMPLETE")
print("="*80)

print(f"\nTotal execution time: {total_time/60:.1f} minutes")
print("\n📊 Generated Outputs:")
print("   Phase 1: Data Analysis")
print("      - 01_price_distribution.png")
print("      - 02_unit_price_distribution.png")
print("      - 03_brand_analysis.png")
print("      - 04_category_analysis.png")
print("      - research_train_enriched.csv")
print("      - research_summary.txt")

print("\n   Phase 2: Image Analysis")
print("      - 05_image_visual_analysis.png")
print("      - 06_montage_*.png (price range montages)")
print("      - image_analysis_results.csv")
print("      - sample_images/ folder")

print("\n   Phase 3: Error Analysis")
print("      - 07_error_analysis_by_range.png")
print("      - 08_error_analysis_by_unit.png")
print("      - error_analysis_by_range.csv")
print("      - error_analysis_by_unit.csv")
print("      - difficult_samples_top50.csv")

print("\n" + "="*80)
print("📋 NEXT STEPS:")
print("="*80)
print("\n1. Review all visualizations in outputs/ folder")
print("2. Read research_summary.txt for key statistics")
print("3. Examine difficult_samples_top50.csv to understand failure cases")
print("4. Check error_analysis_*.csv to identify improvement opportunities")
print("\n5. Based on insights, we'll implement:")
print("   - Advanced feature engineering")
print("   - Multi-modal fusion architecture")
print("   - Stratified modeling")
print("   - Ensemble strategies")
print("\n💡 Review the outputs and share your insights to proceed with implementation!")
