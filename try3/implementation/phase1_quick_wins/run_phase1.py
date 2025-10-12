"""
PHASE 1: QUICK WINS - MASTER RUNNER
====================================

This script runs all Phase 1 improvements in sequence:
1. Log Transform Ensemble
2. Unit Standardization  
3. Advanced Feature Engineering

Expected Total Improvement: 53.6% → 45-46% SMAPE
Time Estimate: 9-13 hours
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE1_DIR = Path(__file__).parent
OUTPUT_DIR = PHASE1_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    {
        'name': '1.1 Log Transform Ensemble',
        'file': '01_log_transform_ensemble.py',
        'expected_time': '4-6 hours',
        'expected_smape': '49-50%',
        'improvement': '-4 to -6 points',
        'description': 'Train DistilBERT on log/sqrt/boxcox transforms and ensemble'
    },
    {
        'name': '1.2 Unit Standardization',
        'file': '02_unit_standardization.py',
        'expected_time': '2-3 hours',
        'expected_smape': '47-48%',
        'improvement': '-2 to -3 points',
        'description': 'Extract and standardize unit variations, handle bulk quantities'
    },
    {
        'name': '1.3 Advanced Features',
        'file': '03_advanced_features.py',
        'expected_time': '3-4 hours',
        'expected_smape': '45-46%',
        'improvement': '-2 to -3 points',
        'description': 'Extract brands, premium signals, text complexity, interactions'
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)

def print_subheader(text):
    """Print formatted subheader"""
    print("\n" + "-"*80)
    print(text)
    print("-"*80)

def run_script(script_info, phase_num):
    """Run a single phase script"""
    script_path = PHASE1_DIR / script_info['file']
    
    print_subheader(f"PHASE {phase_num}: {script_info['name']}")
    
    print(f"\n📝 Description: {script_info['description']}")
    print(f"⏱️  Expected Time: {script_info['expected_time']}")
    print(f"🎯 Expected SMAPE: {script_info['expected_smape']}")
    print(f"📊 Expected Improvement: {script_info['improvement']}")
    print(f"\n🚀 Running: {script_info['file']}")
    print("-"*80)
    
    start_time = datetime.now()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("-"*80)
        print(f"✅ Phase {phase_num} completed successfully!")
        print(f"⏱️  Duration: {duration}")
        
        return {
            'success': True,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds()
        }
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("-"*80)
        print(f"❌ Phase {phase_num} failed!")
        print(f"Error: {str(e)}")
        print(f"⏱️  Duration before failure: {duration}")
        
        return {
            'success': False,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'error': str(e)
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print_header("PHASE 1: QUICK WINS - MASTER RUNNER")
    
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {PHASE1_DIR}")
    print(f"📊 Baseline SMAPE: 53.636%")
    print(f"🎯 Target SMAPE: 45-46%")
    print(f"📈 Expected Improvement: -7 to -9 SMAPE points")
    print(f"⏱️  Total Estimated Time: 9-13 hours")
    
    print("\n📋 Phase 1 Pipeline:")
    for i, script in enumerate(SCRIPTS, 1):
        print(f"   {i}. {script['name']}")
        print(f"      → {script['description']}")
        print(f"      → Time: {script['expected_time']}, Improvement: {script['improvement']}")
    
    input("\n⚠️  Press Enter to start Phase 1 execution (or Ctrl+C to cancel)...")
    
    # Track overall progress
    overall_start = datetime.now()
    results = []
    
    # Run each phase
    for i, script in enumerate(SCRIPTS, 1):
        result = run_script(script, f"1.{i}")
        results.append({
            'phase': f"1.{i}",
            'name': script['name'],
            'file': script['file'],
            **result
        })
        
        # Stop if a phase fails
        if not result['success']:
            print(f"\n❌ Stopping execution due to failure in Phase 1.{i}")
            break
        
        # Pause between phases (except after last one)
        if i < len(SCRIPTS):
            print(f"\n⏸️  Pausing before next phase...")
            print(f"   Next: Phase 1.{i+1} - {SCRIPTS[i]['name']}")
            input(f"   Press Enter to continue (or Ctrl+C to stop)...")
    
    overall_end = datetime.now()
    total_duration = overall_end - overall_start
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print_header("PHASE 1 EXECUTION SUMMARY")
    
    print(f"\n⏱️  Total Duration: {total_duration}")
    print(f"📅 Started:  {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Finished: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Phase Results:")
    all_success = True
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = result['duration_seconds'] / 60  # Convert to minutes
        print(f"\n   Phase {result['phase']}: {result['name']}")
        print(f"   Status: {status}")
        print(f"   Duration: {duration:.1f} minutes")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            all_success = False
    
    # Save results
    results_file = OUTPUT_DIR / 'phase1_execution_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'overall_start': overall_start.isoformat(),
            'overall_end': overall_end.isoformat(),
            'total_duration_seconds': total_duration.total_seconds(),
            'all_phases_successful': all_success,
            'phases': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if all_success:
        print_header("🎉 PHASE 1 COMPLETE - ALL IMPROVEMENTS APPLIED!")
        
        print(f"""
📊 Expected Results:
   Baseline SMAPE:    53.636%
   Phase 1.1 (Log):   49-50%  (-4 to -6 pts)
   Phase 1.2 (Units): 47-48%  (-2 to -3 pts)
   Phase 1.3 (Feat):  45-46%  (-2 to -3 pts)
   
   ✨ Total Expected Improvement: -7 to -9 SMAPE points
   
🎯 Next Steps:
   1. Train final model with all features
   2. Validate on holdout set
   3. Generate submission file
   4. If SMAPE > 46%, proceed to Phase 2 (Stratified Models)
   
📁 Output Files:
   - try3/outputs/phase1_log_transform/
   - try3/outputs/phase1_unit_standardization/
   - try3/outputs/phase1_advanced_features/
   
🚀 Ready to train final Phase 1 model!
        """)
    else:
        print_header("⚠️ PHASE 1 INCOMPLETE")
        print("\nSome phases failed. Please check the errors above and:")
        print("1. Fix any issues")
        print("2. Re-run this script")
        print("3. Or run individual phase scripts")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user")
        print("Progress has been saved. You can resume by re-running this script.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
