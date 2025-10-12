"""
MEMORY-EFFICIENT IMAGE PIPELINE - MASTER SCRIPT

Runs the complete image pipeline:
1. Download images with retry logic
2. Extract ResNet50 features
3. Train image-only model
4. Ensemble with text model

Designed for 16GB RAM on SageMaker
"""

import subprocess
import sys
from pathlib import Path
import time

# Import auto-config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config_auto import OUTPUT_DIR

print("="*80)
print("MEMORY-EFFICIENT IMAGE PIPELINE")
print("="*80)
print()

PIPELINE_DIR = Path(__file__).parent

STEPS = [
    {
        'name': 'Download Images',
        'script': 'download_images_efficient.py',
        'description': 'Download images in batches with retry logic',
        'estimated_time': '30-60 min',
    },
    {
        'name': 'Extract Features',
        'script': 'extract_image_features_efficient.py',
        'description': 'Extract ResNet50 features (2048-dim)',
        'estimated_time': '20-30 min',
    },
    {
        'name': 'Train Image Model',
        'script': 'train_image_model_efficient.py',
        'description': 'Train LightGBM on image features',
        'estimated_time': '5-10 min',
    },
    {
        'name': 'Ensemble Models',
        'script': 'ensemble_text_image.py',
        'description': 'Combine text and image models',
        'estimated_time': '1-2 min',
    },
]

# ============================================================================
# RUN PIPELINE
# ============================================================================
print(f"📋 Pipeline Steps:")
for i, step in enumerate(STEPS, 1):
    print(f"   {i}. {step['name']} (~{step['estimated_time']})")
print()

total_start = time.time()
completed_steps = []

for i, step in enumerate(STEPS, 1):
    print("="*80)
    print(f"STEP {i}/{len(STEPS)}: {step['name'].upper()}")
    print("="*80)
    print(f"Description: {step['description']}")
    print(f"Estimated time: {step['estimated_time']}")
    print()
    
    script_path = PIPELINE_DIR / step['script']
    
    # Check if script exists
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        print(f"Pipeline stopped at step {i}")
        break
    
    # Run script
    step_start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,  # Show output in real-time
        )
        
        step_time = time.time() - step_start
        print()
        print(f"✅ Step {i} complete! ({step_time/60:.1f} minutes)")
        print()
        
        completed_steps.append({
            'step': i,
            'name': step['name'],
            'time': step_time
        })
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Step {i} failed!")
        print(f"Error: {e}")
        print(f"Pipeline stopped at step {i}")
        break
    
    except KeyboardInterrupt:
        print()
        print(f"⚠️  Pipeline interrupted by user at step {i}")
        break

# ============================================================================
# SUMMARY
# ============================================================================
total_time = time.time() - total_start

print()
print("="*80)
print("PIPELINE SUMMARY")
print("="*80)
print()

if len(completed_steps) == len(STEPS):
    print("✅ All steps completed successfully!")
    print()
    
    print(f"⏱️  Timing Breakdown:")
    for step_info in completed_steps:
        print(f"   Step {step_info['step']}: {step_info['name']} - {step_info['time']/60:.1f} min")
    print(f"   Total: {total_time/60:.1f} min")
    print()
    
    print(f"📁 Final Output:")
    ensemble_dir = OUTPUT_DIR / "ensemble_text_image"
    submission_file = ensemble_dir / "submission.csv"
    print(f"   {submission_file}")
    print()
    
    print(f"🎯 Next Step:")
    print(f"   Copy submission file to submission folder:")
    print(f"   cp {submission_file} {OUTPUT_DIR.parent.parent}/submission/test_out.csv")
    print()
    
else:
    print(f"⚠️  Pipeline incomplete!")
    print(f"   Completed: {len(completed_steps)}/{len(STEPS)} steps")
    print(f"   Time elapsed: {total_time/60:.1f} min")
    print()
    
    if len(completed_steps) > 0:
        print(f"✓ Completed steps:")
        for step_info in completed_steps:
            print(f"   {step_info['step']}. {step_info['name']}")
        print()
    
    remaining = len(STEPS) - len(completed_steps)
    if remaining > 0:
        next_step = STEPS[len(completed_steps)]
        print(f"❌ Failed/Remaining steps:")
        for i in range(len(completed_steps), len(STEPS)):
            print(f"   {i+1}. {STEPS[i]['name']}")
        print()
        print(f"💡 To resume, run:")
        print(f"   python {next_step['script']}")
        print()
