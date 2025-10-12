"""
MEMORY-EFFICIENT IMAGE DOWNLOAD WITH RETRY LOGIC

Strategy:
1. Download images in small batches (500 at a time)
2. Retry failed downloads at the end
3. Save progress after each batch
4. Memory-efficient processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
import json

# Import auto-config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config_auto import DATA_DIR, OUTPUT_DIR

print("="*80)
print("MEMORY-EFFICIENT IMAGE DOWNLOAD WITH RETRY")
print("="*80)
print()

# Configuration
CONFIG = {
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR / "images_efficient",
    'batch_size': 500,  # Process 500 images at a time
    'max_retries': 3,
    'retry_delay': 2,  # seconds
    'timeout': 10,  # seconds per image
    'image_size': (224, 224),  # ResNet input size
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
(CONFIG['output_dir'] / 'train').mkdir(exist_ok=True)
(CONFIG['output_dir'] / 'test').mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load CSV files
train_files = [CONFIG['data_dir'] / 'train1.csv', CONFIG['data_dir'] / 'train2.csv']
test_files = [CONFIG['data_dir'] / 'test1.csv', CONFIG['data_dir'] / 'test2.csv']

train_dfs = []
for f in train_files:
    if f.exists():
        df = pd.read_csv(f)
        train_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

test_dfs = []
for f in test_files:
    if f.exists():
        df = pd.read_csv(f)
        test_dfs.append(df)
        print(f"✓ Loaded {f.name}: {len(df):,} rows")

train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\n✓ Total training samples: {len(train_df):,}")
print(f"✓ Total test samples: {len(test_df):,}")
print()

# ============================================================================
# STEP 2: DOWNLOAD FUNCTION WITH RETRY
# ============================================================================
def download_image(url, sample_id, save_dir, max_retries=3, timeout=10):
    """
    Download and save image with retry logic
    
    Returns:
        (success: bool, error_msg: str or None)
    """
    save_path = save_dir / f"{sample_id}.jpg"
    
    # Skip if already downloaded
    if save_path.exists():
        return True, None
    
    for attempt in range(max_retries):
        try:
            # Download image
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Open and validate image
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB
            
            # Resize to save space
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Save
            img.save(save_path, 'JPEG', quality=85)
            
            return True, None
            
        except requests.exceptions.Timeout:
            error = f"Timeout (attempt {attempt+1}/{max_retries})"
            if attempt < max_retries - 1:
                time.sleep(CONFIG['retry_delay'])
            else:
                return False, error
                
        except requests.exceptions.RequestException as e:
            error = f"Request error: {str(e)[:50]}"
            if attempt < max_retries - 1:
                time.sleep(CONFIG['retry_delay'])
            else:
                return False, error
                
        except Exception as e:
            error = f"Image error: {str(e)[:50]}"
            if attempt < max_retries - 1:
                time.sleep(CONFIG['retry_delay'])
            else:
                return False, error
    
    return False, "Max retries exceeded"

# ============================================================================
# STEP 3: DOWNLOAD TRAINING IMAGES IN BATCHES
# ============================================================================
print("="*80)
print("STEP 3: DOWNLOADING TRAINING IMAGES")
print("="*80)

train_save_dir = CONFIG['output_dir'] / 'train'
train_progress_file = CONFIG['output_dir'] / 'train_progress.json'

# Load progress if exists
if train_progress_file.exists():
    with open(train_progress_file, 'r') as f:
        progress = json.load(f)
    downloaded = set(progress['downloaded'])
    failed = progress.get('failed', {})
    print(f"✓ Resuming from previous progress: {len(downloaded)} already downloaded")
else:
    downloaded = set()
    failed = {}

# Process in batches
n_batches = (len(train_df) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
print(f"✓ Processing {len(train_df):,} images in {n_batches} batches")
print()

for batch_idx in range(n_batches):
    start_idx = batch_idx * CONFIG['batch_size']
    end_idx = min((batch_idx + 1) * CONFIG['batch_size'], len(train_df))
    batch_df = train_df.iloc[start_idx:end_idx]
    
    print(f"📦 Batch {batch_idx+1}/{n_batches} ({start_idx}-{end_idx})")
    
    batch_success = 0
    batch_failed = 0
    
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Downloading"):
        sample_id = row['sample_id']
        
        # Skip if already downloaded
        if sample_id in downloaded:
            batch_success += 1
            continue
        
        # Download
        success, error = download_image(
            row['image_link'],
            sample_id,
            train_save_dir,
            max_retries=CONFIG['max_retries'],
            timeout=CONFIG['timeout']
        )
        
        if success:
            downloaded.add(sample_id)
            batch_success += 1
        else:
            failed[sample_id] = {
                'url': row['image_link'],
                'error': error,
                'batch': batch_idx + 1
            }
            batch_failed += 1
    
    print(f"   ✓ Success: {batch_success}/{len(batch_df)}")
    print(f"   ❌ Failed: {batch_failed}/{len(batch_df)}")
    
    # Save progress
    progress = {
        'downloaded': list(downloaded),
        'failed': failed,
        'batch_completed': batch_idx + 1,
        'total_batches': n_batches
    }
    with open(train_progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"   💾 Progress saved")
    print()

print(f"✅ Training images download complete!")
print(f"   Total downloaded: {len(downloaded):,}")
print(f"   Total failed: {len(failed):,}")
print()

# ============================================================================
# STEP 4: RETRY FAILED TRAINING IMAGES
# ============================================================================
if len(failed) > 0:
    print("="*80)
    print("STEP 4: RETRYING FAILED TRAINING IMAGES")
    print("="*80)
    print(f"Retrying {len(failed)} failed downloads...")
    print()
    
    retry_success = 0
    retry_failed = {}
    
    for sample_id, info in tqdm(failed.items(), desc="Retrying"):
        success, error = download_image(
            info['url'],
            sample_id,
            train_save_dir,
            max_retries=CONFIG['max_retries'] + 2,  # More retries
            timeout=CONFIG['timeout'] + 5  # Longer timeout
        )
        
        if success:
            downloaded.add(sample_id)
            retry_success += 1
        else:
            retry_failed[sample_id] = {
                'url': info['url'],
                'error': error,
                'original_error': info['error']
            }
    
    print(f"✅ Retry complete!")
    print(f"   Recovered: {retry_success}")
    print(f"   Still failed: {len(retry_failed)}")
    
    # Update progress
    progress = {
        'downloaded': list(downloaded),
        'failed': retry_failed,
        'batch_completed': n_batches,
        'total_batches': n_batches,
        'retry_completed': True
    }
    with open(train_progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print()

# ============================================================================
# STEP 5: DOWNLOAD TEST IMAGES (SAME PROCESS)
# ============================================================================
print("="*80)
print("STEP 5: DOWNLOADING TEST IMAGES")
print("="*80)

test_save_dir = CONFIG['output_dir'] / 'test'
test_progress_file = CONFIG['output_dir'] / 'test_progress.json'

# Load progress if exists
if test_progress_file.exists():
    with open(test_progress_file, 'r') as f:
        progress = json.load(f)
    downloaded_test = set(progress['downloaded'])
    failed_test = progress.get('failed', {})
    print(f"✓ Resuming from previous progress: {len(downloaded_test)} already downloaded")
else:
    downloaded_test = set()
    failed_test = {}

# Process in batches
n_batches_test = (len(test_df) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']
print(f"✓ Processing {len(test_df):,} images in {n_batches_test} batches")
print()

for batch_idx in range(n_batches_test):
    start_idx = batch_idx * CONFIG['batch_size']
    end_idx = min((batch_idx + 1) * CONFIG['batch_size'], len(test_df))
    batch_df = test_df.iloc[start_idx:end_idx]
    
    print(f"📦 Batch {batch_idx+1}/{n_batches_test} ({start_idx}-{end_idx})")
    
    batch_success = 0
    batch_failed = 0
    
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Downloading"):
        sample_id = row['sample_id']
        
        if sample_id in downloaded_test:
            batch_success += 1
            continue
        
        success, error = download_image(
            row['image_link'],
            sample_id,
            test_save_dir,
            max_retries=CONFIG['max_retries'],
            timeout=CONFIG['timeout']
        )
        
        if success:
            downloaded_test.add(sample_id)
            batch_success += 1
        else:
            failed_test[sample_id] = {
                'url': row['image_link'],
                'error': error,
                'batch': batch_idx + 1
            }
            batch_failed += 1
    
    print(f"   ✓ Success: {batch_success}/{len(batch_df)}")
    print(f"   ❌ Failed: {batch_failed}/{len(batch_df)}")
    
    # Save progress
    progress = {
        'downloaded': list(downloaded_test),
        'failed': failed_test,
        'batch_completed': batch_idx + 1,
        'total_batches': n_batches_test
    }
    with open(test_progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"   💾 Progress saved")
    print()

print(f"✅ Test images download complete!")
print(f"   Total downloaded: {len(downloaded_test):,}")
print(f"   Total failed: {len(failed_test):,}")
print()

# ============================================================================
# STEP 6: RETRY FAILED TEST IMAGES
# ============================================================================
if len(failed_test) > 0:
    print("="*80)
    print("STEP 6: RETRYING FAILED TEST IMAGES")
    print("="*80)
    print(f"Retrying {len(failed_test)} failed downloads...")
    print()
    
    retry_success_test = 0
    retry_failed_test = {}
    
    for sample_id, info in tqdm(failed_test.items(), desc="Retrying"):
        success, error = download_image(
            info['url'],
            sample_id,
            test_save_dir,
            max_retries=CONFIG['max_retries'] + 2,
            timeout=CONFIG['timeout'] + 5
        )
        
        if success:
            downloaded_test.add(sample_id)
            retry_success_test += 1
        else:
            retry_failed_test[sample_id] = {
                'url': info['url'],
                'error': error,
                'original_error': info['error']
            }
    
    print(f"✅ Retry complete!")
    print(f"   Recovered: {retry_success_test}")
    print(f"   Still failed: {len(retry_failed_test)}")
    
    # Update progress
    progress = {
        'downloaded': list(downloaded_test),
        'failed': retry_failed_test,
        'batch_completed': n_batches_test,
        'total_batches': n_batches_test,
        'retry_completed': True
    }
    with open(test_progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✅ DOWNLOAD COMPLETE!")
print("="*80)
print()
print(f"📊 Final Statistics:")
print(f"   Training images:")
print(f"      Downloaded: {len(downloaded):,} / {len(train_df):,}")
print(f"      Success rate: {len(downloaded)/len(train_df)*100:.2f}%")
print(f"   ")
print(f"   Test images:")
print(f"      Downloaded: {len(downloaded_test):,} / {len(test_df):,}")
print(f"      Success rate: {len(downloaded_test)/len(test_df)*100:.2f}%")
print()
print(f"📁 Images saved to:")
print(f"   Train: {train_save_dir}")
print(f"   Test: {test_save_dir}")
print()
print(f"📝 Progress files:")
print(f"   Train: {train_progress_file}")
print(f"   Test: {test_progress_file}")
print()
print(f"🚀 Next step: Run image feature extraction")
print(f"   python extract_image_features_efficient.py")
print()
