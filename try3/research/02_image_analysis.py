"""
IMAGE ANALYSIS RESEARCH - PHASE 2
==================================
Goal: Understand what visual information images provide for price prediction

Research Questions:
1. Image availability and quality
2. Visual patterns vs price ranges
3. Package size/multipack detection
4. Color and design patterns
5. Image-text correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("../../try2/dataset")
OUTPUT_DIR = Path("outputs")
IMAGE_CACHE_DIR = OUTPUT_DIR / "sample_images"
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

print("="*80)
print("IMAGE ANALYSIS RESEARCH - PHASE 2")
print("="*80)

# Load research data from Phase 1
print("\n📂 Loading enriched research data...")
train = pd.read_csv(OUTPUT_DIR / 'research_train_enriched.csv')
print(f"   ✓ Loaded {len(train):,} samples")

# ============================================================================
# STEP 1: IMAGE AVAILABILITY CHECK
# ============================================================================
print("\n" + "="*80)
print("STEP 1: IMAGE AVAILABILITY & ACCESSIBILITY")
print("="*80)

print("\n🔗 Checking image links...")
train['has_image_link'] = train['image_link'].notna()
print(f"   Samples with image links: {train['has_image_link'].sum():,} ({train['has_image_link'].sum()/len(train)*100:.2f}%)")

# Sample different price ranges for image analysis
print("\n📊 Sampling images from different price ranges...")
sample_strategy = {
    'Budget ($0-$10)': (0, 10, 30),
    'Economy ($10-$20)': (10, 20, 30),
    'Mid-Range ($20-$50)': (20, 50, 30),
    'Premium ($50-$100)': (50, 100, 20),
    'Luxury ($100+)': (100, float('inf'), 10)
}

sampled_data = []
for range_name, (min_price, max_price, n_samples) in sample_strategy.items():
    range_data = train[(train['price'] >= min_price) & (train['price'] < max_price)]
    if len(range_data) > 0:
        sample = range_data.sample(n=min(n_samples, len(range_data)), random_state=42)
        sample['price_category'] = range_name
        sampled_data.append(sample)
        print(f"   {range_name:20s}: Sampled {len(sample):>3} images from {len(range_data):>6,} products")

sample_df = pd.concat(sampled_data, ignore_index=True)
print(f"\n   Total images to analyze: {len(sample_df)}")

# ============================================================================
# STEP 2: DOWNLOAD SAMPLE IMAGES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DOWNLOADING SAMPLE IMAGES")
print("="*80)

def download_image(url, save_path, max_retries=3):
    """Download image with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(save_path)
                return True, img.size, img.mode
            else:
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, None, str(e)
            time.sleep(2)
    return False, None, "Max retries exceeded"

print("\n⬇️ Downloading images (this may take a few minutes)...")
download_results = []

for idx, row in sample_df.iterrows():
    sample_id = row['sample_id']
    image_url = row['image_link']
    
    if pd.isna(image_url):
        continue
    
    save_path = IMAGE_CACHE_DIR / f"{sample_id}.jpg"
    
    # Skip if already downloaded
    if save_path.exists():
        try:
            img = Image.open(save_path)
            download_results.append({
                'sample_id': sample_id,
                'success': True,
                'width': img.size[0],
                'height': img.size[1],
                'mode': img.mode
            })
        except:
            pass
        continue
    
    success, size, mode = download_image(image_url, save_path)
    
    result = {
        'sample_id': sample_id,
        'success': success,
        'width': size[0] if success else None,
        'height': size[1] if success else None,
        'mode': mode if success else None
    }
    download_results.append(result)
    
    if len(download_results) % 20 == 0:
        print(f"   Progress: {len(download_results)}/{len(sample_df)} images")
        time.sleep(0.5)  # Rate limiting

download_df = pd.DataFrame(download_results)
success_rate = download_df['success'].sum() / len(download_df) * 100
print(f"\n   ✓ Successfully downloaded: {download_df['success'].sum()}/{len(download_df)} ({success_rate:.1f}%)")

# Merge download results with sample data
sample_df = sample_df.merge(download_df, on='sample_id', how='left')

# ============================================================================
# STEP 3: IMAGE PROPERTIES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: IMAGE PROPERTIES ANALYSIS")
print("="*80)

successful_images = sample_df[sample_df['success'] == True]

if len(successful_images) > 0:
    print(f"\n📐 Image Dimensions:")
    print(f"   Average width: {successful_images['width'].mean():.0f} px")
    print(f"   Average height: {successful_images['height'].mean():.0f} px")
    print(f"   Most common size: {successful_images['width'].mode().values[0]:.0f} x {successful_images['height'].mode().values[0]:.0f} px")
    
    print(f"\n🎨 Color Modes:")
    mode_counts = successful_images['mode'].value_counts()
    for mode, count in mode_counts.items():
        pct = (count / len(successful_images)) * 100
        print(f"   {mode}: {count} images ({pct:.1f}%)")
    
    # Aspect ratio analysis
    successful_images['aspect_ratio'] = successful_images['width'] / successful_images['height']
    print(f"\n📏 Aspect Ratios:")
    print(f"   Mean: {successful_images['aspect_ratio'].mean():.2f}")
    print(f"   Median: {successful_images['aspect_ratio'].median():.2f}")
    print(f"   Std: {successful_images['aspect_ratio'].std():.2f}")

# ============================================================================
# STEP 4: VISUAL PATTERN EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: BASIC VISUAL PATTERN EXTRACTION")
print("="*80)

def analyze_image_basic(image_path):
    """Extract basic visual features from image"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        features = {
            'width': img.size[0],
            'height': img.size[1],
            'aspect_ratio': img.size[0] / img.size[1],
            'total_pixels': img.size[0] * img.size[1],
        }
        
        # Color analysis (if RGB)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            features['mean_red'] = img_array[:,:,0].mean()
            features['mean_green'] = img_array[:,:,1].mean()
            features['mean_blue'] = img_array[:,:,2].mean()
            features['brightness'] = img_array.mean()
            features['color_variance'] = img_array.std()
        else:
            features['mean_red'] = None
            features['mean_green'] = None
            features['mean_blue'] = None
            features['brightness'] = img_array.mean() if len(img_array.shape) == 2 else None
            features['color_variance'] = img_array.std() if len(img_array.shape) == 2 else None
        
        return features
    except Exception as e:
        return None

print("\n🔍 Extracting visual features from sample images...")
visual_features = []

for idx, row in successful_images.iterrows():
    sample_id = row['sample_id']
    image_path = IMAGE_CACHE_DIR / f"{sample_id}.jpg"
    
    if image_path.exists():
        features = analyze_image_basic(image_path)
        if features:
            features['sample_id'] = sample_id
            visual_features.append(features)

visual_df = pd.DataFrame(visual_features)
print(f"   ✓ Extracted features from {len(visual_df)} images")

# Merge with sample data
analysis_df = sample_df.merge(visual_df, on='sample_id', how='left')

# ============================================================================
# STEP 5: PRICE VS VISUAL PATTERNS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: PRICE VS VISUAL PATTERNS CORRELATION")
print("="*80)

if len(visual_df) > 0:
    # Correlations
    print("\n🔗 Visual Features vs Price Correlation:")
    for feature in ['brightness', 'color_variance', 'aspect_ratio']:
        if feature in analysis_df.columns:
            corr = analysis_df[['price', feature]].corr().iloc[0, 1]
            print(f"   {feature:20s}: {corr:>7.4f}")
    
    # Brightness by price range
    print("\n💡 Average Brightness by Price Range:")
    for price_cat in analysis_df['price_category'].unique():
        cat_data = analysis_df[analysis_df['price_category'] == price_cat]
        avg_brightness = cat_data['brightness'].mean()
        if not pd.isna(avg_brightness):
            print(f"   {price_cat:20s}: {avg_brightness:>7.2f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price vs Brightness
    axes[0, 0].scatter(analysis_df['brightness'], analysis_df['price'], alpha=0.6)
    axes[0, 0].set_xlabel('Image Brightness', fontsize=12)
    axes[0, 0].set_ylabel('Price ($)', fontsize=12)
    axes[0, 0].set_title('Price vs Image Brightness', fontsize=14, fontweight='bold')
    
    # Price vs Color Variance
    axes[0, 1].scatter(analysis_df['color_variance'], analysis_df['price'], alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Color Variance (Complexity)', fontsize=12)
    axes[0, 1].set_ylabel('Price ($)', fontsize=12)
    axes[0, 1].set_title('Price vs Image Complexity', fontsize=14, fontweight='bold')
    
    # Aspect Ratio Distribution
    axes[1, 0].hist(analysis_df['aspect_ratio'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Image Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Square')
    axes[1, 0].legend()
    
    # Price distribution by category
    price_by_cat = [analysis_df[analysis_df['price_category'] == cat]['price'].values 
                    for cat in analysis_df['price_category'].unique()]
    axes[1, 1].boxplot(price_by_cat, labels=analysis_df['price_category'].unique())
    axes[1, 1].set_xlabel('Price Category', fontsize=12)
    axes[1, 1].set_ylabel('Price ($)', fontsize=12)
    axes[1, 1].set_title('Price Distribution by Category (Sample)', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_image_visual_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n   ✓ Saved visualization: {OUTPUT_DIR / '05_image_visual_analysis.png'}")

# ============================================================================
# STEP 6: CREATE VISUAL MONTAGE
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING VISUAL MONTAGE BY PRICE RANGE")
print("="*80)

def create_montage(image_paths, titles, save_path, grid_size=(5, 6)):
    """Create a montage of images"""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (img_path, title) in enumerate(zip(image_paths[:rows*cols], titles[:rows*cols])):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=8)
            axes[idx].axis('off')
        except:
            axes[idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(len(image_paths), rows*cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

print("\n🖼️ Creating price range montages...")

for price_cat in sample_df['price_category'].unique():
    cat_data = sample_df[(sample_df['price_category'] == price_cat) & (sample_df['success'] == True)]
    
    if len(cat_data) > 0:
        image_paths = [IMAGE_CACHE_DIR / f"{sid}.jpg" for sid in cat_data['sample_id']]
        titles = [f"${p:.2f}\n{brand[:20]}" for p, brand in zip(cat_data['price'], cat_data['brand'])]
        
        safe_filename = price_cat.replace('$', '').replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        save_path = OUTPUT_DIR / f'06_montage_{safe_filename}.png'
        
        create_montage(image_paths, titles, save_path, grid_size=(5, 6))
        print(f"   ✓ Created montage: {save_path.name}")

# ============================================================================
# SAVE ANALYSIS RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING IMAGE ANALYSIS RESULTS")
print("="*80)

# Save analysis dataframe
analysis_df.to_csv(OUTPUT_DIR / 'image_analysis_results.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'image_analysis_results.csv'}")

# Save summary report
with open(OUTPUT_DIR / 'image_analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("IMAGE ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("IMAGE AVAILABILITY\n")
    f.write("-"*80 + "\n")
    f.write(f"Total samples analyzed: {len(sample_df)}\n")
    f.write(f"Successfully downloaded: {download_df['success'].sum()} ({success_rate:.1f}%)\n\n")
    
    if len(successful_images) > 0:
        f.write("IMAGE PROPERTIES\n")
        f.write("-"*80 + "\n")
        f.write(f"Average dimensions: {successful_images['width'].mean():.0f} x {successful_images['height'].mean():.0f} px\n")
        f.write(f"Average aspect ratio: {successful_images['aspect_ratio'].mean():.2f}\n\n")
    
    if len(visual_df) > 0:
        f.write("VISUAL FEATURES\n")
        f.write("-"*80 + "\n")
        f.write(f"Features extracted: {len(visual_df)} images\n")
        f.write(f"Average brightness: {visual_df['brightness'].mean():.2f}\n")
        f.write(f"Average color variance: {visual_df['color_variance'].mean():.2f}\n")

print(f"✓ Saved: {OUTPUT_DIR / 'image_analysis_summary.txt'}")

print("\n" + "="*80)
print("✅ PHASE 2 IMAGE ANALYSIS COMPLETE")
print("="*80)
print("\nKey Outputs:")
print("   1. image_analysis_results.csv - Detailed analysis data")
print("   2. 05_image_visual_analysis.png - Visual patterns analysis")
print("   3. 06_montage_*.png - Price range image montages")
print("   4. image_analysis_summary.txt - Summary report")
print(f"   5. sample_images/ - {download_df['success'].sum()} downloaded images")
print("\n📋 Next: Review insights and proceed to Phase 3 (Advanced Feature Engineering)")
