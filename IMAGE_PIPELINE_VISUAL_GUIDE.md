# 📊 Image Features Pipeline - Visual Guide

```
╔══════════════════════════════════════════════════════════════════════════╗
║                   IMAGE FEATURES PIPELINE OVERVIEW                        ║
╚══════════════════════════════════════════════════════════════════════════╝

CURRENT STATE (V3 Clean):
┌────────────────────────────────────────────────────────────────────────┐
│  Text Features Only (47 features)                                      │
│  Test SMAPE: 63.28%                                                    │
│  Status: ✅ Working, but can be improved                              │
└────────────────────────────────────────────────────────────────────────┘

TARGET STATE (V5 Text + Image):
┌────────────────────────────────────────────────────────────────────────┐
│  Text + Image Features (204 features)                                  │
│  Expected SMAPE: 55-60%                                                │
│  Improvement: 5-10 percentage points                                   │
│  Status: 🎯 Ready to implement                                        │
└────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
STEP-BY-STEP PIPELINE FLOW
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: IMAGE FEATURE EXTRACTION (40-90 minutes)                        │
│ File: image_feature_extraction.py                                       │
└─────────────────────────────────────────────────────────────────────────┘

Input:
┌────────────┐    ┌────────────┐    ┌───────────┐    ┌───────────┐
│ train1.csv │    │ train2.csv │    │ test1.csv │    │ test2.csv │
│  37.5K     │ + │  37.5K     │    │  37.5K    │ + │  37.5K    │
└────────────┘    └────────────┘    └───────────┘    └───────────┘
       │                 │                  │               │
       └─────────────────┴──────────────────┴───────────────┘
                              ↓
                   ┌──────────────────────┐
                   │  Combine by sample_id │
                   └──────────────────────┘
                              ↓
               ┌─────────────────────────────┐
               │ 150,000 image URLs           │
               │ (image_link column)          │
               └─────────────────────────────┘
                              ↓
             ┌────────────────────────────────┐
             │ For each image URL:            │
             │ 1. Download image (retry 3x)   │
             │ 2. Resize to 224x224           │
             │ 3. Extract features ──────────→│
             └────────────────────────────────┘
                              ↓
        ┌─────────────────────┬──────────────────────┬───────────────────┐
        │                     │                      │                   │
   ┌────▼─────┐      ┌───────▼──────┐      ┌───────▼──────┐   ┌───────▼──────┐
   │ ResNet50 │      │ Color Stats  │      │ Quality      │   │ Metadata     │
   │ Features │      │ (RGB, HSV)   │      │ Metrics      │   │              │
   │ 2048-dim │      │              │      │              │   │              │
   └────┬─────┘      └───────┬──────┘      └───────┬──────┘   └───────┬──────┘
        │                    │                     │                  │
   ┌────▼─────┐             │                     │                  │
   │   PCA    │             │                     │                  │
   │ Reduce to│             │                     │                  │
   │ 128-dim  │             │                     │                  │
   └────┬─────┘             │                     │                  │
        │                   │                     │                  │
        └───────────────────┴─────────────────────┴──────────────────┘
                              ↓
                   ┌──────────────────────┐
                   │  Combine all features │
                   └──────────────────────┘
                              ↓
Output:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ image_features_train.csv    │    │ image_features_test.csv     │
│ 75,000 × 158 features       │    │ 75,000 × 158 features       │
│                             │    │                             │
│ - 128 ResNet embeddings     │    │ - 128 ResNet embeddings     │
│ - 21 color features         │    │ - 21 color features         │
│ - 7 quality features        │    │ - 7 quality features        │
│ - 1 availability flag       │    │ - 1 availability flag       │
│ - 1 sample_id               │    │ - 1 sample_id               │
└─────────────────────────────┘    └─────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: COMBINE FEATURES (2 minutes)                                    │
│ File: combine_features_with_images.py                                   │
└─────────────────────────────────────────────────────────────────────────┘

Input A (Text Features - V3 Clean):
┌──────────────────────────────────┐
│ features_v3_clean_train.csv      │
│ 75,000 × 49 columns              │
│                                  │
│ - 47 text features (NO LEAKAGE!) │
│ - 1 sample_id                    │
│ - 1 price                        │
└──────────────────────────────────┘

Input B (Image Features):
┌──────────────────────────────────┐
│ image_features_train.csv         │
│ 75,000 × 158 columns             │
│                                  │
│ - 157 image features             │
│ - 1 sample_id                    │
└──────────────────────────────────┘

         ↓                  ↓
         └─────────┬────────┘
                   ↓
        ┌──────────────────────┐
        │ Merge on sample_id   │
        └──────────────────────┘
                   ↓
Output:
┌──────────────────────────────────────┐
│ features_v5_text_image_train.csv     │
│ 75,000 × 206 columns                 │
│                                      │
│ - 47 text features                   │
│ - 157 image features                 │
│ - 1 sample_id                        │
│ - 1 price                            │
│                                      │
│ Total: 204 features!                 │
└──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: TRAIN MODELS (15 minutes)                                      │
│ File: train_v5_text_image.py                                            │
└─────────────────────────────────────────────────────────────────────────┘

Input:
┌──────────────────────────────────────┐
│ features_v5_text_image_train.csv     │
│ 75,000 samples × 204 features        │
└──────────────────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ Stratified Split     │
        │ (by price quantiles) │
        └──────────────────────┘
                   ↓
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐    ┌───▼────┐    ┌───▼────┐
│ Train  │    │  Val   │    │  Test  │
│  60%   │    │  15%   │    │  25%   │
│ 45K    │    │ 11.25K │    │ 18.75K │
└───┬────┘    └───┬────┘    └───┬────┘
    │             │              │
    └─────────────┴──────────────┘
                   ↓
         ┌─────────────────────┐
         │ Train 3 Models:     │
         │ - XGBoost           │
         │ - LightGBM          │
         │ - CatBoost          │
         └─────────────────────┘
                   ↓
         ┌─────────────────────┐
         │ Evaluate on Test    │
         │ Set (SMAPE)         │
         └─────────────────────┘
                   ↓
         ┌─────────────────────┐
         │ Select Best Model   │
         └─────────────────────┘
                   ↓
         ┌─────────────────────┐
         │ Predict on 75K      │
         │ Test Samples        │
         └─────────────────────┘
                   ↓
Output:
┌──────────────────────────────────────┐
│ test_out_v5_text_image.csv           │
│ 75,000 predictions                   │
│                                      │
│ Columns:                             │
│ - sample_id                          │
│ - price (predicted)                  │
│                                      │
│ Ready for submission! ✅             │
└──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
FEATURE BREAKDOWN
═══════════════════════════════════════════════════════════════════════════

TEXT FEATURES (47 total - from V3 Clean):
┌──────────────────────────────────────────────────────────────────────┐
│ ✅ Text Features (7)                                                 │
│    title_length, title_word_count, desc_length, capital_ratio...    │
│                                                                      │
│ ✅ Quantity Features (11)                                           │
│    ipq, multipack_size, weight_value, volume_value...               │
│                                                                      │
│ ✅ Brand Features (5) - NO PRICE INFO!                             │
│    brand_frequency, brand_tier, is_popular_brand...                 │
│                                                                      │
│ ✅ Category Features (3)                                            │
│    category_size, is_large_category...                              │
│                                                                      │
│ ✅ Interaction Features (5)                                         │
│    brand_category_frequency, tier_multipack_interaction...          │
│                                                                      │
│ ✅ Quality Indicators (7)                                           │
│    premium_keyword_count, value_keyword_count...                    │
│                                                                      │
│ ✅ Size Categories (4)                                              │
│    is_travel_size, is_family_size, is_multipack...                  │
│                                                                      │
│ ✅ Advanced Text Patterns (6)                                       │
│    special_char_count, numeric_density...                           │
└──────────────────────────────────────────────────────────────────────┘

IMAGE FEATURES (157 total - NEW!):
┌──────────────────────────────────────────────────────────────────────┐
│ 🆕 ResNet50 Embeddings (128)                                         │
│    img_embed_0, img_embed_1, ..., img_embed_127                     │
│    Captures: shapes, textures, patterns, objects                    │
│                                                                      │
│ 🆕 Color Features (21)                                              │
│    RGB: mean_r, mean_g, mean_b, std_r, std_g, std_b                │
│    HSV: mean_hue, mean_saturation, mean_value, std_*               │
│    Dominant: dominant_color_1_r/g/b, dominant_color_2_r/g/b...     │
│    Captures: packaging colors, brand schemes                        │
│                                                                      │
│ 🆕 Quality Features (7)                                             │
│    Dimensions: image_width, image_height, aspect_ratio, image_area │
│    Metrics: sharpness, contrast, brightness                         │
│    Captures: image quality, product photography                     │
│                                                                      │
│ 🆕 Metadata (1)                                                     │
│    image_available (0 or 1)                                         │
│    Indicates if image was successfully downloaded                   │
└──────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
PERFORMANCE TIMELINE
═══════════════════════════════════════════════════════════════════════════

    Baseline (INVALID)          V3 Clean            V5 Text+Image (TARGET)
    ─────────────────          ─────────          ───────────────────────
         47.52%                 63.28%                  55-60%
    (has leakage ❌)     (no leakage ✅)          (no leakage ✅)
                                  │
                                  │  Add 157 image features
                                  ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
    │ 35 features  │      │ 47 features  │      │ 204 features     │
    │ Text only    │      │ Text only    │      │ Text + Image     │
    │ + leakage    │      │ (clean)      │      │ (clean)          │
    └──────────────┘      └──────────────┘      └──────────────────┘
                                  │
                                  │  Expected improvement
                                  ▼
                          ┌──────────────┐
                          │ 5-10% better │
                          │ SMAPE        │
                          └──────────────┘


═══════════════════════════════════════════════════════════════════════════
WHY IMAGES SHOULD HELP
═══════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────┐
│ 1. PACKAGE SIZE VISUAL CUES                                            │
│    ┌───────┐  ┌──────────────┐  ┌────────────────────────┐           │
│    │Single │  │  6-Pack      │  │  Family Size (24-pack) │           │
│    │$2-5   │  │  $10-20      │  │  $25-50                │           │
│    └───────┘  └──────────────┘  └────────────────────────┘           │
│                                                                        │
│ 2. BRAND PACKAGING QUALITY                                            │
│    ┌─────────────┐           ┌─────────────┐                         │
│    │  Premium    │           │  Economy    │                         │
│    │  - Gold foil│           │  - Plain    │                         │
│    │  - Clean    │           │  - Cluttered│                         │
│    │  Higher $   │           │  Lower $    │                         │
│    └─────────────┘           └─────────────┘                         │
│                                                                        │
│ 3. PRODUCT CATEGORY VISUAL PATTERNS                                   │
│    Food Packaging    Personal Care      Supplements                  │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
│    │ Colorful │      │ Clean,   │      │ Bottle   │                 │
│    │ Photos   │      │ White    │      │ + Pills  │                 │
│    └──────────┘      └──────────┘      └──────────┘                 │
│                                                                        │
│ 4. IMAGE QUALITY = PRODUCT QUALITY                                    │
│    Professional Photo     Amateur Photo                               │
│    ┌──────────────┐      ┌──────────────┐                           │
│    │ Sharp        │      │ Blurry       │                           │
│    │ High contrast│      │ Low contrast │                           │
│    │ Premium      │      │ Budget       │                           │
│    └──────────────┘      └──────────────┘                           │
└────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
EXECUTION OPTIONS
═══════════════════════════════════════════════════════════════════════════

OPTION 1: AUTOMATED (Recommended)
┌────────────────────────────────────────────────────────────────────────┐
│ $ cd try2                                                              │
│ $ python run_image_pipeline.py                                        │
│                                                                        │
│ ✅ Runs all 3 steps automatically                                     │
│ ✅ Pre-flight checks                                                  │
│ ✅ Progress tracking                                                  │
│ ✅ Error handling                                                     │
│ ⏱  Total time: 1-2 hours                                              │
└────────────────────────────────────────────────────────────────────────┘

OPTION 2: MANUAL STEP-BY-STEP
┌────────────────────────────────────────────────────────────────────────┐
│ $ cd try2                                                              │
│                                                                        │
│ # Step 1: Extract image features (40-90 min)                          │
│ $ python image_feature_extraction.py                                  │
│                                                                        │
│ # Step 2: Combine features (2 min)                                    │
│ $ python combine_features_with_images.py                              │
│                                                                        │
│ # Step 3: Train models (15 min)                                       │
│ $ python train_v5_text_image.py                                       │
└────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════

TRAINING OUTPUT (what you'll see):
┌────────────────────────────────────────────────────────────────────────┐
│ MODEL COMPARISON                                                       │
│ ──────────────────────────────────────────────────────────────────────│
│ Model      Train SMAPE    Val SMAPE    Test SMAPE                     │
│ XGBoost       XX.XX%       XX.XX%       XX.XX%  ← Check this value!   │
│ LightGBM      XX.XX%       XX.XX%       XX.XX%                         │
│ CatBoost      XX.XX%       XX.XX%       XX.XX%                         │
│                                                                        │
│ 🏆 Best Model: XGBoost                                                │
│    Test SMAPE: XX.XX%                                                  │
│                                                                        │
│ COMPARISON WITH PREVIOUS VERSIONS:                                     │
│ V3 Clean (Text only):    63.28%                                        │
│ V5 (Text + Image):       XX.XX%  ← Target: 55-60%                     │
│ Improvement:             XX.XX% ↓                                      │
└────────────────────────────────────────────────────────────────────────┘

SUCCESS CRITERIA:
┌────────────────────────────────────────────────────────────────────────┐
│ ✅ Minimum:  V5 < 63.28% (any improvement)                            │
│ ✅ Good:     V5 = 57-60% (5-6% improvement)                           │
│ ✅ Great:    V5 = 55-57% (8-10% improvement)                          │
│ ✅ Excellent: V5 < 55% (>10% improvement)                             │
└────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING GUIDE
═══════════════════════════════════════════════════════════════════════════

ISSUE: Many images fail to download
┌────────────────────────────────────────────────────────────────────────┐
│ SYMPTOMS:                                                              │
│ - "Failed to download..." warnings                                    │
│ - Image success rate < 80%                                            │
│                                                                        │
│ SOLUTIONS:                                                             │
│ 1. Check internet connection                                          │
│ 2. Increase timeout in image_feature_extraction.py:                   │
│    DOWNLOAD_TIMEOUT = 20  (default: 10)                               │
│    MAX_RETRIES = 5        (default: 3)                                │
│ 3. Re-run - already processed images are skipped                      │
└────────────────────────────────────────────────────────────────────────┘

ISSUE: Out of memory (GPU)
┌────────────────────────────────────────────────────────────────────────┐
│ SYMPTOMS:                                                              │
│ - CUDA out of memory error                                            │
│                                                                        │
│ SOLUTION:                                                              │
│ Force CPU mode in image_feature_extraction.py:                        │
│ device = 'cpu'  # Around line 490                                     │
└────────────────────────────────────────────────────────────────────────┘

ISSUE: No improvement in SMAPE
┌────────────────────────────────────────────────────────────────────────┐
│ SYMPTOMS:                                                              │
│ - V5 SMAPE ≈ V3 SMAPE (63%)                                           │
│                                                                        │
│ SOLUTIONS:                                                             │
│ 1. Check feature importance - are image features used?                │
│ 2. Try EfficientNet instead of ResNet50                               │
│ 3. Fine-tune image model on this dataset                              │
│ 4. Add object detection features                                      │
│ 5. Use CLIP embeddings instead                                        │
└────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
FILES CREATED
═══════════════════════════════════════════════════════════════════════════

📁 try2/
  ├── 🆕 image_feature_extraction.py        (Main extraction script)
  ├── 🆕 combine_features_with_images.py    (Feature merger)
  ├── 🆕 train_v5_text_image.py             (Model training)
  ├── 🆕 run_image_pipeline.py              (Master runner)
  └── 🆕 IMAGE_FEATURES_README.md           (Technical docs)

📁 Project root/
  ├── 🆕 IMAGE_PIPELINE_QUICKSTART.md       (Quick reference)
  ├── 🆕 IMAGE_FEATURES_IMPLEMENTATION_SUMMARY.md
  └── ✏️ requirements.txt (updated)


═══════════════════════════════════════════════════════════════════════════
READY TO RUN! 🚀
═══════════════════════════════════════════════════════════════════════════

Quick Start:
$ cd try2
$ python run_image_pipeline.py

Expected result: 55-60% SMAPE (improvement from 63.28%)
Time: 1-2 hours total
```
