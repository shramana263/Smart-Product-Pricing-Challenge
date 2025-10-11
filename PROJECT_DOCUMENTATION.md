# 🏆 Amazon ML Hackathon - Smart Product Pricing Challenge
## Complete Project Documentation & Progress Report

**Last Updated:** October 12, 2025  
**Current Status:** Target Leakage Discovered & Fixed, Clean Baseline Established  
**Best Valid SMAPE:** 63.28% (Clean Features V3)  
**Project Goal:** <45% SMAPE without target leakage

---

## 📊 Executive Summary

### Challenge Overview
- **Task:** Predict product prices from catalog text (title, description, images)
- **Dataset:** 75,000 training samples + 75,000 test samples
- **Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- **Key Discovery:** Original baseline (47.52%) had severe target leakage!

### Major Milestones
1. ✅ **Phase 1:** Explored sentence embeddings (all-MiniLM-L6-v2) → 69.90% SMAPE (failed)
2. ✅ **Phase 2:** Built 41 advanced hand-crafted features → 3.73% SMAPE (invalid - had leakage!)
3. ✅ **Phase 3:** Discovered target leakage in baseline (47.52%)
4. ✅ **Phase 4:** Created clean features (47 features) → **63.28% SMAPE** (valid!)
5. 🔄 **Phase 5 (In Progress):** Implementing safe target encoding + planning fine-tuned LLM

---

## 🗂️ Project Structure

```
Amazon_ML_hackathon/code/
├── try2/                                    # Main working directory
│   ├── dataset/                             # Raw data
│   │   ├── train1.csv                       # 37,500 training samples
│   │   ├── train2.csv                       # 37,500 training samples
│   │   ├── test1.csv                        # 37,500 test samples
│   │   └── test2.csv                        # 37,500 test samples
│   │
│   ├── preparation/                         # Feature engineering outputs
│   │   ├── features_clean.csv               # ⚠️ BASELINE (has leakage: brand_target)
│   │   ├── features_v2_train.csv            # ⚠️ 41 features (has leakage: estimated_unit_price)
│   │   ├── features_v2_test.csv
│   │   ├── features_v3_clean_train.csv      # ✅ 47 features (NO leakage!)
│   │   ├── features_v3_clean_test.csv       # ✅ 47 features (NO leakage!)
│   │   ├── features_v4_safe_target_train.csv # ✅ 49 features (CV-based target encoding)
│   │   ├── features_v4_safe_target_test.csv  # ✅ 49 features (CV-based target encoding)
│   │   ├── labels.csv                       # Price targets
│   │   └── feature_names.txt                # Feature documentation
│   │
│   ├── modeling/                            # Model training outputs
│   │   ├── improved_modeling.ipynb          # ⚠️ Baseline notebook (47.52% - has leakage)
│   │   ├── test_out_v2.csv                  # ⚠️ Predictions with leakage (invalid)
│   │   ├── test_out_v3_clean.csv            # ✅ Predictions from clean features (85/15 split)
│   │   ├── test_out_v3_fair.csv             # ✅ Predictions from clean features (60/15/25 split)
│   │   ├── feature_importance_v3_clean.csv  # Feature analysis
│   │   └── feature_importance_v3_fair.csv   # Feature analysis (fair comparison)
│   │
│   ├── preprocessing/                       # Text preprocessing
│   │   ├── TextPreprocessing.ipynb          # Initial text cleaning
│   │   └── processed_features.csv           # Preprocessed text features
│   │
│   ├── LEAKAGE_INVESTIGATION_REPORT.md      # 🔍 CRITICAL: Full leakage analysis
│   │
│   ├── advanced_feature_engineering.py      # ⚠️ V2 features (has leakage)
│   ├── advanced_feature_engineering_clean.py # ✅ V3 features (NO leakage)
│   ├── safe_target_encoding.py              # ✅ V4 features (CV-based encoding)
│   │
│   ├── train_v2_features.py                 # ⚠️ Training with leakage (invalid)
│   ├── train_v3_clean.py                    # ✅ Training clean features (85/15 split)
│   └── train_v3_fair_comparison.py          # ✅ Training clean features (60/15/25 split)
│
├── embeddings_data/                         # Sentence transformer outputs
│   ├── train_embeddings.csv                 # 75K × 130 (128 dims + sample_id + price)
│   └── test_embeddings.csv                  # 75K × 129 (128 dims + sample_id)
│
├── text_embeddings.py                       # Sentence embedding extraction
├── train_with_embeddings.py                 # Training with embeddings (69.90% SMAPE)
└── venv/                                    # Python virtual environment
```

---

## 📈 Performance Evolution Timeline

### Baseline (Original - INVALID)
- **File:** `try2/preparation/features_clean.csv` + `try2/modeling/improved_modeling.ipynb`
- **SMAPE:** 47.52%
- **Features:** 35 (including `brand_target`, `brand_target_log`)
- **Split:** 60% train / 15% val / 25% test (stratified by price quantiles)
- **Status:** ⚠️ **INVALID - Contains target leakage!**
- **Issue:** `brand_target = groupby('brand')['price'].mean()` calculated BEFORE train/test split

### Attempt 1: Sentence Embeddings (FAILED)
- **File:** `text_embeddings.py` + `train_with_embeddings.py`
- **Model:** all-MiniLM-L6-v2 (384 dims → 128 dims via PCA)
- **Results:**
  - Embeddings only: **69.90% SMAPE**
  - Embeddings + existing features: **55.63% SMAPE**
- **Conclusion:** Generic embeddings underperform hand-crafted features
- **Processing Time:** ~68 minutes for 150K samples

### Attempt 2: Advanced Features V2 (INVALID - Leakage)
- **File:** `try2/advanced_feature_engineering.py` + `try2/train_v2_features.py`
- **SMAPE:** 3.73% (XGBoost), 5.06% (LightGBM), 5.74% (CatBoost)
- **Features:** 41 features
- **Status:** ⚠️ **INVALID - Contains target leakage!**
- **Issue:** `estimated_unit_price = price / total_quantity` uses target price!
- **Discovery:** Test predictions were $0.01-$87.74 (suspiciously low) → Led to leakage investigation

### ✅ Current Best: Clean Features V3 (VALID)
- **Files:** 
  - Feature Engineering: `try2/advanced_feature_engineering_clean.py`
  - Training: `try2/train_v3_fair_comparison.py`
  - Features: `try2/preparation/features_v3_clean_train.csv` / `test.csv`
- **SMAPE:** **63.28%** (XGBoost), 63.44% (LightGBM), 64.69% (CatBoost)
- **Features:** 47 features (NO leakage!)
- **Split:** 60% train / 15% val / 25% test (same as baseline for fair comparison)
- **Status:** ✅ **VALID - No target leakage, deployable**

### 🔄 Next: Safe Target Encoding V4 (In Progress)
- **Files:**
  - Feature Engineering: `try2/safe_target_encoding.py`
  - Features: `try2/preparation/features_v4_safe_target_train.csv` / `test.csv`
- **SMAPE:** Not yet trained
- **Features:** 49 features (47 clean + 2 CV-based target encodings)
- **Status:** ✅ Features created, ready to train
- **Expected:** 55-58% SMAPE (5-10% improvement)

---

## 🔧 Feature Engineering Details

### V3 Clean Features (47 Features - NO LEAKAGE)

#### 1. Text Features (7 features)
```python
- title_length              # Character count
- title_word_count          # Word count
- desc_length               # Description character count
- desc_word_count           # Description word count
- has_description           # Boolean: has description?
- capital_ratio             # Ratio of capital letters
- number_count              # Count of numbers in title
```

#### 2. Quantity Features (11 features)
```python
- ipq                       # Item Pack Quantity (extracted from catalog)
- multipack_size            # Detected multipack size (e.g., "Pack of 6")
- has_multipack             # Boolean: is multipack?
- weight_value              # Extracted weight (oz, lb, kg)
- has_weight                # Boolean: has weight?
- volume_value              # Extracted volume (fl oz, ml, L)
- has_volume                # Boolean: has volume?
- total_numbers_in_title    # Sum of all numbers mentioned
- total_quantity            # ipq × multipack_size
- log_total_quantity        # Log transform
- log_weight, log_volume    # Log transforms
```

#### 3. Brand Features (5 features) - NO PRICE INFO!
```python
- brand_frequency           # Count of brand occurrences
- brand_tier               # Brand cluster (by frequency, NOT price)
- is_popular_brand         # Top 25% by frequency
- brand_name_length        # Length of brand name
- brand_in_title           # Boolean: brand name in title?
```

#### 4. Category Features (3 features) - NO PRICE INFO!
```python
- category_size            # Number of products in category
- is_large_category        # Boolean: large category?
- category_in_title        # Boolean: category keyword in title?
```

**Categories Detected:**
- supplement, beverage, snack, condiment
- personal_care, grain, canned, frozen, dairy, other

#### 5. Interaction Features (5 features)
```python
- brand_category_frequency      # Brand × category co-occurrence
- quantity_category_ratio       # total_quantity / category_size
- tier_multipack_interaction    # brand_tier × has_multipack
- brand_tier_quantity           # brand_tier × log(total_quantity)
- weight_brand_interaction      # log_weight × brand_tier
```

#### 6. Quality Indicators (7 features)
```python
- premium_keyword_count    # "premium", "organic", "natural", etc.
- has_premium_keywords     # Boolean
- value_keyword_count      # "value", "economy", "bulk", etc.
- quality_keyword_count    # "quality", "fresh", "authentic", etc.
- premium_score            # premium + brand_tier + quality
- value_score              # value + has_multipack
```

#### 7. Size Categories (4 features)
```python
- is_travel_size          # "mini", "travel", "pocket"
- is_family_size          # "family", "jumbo", "large"
- is_multipack            # Detected multipack
- is_standard_size        # Not travel/family/multipack
```

#### 8. Advanced Text Patterns (6 features)
```python
- special_char_count      # Count of special characters
- has_parentheses         # Boolean: contains ()
- has_ampersand           # Boolean: contains &
- title_complexity        # Avg word length in title
- numeric_density         # Numbers per word ratio
```

### V4 Safe Target Encoding (49 Features)

**Added Features (K-Fold CV-based):**
```python
- brand_target_enc        # Safe brand→price encoding (CV-based)
- category_target_enc     # Safe category→price encoding (CV-based)
```

**Key Innovation:**
- Uses 5-fold cross-validation
- Each fold calculates brand/category averages using ONLY training folds
- Prevents leakage while capturing price patterns
- Formula: `(n * mean + smoothing * global_mean) / (n + smoothing)`

---

## 🔍 Target Leakage Investigation

### Problem Discovery

**Suspicious Sign:** V2 predictions were extremely low ($0.01-$87.74) despite training range $0.13-$2,796

**Root Cause Analysis:**

#### Leakage Source 1: Baseline `brand_target`
```python
# From try2/preparation/data_preparation.ipynb (lines 147-152)
# ⚠️ WRONG: Done BEFORE train/test split!
target_mean = df_encoded.groupby('brand')['price'].mean()  
df_encoded['brand_target'] = df_encoded['brand'].map(target_mean)
```

**Impact:**
- Feature used average prices from ALL 75K samples (including validation & test)
- Model learned: "If brand='Coca-Cola', predict ~$12.50"
- Not machine learning, just memorization!
- Feature importance: 0.77 (highest correlation with target)

#### Leakage Source 2: V2 `estimated_unit_price`
```python
# From try2/advanced_feature_engineering.py (line 270)
# ⚠️ WRONG: Uses target price directly!
df['estimated_unit_price'] = df['price'] / df['total_quantity']
```

**Impact:**
- Training: Feature = actual_price / quantity (perfect information!)
- Test: Feature = 0 (no price available)
- Model relied heavily on this feature, failed on test data

### Evidence

**Feature Importance (from leaky baseline):**
```
brand_target: 0.7710       ← HIGHEST! (direct price encoding)
brand_target_log: 0.7632   ← SECOND! (log of prices)
```

**Feature Importance (from V2 with leakage):**
```
estimated_unit_price: 182,956  ← HIGHEST! (uses target)
has_weight: 53,658
number_count: 49,905
```

**Feature Importance (from V3 clean):**
```
total_numbers_in_title: 23,518  ← Text pattern
special_char_count: 17,012
has_weight: 16,786              ← Legitimate feature
```

### Performance Impact

| Version | SMAPE | Leakage? | Valid? |
|---------|-------|----------|--------|
| Baseline | 47.52% | ✅ YES | ❌ INVALID |
| V2 | 3.73% | ✅ YES | ❌ INVALID |
| V3 Clean | 63.28% | ❌ NO | ✅ VALID |

**Gap Analysis:**
- 47.52% → 63.28% = **15.76% "cost of honesty"**
- This gap represents the unfair advantage from leakage

---

## 🚀 Next Steps & Roadmap

### Immediate Next Steps

#### 1. Train V4 Models with Safe Target Encoding ⭐⭐⭐
**Status:** Features created, ready to train  
**Expected:** 55-58% SMAPE (5-10% improvement)  
**Files:**
- Features: `try2/preparation/features_v4_safe_target_train.csv`
- TODO: Create `train_v4_safe_target.py`

#### 2. Fine-Tune Language Model ⭐⭐⭐ (NEW REQUEST!)
**Task:** Fine-tune DistilBERT/MPNet on product text + price  
**Approach:**
- Use regression head on top of transformer
- Train on catalog_content → price directly
- May outperform generic embeddings (all-MiniLM-L6-v2: 69.90%)

**Implementation Plan:**
```python
# Pseudo-code
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',  # or 'microsoft/mpnet-base'
    num_labels=1  # Regression
)

# Fine-tune on:
# Input: catalog_content (title + description)
# Output: price (continuous)
# Loss: MSE or MAE
```

**Advantages:**
- Task-specific: Learns price prediction directly
- Better than generic embeddings
- Can combine with hand-crafted features

**TODO:**
- Create `finetune_transformer.py`
- Implement custom regression head
- Train/validate on same 60/15/25 split

#### 3. Image Features ⭐⭐
**Status:** Not started  
**Approach:**
- Extract ResNet/EfficientNet embeddings from `image_link`
- Combine with text features
- Expected: 10-15% improvement

#### 4. Ensemble Methods ⭐
**Combine:**
- XGBoost (63.28% clean)
- Fine-tuned transformer (TBD)
- With image features (TBD)

### Performance Targets

| Milestone | Target SMAPE | Features |
|-----------|--------------|----------|
| ✅ V3 Clean | 63.28% | 47 hand-crafted (NO leakage) |
| 🔄 V4 Safe Target | <58% | + Safe target encoding |
| 📋 V5 Fine-tuned LLM | <55% | + Fine-tuned transformer |
| 📋 V6 Image | <50% | + Image embeddings |
| 🎯 Final Goal | <45% | Ensemble of all |

---

## 💻 Code Files Reference

### Feature Engineering Scripts

#### ⚠️ `try2/advanced_feature_engineering.py` (DO NOT USE)
**Status:** Has target leakage  
**Issue:** `estimated_unit_price = price / total_quantity`  
**Output:** `features_v2_train.csv`, `features_v2_test.csv`

#### ✅ `try2/advanced_feature_engineering_clean.py` (USE THIS)
**Status:** Clean, no leakage  
**Features:** 47 hand-crafted features  
**Output:** `features_v3_clean_train.csv`, `features_v3_clean_test.csv`  
**Key Methods:**
- `_parse_catalog_content()`: Extract title, description, IPQ, brand
- `_add_basic_text_features()`: Length, word count, capitals
- `_add_advanced_quantity_features()`: Multipack, weight, volume
- `_add_brand_features()`: Frequency-based (NO price info)
- `_add_category_features()`: Category detection from keywords

#### ✅ `try2/safe_target_encoding.py` (USE THIS)
**Status:** Safe, CV-based encoding  
**Features:** +2 target encodings (brand, category)  
**Output:** `features_v4_safe_target_train.csv`, `features_v4_safe_target_test.csv`  
**Key Class:**
- `SafeTargetEncoder`: K-Fold CV-based target encoding
- `fit_transform()`: Creates encodings using only train folds
- `_calculate_encoding()`: Smoothing formula for rare categories

### Model Training Scripts

#### ⚠️ `try2/train_v2_features.py` (DO NOT USE)
**Status:** Trains on leaked features  
**Results:** 3.73% SMAPE (invalid)

#### ✅ `try2/train_v3_clean.py` (REFERENCE)
**Status:** Clean training (85/15 split)  
**Results:** 63.56% SMAPE  
**Split:** 85% train, 15% validation

#### ✅ `try2/train_v3_fair_comparison.py` (USE THIS)
**Status:** Clean training (60/15/25 split)  
**Results:** **63.28% SMAPE** (best valid)  
**Split:** 60% train, 15% val, 25% test (same as baseline for fair comparison)  
**Models:** LightGBM, XGBoost, CatBoost

### Embedding Scripts

#### `text_embeddings.py`
**Model:** all-MiniLM-L6-v2 (sentence-transformers)  
**Dimensions:** 384 → 128 (PCA)  
**Results:** 69.90% SMAPE (embeddings only)  
**Output:** `embeddings_data/train_embeddings.csv`, `test_embeddings.csv`

#### `train_with_embeddings.py`
**Approach:** Train with embeddings ± existing features  
**Results:**
- Embeddings only: 69.90% SMAPE
- Embeddings + features: 55.63% SMAPE (test merge failed)

---

## 📝 Important Files to Read

### 1. 🔍 `try2/LEAKAGE_INVESTIGATION_REPORT.md`
**CRITICAL:** Full documentation of target leakage discovery  
**Contents:**
- How leakage was created
- Feature importance evidence
- Performance impact analysis
- Recommendations for future work

### 2. `try2/preparation/data_preparation.ipynb`
**Contains:** Baseline feature engineering (with leakage)  
**Key Section:** Lines 145-152 (target encoding BEFORE split)

### 3. `try2/modeling/improved_modeling.ipynb`
**Contains:** Baseline model training (47.52% SMAPE)  
**Status:** Results are invalid due to leakage

---

## 🛠️ Development Environment

### Python Packages Installed
```
pandas
numpy
scikit-learn
lightgbm==4.6.0
xgboost==3.0.5
catboost==1.2.8
sentence-transformers==5.1.1
transformers
torch==2.8.0
optuna  # For hyperparameter tuning
matplotlib
seaborn
scipy
```

### Virtual Environment
**Location:** `venv/`  
**Python Version:** 3.11.0  
**Activation:** 
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 📊 Data Schema

### Training Data (`dataset/train1.csv`, `train2.csv`)
```csv
sample_id,catalog_content,image_link,price
33127,"Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)
Value: 72.0
Unit: Fl Oz
Item Pack Quantity: 1
...",https://...,4.89
```

**Columns:**
- `sample_id`: Unique identifier (int)
- `catalog_content`: Multi-line text with Item Name, Value, Unit, etc.
- `image_link`: URL to product image
- `price`: Target variable (float, $0.13 - $2,796.00)

### Test Data (`dataset/test1.csv`, `test2.csv`)
Same schema as training, but **NO `price` column**.

### Feature Files Schema

#### Clean Features V3 (47 features)
**File:** `preparation/features_v3_clean_train.csv`

```csv
sample_id,price,ipq,title_length,title_word_count,desc_length,desc_word_count,
has_description,capital_ratio,number_count,multipack_size,has_multipack,
weight_value,has_weight,volume_value,has_volume,total_numbers_in_title,
total_quantity,log_total_quantity,log_weight,log_volume,brand_frequency,
brand_tier,is_popular_brand,brand_name_length,brand_in_title,category_size,
is_large_category,brand_category_frequency,quantity_category_ratio,
tier_multipack_interaction,brand_tier_quantity,weight_brand_interaction,
premium_keyword_count,has_premium_keywords,value_keyword_count,
quality_keyword_count,premium_score,value_score,is_travel_size,is_family_size,
is_multipack,is_standard_size,special_char_count,has_parentheses,has_ampersand,
title_complexity,numeric_density,category_in_title
```

---

## 🎯 Key Learnings & Best Practices

### ✅ DO's

1. **Always split data BEFORE feature engineering**
   - Calculate statistics only from training data
   - Use cross-validation for target encoding

2. **Validate features for leakage**
   - Check if feature uses target information
   - Test on truly unseen data

3. **Use cross-validation for target encoding**
   - Prevents overfitting
   - More generalizable than global encoding

4. **Document feature creation process**
   - Track which features have leakage risk
   - Maintain clean baseline

5. **Compare predictions to training distribution**
   - Suspiciously different ranges indicate problems
   - Test predictions should be similar to training

### ❌ DON'Ts

1. **Never use target variable in feature engineering**
   - No `price` in calculations
   - No `groupby(...)[target].mean()` before split

2. **Never calculate statistics from full dataset before split**
   - Brand averages must be from training only
   - Use CV for any target-related encoding

3. **Never trust too-good-to-be-true results**
   - 3.73% SMAPE was suspicious → Led to leakage discovery
   - Always investigate unusually good performance

4. **Never deploy models with leakage**
   - Production data won't have target information
   - Model will fail spectacularly

---

## 📖 How to Continue This Project

### Option 1: Train V4 Models (Immediate)
```powershell
cd try2
python train_v4_safe_target.py  # TODO: Create this file
```
**Expected Outcome:** 55-58% SMAPE

### Option 2: Fine-Tune Transformer (NEW!)
```powershell
cd try2
python finetune_transformer.py  # TODO: Create this file
```
**Models to Try:**
- `distilbert-base-uncased` (smaller, faster)
- `microsoft/mpnet-base` (better quality)
- `roberta-base` (robust alternative)

**Expected Outcome:** 50-60% SMAPE (combined with hand-crafted features)

### Option 3: Add Image Features
```powershell
cd try2
python extract_image_features.py  # TODO: Create this file
python train_v5_with_images.py    # TODO: Create this file
```
**Expected Outcome:** <50% SMAPE

### Option 4: Ensemble Everything
Combine:
- Clean features (63.28%)
- Safe target encoding (TBD)
- Fine-tuned transformer (TBD)
- Image features (TBD)

**Expected Outcome:** <45% SMAPE (goal!)

---

## 🔬 Experimental Results Summary

| Experiment | SMAPE | Valid? | Notes |
|------------|-------|--------|-------|
| Baseline (with leakage) | 47.52% | ❌ | `brand_target` leakage |
| Sentence embeddings only | 69.90% | ✅ | all-MiniLM-L6-v2, worse than features |
| Embeddings + features | 55.63% | ❌ | Test merge failed |
| V2 Advanced features | 3.73% | ❌ | `estimated_unit_price` leakage |
| **V3 Clean features** | **63.28%** | ✅ | **Current best valid** |
| V4 Safe target encoding | TBD | ✅ | Features ready, not trained yet |
| V5 Fine-tuned LLM | TBD | ✅ | TODO |
| V6 With images | TBD | ✅ | TODO |

---

## 📞 Quick Reference Commands

### Run Feature Engineering
```powershell
cd try2
python advanced_feature_engineering_clean.py  # V3 clean
python safe_target_encoding.py                # V4 safe target enc
```

### Train Models
```powershell
cd try2
python train_v3_fair_comparison.py  # Current best (63.28%)
```

### Check Files
```powershell
# Training features
head preparation/features_v3_clean_train.csv

# Test features  
head preparation/features_v3_clean_test.csv

# Predictions
head modeling/test_out_v3_fair.csv
```

---

## 🎓 Technical Concepts Explained

### SMAPE (Symmetric Mean Absolute Percentage Error)
```python
SMAPE = (100 / n) * Σ(2 * |pred - actual| / (|pred| + |actual|))
```
- Lower is better (0% = perfect)
- Symmetric: Treats over/under-prediction equally
- Handles zero values better than MAPE

### Target Encoding
**Unsafe (Leaky):**
```python
brand_mean = data.groupby('brand')['price'].mean()  # Uses ALL data!
```

**Safe (CV-based):**
```python
for train_idx, val_idx in kfold.split(data):
    brand_mean = data[train_idx].groupby('brand')['price'].mean()
    data[val_idx]['brand_enc'] = data[val_idx]['brand'].map(brand_mean)
```

### Feature Leakage
**Definition:** When features contain information from the target that wouldn't be available at prediction time.

**Examples:**
- ❌ `estimated_unit_price = price / quantity` (uses target!)
- ❌ `brand_avg_price = groupby('brand')['price'].mean()` (before split)
- ✅ `brand_frequency = brand.value_counts()` (no price info)

---

## 📚 References & Resources

### Papers & Techniques Used
1. Target Encoding with Cross-Validation
2. SMAPE metric for price prediction
3. Sentence transformers for text embedding
4. Feature importance analysis

### Useful Links
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Target Encoding Best Practices](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)

---

## ✍️ Author Notes

**Current Status:** Solid foundation established with clean features (63.28% SMAPE)

**Next Priority:** Fine-tune transformer on product text + price regression

**Final Goal:** <45% SMAPE through ensemble of:
1. Clean hand-crafted features
2. Safe target encoding
3. Fine-tuned language model
4. Image features

**Key Insight:** The 47.52% baseline was artificially good due to target leakage. Our 63.28% represents honest, deployable performance.

---

**End of Documentation**  
Last Updated: October 12, 2025  
Project Status: Active Development  
Best Valid SMAPE: 63.28% (Clean V3)
