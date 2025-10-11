# 🚀 Quick Start Guide - Amazon ML Hackathon

## Current Project State

**Best Valid Model:** 63.28% SMAPE (Clean V3 features)  
**Status:** ✅ Target leakage discovered & fixed  
**Next Steps:** Fine-tune transformer OR train V4 safe target encoding

---

## 📂 Important Files

### ✅ USE THESE (No Leakage)
```
try2/advanced_feature_engineering_clean.py    → Creates V3 features (47 features)
try2/safe_target_encoding.py                  → Creates V4 features (+2 target enc)
try2/train_v3_fair_comparison.py              → Training script (63.28% SMAPE)

try2/preparation/features_v3_clean_train.csv  → Clean training features
try2/preparation/features_v3_clean_test.csv   → Clean test features
try2/preparation/features_v4_safe_target_*.csv → With safe target encoding

try2/modeling/test_out_v3_fair.csv            → Valid predictions
try2/LEAKAGE_INVESTIGATION_REPORT.md          → Detailed leakage analysis
```

### ⚠️ DO NOT USE (Has Leakage)
```
try2/advanced_feature_engineering.py          → V2 features (estimated_unit_price leakage)
try2/train_v2_features.py                     → Training with leaked features
try2/preparation/features_clean.csv           → Baseline (brand_target leakage)
try2/modeling/improved_modeling.ipynb         → Baseline notebook (47.52% invalid)
```

---

## 🎯 Quick Commands

### Train Current Best Model
```powershell
cd try2
C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\venv\Scripts\python.exe train_v3_fair_comparison.py
```
**Output:** 63.28% SMAPE

### Create V4 Features (Safe Target Encoding)
```powershell
cd try2
C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\venv\Scripts\python.exe safe_target_encoding.py
```
**Output:** `features_v4_safe_target_train.csv`, `features_v4_safe_target_test.csv`

### Recreate V3 Clean Features
```powershell
cd try2
C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\venv\Scripts\python.exe advanced_feature_engineering_clean.py
```
**Output:** `features_v3_clean_train.csv`, `features_v3_clean_test.csv`

---

## 📊 Performance Comparison

| Version | SMAPE | Leakage? | Use? |
|---------|-------|----------|------|
| Baseline | 47.52% | ✅ YES | ❌ NO |
| V2 | 3.73% | ✅ YES | ❌ NO |
| **V3 Clean** | **63.28%** | ❌ NO | ✅ YES |
| V4 Safe Target | TBD | ❌ NO | ✅ YES |

---

## 🔥 Next Steps (Choose One)

### Option 1: Train V4 with Safe Target Encoding (Fastest)
**Expected:** 55-58% SMAPE  
**Time:** ~10 minutes  
**TODO:** Create `train_v4_safe_target.py` (copy from `train_v3_fair_comparison.py`)

### Option 2: Fine-Tune Transformer (Your Request!)
**Expected:** 50-60% SMAPE  
**Time:** ~2-3 hours  
**TODO:** Create `finetune_transformer.py`

**Models to try:**
- `distilbert-base-uncased` (fast)
- `microsoft/mpnet-base` (better)
- `roberta-base` (alternative)

### Option 3: Add Image Features
**Expected:** <50% SMAPE  
**Time:** ~1-2 hours  
**TODO:** Create `extract_image_features.py`

---

## 🔍 Key Findings

### Target Leakage Discovered!

**Baseline (47.52%):**
```python
# ⚠️ WRONG! Uses future data
brand_target = df.groupby('brand')['price'].mean()  # Before split!
```

**V2 (3.73%):**
```python
# ⚠️ WRONG! Uses target directly
estimated_unit_price = price / total_quantity
```

**V3 Clean (63.28%):**
```python
# ✅ CORRECT! No price information
brand_frequency = brand.value_counts()  # Frequency only
```

**Gap:** 47.52% → 63.28% = 15.76% "cost of honesty"

---

## 💡 Feature Engineering Cheat Sheet

### V3 Clean Features (47 features)
- **Text:** length, word count, capitals, numbers
- **Quantity:** multipack, weight, volume (extracted from text)
- **Brand:** frequency, tier (by count NOT price)
- **Category:** detected from keywords (10 categories)
- **Interactions:** brand × quantity, tier × multipack
- **Quality:** premium/value/quality keywords

### V4 Safe Target Encoding (+2 features)
- **brand_target_enc:** CV-based brand→price (safe!)
- **category_target_enc:** CV-based category→price (safe!)

---

## 🚨 Common Mistakes to Avoid

1. ❌ Using `price` in feature calculations
2. ❌ Calculating statistics before train/test split
3. ❌ Using baseline features (`features_clean.csv`)
4. ❌ Trusting V2 results (3.73% too good to be true!)
5. ✅ Always validate: Do predictions match training distribution?

---

## 📖 Full Documentation

See `PROJECT_DOCUMENTATION.md` for:
- Complete file structure
- Detailed feature descriptions
- Target leakage analysis
- Performance timeline
- Technical concepts
- How to continue

---

## 🎓 Terminology

**SMAPE:** Symmetric Mean Absolute Percentage Error (lower is better)  
**Target Leakage:** Features that contain target information  
**CV:** Cross-Validation  
**V3:** Version 3 (clean features)  
**V4:** Version 4 (V3 + safe target encoding)

---

## 📞 Quick Help

**Problem:** Features have leakage  
**Solution:** Use `advanced_feature_engineering_clean.py`

**Problem:** Model training fails  
**Solution:** Check file paths, activate venv

**Problem:** Predictions are weird  
**Solution:** Check if using clean features (V3/V4)

**Problem:** Want to improve performance  
**Solution:** Try V4 → Fine-tune transformer → Add images

---

**Current Best:** 63.28% SMAPE (XGBoost, V3 Clean)  
**Goal:** <45% SMAPE without leakage  
**Status:** Ready for next iteration! 🚀
