# Implementation Guide - Smart Product Pricing Challenge

**Goal:** Improve SMAPE from 53.636% to <42% (top 3 competitive)

**Status:** Phase 1 implementation ready, awaiting execution

---

## 📊 Current Situation

- **Baseline Model:** DistilBERT text-only
- **Current SMAPE:** 53.636%
- **Target SMAPE:** <42%
- **Gap to Close:** ~12 SMAPE points
- **Competition Top 3:** 41.28% - 42.31%

---

## 🎯 Implementation Strategy

Based on comprehensive research (see `../research/` folder), we've identified 6 major improvements that can close the gap:

| Improvement | Expected Gain | Priority |
|------------|--------------|----------|
| 1. Log Transform Ensemble | -4 to -6 pts | 🔴 CRITICAL |
| 2. Unit Standardization | -2 to -3 pts | 🔴 CRITICAL |
| 3. Advanced Features | -2 to -3 pts | 🟡 HIGH |
| 4. Stratified Models | -3 to -4 pts | 🟡 HIGH |
| 5. Feature Interactions | -1 to -2 pts | 🟢 MEDIUM |
| 6. Image Integration | -2 to -3 pts | 🟢 MEDIUM |

**Total Potential:** -13 to -16 SMAPE points → **40-41% final SMAPE** ✨

---

## 📁 Folder Structure

```
implementation/
├── README.md                           # This file
├── IMPLEMENTATION_ROADMAP.md           # Detailed 12-day plan
├── status_tracker.py                   # Track progress
├── status.json                         # Current status (auto-generated)
│
├── phase1_quick_wins/                  # Days 1-4
│   ├── run_phase1.py                   # Master runner
│   ├── 01_log_transform_ensemble.py    # Log/sqrt/boxcox transforms
│   ├── 02_unit_standardization.py      # Unit normalization
│   └── 03_advanced_features.py         # Brand, premium, text features
│
├── phase2_stratified/                  # Days 5-8 (NOT YET CREATED)
│   ├── 01_range_classification.py
│   ├── 02_range_specific_models.py
│   └── 03_feature_interactions.py
│
└── phase3_advanced/                    # Days 9-12 (NOT YET CREATED)
    ├── 01_image_integration.py
    ├── 02_multimodal_ensemble.py
    └── 03_final_tuning.py
```

---

## 🚀 Quick Start

### Option 1: Run Full Phase 1 (Recommended)

```bash
cd phase1_quick_wins
python run_phase1.py
```

This will run all 3 Phase 1 scripts in sequence:
- 1.1 Log Transform Ensemble (~4-6 hours)
- 1.2 Unit Standardization (~2-3 hours)
- 1.3 Advanced Features (~3-4 hours)

### Option 2: Run Individual Scripts

```bash
cd phase1_quick_wins

# Step 1: Log Transform Ensemble
python 01_log_transform_ensemble.py

# Step 2: Unit Standardization
python 02_unit_standardization.py

# Step 3: Advanced Features
python 03_advanced_features.py
```

### Option 3: Track Progress

```bash
# Initialize status tracker
python status_tracker.py init

# View current status
python status_tracker.py

# Update status after completing a step
python status_tracker.py update phase1_quick_wins 1.1_log_transform COMPLETED 49.5
```

---

## 📋 Phase 1: Quick Wins (Days 1-4)

### 1.1 Log Transform Ensemble

**Script:** `phase1_quick_wins/01_log_transform_ensemble.py`

**What it does:**
- Trains 3 DistilBERT models on different target transformations:
  - Log transform: `log(price + 1)`
  - Square root: `sqrt(price)`
  - Box-Cox: Yeo-Johnson transformation
- Ensembles predictions with optimized weights
- Handles extreme price skewness (skewness=13.60)

**Expected improvement:** 53.6% → 49-50% SMAPE (-4 to -6 points)

**Time:** 4-6 hours (with GPU)

**Output files:**
- `outputs/phase1_log_transform/training_template.py`
- `outputs/phase1_log_transform/ensemble_strategy.py`
- `outputs/phase1_log_transform/create_submission.py`

**Note:** Requires GPU for training. Script provides framework; actual training needs GPU environment.

---

### 1.2 Unit Standardization

**Script:** `phase1_quick_wins/02_unit_standardization.py`

**What it does:**
- Standardizes 156 unit variations (oz/Oz/ounce → oz)
- Extracts nested quantities (e.g., "24 per pack × 6 cases" = 144)
- Calculates per-unit prices for bulk items
- Creates unit category features (weight/volume/count)

**Research finding:** Fixes errors like $691 actual vs $30 predicted (missing "per case")

**Expected improvement:** 49-50% → 47-48% SMAPE (-2 to -3 points)

**Time:** 2-3 hours

**Output files:**
- `outputs/phase1_unit_standardization/train_with_units.csv`
- `outputs/phase1_unit_standardization/test_with_units.csv`
- `outputs/phase1_unit_standardization/unit_mappings.json`
- `outputs/phase1_unit_standardization/integration_guide.txt`

**New features added:**
- `qty` - Primary quantity (e.g., 16 from "16 oz")
- `unit` - Standardized unit (e.g., "oz")
- `total_qty` - Nested quantity (e.g., 144 from "24 × 6")
- `multiplier` - Bulk factor (total_qty / qty)
- `price_per_unit` - Price / multiplier
- `unit_category` - Broader category (weight/volume/count)

---

### 1.3 Advanced Feature Engineering

**Script:** `phase1_quick_wins/03_advanced_features.py`

**What it does:**
- Extracts brand names and creates brand tiers (budget/mid/premium/luxury)
- Identifies premium signals (organic, premium, luxury keywords)
- Identifies budget signals (value, economy, basic keywords)
- Calculates text complexity metrics (word count, vocabulary richness)
- Infers product categories (electronics, food, health, etc.)
- Creates interaction features (qty × premium, brand × category)

**Research findings:**
- Top brands have 30% lower price variance
- Premium keywords add 40% to price on average
- Text complexity correlates with price (r=0.31)
- Electronics 2x higher price than food/beverage

**Expected improvement:** 47-48% → 45-46% SMAPE (-2 to -3 points)

**Time:** 3-4 hours

**Output files:**
- `outputs/phase1_advanced_features/train_with_advanced_features.csv`
- `outputs/phase1_advanced_features/test_with_advanced_features.csv`
- `outputs/phase1_advanced_features/feature_info.json`
- `outputs/phase1_advanced_features/integration_guide.txt`

**New features added:** 21 features across 5 categories
- Brand: `brand`, `brand_tier`
- Premium/Budget: `premium_count`, `budget_count`, `premium_signal`
- Text: `text_char_count`, `text_word_count`, `text_unique_word_ratio`, etc.
- Category: `category`
- Interactions: `qty_premium_interaction`, `brand_category`, `is_bulk`, etc.

---

## 🎯 Expected Results After Phase 1

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SMAPE | 53.636% | 45-46% | -7 to -9 points |
| Gap to Top 3 | 12 pts | 4 pts | 66% closed |

**If Phase 1 achieves 45-46% SMAPE:** Phase 2 (Stratified Models) can get us to ~42-44%

**If Phase 1 achieves only 47-48% SMAPE:** Phase 2 and 3 both needed to reach <42%

---

## 📝 Integration Approaches

After Phase 1 completes, you have 3 options for model training:

### Option 1: Feature-Enriched Text (Simple)
Add feature info to text before feeding to DistilBERT:
```python
text = "Product description [Brand: premium] [Category: electronics] [Bulk: 24x]"
```

### Option 2: Two-Stage Model (Recommended)
1. DistilBERT predicts from text → text_pred
2. LightGBM predicts from text_pred + all features → final_pred

**Pros:** Combines deep learning + gradient boosting strengths

### Option 3: Custom Architecture (Advanced)
Modify DistilBERT to accept both text embeddings and numerical features

---

## 🔄 Workflow

```
1. Run Phase 1 scripts
   ↓
2. Validate results on holdout set
   ↓
3. If SMAPE > 46%, proceed to Phase 2
   ↓
4. If SMAPE 42-46%, generate submission & evaluate
   ↓
5. If SMAPE < 42%, SUCCESS! Generate final submission
```

---

## 📊 Validation Strategy

After each phase, validate on stratified holdout set:

```python
from sklearn.model_selection import StratifiedKFold

# Create price bins
bins = [0, 10, 20, 30, 50, 100, float('inf')]
train['price_bin'] = pd.cut(train['price'], bins=bins)

# Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['price_bin'])):
    # Train and validate
    fold_smape = validate(model, train.iloc[val_idx])
    print(f"Fold {fold}: {fold_smape:.2f}%")
```

---

## 🚨 Important Notes

### GPU Requirements

Phase 1.1 (Log Transform Ensemble) requires GPU for DistilBERT training:
- **Minimum:** NVIDIA GPU with 8GB VRAM
- **Recommended:** 16GB+ VRAM
- **Alternative:** Use cloud GPU (Google Colab Pro, AWS SageMaker)

If no GPU available:
- Skip to Phase 1.2 and 1.3 (feature engineering)
- Use pre-trained DistilBERT from existing model
- Focus on improving features rather than model architecture

### Data Location

Scripts assume data is in `../../try2/dataset/`:
- `train1.csv` and `train2.csv` (75,000 total samples)
- `test1.csv` and `test2.csv` (75,000 total samples)

If data is elsewhere, update the `CONFIG['data_dir']` in each script.

### Time Estimates

All time estimates assume:
- GPU available for Phase 1.1
- CPU: 8+ cores for parallel processing
- RAM: 16GB+ for large datasets
- No major debugging needed

Add 20-50% buffer time for unexpected issues.

---

## 📚 Additional Resources

| Document | Purpose |
|----------|---------|
| `IMPLEMENTATION_ROADMAP.md` | Detailed 12-day plan with code examples |
| `../research/RESEARCH_STATUS.md` | Research findings and visualizations |
| `../research/outputs/` | Research charts and analysis CSVs |
| `../README.md` | Try3 folder overview and strategy |

---

## 🆘 Troubleshooting

### Issue: "No GPU available"
**Solution:** 
- Use Google Colab Pro (free GPU)
- Or skip Phase 1.1 and focus on features (Phases 1.2-1.3)

### Issue: "Out of memory"
**Solution:**
- Reduce batch size in Phase 1.1 (try `batch_size=8`)
- Use gradient accumulation
- Split training into smaller chunks

### Issue: "Unit extraction not working"
**Solution:**
- Check if catalog_content has expected format
- Run sample extraction: `extract_quantity_and_unit("16 Ounce")`
- Review unit_mappings.json for missing variations

### Issue: "SMAPE not improving"
**Solution:**
- Validate on holdout set, not training set
- Check for data leakage
- Ensure inverse transforms are correct
- Review error analysis by price range

---

## ✅ Success Criteria

**Phase 1 Success:**
- All 3 scripts execute without errors
- Features extracted for >90% of samples
- Validation SMAPE ≤ 46%
- Ready to proceed to Phase 2 or submission

**Overall Success:**
- Final SMAPE < 42%
- Submission ranks in top 3
- Model generalizes well (train/val gap < 2%)

---

## 🎉 Next Steps

1. **Now:** Run Phase 1 scripts
2. **After Phase 1:** Validate results, check SMAPE
3. **If SMAPE > 46%:** Implement Phase 2 (Stratified Models)
4. **If SMAPE 42-46%:** Generate submission, evaluate
5. **If SMAPE < 42%:** Success! Optimize further if time permits

---

**Questions? Check:**
- Research findings: `../research/outputs/research_summary.txt`
- Detailed plan: `IMPLEMENTATION_ROADMAP.md`
- Quick reference: `../QUICK_START.md`

**Ready to improve from 53.6% to 45%? Let's go! 🚀**
