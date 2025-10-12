# 🚀 Implementation Roadmap: 53.6% → 41% SMAPE

**Current Score**: 53.636% (DistilBERT text-only)  
**Target Score**: 40-42% (Top 3 competitive)  
**Gap to Close**: ~12 SMAPE points

---

## 📊 Research-Backed Improvement Plan

Based on comprehensive research of 75,000 samples:

| Improvement | Expected Gain | Difficulty | Priority |
|-------------|---------------|------------|----------|
| 1. Log Transform Ensemble | -6 to -8 pts | Low | 🔥 CRITICAL |
| 2. Unit Standardization | -3 to -4 pts | Low | 🔥 CRITICAL |
| 3. Stratified Models | -4 to -6 pts | Medium | ⭐ HIGH |
| 4. Feature Interactions | -2 to -3 pts | Medium | ⭐ HIGH |
| 5. Multi-Modal Fusion | -2 to -3 pts | High | 💡 NICE-TO-HAVE |
| 6. Ensemble & Tuning | -2 to -3 pts | Medium | 💡 FINAL |

---

## 🎯 Phase 1: Quick Wins (Days 1-4) → 53.6% to 46-48%

### ✅ Step 1.1: Log Target Transformation (Day 1)
**Expected**: 53.6% → 49-50%  
**Time**: 4-6 hours  
**Files**: `phase1_quick_wins/01_log_transform_ensemble.py`

**Implementation**:
```python
# Train 3 DistilBERT models:
1. Predict log(price + 1)
2. Predict sqrt(price)  
3. Predict raw price

# Ensemble with weights:
final = 0.5 * exp(pred_log) + 0.3 * (pred_sqrt)^2 + 0.2 * pred_raw
```

**Why This Works**:
- Price distribution is HIGHLY skewed (skewness=13.60, kurtosis=736!)
- Log transform makes distribution nearly normal
- SMAPE penalizes relative errors - log space handles better
- Q-Q plot shows clear log-normal pattern

**Validation Strategy**:
- Use stratified K-fold by price bins
- Ensure all bins represented in each fold

---

### ✅ Step 1.2: Unit Standardization (Day 2)
**Expected**: 49-50% → 47-48%  
**Time**: 3-4 hours  
**Files**: `phase1_quick_wins/02_unit_standardization.py`

**Problem Identified**:
```
"Oz" (85.3% SMAPE) ≠ "oz" (82.5%) ≠ "Ounce" (77.7%) ≠ "ounce" (75.1%)
But they're ALL the SAME UNIT!
```

**Implementation**:
```python
unit_mapping = {
    # Ounce variations
    'oz': 'ounce', 'Oz': 'ounce', 'OZ': 'ounce',
    'ounce': 'ounce', 'Ounce': 'ounce',
    
    # Fluid ounce variations
    'fl oz': 'fluid_ounce', 'Fl Oz': 'fluid_ounce',
    'FL Oz': 'fluid_ounce', 'Fluid Ounce': 'fluid_ounce',
    
    # Count variations
    'count': 'count', 'Count': 'count', 'COUNT': 'count',
    'ct': 'count',
    
    # Pound variations
    'pound': 'pound', 'Pound': 'pound', 'lb': 'pound',
    
    # Add None handling
    None: 'unit_missing', 'None': 'unit_missing'
}
```

**Also Extract Nested Quantities**:
```python
# Current: "18 count per pack -- 16 per case" → value=16
# Should be: 18 * 16 = 288 total items

def extract_total_quantity(text):
    # Pattern 1: "X per pack" or "pack of X"
    pack_size = extract_pack_size(text)
    
    # Pattern 2: "Y per case"
    case_size = extract_case_size(text)
    
    # Pattern 3: value from catalog
    base_value = extract_value(text)
    
    return pack_size * case_size * base_value
```

**Impact**: Fixes worst error (691 vs 30 prediction)

---

### ✅ Step 1.3: Advanced Text Features (Day 3-4)
**Expected**: 47-48% → 46-47%  
**Time**: 6-8 hours  
**Files**: `phase1_quick_wins/03_advanced_features.py`

**Research-Backed Features**:

**1. Multi-Pack Intelligence**
```python
# 35.26% of products have pack info
# Most common: 6, 12, 2, 3, 1, 4, 24
features['pack_multiplier'] = {
    1: 1.0,
    2: 1.85,    # Not quite 2x
    3: 2.65,    # Bulk discount
    4: 3.45,
    6: 5.20,    # Common pack size
    12: 9.80,   # Wholesale
    24: 18.50   # Case pricing
}[pack_size]
```

**2. Premium Signals** (Research findings)
```python
# Average price differences:
'is_organic':     +$1.86   (10.66% of products)
'is_premium':    +$12.05   (3.02% of products)
'has_gourmet':    +$8.32
'is_family_size': -$2.73   (cheaper per unit!)
```

**3. Brand Tier Encoding**
```python
# From research: 445 brands with 10+ products
brand_tiers = {
    'Budget':     [list of 86 brands],  # Avg <$10
    'Economy':    [list of 186 brands], # Avg $10-20
    'Mid-Range':  [list of 95 brands],  # Avg $20-30
    'Premium':    [list of 56 brands],  # Avg $30-50
    'Luxury':     [list of 22 brands]   # Avg >$50
}
```

**4. Category-Specific Features**
```python
# Food & Beverage (53% of data) specific:
- Organic certification
- Serving size per container
- Perishable vs shelf-stable
- Liquid vs solid

# Health & Wellness (8% of data):
- Supplement facts
- Dosage information
- Certifications (USDA, Non-GMO)
```

---

## 🎯 Phase 2: Stratified Modeling (Days 5-8) → 46-47% to 42-44%

### ✅ Step 2.1: Price Range Classification (Day 5)
**Expected**: Setup for stratification  
**Files**: `phase2_stratified/01_range_classifier.py`

**Why Needed** (from error analysis):
```
Budget ($0-$10):    122.6% SMAPE  ← 38% of data!
Economy ($10-$20):   46.5% SMAPE
Mid-Range ($20-$30): 17.6% SMAPE  ← Model works here!
Premium ($30-$50):   45.9% SMAPE
High-End ($50-$100): 91.2% SMAPE
Luxury ($100+):     140.8% SMAPE  ← Complete failure
```

**Implementation**:
```python
# Train classifier to predict price range
ranges = ['budget', 'economy', 'mid', 'premium', 'high', 'luxury']
classifier = train_multiclass_classifier(features, ranges)

# Confidence threshold approach:
if confidence < 0.7:
    # Use ensemble of adjacent ranges
    pred = weighted_avg([range_model[i-1], range_model[i], range_model[i+1]])
```

---

### ✅ Step 2.2: Range-Specific DistilBERT Models (Days 6-7)
**Expected**: 46-47% → 43-45%  
**Files**: `phase2_stratified/02_range_specific_models.py`

**Train 6 Separate Models**:
```python
models = {
    'budget': DistilBERT(train[price < 10]),      # 28,650 samples
    'economy': DistilBERT(train[10 ≤ price < 20]), # 19,285 samples
    'mid': DistilBERT(train[20 ≤ price < 30]),     # 9,838 samples
    'premium': DistilBERT(train[30 ≤ price < 50]), # 9,023 samples
    'high': DistilBERT(train[50 ≤ price < 100]),   # 6,249 samples
    'luxury': DistilBERT(train[price ≥ 100])       # 1,955 samples
}
```

**Range-Specific Optimizations**:
```python
# Budget model: Focus on pack size
budget_features += ['pack_size', 'is_multipack', 'total_quantity']

# Luxury model: Focus on brand
luxury_features += ['brand_tier', 'is_premium', 'brand_reputation']

# Mid-range: Balanced
mid_features += ['brand', 'category', 'quantity', 'quality_signals']
```

---

### ✅ Step 2.3: Feature Interactions (Day 8)
**Expected**: 43-45% → 42-43%  
**Files**: `phase2_stratified/03_feature_interactions.py`

**Critical Interactions**:
```python
# 1. Brand × Category
# "Food to Live" in organic vs snacks pricing differs
df['brand_cat'] = brand_encoding * category_encoding

# 2. Brand × Quantity
# Premium brands: linear scaling
# Budget brands: bulk discounts (non-linear)
df['brand_qty'] = brand_tier * log(quantity + 1)

# 3. Unit × Value
# Price per ounce varies by unit type
df['unit_value'] = value / unit_standard_size[unit]

# 4. Category × Quantity Ratio
# Compare to category median
df['qty_ratio'] = quantity / category_median_quantity

# 5. 3-Way Interaction
df['brand_cat_qty'] = (brand_tier * 
                       category_encoding * 
                       log(quantity + 1))
```

---

## 🎯 Phase 3: Advanced Techniques (Days 9-12) → 42-43% to 40-41%

### ✅ Step 3.1: Multi-Modal Fusion (Days 9-10) [OPTIONAL]
**Expected**: 42-43% → 41-42%  
**Files**: `phase3_advanced/01_multimodal_fusion.py`

**Warning**: Images showed WEAK correlation in research  
**Only proceed if time permits**

**Simple Approach** (if needed):
```python
# Extract CNN features
image_features = EfficientNet_B0(images)  # 1280-dim

# Late fusion
text_pred = distilbert_model(text)
image_pred = cnn_regressor(image_features)

# Learned weights
final = alpha * text_pred + beta * image_pred
# Where alpha ≈ 0.85, beta ≈ 0.15 (text dominates)
```

---

### ✅ Step 3.2: Model Ensemble (Days 11-12)
**Expected**: 41-42% → 40-41%  
**Files**: `phase3_advanced/02_ensemble.py`

**Diverse Model Stack**:
```python
# Level 1: Base models
models = [
    DistilBERT_log_transform,
    DistilBERT_sqrt_transform,
    DistilBERT_raw,
    CatBoost(category_aware=True),
    LightGBM(boosting='dart'),
    XGBoost(objective='reg:gamma'),
    TabNet(tabular_features)
]

# Level 2: Meta-learner
meta = Ridge(alpha=1.0)
final_pred = meta.predict(level1_predictions)
```

**Stacking Strategy**:
- Use 5-fold CV to generate meta-features
- Prevent overfitting with regularization
- Weight by validation SMAPE

---

### ✅ Step 3.3: Final Tuning
**Expected**: 40-41% final score  
**Files**: `phase3_advanced/03_final_tuning.py`

**Techniques**:
1. **Hyperparameter Optimization** (Optuna)
2. **Prediction Calibration** (Isotonic regression)
3. **Pseudo-Labeling** (high-confidence test predictions)
4. **Test-Time Augmentation** (for images, if used)

---

## 📈 Expected Progress Timeline

```
Day  | Task                        | Expected SMAPE | Cumulative Gain
-----|----------------------------|----------------|----------------
0    | Current (DistilBERT)        | 53.6%          | Baseline
1    | Log Transform              | 49-50%         | -4 to -5 pts
2    | Unit Standardization       | 47-48%         | -6 to -8 pts
3-4  | Advanced Features          | 46-47%         | -7 to -10 pts
5    | Range Classifier           | 46-47%         | Setup
6-7  | Stratified Models          | 43-45%         | -9 to -13 pts
8    | Feature Interactions       | 42-43%         | -11 to -14 pts
9-10 | Multi-Modal (optional)     | 41-42%         | -12 to -15 pts
11-12| Ensemble & Tuning          | 40-41%         | -13 to -16 pts
```

**Conservative Estimate**: 42-43% (Top 5)  
**Optimistic Estimate**: 40-41% (Top 3)

---

## ⚠️ Critical Success Factors

### ✅ DO:
1. **Validate after each change** - Track SMAPE on holdout set
2. **Use stratified CV** - Ensure all price ranges represented
3. **Version control** - Commit after each improvement
4. **Document learnings** - What worked, what didn't
5. **Check data leakage** - No test data in training

### ❌ DON'T:
1. **Don't skip log transform** - Easiest 5+ point gain
2. **Don't ignore budget category** - 38% of data
3. **Don't treat units inconsistently** - "Oz" ≠ "oz"
4. **Don't over-engineer images** - Weak signal (2-3 pts max)
5. **Don't overfit** - Simple baseline first, then improve

---

## 🎯 Success Metrics

**Minimum Viable Improvement** (Week 1):
- [ ] Log transform working: <50% SMAPE
- [ ] Unit standardization done: <48% SMAPE
- [ ] Advanced features added: <47% SMAPE

**Target Achievement** (Week 2):
- [ ] Stratified models trained: <44% SMAPE
- [ ] Feature interactions working: <43% SMAPE

**Stretch Goal** (Week 3):
- [ ] Ensemble deployed: <41% SMAPE
- [ ] Final tuning complete: ~40-41% SMAPE

---

## 📁 File Structure

```
try3/implementation/
├── IMPLEMENTATION_ROADMAP.md          (This file)
├── phase1_quick_wins/
│   ├── 01_log_transform_ensemble.py
│   ├── 02_unit_standardization.py
│   └── 03_advanced_features.py
├── phase2_stratified/
│   ├── 01_range_classifier.py
│   ├── 02_range_specific_models.py
│   └── 03_feature_interactions.py
├── phase3_advanced/
│   ├── 01_multimodal_fusion.py
│   ├── 02_ensemble.py
│   └── 03_final_tuning.py
└── utils/
    ├── feature_engineering.py
    ├── model_training.py
    └── evaluation.py
```

---

## 🚀 Let's Start!

**Next Step**: Implement Phase 1.1 - Log Transform Ensemble

Run:
```bash
cd try3/implementation/phase1_quick_wins
python 01_log_transform_ensemble.py
```

**Expected Time**: 4-6 hours  
**Expected Result**: 53.6% → 49-50% SMAPE

Let's go! 🎯
