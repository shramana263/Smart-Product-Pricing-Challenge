# 🚀 DistilBERT + Image Integration Plan

## Current Status Summary

### ✅ Text Model (DistilBERT Fine-tuned)
- **Current SMAPE:** 53.78%
- **Baseline SMAPE:** 63.28%
- **Improvement:** 9.50% (15% relative improvement) 🎉
- **Model:** distilbert-base-uncased
- **Status:** ✅ COMPLETE - Model fine-tuned and working excellently!

### 📊 Existing Image Pipeline (ResNet50 + Traditional ML)
- **Architecture:** ResNet50 → PCA → XGBoost/LightGBM/CatBoost
- **Features:** 157 image features (128 embeddings + 21 color + 7 quality)
- **Expected SMAPE:** 55-60% (with text features ~47 combined)
- **Status:** ⚠️ Ready but not tested

---

## 🎯 Recommendation: Two-Stage Integration Approach

### Strategy Overview

**Your DistilBERT model is performing EXCELLENTLY (53.78% SMAPE)!** 

I recommend a **cautious, two-stage approach** to integrate images without hurting your strong text model:

---

## 📋 Stage 1: Quick Win - Late Fusion (Recommended First)

### Approach: Ensemble DistilBERT + Image Model

**How it works:**
1. Keep your fine-tuned DistilBERT as-is (53.78% SMAPE)
2. Train a separate image-only model using existing pipeline
3. Combine predictions using weighted average

**Why this is safe:**
- ✅ No risk to your existing strong model
- ✅ Fast to implement (2-3 hours)
- ✅ Easy to tune ensemble weights
- ✅ Can quickly test if images help
- ✅ Reversible if images don't help

### Implementation Steps

#### Step 1: Extract Image Features Only (40-90 min)
```powershell
cd try2/images
python image_feature_extraction.py
```

This creates:
- `preparation/image_features_train.csv` (157 features)
- `preparation/image_features_test.csv` (157 features)

#### Step 2: Train Image-Only Model (15 min)
```python
# New file: train_image_only_model.py
# Train XGBoost/LightGBM on ONLY image features
# No text features - pure image model
```

#### Step 3: Ensemble Predictions (5 min)
```python
# New file: ensemble_distilbert_image.py

# Load predictions
distilbert_pred = pd.read_csv('modeling/distilbert_sagemaker/test_out_distilbert_sagemaker.csv')
image_pred = pd.read_csv('modeling/test_out_image_only.csv')

# Weighted ensemble (test different weights)
weights = [
    (0.9, 0.1),  # 90% DistilBERT, 10% Image
    (0.8, 0.2),  # 80% DistilBERT, 20% Image
    (0.7, 0.3),  # 70% DistilBERT, 30% Image
]

for w_text, w_img in weights:
    ensemble['price'] = w_text * distilbert_pred['price'] + w_img * image_pred['price']
    # Calculate SMAPE on validation set
```

**Expected Results:**
- **If images help:** 51-53% SMAPE (1-3% improvement)
- **If images don't help:** Stick with DistilBERT (53.78%)
- **Time investment:** 2-3 hours
- **Risk:** Zero (can always fall back to pure DistilBERT)

---

## 📋 Stage 2: Deep Integration - Multimodal Fusion (If Stage 1 Works)

Only proceed if Stage 1 shows images help!

### Option 2A: Early Fusion - Feature Concatenation

**How it works:**
Combine DistilBERT embeddings + Image features → Train final regression head

```python
# Architecture:
# Text → DistilBERT → [768-dim embeddings]
#                                           ↓
#                                      [Concatenate] → MLP → Price
#                                           ↑
# Image → ResNet50 → [128-dim embeddings]
```

**Pros:**
- ✅ Learns joint representations
- ✅ Can capture text-image interactions
- ✅ More sophisticated

**Cons:**
- ⚠️ Requires retraining/fine-tuning
- ⚠️ Risk of hurting text performance
- ⚠️ More complex to implement
- ⚠️ Takes 2-3 days to get right

### Option 2B: Attention-Based Fusion (Advanced)

**How it works:**
Use cross-attention between text and image embeddings

```python
# Architecture:
# Text → DistilBERT → [768-dim]
#                           ↓
#                   [Cross-Attention] → Fusion → Price
#                           ↑
# Image → ResNet50 → [2048-dim]
```

**Pros:**
- ✅ State-of-the-art approach
- ✅ Learns what to focus on (text vs image)
- ✅ Best potential performance

**Cons:**
- ⚠️ Complex implementation
- ⚠️ Requires significant compute
- ⚠️ Risk of overfitting
- ⚠️ Takes 3-5 days to tune

---

## 🎯 My Strong Recommendation

### Start with **Stage 1: Late Fusion Ensemble**

**Why?**

1. **Your DistilBERT is already excellent (53.78%)**
   - Top 15% improvement over baseline
   - Don't risk breaking what works

2. **Quick validation if images help**
   - 2-3 hours to know if images are useful
   - If they don't help, you saved days of work

3. **Low risk, good potential**
   - Worst case: No improvement (keep 53.78%)
   - Best case: 1-3% improvement (50-52% SMAPE)
   - Expected: 51-53% SMAPE

4. **Easy to optimize**
   - Just tune ensemble weights
   - No retraining needed
   - Can submit within a day

---

## 📝 Detailed Implementation Plan

### Phase 1: Image-Only Model (TODAY)

```powershell
# 1. Extract image features (40-90 min)
cd try2/images
python image_feature_extraction.py

# 2. Train image-only model (create new script)
python train_image_only_model.py
```

**Create: `try2/images/train_image_only_model.py`**
```python
"""
Train image-only model for ensemble with DistilBERT
"""
import pandas as pd
import xgboost as xgb
# Load ONLY image features
# Train XGBoost/LightGBM
# Save predictions
```

### Phase 2: Ensemble Testing (TODAY)

**Create: `try2/modeling/ensemble_distilbert_image.py`**
```python
"""
Ensemble DistilBERT + Image predictions
Test different weight combinations
"""

# Load both predictions
# Try weights: [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]
# Calculate validation SMAPE for each
# Pick best weights
# Generate final submission
```

### Phase 3: Validation & Submission (TODAY)

1. **Validate on holdout set**
   - Compare ensemble vs pure DistilBERT
   - Only submit if improvement > 0.5%

2. **Generate submission**
   - Use best ensemble weights
   - Save as `test_out_distilbert_image_ensemble.csv`

3. **Document results**
   - Record SMAPE improvements
   - Note optimal ensemble weights

---

## 🔧 Practical Script Templates

### Script 1: `train_image_only_model.py`

```python
"""
Image-Only Model for Ensemble
Trains on pure image features (no text)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load image features
train_img = pd.read_csv('./preparation/image_features_train.csv')
test_img = pd.read_csv('./preparation/image_features_test.csv')

# Load original train for prices
train_orig = pd.concat([
    pd.read_csv('./dataset/train1.csv'),
    pd.read_csv('./dataset/train2.csv')
])

# Merge to get prices
train_df = train_img.merge(train_orig[['sample_id', 'price']], on='sample_id')

# Prepare features
X = train_df.drop(['sample_id', 'price'], axis=1)
y = train_df['price'].values

# Same split as DistilBERT (for fair validation)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42
)

# Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50
)

# Predict on test set
X_test_final = test_img.drop(['sample_id'], axis=1)
dtest = xgb.DMatrix(X_test_final)
predictions = model.predict(dtest)

# Save
submission = pd.DataFrame({
    'sample_id': test_img['sample_id'],
    'price': predictions
})
submission.to_csv('./modeling/test_out_image_only.csv', index=False)

print(f"Image-only model predictions saved!")
```

### Script 2: `ensemble_distilbert_image.py`

```python
"""
Ensemble DistilBERT + Image Model
Finds optimal weighting
"""

import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100

# Load predictions
distilbert = pd.read_csv('./modeling/distilbert_sagemaker/test_out_distilbert_sagemaker.csv')
image_only = pd.read_csv('./modeling/test_out_image_only.csv')

# For validation, load actual test split
# (You need to save this from training script)
# val_actual = pd.read_csv('./modeling/validation_actuals.csv')

# Test different weights
results = []
for w_text in np.arange(0.5, 1.0, 0.05):
    w_img = 1.0 - w_text
    
    ensemble_pred = w_text * distilbert['price'] + w_img * image_only['price']
    
    # Calculate SMAPE on validation (if available)
    # val_smape = smape(val_actual['price'], ensemble_pred)
    
    results.append({
        'weight_distilbert': w_text,
        'weight_image': w_img,
        # 'val_smape': val_smape
    })
    
results_df = pd.DataFrame(results)
print(results_df)

# Pick best weights (for now, start with 0.8/0.2)
best_w_text = 0.8
best_w_img = 0.2

# Generate final ensemble
final_pred = best_w_text * distilbert['price'] + best_w_img * image_only['price']

submission = pd.DataFrame({
    'sample_id': distilbert['sample_id'],
    'price': final_pred
})
submission.to_csv('./modeling/test_out_distilbert_image_ensemble.csv', index=False)

print(f"Ensemble predictions saved!")
print(f"Weights: {best_w_text:.2f} DistilBERT + {best_w_img:.2f} Image")
```

---

## 🎓 Why Traditional Image Pipeline Works

### Current Plan Analysis

**Your existing image pipeline (ResNet50 → XGBoost) is actually EXCELLENT for this use case:**

✅ **Pros:**
1. **Proven approach** - ResNet50 features work well for product images
2. **Fast training** - No need to fine-tune deep models
3. **Interpretable** - Can see which image features matter
4. **Low risk** - Won't hurt your DistilBERT performance
5. **Memory efficient** - PCA reduces dimensions well
6. **Good features:**
   - 128 ResNet embeddings capture visual patterns
   - Color features (brand, premium vs budget)
   - Quality metrics (professional photos = higher prices)

⚠️ **Cons:**
1. **Limited improvement** - Might only add 1-3% gain
2. **Fixed features** - ResNet not trained on product pricing task
3. **No text-image interaction** - Late fusion doesn't learn joint patterns

### When to Use Different Approaches

**Stick with ResNet50 + Traditional ML if:**
- ✅ You want results fast (today)
- ✅ You want low risk
- ✅ You want interpretability
- ✅ Images are secondary to text

**Consider Deep Integration if:**
- ⚠️ ResNet ensemble gives >2% improvement
- ⚠️ You have 3-5 days to experiment
- ⚠️ You have strong GPU resources
- ⚠️ You want to squeeze every 0.1%

---

## 📊 Expected Performance

### Conservative Estimate

| Model | SMAPE | Notes |
|-------|-------|-------|
| DistilBERT (current) | 53.78% | ✅ Strong baseline |
| Image-only | 70-75% | Worse than text (expected) |
| Ensemble (0.8/0.2) | 52-53% | 0.5-1.5% improvement |

### Optimistic Estimate

| Model | SMAPE | Notes |
|-------|-------|-------|
| DistilBERT (current) | 53.78% | ✅ Strong baseline |
| Image-only | 65-70% | Better than expected |
| Ensemble (0.7/0.3) | 51-52% | 2-3% improvement |

---

## 🚦 Decision Tree

```
Start Here
    ↓
Has limited time (< 1 day)?
    ├─ YES → Use ONLY DistilBERT (53.78%) ✅ SAFE
    └─ NO → Continue
        ↓
    Willing to risk no improvement?
        ├─ NO → Use ONLY DistilBERT (53.78%) ✅ SAFE
        └─ YES → Continue
            ↓
        Try Late Fusion Ensemble (2-3 hours)
            ↓
        Improvement > 0.5%?
            ├─ YES → Submit ensemble! 🎉
            └─ NO → Stick with DistilBERT
                ↓
            Want to try deep integration?
                ├─ YES → Spend 3-5 days on Stage 2
                └─ NO → Submit DistilBERT ✅
```

---

## 🎯 Final Recommendation

### **Option 1: SAFE (Recommended for Competition)**

**Submit pure DistilBERT (53.78%)**

**Why:**
- ✅ Already excellent performance (15% better than baseline)
- ✅ Zero risk
- ✅ Proven to work
- ✅ Can submit today

**When to choose:**
- Deadline is soon
- Want guaranteed strong result
- Don't want to risk breaking what works

---

### **Option 2: BALANCED (Recommended for Learning)**

**Try Late Fusion Ensemble**

**Timeline:**
- Today: Extract image features (1-2 hours)
- Today: Train image model (30 min)
- Today: Test ensemble (30 min)
- Today: Submit if improved, else use pure DistilBERT

**Expected outcome:**
- 51-53% SMAPE (1-3% improvement)
- Low risk
- Good learning experience

**When to choose:**
- Have 1 day to spare
- Want to try multimodal
- Can afford to test

---

### **Option 3: AMBITIOUS (For Research/Learning)**

**Deep Multimodal Integration**

**Timeline:**
- Week 1: Implement early/cross-attention fusion
- Week 1: Fine-tune end-to-end
- Week 2: Hyperparameter optimization
- Week 2: Extensive validation

**Expected outcome:**
- 50-52% SMAPE (2-4% improvement)
- High risk
- Complex implementation
- Great learning

**When to choose:**
- Have 2+ weeks
- Want to learn advanced techniques
- Competition is not primary goal
- Want publication-quality approach

---

## 📝 Action Plan for TODAY

If you choose **Option 2 (Balanced)**, here's your action plan:

### ⏰ Hour 1-2: Extract Image Features
```powershell
cd try2/images
python image_feature_extraction.py
```

### ⏰ Hour 2.5: Create & Run Image-Only Model
Create `train_image_only_model.py` (see template above)
```powershell
python train_image_only_model.py
```

### ⏰ Hour 3: Create & Test Ensemble
Create `ensemble_distilbert_image.py` (see template above)
```powershell
python ensemble_distilbert_image.py
```

### ⏰ Hour 3.5: Decision
- If ensemble SMAPE < 53.3%: Submit ensemble! 🎉
- If ensemble SMAPE ≥ 53.3%: Stick with DistilBERT (53.78%)

---

## 🎓 Key Insights

1. **Your DistilBERT model is already excellent** - Don't underestimate this achievement!

2. **Images are supplementary** - For text-heavy product descriptions, text models dominate

3. **Late fusion is underrated** - Simple weighted averaging often beats complex fusion

4. **Risk management matters** - In competitions, don't break what works

5. **Know when to stop** - 53.78% is already top-tier performance

---

## 📚 References & Further Reading

- **Late Fusion for Multimodal Learning** - Often outperforms complex approaches
- **Ensemble Methods** - Wisdom of crowds applies to ML models
- **ResNet50 Transfer Learning** - Proven effective for product images
- **DistilBERT Fine-tuning** - Your current approach is state-of-the-art

---

**Status:** ✅ Ready to implement  
**Priority:** ⭐⭐⭐ Try if time permits  
**Risk:** 🟢 LOW (Late fusion) | 🟡 MEDIUM (Deep integration)  
**Expected ROI:** 1-3% SMAPE improvement  

**Last Updated:** October 12, 2025
