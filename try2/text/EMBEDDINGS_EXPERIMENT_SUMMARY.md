# 📊 Text Embeddings Experiment Results

## Date: October 11, 2025
## Approach: Sentence Transformers for Price Prediction

---

## ✅ What We Did

### 1. Text Embedding Extraction
- **Model Used**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Dimensionality Reduction**: PCA to 128 components (87.24% variance retained)
- **Processing Time**: ~68 minutes (34 min train + 34 min test)
- **Data Processed**: 150,000 samples (75k train + 75k test)

### 2. Model Training Approaches

#### Approach A: Embeddings Only
**Models Tested**: LightGBM, XGBoost, CatBoost

**Results**:
| Model | Val SMAPE | Test SMAPE |
|-------|-----------|------------|
| **XGBoost** | 68.72% | **69.90%** |
| LightGBM | 69.26% | 70.16% |
| CatBoost | 69.05% | 70.48% |

**Conclusion**: ❌ **Worse than baseline (47.52%)**

#### Approach B: Embeddings + Existing Features
**Combined**: 128 embedding dims + 35 hand-crafted features = 163 total features

**Results**:
| Model | Val SMAPE | Test SMAPE | MAE | RMSE |
|-------|-----------|------------|-----|------|
| **XGBoost** | 54.85% | **55.63%** | $11.84 | $22.29 |
| CatBoost | 55.00% | 56.03% | $11.85 | $21.95 |
| LightGBM | 55.17% | 56.36% | $12.02 | $22.03 |

**Conclusion**: ⚠️ **Still worse than baseline, but better than embeddings-only**

#### Approach C: Optimized Embeddings (Optuna)
**Status**: Currently running with hyperparameter optimization

---

## 🤔 Why Embeddings Underperformed

### Problem Analysis:

1. **Embeddings Lack Explicit Signals**
   - Embeddings capture semantics but miss explicit features
   - Example: `brand_target` encoding (most important feature in your baseline) = direct price signal
   - Embeddings don't capture this as strongly

2. **Your Hand-Crafted Features Are Very Good!**
   - `brand_target`: Direct price-per-brand encoding
   - `normalized_quantity`: Explicit size information
   - `unit` encoding: Critical for price calculation
   
   These are **hard** signals that embeddings can't easily capture

3. **Catalog Text May Be Too Generic**
   - Product descriptions might be similar across price ranges
   - Example: "Premium Organic Coffee" appears in both $10 and $30 products
   - Embeddings treat similar text similarly, regardless of brand prestige

4. **Missing Visual Information**
   - Text alone can't distinguish package size, multi-packs, etc.
   - Images would help significantly

---

## 💡 Key Learnings

### What Worked in Your Baseline (47.52%):
✅ **Brand Target Encoding** - Direct price relationship
✅ **Explicit Quantity/Unit Features** - Critical for pricing
✅ **Category Encoding** - Price varies significantly by category
✅ **Feature Engineering** - Your manual features capture domain knowledge

### What Embeddings Provide:
✅ Semantic understanding of product descriptions
✅ Context-aware representations
✅ Generalization to unseen text patterns
❌ But lack explicit numerical relationships

---

## 🎯 Recommendations Going Forward

### Option 1: Improve Current Baseline (RECOMMENDED)
**Why**: Your hand-crafted features are working well!

**Actions**:
1. ✅ **Add More Interaction Features**
   ```python
   - brand_target × category
   - value × unit interaction
   - brand_freq × normalized_quantity
   ```

2. ✅ **Better Brand Encoding**
   ```python
   - Hierarchical brand clustering
   - Brand price tier classification
   - Brand-category-specific encoding
   ```

3. ✅ **Quantity Engineering**
   ```python
   - Extract all numeric values from text
   - Better multi-pack detection
   - Size category classification (travel/standard/family/bulk)
   ```

4. ✅ **Ensemble Different Transformations**
   ```python
   - Model on log(price)
   - Model on sqrt(price)
   - Model on raw price
   - Weighted average
   ```

**Expected Improvement**: 47.52% → **42-45% SMAPE**

---

### Option 2: Add Image Features (HIGH IMPACT)
**Why**: Images contain information text can't capture

**What Images Provide**:
- Package size (visual)
- Multi-pack indicators
- Brand logos (premium vs budget)
- Product quality signals

**Implementation**:
1. Download images using `src/utils.py`
2. Extract features with EfficientNet/ResNet
3. Combine: hand-crafted features + image embeddings
4. Train ensemble model

**Expected Improvement**: 47.52% → **38-42% SMAPE**

---

### Option 3: Hybrid Approach
**Combine**:
- Your hand-crafted features (35 dims) - **70% weight**
- Text embeddings (128 dims) - **15% weight**
- Image embeddings (when available) - **15% weight**

**Rational**:
- Hand-crafted features for explicit signals
- Embeddings for semantic nuances
- Images for visual confirmation

---

## 📈 Next Steps (Priority Order)

### 🥇 Priority 1: Improve Hand-Crafted Features (This Week)
**Time**: 1-2 days  
**Expected SMAPE**: 42-45%

Tasks:
- [ ] Add brand × category interactions
- [ ] Better quantity extraction from text
- [ ] Create price tier features
- [ ] Ensemble with different target transformations
- [ ] Hyperparameter tuning with Optuna (longer trials)

### 🥈 Priority 2: Add Image Features (Next Week)
**Time**: 2-3 days  
**Expected SMAPE**: 38-42%

Tasks:
- [ ] Download product images
- [ ] Extract CNN features (EfficientNet-B0)
- [ ] Combine with text features
- [ ] Train multi-modal model

### 🥉 Priority 3: Advanced Techniques (If Time)
**Time**: 2-3 days  
**Expected SMAPE**: 36-40%

Tasks:
- [ ] Stacked ensemble (multiple models)
- [ ] Pseudo-labeling on test set
- [ ] Category-specific models
- [ ] Price range binning

---

## 📊 Comparison: Approaches Ranked

| Approach | SMAPE | Pros | Cons | Recommendation |
|----------|-------|------|------|----------------|
| **Your Baseline** | **47.52%** | ✅ Explicit features<br>✅ Fast training<br>✅ Interpretable | ⚠️ Manual engineering | ⭐⭐⭐⭐⭐ Keep & improve |
| Hand-Crafted + Images | 38-42% (est.) | ✅ Multi-modal<br>✅ Visual info | ⚠️ Slower<br>⚠️ Complex | ⭐⭐⭐⭐ Next priority |
| Embeddings + Hand-Crafted | 55.63% | ✅ Semantic understanding | ❌ Worse than baseline | ⭐⭐ Not recommended alone |
| Embeddings Only | 69.90% | ✅ Auto features | ❌ Much worse | ⭐ Don't use |

---

## 🎓 Conclusion

### Key Takeaway:
**Your hand-crafted feature engineering is EXCELLENT!** The embeddings experiment confirmed that your domain knowledge and explicit feature extraction are capturing the right signals for price prediction.

### Best Path Forward:
1. **Stick with your successful approach**
2. **Add more interaction features** (brand × category, value × unit)
3. **Add image features** (biggest opportunity)
4. **Use embeddings sparingly** (maybe as 10-15% of features in ensemble)

### Expected Timeline to <40% SMAPE:
- **Week 1**: Improve features → 42-45%
- **Week 2**: Add images → 38-42%
- **Week 3**: Final tuning → 36-40%

---

## 📁 Files Created

```
try2/
├── text_embeddings.py              # Embedding extraction script
├── train_with_embeddings.py        # Training script
├── train_embeddings_optimized.py   # Optuna optimization
├── embeddings_data/
│   ├── train_embeddings.csv        # 75k × 130 (128 emb + id + price)
│   ├── test_embeddings.csv         # 75k × 129 (128 emb + id)
│   ├── pca_model.pkl               # PCA transformer
│   └── embedding_info.txt          # Metadata
├── modeling/
│   ├── test_out_embeddings.csv     # Predictions (embeddings only)
│   └── embedding_model_results.csv # Model comparison
└── EMBEDDINGS_EXPERIMENT_SUMMARY.md  # This file
```

---

**Bottom Line**: Embeddings are powerful but not a silver bullet. Your hand-crafted features + images will give you better results! 🎯

---

Generated: October 11, 2025
