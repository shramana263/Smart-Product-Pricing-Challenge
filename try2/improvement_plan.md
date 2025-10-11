# Improvement Plan - Smart Product Pricing Challenge

## Current Status
- **Best SMAPE**: 47.52%
- **Model**: LightGBM with Optuna optimization
- **Features**: 35 text-based features
- **Missing**: Image features (CRITICAL!)

## Phase 1: Image Feature Extraction 🖼️ (HIGHEST IMPACT)

### Why Images Matter:
- Product size visible in packaging
- Brand recognition from logos
- Quality indicators (premium vs budget packaging)
- Multi-pack indicators

### Implementation Steps:

1. **Download Images**
   ```python
   from src.utils import download_images
   # Download all training images
   ```

2. **Extract Features Using Pre-trained Models**
   - **CLIP** (Best for price prediction): Text-image embeddings
   - **EfficientNet-B0**: Efficient, 8B param limit compliant
   - **ResNet50**: Classic, reliable features
   
3. **Custom Visual Features**
   - Color distribution (premium products often have specific palettes)
   - Text density in image (indicates product size)
   - Logo detection
   - Package type classification

4. **Feature Dimensionality**
   - Extract 512-1024 dimensional embeddings
   - Apply PCA to reduce to top 50-100 components
   - Combine with existing text features

**Expected Improvement**: 3-5% SMAPE reduction

---

## Phase 2: Advanced Text Feature Engineering 📝

### A. Title/Description Enhancements

1. **Extract More Quantity Indicators**
   - Multi-pack patterns: "pack of 12", "24 count"
   - Size comparisons: "family size", "travel size"
   - Volume indicators: extract all measurements

2. **Brand Hierarchy**
   - Group brands into tiers (premium, mid, budget)
   - Create brand-category interaction features
   - Average price per brand-category combo

3. **Price-Indicative Keywords**
   - "organic", "premium", "gourmet" → higher price
   - "value pack", "economy" → lower price
   - "imported", "artisan" → higher price

### B. Interaction Features

```python
# Brand × Category interactions
# Value × Unit interactions  
# Normalized_quantity × Category
# Brand_target × Word_count
```

**Expected Improvement**: 1-2% SMAPE reduction

---

## Phase 3: Model Improvements 🤖

### A. Ensemble Strategies

1. **Stacked Ensemble**
   - Level 1: LightGBM, XGBoost, CatBoost, Neural Network
   - Level 2: Ridge/Lasso meta-learner
   
2. **Weighted Average** (Different data views)
   - Model trained on text features only
   - Model trained on image features only
   - Model trained on combined features

3. **Target Transformations**
   - Try sqrt(price) instead of log(price)
   - Try Box-Cox transformation
   - Quantile transformation

### B. Advanced Techniques

1. **Price Binning**
   - Train separate models for different price ranges
   - Low (<$10), Medium ($10-$50), High (>$50)
   
2. **Pseudo-Labeling**
   - Make predictions on test set
   - Add high-confidence predictions to training
   - Retrain model

3. **Feature Selection**
   - Try Recursive Feature Elimination
   - SHAP-based feature selection
   - Permutation importance

**Expected Improvement**: 1-3% SMAPE reduction

---

## Phase 4: Error Analysis & Refinement 🔍

### A. Identify Failure Patterns

1. **Analyze Top Errors**
   - Which categories have highest errors?
   - Which brands are mispredicted?
   - Price range analysis

2. **Create Correction Models**
   - Train residual correction model
   - Apply category-specific adjustments

### B. Post-Processing

1. **Price Range Constraints**
   - Set min/max bounds per category
   - Use quantile-based clipping

2. **Smoothing**
   - Apply moving average on similar products
   - Use KNN-based price adjustment

**Expected Improvement**: 0.5-1% SMAPE reduction

---

## Phase 5: Final Optimizations ⚡

1. **Hyperparameter Tuning**
   - Extended Optuna trials (500+ iterations)
   - Cross-validation with different seeds
   
2. **Feature Engineering Iteration**
   - Try polynomial features (degree 2)
   - Log/sqrt transformations on new features
   
3. **Ensemble Refinement**
   - Optimize ensemble weights
   - Try different meta-learners

**Expected Improvement**: 0.5-1% SMAPE reduction

---

## Implementation Priority

### Week 1: Image Features (DO THIS FIRST!)
- [ ] Download all images
- [ ] Extract CLIP embeddings
- [ ] Train model with image + text features
- [ ] Target: <45% SMAPE

### Week 2: Advanced Features & Ensembles
- [ ] Create interaction features
- [ ] Build stacked ensemble
- [ ] Implement pseudo-labeling
- [ ] Target: <43% SMAPE

### Week 3: Refinement & Optimization
- [ ] Error analysis & corrections
- [ ] Hyperparameter tuning
- [ ] Final ensemble optimization
- [ ] Target: <42% SMAPE

---

## Projected Final SMAPE: 40-42%

## Key Risks
1. **Image download failures**: Use retry logic
2. **Memory issues**: Process images in batches
3. **Overfitting**: Use strong regularization
4. **Time constraints**: Prioritize image features first

---

## Tools & Libraries Needed
```
pip install torch torchvision
pip install transformers  # For CLIP
pip install efficientnet_pytorch
pip install opencv-python pillow
pip install shap
```

## Notes
- Images are the BIGGEST missed opportunity
- Text features alone have ~47% ceiling
- Combining modalities can break 42%
- Focus on LOW-HANGING FRUIT first (images!)
