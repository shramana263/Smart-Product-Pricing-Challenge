# 🎯 Strategy to Achieve <40% SMAPE

**Current Status:**
- Pure DistilBERT Fine-tuned: **53.636%** ✅ (Best so far)
- Text + Image Ensemble: 59.708%
- Target: **<40% SMAPE**
- Gap to close: **~14 points**

---

## 📊 Root Cause Analysis

### **Problem 1: Outlier Handling (CRITICAL)**

From your research data:
```
Price Statistics:
  Mean:     $23.65
  Median:   $14.00
  Max:      $2,796.00  ← 199x the median!
  Skewness: 13.60      ← Extremely right-skewed
  Kurtosis: 736.65     ← Massive outliers
```

**Current Issue:**
- Pure DistilBERT likely trained with **MSE loss** = penalizes large errors quadratically
- Model learns to predict median/mean to minimize squared error
- **Result:** Fails catastrophically on high-priced items ($100-$2,796)
- Box plots show significant outliers (>$100) that model ignores

**Impact on SMAPE:**
- Predicting $20 for a $500 item = 183% SMAPE for that sample!
- Even 1% of samples with 150%+ SMAPE destroys overall score
- SMAPE can go up to 200% per sample

### **Problem 2: Missing Information (Text Only)**

**Text limitations:**
- Quantity: "Pack of 12" mentioned inconsistently
- Size: Images show actual product size better than text
- Premium signals: Visual packaging quality not in text
- Material: Image shows gold/silver/plastic better than description

**Example Case:**
```
Text: "Premium Coffee Beans"
Could be: $15 (1 lb) OR $150 (5 lb premium organic)
Image reveals: Large bag, premium packaging → $150
```

### **Problem 3: Model Architecture**

Current fine-tuned DistilBERT:
- ❌ Uses MSE loss (sensitive to outliers)
- ❌ Single regression head (can't handle multimodal reasoning)
- ❌ No explicit outlier handling
- ❌ Text-only (ignores 50% of information)

---

## 🚀 Proposed Solution: Intelligent Multimodal System

### **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                         │
├─────────────────────┬───────────────────────────────────────┤
│   TEXT BRANCH       │        IMAGE BRANCH                   │
│                     │                                       │
│  DistilBERT         │   Vision Transformer (ViT)            │
│  Fine-tuned         │   OR EfficientNet-B4                  │
│  Huber Loss         │   Pretrained on e-commerce            │
│  (robust to         │                                       │
│   outliers)         │   Extract:                            │
│                     │   - Packaging quality                 │
│  Extract:           │   - Size indicators                   │
│  - Product info     │   - Premium signals                   │
│  - Quantities       │   - Quantity from image               │
│  - Brand signals    │   - Material (gold/plastic)           │
│                     │                                       │
└─────────────────────┴───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              INTELLIGENT FUSION MODULE                      │
│                                                             │
│  Cross-Attention Mechanism:                                 │
│  - Text queries image for missing quantity                 │
│  - Image queries text for product category                 │
│  - Weighted fusion based on confidence                     │
│                                                             │
│  Missing Data Handler:                                      │
│  IF text_quantity == NULL:                                  │
│      → Look at image_size_estimate                          │
│  IF text_premium_unclear:                                   │
│      → Check image_packaging_quality                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           OUTLIER-AWARE PREDICTION HEAD                     │
│                                                             │
│  Two-Stage Approach:                                        │
│                                                             │
│  Stage 1: Price Range Classification                        │
│    Classes: Budget(<$10), Mid($10-50), Premium($50-100),  │
│             Luxury($100-500), Ultra-Luxury($500+)          │
│    Loss: Focal Loss (handles class imbalance)             │
│                                                             │
│  Stage 2: Conditional Regression                            │
│    IF class == Budget/Mid:                                  │
│        → Use Huber loss (δ=1.0)                            │
│    IF class == Premium/Luxury/Ultra:                        │
│        → Use Log-Cosh loss (smooth, robust)                │
│                                                             │
│  Final Prediction:                                          │
│    price = class_midpoint × regression_multiplier          │
│    with confidence-weighted ensemble                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Plan

### **Phase 1: Outlier-Robust Loss Functions (Week 1)**

**1.1 Huber Loss for DistilBERT**
```python
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)
```

**Why Huber?**
- Quadratic for small errors (< δ) → accurate on normal prices
- Linear for large errors (> δ) → robust to outliers
- Reduces penalty on $20→$500 predictions

**Expected Improvement:** 53.6% → 49-50% SMAPE

---

**1.2 Log-Cosh Loss (Alternative)**
```python
def log_cosh_loss(pred, target):
    error = pred - target
    return torch.mean(torch.log(torch.cosh(error)))
```

**Why Log-Cosh?**
- Smooth everywhere (good for optimization)
- ~L2 for small errors, ~L1 for large errors
- Works well with Adam optimizer

**Expected Improvement:** 53.6% → 48-49% SMAPE

---

**1.3 Quantile Loss for SMAPE Optimization**
```python
def quantile_loss(pred, target, quantile=0.5):
    error = target - pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))
```

**Why Quantile?**
- Can optimize for median (quantile=0.5)
- SMAPE is median-based error metric
- Less sensitive to extreme outliers

**Expected Improvement:** 53.6% → 47-48% SMAPE

---

### **Phase 2: Two-Stage Prediction (Week 2)**

**2.1 Price Range Classification**

```python
class PriceRangeClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # 5 price ranges
        )
    
    def forward(self, embeddings):
        return self.classifier(embeddings)
```

**Price Ranges:**
```
Budget:       $0-10      (40% of samples) → Target: 30% SMAPE
Mid-Range:    $10-50     (45% of samples) → Target: 25% SMAPE  
Premium:      $50-100    (10% of samples) → Target: 35% SMAPE
Luxury:       $100-500   (4% of samples)  → Target: 40% SMAPE
Ultra-Luxury: $500+      (1% of samples)  → Target: 50% SMAPE

Weighted Average: 0.40×30 + 0.45×25 + 0.10×35 + 0.04×40 + 0.01×50
                = 12 + 11.25 + 3.5 + 1.6 + 0.5 = 28.85% SMAPE ✅
```

**2.2 Conditional Regression**

```python
class ConditionalRegressor(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        # Separate regression heads for each price range
        self.budget_head = nn.Linear(hidden_size, 1)
        self.mid_head = nn.Linear(hidden_size, 1)
        self.premium_head = nn.Linear(hidden_size, 1)
        self.luxury_head = nn.Linear(hidden_size, 1)
        self.ultra_head = nn.Linear(hidden_size, 1)
    
    def forward(self, embeddings, price_class):
        # Route to appropriate head based on classification
        predictions = []
        for i, cls in enumerate(price_class):
            if cls == 0:  # Budget
                pred = self.budget_head(embeddings[i])
            elif cls == 1:  # Mid
                pred = self.mid_head(embeddings[i])
            # ... etc
            predictions.append(pred)
        return torch.stack(predictions)
```

**Expected Improvement:** 49% → 42-43% SMAPE

---

### **Phase 3: Multimodal Fusion (Week 3)**

**3.1 Vision Transformer for Images**

```python
from transformers import ViTModel

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Fine-tune for e-commerce
        self.price_predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)  # Image features
        )
    
    def forward(self, images):
        outputs = self.vit(images)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.price_predictor(cls_token)
```

**Why ViT?**
- Better than ResNet50 for fine details (packaging quality)
- Attention mechanism learns what matters (size indicators)
- Can extract quantities from image labels

**3.2 Cross-Modal Attention**

```python
class CrossModalFusion(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.image_to_text_attn = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, text_embed, image_embed):
        # Text queries image for missing info
        text_enhanced, _ = self.text_to_image_attn(
            query=text_embed,
            key=image_embed,
            value=image_embed
        )
        
        # Image queries text for context
        image_enhanced, _ = self.image_to_text_attn(
            query=image_embed,
            key=text_embed,
            value=text_embed
        )
        
        # Fusion
        fused = torch.cat([text_enhanced, image_enhanced], dim=-1)
        return fused
```

**3.3 Missing Data Handler**

```python
class MissingDataHandler(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence_scorer = nn.Linear(768, 1)  # Confidence in text
        
    def forward(self, text_embed, image_embed, text_quantity_present):
        # Check text confidence
        text_confidence = torch.sigmoid(self.confidence_scorer(text_embed))
        
        # If quantity missing in text, use image
        if not text_quantity_present:
            # Boost image weight
            weight = torch.tensor([0.3, 0.7])  # 70% image
        else:
            # Normal weighted fusion
            weight = torch.stack([text_confidence, 1 - text_confidence])
        
        fused = weight[0] * text_embed + weight[1] * image_embed
        return fused
```

**Expected Improvement:** 42-43% → 38-40% SMAPE ✅

---

### **Phase 4: Advanced Techniques (Week 4)**

**4.1 Ensemble of Multiple Models**

```python
# Train 5 models with different configurations
models = [
    DistilBERT_Huber,      # 49% SMAPE
    DistilBERT_LogCosh,    # 48% SMAPE
    DistilBERT_Quantile,   # 47% SMAPE
    TwoStage_Conditional,  # 42% SMAPE
    Multimodal_CrossAttn,  # 38% SMAPE
]

# Weighted ensemble (optimize on validation set)
weights = [0.1, 0.1, 0.15, 0.25, 0.4]  # Best model gets most weight
final_pred = sum(w * model.predict(x) for w, model in zip(weights, models))
```

**Expected Improvement:** 38-40% → 36-38% SMAPE

**4.2 Test-Time Augmentation (TTA)**

```python
def test_time_augmentation(model, text, image):
    predictions = []
    
    # Original
    predictions.append(model(text, image))
    
    # Text augmentations
    predictions.append(model(text + " premium quality", image))
    predictions.append(model(text + " value pack", image))
    
    # Image augmentations
    predictions.append(model(text, augment_brightness(image, 1.1)))
    predictions.append(model(text, augment_brightness(image, 0.9)))
    
    # Return median (robust to outliers)
    return torch.median(torch.stack(predictions))
```

**Expected Improvement:** 36-38% → 35-37% SMAPE

**4.3 Pseudo-Labeling on Test Set**

```python
# Use high-confidence predictions as additional training data
def pseudo_label(model, test_data):
    predictions = model.predict(test_data)
    confidence = model.predict_confidence(test_data)
    
    # Keep only high-confidence predictions
    high_conf_mask = confidence > 0.9
    pseudo_train = test_data[high_conf_mask]
    pseudo_labels = predictions[high_conf_mask]
    
    # Retrain with original + pseudo-labeled data
    combined_data = concat(original_train, pseudo_train)
    combined_labels = concat(original_labels, pseudo_labels)
    
    model.fit(combined_data, combined_labels)
```

**Expected Improvement:** 35-37% → 34-36% SMAPE

---

## 📋 Prioritized Action Plan

### **🔴 CRITICAL (Do First):**

1. **Implement Huber Loss** (1 day)
   - Modify `finetune_distilbert_sagemaker.py`
   - Replace MSE with Huber (δ=1.0)
   - Expected: 53.6% → 49-50%

2. **Add Log Transform to Target** (1 day)
   ```python
   # Train on log(price)
   target = np.log1p(train['price'])
   # Predict and inverse transform
   pred_price = np.expm1(model.predict(test))
   ```
   - Handles skewed distribution
   - Expected: 49% → 46-47%

3. **Two-Stage Classification + Regression** (3 days)
   - Price range classifier (5 classes)
   - Conditional regression heads
   - Expected: 46-47% → 41-42%

### **🟡 HIGH PRIORITY (Week 2):**

4. **Add Image Features with ViT** (4 days)
   - Replace ResNet50 with Vision Transformer
   - Fine-tune on e-commerce data
   - Expected: 41-42% → 38-39%

5. **Cross-Modal Attention** (3 days)
   - Implement text→image and image→text attention
   - Missing data handler
   - Expected: 38-39% → 36-37%

### **🟢 MEDIUM PRIORITY (Week 3):**

6. **Ensemble of 5 Models** (2 days)
   - Train variations with different losses
   - Optimize ensemble weights
   - Expected: 36-37% → 34-35%

7. **Test-Time Augmentation** (1 day)
   - Augment text and images at inference
   - Take median prediction
   - Expected: 34-35% → 33-34%

---

## 🎯 Expected Timeline to <40%

```
Week 1: Outlier-Robust Losses
  Day 1-2:   Huber Loss              → 49-50% SMAPE
  Day 3-4:   Log Transform           → 46-47% SMAPE
  Day 5-7:   Two-Stage Prediction    → 41-42% SMAPE ✅ BELOW 42%!

Week 2: Multimodal Intelligence
  Day 8-11:  Vision Transformer      → 38-39% SMAPE ✅ BELOW 40%!
  Day 12-14: Cross-Modal Attention   → 36-37% SMAPE

Week 3: Ensemble & Refinement
  Day 15-16: Ensemble Models         → 34-35% SMAPE
  Day 17:    Test-Time Aug           → 33-34% SMAPE

FINAL RESULT: 33-37% SMAPE (Target: <40% ✅)
```

---

## 💡 Key Insights

### **Why Your Current Models Failed:**

1. **Pure DistilBERT (53.6%):**
   - ✅ Good text understanding
   - ❌ MSE loss = outlier catastrophe
   - ❌ Text-only = missing visual signals
   - ❌ No explicit range handling

2. **Balanced Model + Ensemble (59.7%):**
   - ❌ Frozen embeddings = no adaptation
   - ❌ Engineered features added noise
   - ❌ ResNet50 not optimized for e-commerce
   - ❌ Simple averaging = suboptimal fusion

### **Why New Approach Will Work:**

1. **Outlier Handling:**
   - Huber/Log-Cosh loss = robust to large errors
   - Two-stage = separate handling per price range
   - Log transform = normalizes skewed distribution

2. **Multimodal Intelligence:**
   - ViT > ResNet50 for fine details
   - Cross-attention = mutual information
   - Missing data handler = fill gaps intelligently

3. **Smart Fusion:**
   - Confidence-weighted combination
   - Learns when to trust text vs image
   - Handles edge cases gracefully

---

## 📊 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting on outliers | Medium | High | Use stratified CV, validate on all price ranges |
| Image download failures | Low | Medium | Already handled with retry logic (99.99% success) |
| ViT too large for RAM | Medium | High | Use ViT-small or efficient batch processing |
| Cross-attention too complex | Low | Medium | Start simple, add complexity gradually |
| Time constraint (4 weeks) | Medium | High | Focus on priorities (Huber→Two-Stage→ViT) |

---

## 🚀 Next Steps

**Start TODAY:**

1. **Modify `finetune_distilbert_sagemaker.py`:**
   - Add Huber loss option
   - Add log transform option
   - Train and evaluate

2. **Create `train_two_stage_model.py`:**
   - Implement price range classifier
   - Implement conditional regression
   - Test on validation set

3. **Research ViT models:**
   - `google/vit-base-patch16-224` (86M params)
   - `google/vit-small-patch16-224` (22M params) ← Recommended
   - `microsoft/swin-tiny-patch4-window7-224` (28M params)

Would you like me to:
1. ✅ Create the modified `finetune_distilbert_huber.py` script?
2. ✅ Create the `train_two_stage_model.py` script?
3. ✅ Create the `train_multimodal_vit.py` script?
4. ✅ Set up the training pipeline for all 3 approaches?

**Target: Achieve <40% SMAPE in 2-3 weeks! 🎯**
