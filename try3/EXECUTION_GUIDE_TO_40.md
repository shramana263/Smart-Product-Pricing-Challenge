# 🎯 Path to <40% SMAPE: Execution Guide

## 📊 Current Status

```
✅ Phase 1 COMPLETE: Huber Loss → 47.378% SMAPE
🔄 Phase 2 READY:    Two-Stage Model (Classification + Regression)
⏳ Phase 3 PLANNED:  Vision Transformer + Multimodal Fusion
```

**Progress Tracker:**
- Starting point: 53.636% SMAPE
- After Phase 1: **47.378% SMAPE** (-6.258 points! ✅)
- **Current gap to 40%: 7.378 points**

---

## 🚀 Phase 2: Two-Stage Model (NEXT STEP)

### Why Two-Stage?

**Current Problem:**
```
Budget items ($0-$10):     122.6% SMAPE ❌
Luxury items ($100+):      140.8% SMAPE ❌
Mid-range items ($20-$30):  17.6% SMAPE ✅
```

Single model predicts ~$20-30 for EVERYTHING!
- Predicts $30 for $5 item → 142% SMAPE
- Predicts $30 for $700 item → 183% SMAPE

**Two-Stage Solution:**
1. **Stage 1**: Classify into price range first
2. **Stage 2**: Use specialized regression head per range

Result: Budget gets budget-specialized model, Luxury gets luxury-specialized model!

---

### How to Run

#### On SageMaker (RECOMMENDED):

```bash
# 1. Upload code to SageMaker
cd /opt/ml/code/try3/implementation

# 2. Run two-stage training
python train_two_stage_model.py

# Expected runtime: ~2-3 hours
# Expected result: 41-42% SMAPE
```

#### Locally:

```bash
cd try3/implementation
python train_two_stage_model.py
```

---

### Expected Output

```
📊 Price Range Distribution:
  Budget      ($0-$10):   5,713 (38%)
  Economy     ($10-$20):  3,863 (26%)
  Mid-Premium ($20-$50):  3,801 (25%)
  High-End    ($50-$100): 1,245 (8%)
  Luxury      ($100+):    378 (3%)

🚀 Starting cross-validation...

Fold 1/5
Epoch 1/5
Training: [Class Acc: 72%, SMAPE: 45%]
Val SMAPE: 43.2%
Val Class Acc: 75%

...

✓ Fold 1 SMAPE: 41.8%
✓ Fold 2 SMAPE: 42.1%
✓ Fold 3 SMAPE: 41.5%
✓ Fold 4 SMAPE: 42.3%
✓ Fold 5 SMAPE: 41.9%

OOF SMAPE: 41.9% ← TARGET!
Improvement: 47.4% → 41.9% = -5.5 points ✅
```

---

### Success Criteria

- [ ] Classification accuracy > 70%
- [ ] Budget range SMAPE < 50% (currently 122.6%)
- [ ] Luxury range SMAPE < 80% (currently 140.8%)
- [ ] **Overall OOF SMAPE < 42%**

If achieved: **Only 2 points away from <40%!** 🎯

---

## 🔮 Phase 3: Vision Transformer (If needed)

### When to do this?

- **If Phase 2 achieves <40%**: Celebrate! 🎉 Skip Phase 3.
- **If Phase 2 achieves 40-42%**: Do Phase 3 to reach <40%.
- **If Phase 2 achieves >42%**: Debug Phase 2 first.

### What is ViT?

Replace ResNet50 (2048-dim) with Vision Transformer (384-dim from ViT-small):

```python
from transformers import ViTModel

vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
```

**Why ViT?**
- Attention mechanism: Focuses on relevant image regions
- Better for packaging/quantity/premium signals
- Pre-trained on ImageNet (transfer learning)

**Expected Improvement:** 41-42% → 38-39% SMAPE

---

## 📈 Projected Timeline

### Week 2: Two-Stage Model
- **Days 1-2**: Train two-stage model on SageMaker
- **Day 3**: Analyze results, tune hyperparameters if needed
- **Expected Result**: 41-42% SMAPE

### Week 3: Vision Transformer (if needed)
- **Days 1-2**: Extract ViT features from images
- **Day 3**: Train multimodal model (text + ViT)
- **Day 4**: Ensemble optimization
- **Expected Result**: 38-39% SMAPE

### Week 4: Final Optimization
- **Days 1-2**: Test-time augmentation
- **Day 3**: Ensemble of top 5 models
- **Expected Result**: 35-37% SMAPE (stretch goal!)

---

## 🎯 Decision Tree

```
Current: 47.4% SMAPE
    ↓
Run Phase 2 (Two-Stage)
    ↓
    ├─ Result < 40%? → ✅ SUCCESS! Submit and celebrate!
    │
    ├─ Result 40-42%? → Run Phase 3 (ViT)
    │                      ↓
    │                      ├─ Result < 40%? → ✅ SUCCESS!
    │                      └─ Result ≥ 40%? → Ensemble + TTA
    │
    └─ Result > 42%? → Debug:
                        - Check class accuracy (should be >70%)
                        - Check per-range SMAPE
                        - Tune loss weights
                        - Try different δ values for Huber
```

---

## 🔍 Debugging Guide

### If Two-Stage Model Fails (>45%):

1. **Check Classification Accuracy**
   ```python
   # Should be >70%
   # If <70%: Try Focal Loss with higher gamma
   ```

2. **Check Per-Range SMAPE**
   ```python
   for range_name, range_smape in range_results:
       print(f"{range_name}: {range_smape}%")
   
   # Budget should be <50%
   # Economy should be <35%
   # Luxury should be <80%
   ```

3. **Tune Loss Weights**
   ```python
   # Try different weights
   classification_weight = 0.2  # was 0.3
   regression_weight = 0.8      # was 0.7
   ```

4. **Adjust Huber Delta per Range**
   ```python
   # Budget: more sensitive (δ=0.3)
   # Economy: balanced (δ=1.0)
   # Luxury: more robust (δ=2.0)
   ```

---

## 📊 Performance Tracking

| Phase | Improvement | SMAPE | Status |
|-------|-------------|-------|--------|
| Baseline (MSE) | - | 53.636% | ✅ |
| Phase 1 (Huber) | -6.258 | **47.378%** | ✅ |
| Phase 2 (Two-Stage) | -5.5 (expected) | 41-42% (target) | 🔄 |
| Phase 3 (ViT) | -2.5 (expected) | 38-39% (target) | ⏳ |
| Phase 4 (Ensemble) | -2.0 (expected) | 36-37% (stretch) | ⏳ |

---

## 🚦 Next Action

**IMMEDIATE NEXT STEP:**

```bash
cd try3/implementation
python train_two_stage_model.py
```

**Expected:**
- Runtime: 2-3 hours on SageMaker ml.g4dn.xlarge
- Output: 41-42% OOF SMAPE
- Files: `two_stage_model/oof_predictions.csv`, `test_predictions.csv`

**After training:**
1. Check OOF SMAPE
2. If < 40%: 🎉 Submit to competition!
3. If 40-42%: Proceed to Phase 3 (ViT)
4. If > 42%: Debug using guide above

---

## 📝 Key Files

```
try3/implementation/
├── train_two_stage_model.py          ← RUN THIS!
├── TWO_STAGE_QUICK_REFERENCE.md      ← Architecture details
├── train_distilbert_huber.py         ← Phase 1 (DONE)
└── two_stage_model/                  ← Output folder
    ├── oof_predictions.csv
    ├── test_predictions.csv
    └── best_model_fold*.pt

try3/
├── STRATEGY_TO_40_PERCENT.md         ← Master strategy
└── research/outputs/
    ├── error_analysis_by_range.csv   ← Why two-stage needed
    └── difficult_samples_top50.csv   ← Worst predictions
```

---

## 💡 Key Insights

1. **Huber Loss is powerful**: Beat expectations by 2 points!
2. **Error analysis reveals patterns**: Budget/Luxury ranges are disasters
3. **Two-stage is essential**: Single model can't handle 199x price range
4. **Classification helps regression**: Knowing range improves prediction
5. **Specialized models work better**: Budget model ≠ Luxury model

---

**🚀 You're 7.4 points away from <40%! Two-stage model should get you there!**

**Good luck! 🎯**
