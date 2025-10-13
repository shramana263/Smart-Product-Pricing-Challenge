# 🚀 3-MODEL ENSEMBLE EXECUTION GUIDE

## 📋 OVERVIEW

**Goal**: Achieve <40% SMAPE on price prediction competition

**Strategy**: Build 3 models with different loss functions, then ensemble

**Current Status**: 47.378% SMAPE (Huber baseline)

**Expected Path**:
```
Huber (47.378%) 
  → Quantile-Huber (~45-46%, -2pts) 
    → Multi-Task (~42-43%, -3pts) 
      → Ensemble (<40%, -2pts) ✅
```

---

## 📁 FILES CREATED

All files are in `try3/implementation/`:

1. ✅ **train_quantile_huber.py** - Quantile-Huber Loss model
2. ✅ **train_multitask.py** - Multi-Task Learning model  
3. ✅ **ensemble_final.py** - Final ensemble combiner

---

## 🎯 EXECUTION PLAN

### **STEP 1: Train Quantile-Huber Model** (~2.5 hours)

**What it does**:
- Combines Quantile Loss (asymmetric, SMAPE-aware) + Huber Loss (outlier-robust)
- Better optimization for SMAPE metric than pure Huber
- Expected: 47.4% → 45-46% SMAPE

**Run command**:
```bash
cd try3/implementation
python train_quantile_huber.py
```

**Expected output**:
```
📊 CROSS-VALIDATION RESULTS
Fold 1  45.234%
Fold 2  45.891%
...
OOF     45.XXX%

📊 COMPARISON:
Huber Loss Baseline:     47.378%
Quantile-Huber:          45.XXX% (BETTER by X.XX% ✅)
```

**Output files** (in `outputs/distilbert_quantile_huber/`):
- `best_model_fold1.pt` to `best_model_fold5.pt` (model checkpoints)
- `oof_predictions.csv` (out-of-fold predictions for ensemble)
- `test_predictions.csv` (test predictions)

---

### **STEP 2: Train Multi-Task Model** (~2.5 hours)

**What it does**:
- Adds auxiliary range classification task (Budget/Mid/Premium)
- Dual-head architecture: shared DistilBERT → range classifier + price regressor
- Soft probabilities guide regression (not hard like two-stage!)
- Expected: 45-46% → 42-43% SMAPE

**Run command**:
```bash
python train_multitask.py
```

**Expected output**:
```
📊 CROSS-VALIDATION RESULTS
Fold 1  42.456%
Fold 2  43.123%
...
OOF     42.XXX%

Range Classification Accuracy:
Fold 1  78.45%
...

📊 COMPARISON:
Huber Loss Baseline:     47.378%
Quantile-Huber:          ~45-46%
Multi-Task (THIS):       42.XXX% (EXCELLENT! ✅✅)
```

**Output files** (in `outputs/distilbert_multitask/`):
- `best_model_fold1.pt` to `best_model_fold5.pt`
- `oof_predictions.csv`
- `test_predictions.csv`

**Key differences from two-stage**:
- ✅ Uses soft probabilities, not hard classification
- ✅ Joint training, not sequential pipeline
- ✅ Range task is auxiliary (10% loss weight), not primary
- ✅ No error cascading from misclassification

---

### **STEP 3: Create Final Ensemble** (~30 minutes)

**What it does**:
- Combines all 3 models: Huber + Quantile-Huber + Multi-Task
- Optimizes weights using 4 methods:
  1. Simple averaging (equal weights)
  2. Inverse SMAPE weighting (better models get higher weight)
  3. Grid search (1000 random trials)
  4. Scipy optimization (gradient-based)
- Selects best ensemble method
- Expected: 42-43% → 38-39% SMAPE

**Run command**:
```bash
python ensemble_final.py
```

**Expected output**:
```
📊 INDIVIDUAL MODEL SCORES
huber                47.378%
quantile_huber       45.XXX%
multitask            42.XXX%

🔍 Searching optimal weights (1000 trials)...

🏆 FINAL RESULTS
All ensemble methods:
  simple       40.234%
  inverse      39.876%
  grid         39.123%
  scipy        38.956% ✅ BEST

🏆 Best method: SCIPY
🎯 Best SMAPE:  38.956%

Best weights:
  huber                0.150
  quantile_huber       0.300
  multitask            0.550

📊 IMPROVEMENT OVER BASELINE:
  Baseline (Huber):  47.378%
  Ensemble (Best):   38.956%
  Improvement:       8.422%

🎉🎉🎉 SUCCESS! SMAPE < 40% ✅✅✅
```

**Output files** (in `outputs/ensemble_final/`):
- `oof_predictions.csv` (OOF ensemble predictions)
- `test_predictions.csv` (final submission file!)
- `ensemble_config.csv` (weights and scores)

---

## 📊 MONITORING PROGRESS

### During training, watch for:

**✅ Good signs**:
- Validation SMAPE decreasing each epoch
- Early stopping triggers (model converged)
- OOF SMAPE better than previous models
- No NaN losses or predictions

**⚠️ Warning signs**:
- Loss stuck or increasing
- SMAPE worse than baseline → investigate
- OOM errors → reduce batch size
- Very slow training → check GPU usage

### Quick checks:

```bash
# Check if models finished training
ls -lh try3/outputs/distilbert_quantile_huber/best_model_fold*.pt
ls -lh try3/outputs/distilbert_multitask/best_model_fold*.pt

# Check OOF scores in each model's stdout
grep "OOF" quantile_huber.log
grep "OOF" multitask.log

# Verify predictions exist
wc -l try3/outputs/*/oof_predictions.csv
wc -l try3/outputs/*/test_predictions.csv
```

---

## 🎯 EXPECTED TIMELINE

| Step | Task | Duration | Cumulative |
|------|------|----------|------------|
| 1 | Quantile-Huber training | 2.5 hrs | 2.5 hrs |
| 2 | Multi-Task training | 2.5 hrs | 5.0 hrs |
| 3 | Ensemble optimization | 0.5 hrs | 5.5 hrs |
| **TOTAL** | **All steps** | **~5.5 hours** | - |

**Timeline breakdown**:
- 5 folds × ~30 min/fold = 2.5 hours per model
- Ensemble has no training, just weight optimization
- Total: 5.5 hours from start to submission file!

---

## 🔧 TROUBLESHOOTING

### Model 1 (Quantile-Huber) worse than baseline?

**Possible causes**:
- Quantile parameter not optimal (default: 0.5)
- Weight balance wrong (default: 0.5 quantile, 0.5 huber)

**Fix**:
```python
# In train_quantile_huber.py, try:
'quantile': 0.6,  # Slightly overpredict (SMAPE asymmetric)
'quantile_weight': 0.6,  # More weight on quantile loss
```

### Model 2 (Multi-Task) not improving?

**Possible causes**:
- Classification task overwhelming regression
- Range boundaries not optimal

**Fix**:
```python
# In train_multitask.py, try:
'task_weights': {
    'regression': 0.95,      # Even more focus on regression
    'classification': 0.05   # Less weight on auxiliary task
}
# Or adjust ranges:
'range_thresholds': [15.0, 40.0],  # Different splits
```

### OOM (Out of Memory) errors?

**Fix**:
```python
# Reduce batch size in CONFIG:
'batch_size': 16,  # Was 32
'gradient_accumulation_steps': 4,  # Was 2 (keeps effective batch same)
```

### Training too slow?

**Check**:
```bash
# Verify GPU is being used
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**If GPU not available**:
- Set `'fp16': False` in CONFIG (CPU doesn't support mixed precision)
- Expect 3-4× longer training time

---

## 📈 NEXT STEPS AFTER ENSEMBLE

### If SMAPE < 40% ✅
1. **Submit** `try3/outputs/ensemble_final/test_predictions.csv`
2. Celebrate! 🎉
3. Document results in submission folder
4. Update README with final scores

### If SMAPE 40-42% (close!)
1. **Hyperparameter tuning**:
   - Adjust quantile (0.5 → 0.55 or 0.6)
   - Adjust task weights (0.9/0.1 → 0.95/0.05)
   - Adjust range thresholds ($20/$50 → $15/$40)

2. **Add 4th model**:
   - Try MAE Loss (Mean Absolute Error)
   - Try different quantiles (0.4, 0.6, 0.7)
   - More diversity = better ensemble

3. **Advanced techniques**:
   - Stacking (train meta-model on OOF predictions)
   - Blending (optimize on separate validation set)

### If SMAPE > 42% (investigate!)
1. **Check logs** for training issues
2. **Verify data** - no leakage, correct preprocessing
3. **Compare predictions** - are they reasonable?
4. **Error analysis** by price range

---

## 📝 LOGGING BEST PRACTICES

Save stdout to logs for later analysis:

```bash
# Model 1
python train_quantile_huber.py 2>&1 | tee quantile_huber.log

# Model 2
python train_multitask.py 2>&1 | tee multitask.log

# Ensemble
python ensemble_final.py 2>&1 | tee ensemble.log
```

---

## 🎓 KEY INSIGHTS

### Why this works:

1. **Diversity**: 3 different loss functions capture different aspects
   - Huber: Robust to outliers
   - Quantile-Huber: Asymmetric, SMAPE-aware
   - Multi-Task: Structure-aware via auxiliary task

2. **No error cascading**: Unlike two-stage, errors don't compound
   - Two-stage: 25% misclassified → wrong regression head → 52-54% SMAPE
   - Multi-Task: Soft probabilities → no hard decisions → 42-43% SMAPE

3. **Complementary strengths**: Ensemble captures best of all
   - Huber excels on mid-range prices
   - Quantile-Huber better on extremes
   - Multi-Task understands price structure

### Why two-stage failed:

- ❌ Hard classification boundaries ($19.99 vs $20.01)
- ❌ Error cascading (25% misclassified → wrong head)
- ❌ Independent regression heads (no shared info)
- ❌ Classification accuracy 74-75% insufficient

### Why multi-task succeeds:

- ✅ Soft probabilities (no hard cutoffs)
- ✅ Joint training (shared representation)
- ✅ Auxiliary task (10% weight, not primary)
- ✅ No error cascading

---

## ✅ SUCCESS CRITERIA

**Primary goal**: OOF SMAPE < 40%

**Milestones**:
- ✅ Model 1 (Quantile-Huber): < 46% (2-point gain)
- ✅ Model 2 (Multi-Task): < 43% (3-point gain)
- ✅ Model 3 (Ensemble): < 40% (2-point gain)

**If achieved**:
- Document approach in submission/Documentation.md
- Update model_card.md with ensemble details
- Submit test_predictions.csv
- Celebrate! 🎉🎉🎉

---

## 🚀 QUICK START (TL;DR)

```bash
cd try3/implementation

# Step 1: Train Quantile-Huber (~2.5 hrs)
python train_quantile_huber.py 2>&1 | tee quantile_huber.log

# Step 2: Train Multi-Task (~2.5 hrs)
python train_multitask.py 2>&1 | tee multitask.log

# Step 3: Create Ensemble (~30 min)
python ensemble_final.py 2>&1 | tee ensemble.log

# Check final score
grep "Final OOF SMAPE" ensemble.log

# Submit
cp try3/outputs/ensemble_final/test_predictions.csv submission/test_out.csv
```

**Total time**: ~5.5 hours from start to finish!

**Expected result**: SMAPE < 40% ✅

---

Good luck! 🚀
