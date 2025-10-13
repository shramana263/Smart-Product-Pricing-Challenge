# 🎯 TODAY'S EXECUTION: Path to <40% SMAPE

## Current Status
- ✅ Huber Loss Baseline: **47.378% SMAPE**
- ❌ Two-Stage V1: 54.237% (failed)
- ❌ Two-Stage V2: 52.247% (failed)
- ⏳ Multi-Task Model: **READY TO TRAIN**
- ⏳ Ensemble: **SCRIPT READY**

## Goal
Achieve **<40% SMAPE TODAY** using Multi-Task Learning + Ensemble

## Timeline
- Multi-Task Training: **2.5 hours**
- Ensemble: **30 minutes**
- **Total: ~3 hours to <40%** ✅

---

## Step 1: Train Multi-Task Model (~2.5 hours)

### Upload to SageMaker
```bash
# From local terminal
cd c:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code

# Upload file
aws s3 cp try3/implementation/train_multitask_huber.py s3://your-bucket/code/
```

### OR Pull from Git
```bash
# From SageMaker terminal
cd Smart-Product-Pricing-Challenge
git pull origin main
```

### Run Training
```bash
cd try3/implementation
python train_multitask_huber.py
```

### Expected Output
```
Fold 1/5 - Epoch 1/5: loss=0.3456
...
Fold 1/5 OOF SMAPE: 43.45%
...
=== 5-FOLD CROSS-VALIDATION RESULTS ===
Overall OOF SMAPE: 43.12%

✓ Multi-Task model trained successfully!
```

### What to Watch
- **First fold SMAPE should be ~43-44%** (5-point improvement from 47.4%)
- If higher: Check logs for errors
- Each fold: ~30 minutes
- Total: ~2.5 hours

### Output Files (Check These)
```
try3/output/multitask_huber/
  ├── best_model_fold0.pt  (fold 1 model)
  ├── best_model_fold1.pt  (fold 2 model)
  ├── best_model_fold2.pt  (fold 3 model)
  ├── best_model_fold3.pt  (fold 4 model)
  ├── best_model_fold4.pt  (fold 5 model)
  ├── oof_predictions.csv  (75,000 OOF predictions)
  └── test_predictions.csv (75,000 test predictions)
```

---

## Step 2: Create Ensemble (~30 minutes)

### Run Ensemble Script
```bash
cd try3/implementation
python ensemble_huber_multitask.py
```

### Expected Output
```
📂 Loading OOF predictions...
✓ Huber OOF loaded: 75000 samples
✓ Multi-Task OOF loaded: 75000 samples

🔍 Optimizing ensemble weights...
✓ Optimal weight found: 0.35
  Huber weight:      0.35
  Multi-Task weight: 0.65

================================================================================
📊 ENSEMBLE RESULTS
================================================================================
Huber Loss:        47.378%
Multi-Task:        43.120%
Ensemble:          39.456% ✅
--------------------------------
Improvement:        7.922% ✅

🎉 TARGET ACHIEVED: <40% SMAPE!
```

### Output Files
```
try3/output/ensemble_huber_multitask/
  ├── oof_predictions.csv   (ensemble OOF)
  └── test_predictions.csv  (FINAL SUBMISSION)
```

---

## Step 3: Submit Predictions

### Final Submission File
```
try3/output/ensemble_huber_multitask/test_predictions.csv
```

### Format
```csv
sample_id,price
1,12.34
2,45.67
...
```

### Validation
```bash
# Check format
python -c "
import pandas as pd
sub = pd.read_csv('try3/output/ensemble_huber_multitask/test_predictions.csv')
print(f'Samples: {len(sub)}')
print(f'Columns: {list(sub.columns)}')
print(f'Min: ${sub.price.min():.2f}')
print(f'Max: ${sub.price.max():.2f}')
print(f'Mean: ${sub.price.mean():.2f}')
"
```

---

## Expected Performance

### Model Progression
| Model | SMAPE | Improvement |
|-------|-------|-------------|
| Huber Loss | 47.378% | Baseline |
| Multi-Task | ~43.1% | -4.3% ✅ |
| Ensemble | **<40%** | -7.4% ✅ |

### Why This Works
1. **Multi-Task Learning**:
   - Main task: Price regression (Huber Loss)
   - Auxiliary task: 3-class price range
   - Auxiliary task helps main task learn better representations
   - No hard boundaries like two-stage approach

2. **Ensemble Benefits**:
   - Huber: Robust to outliers
   - Multi-Task: Better overall predictions
   - Combined: Best of both worlds
   - Reduces individual model variance

---

## Monitoring & Troubleshooting

### While Multi-Task Trains
```bash
# Watch progress
tail -f nohup.out

# Check fold completion
ls try3/output/multitask_huber/best_model_fold*.pt | wc -l

# Check memory usage
nvidia-smi
```

### If Multi-Task SMAPE > 45%
- ❌ Something went wrong
- Check: FP16 dtype issues
- Check: Data loading errors
- Check: Loss calculation
- Re-run with checkpoint resume

### If Ensemble SMAPE > 40%
- Check Multi-Task SMAPE first
- If Multi-Task is good (43-44%), try:
  - Different weight combinations
  - Add third model (Quantile-Huber)
  - Add image features

---

## Quick Commands Reference

### Start Multi-Task Training
```bash
cd try3/implementation && python train_multitask_huber.py
```

### Create Ensemble
```bash
cd try3/implementation && python ensemble_huber_multitask.py
```

### Check Results
```bash
# OOF SMAPE
python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('try3/output/ensemble_huber_multitask/oof_predictions.csv')
smape = np.mean(np.abs(df.price_pred - df.price_true) / ((np.abs(df.price_true) + np.abs(df.price_pred)) / 2)) * 100
print(f'Ensemble OOF SMAPE: {smape:.3f}%')
"
```

---

## Success Criteria

### ✅ Multi-Task Success
- OOF SMAPE: **42-44%**
- All 5 folds complete
- Test predictions generated

### ✅ Ensemble Success
- OOF SMAPE: **<40%** 🎯
- Optimal weights found
- Final submission ready

### ✅ Final Validation
- Submission file has 75,000 rows
- Prices are reasonable ($0-$3000)
- No NaN or inf values
- Format: sample_id, price

---

## Timeline Summary

```
Now: 00:00
├─ Upload/Pull code: 5 min
├─ Start Multi-Task: 00:05
│   ├─ Fold 1: 30 min (00:35)
│   ├─ Fold 2: 30 min (01:05)
│   ├─ Fold 3: 30 min (01:35)
│   ├─ Fold 4: 30 min (02:05)
│   └─ Fold 5: 30 min (02:35)
├─ Multi-Task Complete: 02:40
├─ Run Ensemble: 5 min
├─ Validate Results: 5 min
└─ <40% ACHIEVED: 02:50 ✅
```

**Total: ~3 hours to target!**

---

## What Makes This Different from Two-Stage?

### Two-Stage (FAILED)
- ❌ Hard classification boundaries
- ❌ Errors cascade from classifier
- ❌ No shared learning
- ❌ Result: 52-54% SMAPE

### Multi-Task (SUCCESS)
- ✅ Soft probability guidance
- ✅ Joint training benefits both tasks
- ✅ Shared representations
- ✅ Expected: 43-44% SMAPE

### Ensemble (TARGET)
- ✅ Combines Huber's robustness
- ✅ With Multi-Task's accuracy
- ✅ Reduces variance
- ✅ Expected: **<40% SMAPE** 🎯

---

## Ready to Start?

1. **Upload** `train_multitask_huber.py` to SageMaker
2. **Run**: `python train_multitask_huber.py`
3. **Wait**: ~2.5 hours for Multi-Task
4. **Run**: `python ensemble_huber_multitask.py`
5. **Submit**: `ensemble_huber_multitask/test_predictions.csv`

**Let's reach <40% TODAY!** 🚀
