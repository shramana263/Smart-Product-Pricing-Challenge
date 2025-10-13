# ✅ 3-MODEL PIPELINE CHECKLIST

## 🎯 GOAL: <40% SMAPE

Current: **47.378%** → Target: **<40%** (need 7.4 points improvement)

---

## 📋 PRE-FLIGHT CHECK

- [ ] AWS SageMaker ml.g4dn.xlarge instance running
- [ ] GPU available: `nvidia-smi` shows T4
- [ ] Data files exist:
  - [ ] `dataset/train1.csv`
  - [ ] `dataset/train2.csv`
  - [ ] `dataset/test1.csv`
  - [ ] `dataset/test2.csv`
- [ ] Baseline model trained: `outputs/distilbert_huber/` exists
- [ ] Three new scripts created:
  - [ ] `implementation/train_quantile_huber.py`
  - [ ] `implementation/train_multitask.py`
  - [ ] `implementation/ensemble_final.py`

---

## 🚀 MODEL 1: QUANTILE-HUBER LOSS

**Expected**: 47.4% → 45-46% SMAPE (-2 points)

### Run:
```bash
cd try3/implementation
python train_quantile_huber.py 2>&1 | tee ../logs/quantile_huber.log
```

### Monitor:
- [ ] Training starts without errors
- [ ] GPU utilization >80% (`nvidia-smi`)
- [ ] Each epoch completes in ~6 minutes
- [ ] Validation SMAPE decreasing
- [ ] Early stopping triggers around epoch 3-4

### Verify:
- [ ] 5 model files created: `best_model_fold1.pt` to `fold5.pt`
- [ ] OOF predictions: `oof_predictions.csv` exists
- [ ] Test predictions: `test_predictions.csv` exists
- [ ] Final OOF SMAPE in output

### Success criteria:
```bash
grep "Final OOF SMAPE" ../logs/quantile_huber.log
# Should show: 45.XXX% (better than 47.378%)
```

**✅ If SMAPE < 46%**: Proceed to Model 2  
**⚠️ If SMAPE > 47%**: Check logs, adjust hyperparameters

---

## 🚀 MODEL 2: MULTI-TASK LEARNING

**Expected**: 45-46% → 42-43% SMAPE (-3 points)

### Run:
```bash
python train_multitask.py 2>&1 | tee ../logs/multitask.log
```

### Monitor:
- [ ] Training starts without errors
- [ ] Both tasks shown: Regression + Classification
- [ ] Range classification accuracy ~75-80%
- [ ] Validation SMAPE decreasing
- [ ] Better than Quantile-Huber model

### Verify:
- [ ] 5 model files created
- [ ] OOF predictions exist
- [ ] Test predictions exist
- [ ] Range accuracy reported

### Success criteria:
```bash
grep "Final OOF SMAPE" ../logs/multitask.log
# Should show: 42.XXX% (better than 45-46%)
```

**✅ If SMAPE < 43%**: Proceed to Ensemble  
**⚠️ If SMAPE > 45%**: Check task weights, adjust if needed

---

## 🚀 MODEL 3: FINAL ENSEMBLE

**Expected**: 42-43% → 38-39% SMAPE (-2 points) = **<40% ✅**

### Run:
```bash
python ensemble_final.py 2>&1 | tee ../logs/ensemble.log
```

### Monitor:
- [ ] All 3 model OOFs loaded successfully
- [ ] Individual scores shown
- [ ] 4 ensemble methods tested
- [ ] Best method selected
- [ ] Final SMAPE calculated

### Verify:
- [ ] Ensemble OOF predictions created
- [ ] Ensemble test predictions created
- [ ] Ensemble config saved (weights)

### Success criteria:
```bash
grep "Final OOF SMAPE" ../logs/ensemble.log
# Target: <40.000%
```

**🎉 If SMAPE < 40%**: MISSION ACCOMPLISHED!  
**📈 If SMAPE 40-42%**: Still good, try tuning  
**⚠️ If SMAPE > 42%**: Investigate, check individual models

---

## 📊 RESULTS VERIFICATION

### Check all scores:
```bash
echo "=== MODEL SCORES ==="
echo "Baseline (Huber):"
grep "OOF SMAPE" outputs/distilbert_huber/training.log

echo "Model 1 (Quantile-Huber):"
grep "Final OOF SMAPE" logs/quantile_huber.log

echo "Model 2 (Multi-Task):"
grep "Final OOF SMAPE" logs/multitask.log

echo "Model 3 (Ensemble):"
grep "Final OOF SMAPE" logs/ensemble.log
```

### Expected progression:
```
Baseline:        47.378%
Quantile-Huber:  45.XXX%  (-2 pts) ✅
Multi-Task:      42.XXX%  (-3 pts) ✅
Ensemble:        38.XXX%  (-2 pts) ✅ <40% TARGET!
```

---

## 📁 OUTPUT FILES CHECK

### Model 1 (Quantile-Huber):
```bash
ls -lh outputs/distilbert_quantile_huber/
# Should see:
# - best_model_fold1.pt (250MB each)
# - best_model_fold2.pt
# - best_model_fold3.pt
# - best_model_fold4.pt
# - best_model_fold5.pt
# - oof_predictions.csv (150K rows)
# - test_predictions.csv (150K rows)
```

### Model 2 (Multi-Task):
```bash
ls -lh outputs/distilbert_multitask/
# Same structure as Model 1
```

### Model 3 (Ensemble):
```bash
ls -lh outputs/ensemble_final/
# Should see:
# - oof_predictions.csv
# - test_predictions.csv (FINAL SUBMISSION!)
# - ensemble_config.csv (weights and scores)
```

---

## 🎯 SUBMISSION PREP

### If SMAPE < 40%:

```bash
# Copy final predictions
cp outputs/ensemble_final/test_predictions.csv submission/test_out.csv

# Verify format
head submission/test_out.csv
# Should show: sample_id,price

# Check stats
python -c "
import pandas as pd
df = pd.read_csv('submission/test_out.csv')
print(f'Rows: {len(df):,}')
print(f'Min:  ${df.price.min():.2f}')
print(f'Med:  ${df.price.median():.2f}')
print(f'Mean: ${df.price.mean():.2f}')
print(f'Max:  ${df.price.max():.2f}')
"
```

### Documentation:
- [ ] Update `submission/Documentation.md` with ensemble details
- [ ] Update `submission/model_card.md` with 3-model approach
- [ ] Update `submission/README.md` with final scores
- [ ] Save training logs for reference

---

## 🔧 TROUBLESHOOTING

### Model 1 not improving?
```python
# Edit train_quantile_huber.py:
'quantile': 0.6,          # Try different quantile
'quantile_weight': 0.6,   # More weight on quantile
```

### Model 2 not improving?
```python
# Edit train_multitask.py:
'task_weights': {
    'regression': 0.95,     # More focus on regression
    'classification': 0.05  # Less on auxiliary task
}
```

### OOM errors?
```python
# Reduce batch size:
'batch_size': 16,                  # Was 32
'gradient_accumulation_steps': 4,  # Was 2
```

### Ensemble not improving?
- Check individual model quality
- Ensure OOF files are correct
- Try different ensemble methods (simple/inverse/grid/scipy)

---

## 📈 TIMELINE TRACKING

| Time | Task | Status |
|------|------|--------|
| T+0:00 | Start Model 1 (Quantile-Huber) | ⏳ |
| T+2:30 | Model 1 complete | ⏳ |
| T+2:30 | Start Model 2 (Multi-Task) | ⏳ |
| T+5:00 | Model 2 complete | ⏳ |
| T+5:00 | Start Ensemble | ⏳ |
| T+5:30 | Ensemble complete | ⏳ |
| T+5:30 | **SUBMISSION READY** | 🎯 |

**Total**: ~5.5 hours start to finish

---

## ✅ SUCCESS CRITERIA

### Must have:
- [x] All 3 models trained successfully
- [x] Each model better than previous
- [x] Ensemble SMAPE < 40%
- [x] Test predictions generated
- [x] No NaN or negative predictions

### Nice to have:
- [ ] Training logs saved
- [ ] Error analysis done
- [ ] Documentation updated
- [ ] Results visualized

---

## 🎉 COMPLETION CHECKLIST

When SMAPE < 40%:

- [ ] Final submission file: `submission/test_out.csv`
- [ ] Training logs: `try3/logs/*.log`
- [ ] Model checkpoints: `outputs/distilbert_*/`
- [ ] Documentation updated
- [ ] README updated with final scores
- [ ] Celebrate! 🎉🎉🎉

---

## 📞 QUICK COMMANDS

```bash
# Check current score
tail -20 logs/ensemble.log

# Re-run just ensemble (if tuning weights)
python implementation/ensemble_final.py

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor training
tail -f logs/quantile_huber.log

# Compare all models
grep -h "Final OOF SMAPE" logs/*.log
```

---

**Start here**: `cd try3/implementation && python train_quantile_huber.py`

**End goal**: SMAPE < 40% in `ensemble.log`

**Good luck!** 🚀
