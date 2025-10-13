# Quick Test Commands for DistilBERT Huber Model

## Test on SageMaker (Linux)
```bash
cd Smart-Product-Pricing-Challenge/try3/implementation
python test_huber_model.py
```

## Test on Local Windows (if you have the files)
```powershell
cd C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\try3\implementation
python test_huber_model.py
```

## What This Script Does

1. ✅ Loads OOF predictions from `distilbert_huber/oof_predictions.csv`
2. ✅ Calculates overall SMAPE
3. ✅ Shows per-fold SMAPE (if available)
4. ✅ Shows per-range SMAPE (Budget/Mid/Luxury)
5. ✅ Error analysis (worst predictions, quantiles)
6. ✅ Test predictions validation
7. ✅ Saves report to `distilbert_huber/model_test_report.txt`

## Expected Output

```
================================================================================
🧪 Testing DistilBERT Huber Model
================================================================================

📂 Loading predictions...
✓ OOF predictions loaded: 75,000 samples
✓ Test predictions loaded: 75,000 samples

================================================================================
📊 OVERALL PERFORMANCE
================================================================================
OOF SMAPE: 47.378%

================================================================================
📊 PER-FOLD PERFORMANCE
================================================================================
Fold 0:  47.234% (15,000 samples)
Fold 1:  47.456% (15,000 samples)
Fold 2:  47.289% (15,000 samples)
Fold 3:  47.512% (15,000 samples)
Fold 4:  47.399% (15,000 samples)

Mean:    47.378%
Std:     0.112%

================================================================================
📊 PERFORMANCE BY PRICE RANGE
================================================================================
Budget_Economy       ($  0-$ 20):  35.23%  (47,869 samples,  63.8%)
Mid_Premium          ($ 20-$100):  52.45%  (25,238 samples,  33.7%)
Luxury               ($100-$999):  78.92%   (1,893 samples,   2.5%)

================================================================================
✅ SUMMARY
================================================================================
✓ Model tested successfully
✓ OOF SMAPE: 47.378%
📍 Good progress. Gap to 40%: 7.378%
```

## Why Test This?

**Before continuing with Multi-Task (which is failing at 54%):**
- Verify Huber baseline is actually 47.378%
- Check if predictions are valid
- See which price ranges need improvement
- Confirm we have a solid baseline to build on

## Next Steps After Testing

Based on test results:
- If Huber is **~47.4%**: Good baseline ✅
- If Multi-Task is **~54%**: BAD approach ❌

**Recommended:** Stop Multi-Task, use alternative approaches:
1. Quantile Regression (expected: 45-46%)
2. Test-Time Augmentation (expected: 46.5%)
3. Ensemble different approaches

**Goal:** Get to <40% SMAPE today!
