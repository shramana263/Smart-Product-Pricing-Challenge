# 🚀 Quick Command Reference - DistilBERT Training

## Virtual Environment

### Activate (if needed)
```powershell
cd C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing
.\venv_distilbert\Scripts\Activate.ps1
```

## Training Commands

### 1. Run Verification (Test Setup)
```powershell
cd try2
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe verify_setup.py
```

### 2. Start Training (Recommended)
```powershell
cd try2
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe finetune_distilbert_simple.py
```

### 3. Start Training (Advanced Version)
```powershell
cd try2
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe finetune_distilbert.py
```

## Monitoring

### Launch TensorBoard
```powershell
cd try2\modeling\distilbert_simple\logs
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe -m tensorboard.main --logdir=.
```
Then open: http://localhost:6006

## Outputs

### Predictions File
```
try2/modeling/test_out_distilbert_simple.csv
```

### Model Checkpoints
```
try2/modeling/distilbert_simple/checkpoint-XXX/
try2/modeling/distilbert_simple/final_model/
```

### TensorBoard Logs
```
try2/modeling/distilbert_simple/logs/
```

## Expected Results

- **Training Time:** 4-6 hours (CPU) or 1-2 hours (GPU)
- **Expected SMAPE:** 50-55%
- **Current Baseline:** 63.28%
- **Expected Improvement:** 8-13%

## Files Created

1. `try2/finetune_distilbert_simple.py` - Main training script
2. `try2/finetune_distilbert.py` - Advanced version
3. `try2/verify_setup.py` - Verification script
4. `DISTILBERT_GUIDE.md` - Complete guide
5. `DISTILBERT_SETUP_SUMMARY.md` - Setup summary
6. `requirements_distilbert.txt` - Dependencies

## Quick Troubleshooting

### Out of Memory
Edit `finetune_distilbert_simple.py`:
```python
BATCH_SIZE = 16  # or 8
MAX_LENGTH = 64  # or 96
```

### Test on Small Dataset
Edit `finetune_distilbert_simple.py`, add after loading data:
```python
train_split = train_split.sample(frac=0.1)  # Use 10%
NUM_EPOCHS = 1
```

### Check Python Version
```powershell
C:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\venv_distilbert\Scripts\python.exe --version
```

## Next Steps After Training

1. Check SMAPE in terminal output
2. Find predictions: `modeling/test_out_distilbert_simple.csv`
3. Compare with V3 baseline (63.28%)
4. If better: Ensemble with XGBoost predictions
5. If not better: Try alternative models (MPNet, RoBERTa)
