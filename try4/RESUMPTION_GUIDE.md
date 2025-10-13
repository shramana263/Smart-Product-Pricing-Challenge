# 🔄 Try4 Pipeline Resumption Guide

## ✅ FIXED: Pipeline Now Supports Resumption!

The Try4 pipeline has been updated to automatically skip completed stages and resume training from the last completed fold.

---

## 🎯 What's Been Fixed

### 1. Stage-Level Resumption (Already Working)
The main pipeline checks if output files exist and skips completed stages:

```
Stage 1: DeBERTa embeddings    → Skips if deberta_train_embeddings.csv exists
Stage 2: CLIP embeddings        → Skips if clip_train_embeddings.csv exists
Stage 3: Tabular features       → Skips if tabular_train_features.csv exists
Stage 4: Outlier detection      → Skips if outlier_detection_results.csv exists
Stage 5: Outlier treatment      → Skips if train_features_treated.csv exists
Stage 6: Model training         → Always runs (now with fold resumption!)
```

### 2. Fold-Level Resumption (NEWLY ADDED) ✨
The fusion model now checks for existing trained folds:

```python
# Checks for existing models:
outputs/models/lightgbm_fold1.txt
outputs/models/lightgbm_fold2.txt
outputs/models/lightgbm_fold3.txt
outputs/models/lightgbm_fold4.txt
outputs/models/lightgbm_fold5.txt
```

**If interrupted:**
- Fold 1-3 trained → Resumes from Fold 4
- Fold 1-2 trained → Resumes from Fold 3
- No folds trained → Starts from Fold 1

---

## 📋 How It Works

### Example: Interrupted After Stage 2

```bash
# First run (interrupted after CLIP):
python main_pipeline.py
# ✅ Stage 1: DeBERTa embeddings - Complete
# ✅ Stage 2: CLIP embeddings - Complete
# ❌ Interrupted! (Ctrl+C or crash)

# Resume:
python main_pipeline.py
# ✓ Stage 1: DeBERTa - SKIPPED (output exists)
# ✓ Stage 2: CLIP - SKIPPED (output exists)
# ▶️ Stage 3: Tabular features - STARTS HERE
# ▶️ Stage 4: Outlier detection
# ▶️ Stage 5: Outlier treatment
# ▶️ Stage 6: Model training
```

### Example: Interrupted During Model Training

```bash
# First run (interrupted during Fold 3):
python main_pipeline.py
# ... (Stages 1-5 complete)
# ✅ Fold 1 trained - Model saved
# ✅ Fold 2 trained - Model saved
# ❌ Interrupted during Fold 3!

# Resume:
python main_pipeline.py
# ✓ Stages 1-5: SKIPPED (outputs exist)
# 💾 Found existing trained folds: [1, 2]
# ✅ Will resume from Fold 3
# ▶️ Fold 3: TRAINING...
# ▶️ Fold 4: Training...
# ▶️ Fold 5: Training...
```

---

## 🔍 Check Resume Status

### See What's Completed:

```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# Check completed stages
echo "=== Completed Stages ==="
ls -lh outputs/embeddings/*.csv 2>/dev/null && echo "✅ Embeddings" || echo "❌ Embeddings"
ls -lh outputs/features/*.csv 2>/dev/null && echo "✅ Features" || echo "❌ Features"
ls -lh outputs/analysis/*.csv 2>/dev/null && echo "✅ Analysis" || echo "❌ Analysis"
ls -lh outputs/models/*.txt 2>/dev/null && echo "✅ Models" || echo "❌ Models"

# Check trained folds
echo ""
echo "=== Trained Folds ==="
ls -lh outputs/models/lightgbm_fold*.txt 2>/dev/null | wc -l || echo "0"
ls outputs/models/lightgbm_fold*.txt 2>/dev/null || echo "None yet"

# Estimate remaining time
COMPLETED_FOLDS=$(ls outputs/models/lightgbm_fold*.txt 2>/dev/null | wc -l || echo 0)
REMAINING_FOLDS=$((5 - COMPLETED_FOLDS))
FOLD_TIME=10  # minutes per fold (approximate)
REMAINING_TIME=$((REMAINING_FOLDS * FOLD_TIME))
echo ""
echo "=== Estimated Time ==="
echo "Completed folds: $COMPLETED_FOLDS/5"
echo "Remaining folds: $REMAINING_FOLDS"
echo "Est. time left: ~$REMAINING_TIME minutes"
```

### Visual Progress Check:

```bash
# Run this in a separate terminal while pipeline runs
watch -n 30 '
echo "=== TRY4 PIPELINE PROGRESS ==="
echo ""
echo "Embeddings:"
ls -lh outputs/embeddings/*.csv 2>/dev/null | tail -n 5
echo ""
echo "Models:"
ls -lh outputs/models/*.txt 2>/dev/null | tail -n 10
echo ""
echo "Predictions:"
ls -lh outputs/predictions/*.csv 2>/dev/null
echo ""
echo "Latest activity:"
ls -lt outputs/**/*.csv outputs/**/*.txt 2>/dev/null | head -n 3
'
```

---

## 🚀 Usage Examples

### Scenario 1: Fresh Start
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py
```
- Runs all 6 stages from scratch
- Trains all 5 folds
- Time: 3-4 hours (ml.g5.xlarge) or 50-70 min (ml.g5.2xlarge)

### Scenario 2: Resume After Interruption
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py
```
- Automatically detects completed work
- Skips completed stages
- Resumes from last incomplete fold
- Time: Depends on where it stopped

### Scenario 3: Force Rerun Specific Stage
```bash
# Delete output to force rerun
rm outputs/embeddings/deberta_train_embeddings.csv
rm outputs/embeddings/deberta_test_embeddings.csv
python main_pipeline.py
```
- Will rerun Stage 1 (DeBERTa)
- Skips other completed stages

### Scenario 4: Force Rerun Model Training Only
```bash
# Delete all fold models
rm outputs/models/lightgbm_fold*.txt
python main_pipeline.py
```
- Skips Stages 1-5 (embeddings/features exist)
- Retrains all 5 folds
- Time: ~30-60 minutes (ml.g5.xlarge)

### Scenario 5: Rerun Specific Fold
```bash
# Delete specific fold model
rm outputs/models/lightgbm_fold3.txt
python main_pipeline.py
```
- Loads Fold 1-2 from disk
- Retrains Fold 3
- Loads/trains Fold 4-5
- Faster than full retrain!

---

## 💡 Pro Tips

### 1. Use Screen for Long Runs
```bash
# Start in screen
screen -S try4
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py

# Detach: Ctrl+A, then D
# Reattach: screen -r try4
```

### 2. Monitor Progress
```bash
# Terminal 1: Run pipeline
python main_pipeline.py

# Terminal 2: Watch progress
watch -n 10 'ls -lh outputs/*/*.csv outputs/*/*.txt | tail -n 20'

# Terminal 3: GPU usage
watch -n 2 nvidia-smi
```

### 3. Save Checkpoints Manually
```bash
# After each stage completes, optionally backup
tar -czf outputs_backup_$(date +%Y%m%d_%H%M%S).tar.gz outputs/
```

### 4. Estimate Completion Time
```bash
# Check fold training time
ls -l --time-style=+%s outputs/models/lightgbm_fold*.txt | \
  awk '{print $6}' | sort -n | \
  awk 'NR>1{print ($0-prev)/60 " minutes"} {prev=$0}'

# Shows time between fold completions
```

---

## 🔍 Troubleshooting

### Issue: Pipeline Doesn't Resume

**Check:**
```bash
# Verify output files exist
ls -lh outputs/embeddings/
ls -lh outputs/models/

# Check file sizes (should be >0)
du -sh outputs/embeddings/*.csv
du -sh outputs/models/*.txt
```

**Fix:**
If files are corrupt (0 bytes), delete and rerun:
```bash
rm outputs/embeddings/deberta_train_embeddings.csv
python main_pipeline.py
```

### Issue: Fold Resumption Not Working

**Check:**
```bash
# Verify model files are valid LightGBM format
file outputs/models/lightgbm_fold*.txt

# Should show: "ASCII text" or similar
```

**Fix:**
If corrupt, delete specific fold:
```bash
rm outputs/models/lightgbm_fold3.txt
python main_pipeline.py
```

### Issue: Pipeline Starts from Beginning

**Reason:** Output directory might be missing or empty

**Fix:**
```bash
# Check output structure
tree outputs/ -L 2

# Should show:
# outputs/
# ├── embeddings/
# ├── features/
# ├── analysis/
# ├── models/
# └── predictions/

# If missing, pipeline will create them automatically
```

---

## 📊 Output Files Reference

### After Stage 1 (DeBERTa):
```
outputs/embeddings/
├── deberta_train_embeddings.csv  (~150MB, 75K rows × 1025 cols)
└── deberta_test_embeddings.csv   (~150MB, 75K rows × 1025 cols)
```

### After Stage 2 (CLIP):
```
outputs/embeddings/
├── clip_train_embeddings.csv     (~115MB, 75K rows × 769 cols)
└── clip_test_embeddings.csv      (~115MB, 75K rows × 769 cols)
```

### After Stage 3 (Tabular):
```
outputs/features/
├── tabular_train_features.csv    (~5MB, 75K rows × 41 cols)
└── tabular_test_features.csv     (~5MB, 75K rows × 41 cols)
```

### After Stage 4 (Outliers):
```
outputs/analysis/
└── outlier_detection_results.csv (~3MB, 75K rows × 10 cols)
```

### After Stage 5 (Treatment):
```
outputs/features/
└── train_features_treated.csv    (~270MB, merged features)
```

### After Stage 6 (Training):
```
outputs/models/
├── lightgbm_fold1.txt            (~5MB each)
├── lightgbm_fold2.txt
├── lightgbm_fold3.txt
├── lightgbm_fold4.txt
└── lightgbm_fold5.txt

outputs/predictions/
├── oof_predictions.csv           (~2MB, validation results)
├── test_predictions.csv          (~2MB, final submission)
└── feature_importance.csv        (~50KB, feature rankings)
```

---

## ✅ Benefits of Resumption

1. **Save Time:** Don't recompute completed work
2. **Save Money:** Shorter runtime = lower AWS costs
3. **Fault Tolerance:** Survive crashes, interruptions, timeouts
4. **Experimentation:** Easily rerun specific stages
5. **Debugging:** Test individual components

---

## 🎉 Ready to Use!

The resumption feature is now active. Just run:

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py
```

The pipeline will automatically:
- ✅ Detect completed stages
- ✅ Skip existing outputs
- ✅ Resume from last incomplete fold
- ✅ Save each fold as it completes
- ✅ Continue from where it left off if interrupted

**No manual intervention needed!** 🚀
