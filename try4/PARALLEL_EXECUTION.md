# 🚀 Parallel Execution Guide

## 🎯 Can We Run Stages in Parallel?

**Short Answer:** Partially - Stages 1 & 2 can run in parallel, but Stages 3-6 must run sequentially.

---

## 📊 Dependency Graph

```
┌─────────────┐                    ┌──────────────┐
│  Stage 1    │                    │   Stage 3    │
│  DeBERTa    │────┐               │   Tabular    │
│ (25-35 min) │    │               │  Features    │
└─────────────┘    │               │  (3-5 min)   │
                   ├──→ Combine ──→└──────────────┘
┌─────────────┐    │                      │
│  Stage 2    │    │                      ↓
│   CLIP      │────┘               ┌──────────────┐
│  (8-12 min) │                    │   Stage 4    │
└─────────────┘                    │   Outlier    │
                                   │  Detection   │
   ⚡ PARALLEL ⚡                    │  (3-5 min)   │
                                   └──────────────┘
                                          │
                                          ↓
                                   ┌──────────────┐
                                   │   Stage 5    │
                                   │   Outlier    │
                                   │  Treatment   │
                                   │  (3-5 min)   │
                                   └──────────────┘
                                          │
                                          ↓
                                   ┌──────────────┐
                                   │   Stage 6    │
                                   │   Training   │
                                   │ (30-45 min)  │
                                   └──────────────┘

                                   ⚙️  SEQUENTIAL ⚙️
```

---

## ⚡ Parallel Execution Options

### Option 1: Automated Parallel Script (Recommended)

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
./run_parallel.sh
```

**What it does:**
- Automatically runs Stage 1 & 2 in parallel (background processes)
- Waits for both to complete
- Then runs Stages 3-6 sequentially
- Logs output to `logs/deberta.log` and `logs/clip.log`

**Time saved:** ~5-10 minutes (depends on which stage finishes first)

---

### Option 2: Manual Parallel Execution (3 Terminals)

**Terminal 1: DeBERTa**
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python feature_extraction/deberta_embeddings.py
```

**Terminal 2: CLIP**
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python feature_extraction/clip_embeddings.py
```

**Terminal 3: Monitor & Continue**
```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# Wait for both to finish, then continue
watch -n 10 'ls -lh outputs/embeddings/*.csv'

# Once both embeddings exist, run sequential stages
python feature_extraction/tabular_features.py
python preprocessing/outlier_detection.py
python preprocessing/outlier_treatment.py
python modeling/fusion_model.py
```

---

### Option 3: Background Jobs (Single Terminal)

```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# Start both in background
python feature_extraction/deberta_embeddings.py > logs/deberta.log 2>&1 &
DEBERTA_PID=$!

python feature_extraction/clip_embeddings.py > logs/clip.log 2>&1 &
CLIP_PID=$!

# Monitor progress
echo "DeBERTa PID: $DEBERTA_PID"
echo "CLIP PID:    $CLIP_PID"

# Check logs
tail -f logs/deberta.log  # Ctrl+C to exit
tail -f logs/clip.log     # Ctrl+C to exit

# Wait for both to complete
wait $DEBERTA_PID
wait $CLIP_PID

# Continue with sequential stages
python feature_extraction/tabular_features.py
python preprocessing/outlier_detection.py
python preprocessing/outlier_treatment.py
python modeling/fusion_model.py
```

---

## 📊 Time Comparison

### Sequential Execution (default):
```
Stage 1: DeBERTa        25-35 min
  ↓
Stage 2: CLIP            8-12 min  (local images)
  ↓
Stage 3-5: Features     10-15 min
  ↓
Stage 6: Training       30-45 min
────────────────────────────────
Total:                  30-50 min
```

### Parallel Execution:
```
Stage 1 (DeBERTa)  │ Stage 2 (CLIP)
25-35 min          │ 8-12 min
                   │
Max(25-35, 8-12) = 25-35 min (CLIP finishes first)
  ↓
Stage 3-5: Features     10-15 min
  ↓
Stage 6: Training       30-45 min
────────────────────────────────
Total:                  22-40 min  ⚡ 8-10 min faster!
```

**Time saved: 8-10 minutes** (CLIP finishes much earlier than DeBERTa)

---

## 🔍 Why Can't All Stages Run in Parallel?

### Dependencies:

1. **Stage 3 (Tabular)** needs:
   - DeBERTa embeddings ✅
   - CLIP embeddings ✅
   - Raw data

2. **Stage 4 (Outliers)** needs:
   - Stage 3 output (combined features)

3. **Stage 5 (Treatment)** needs:
   - Stage 4 output (outlier labels)

4. **Stage 6 (Training)** needs:
   - Stage 5 output (treated features)

**Conclusion:** Stages 3-6 form a sequential chain.

---

## 💡 GPU Considerations

### Single GPU (ml.g5.xlarge, ml.g5.2xlarge):
**Problem:** DeBERTa and CLIP both use GPU
- DeBERTa: ~8-10GB VRAM
- CLIP: ~2-3GB VRAM
- **Total: ~10-13GB VRAM** (fits on 24GB A10G ✅)

**Recommendation:** 
- ✅ Safe to run in parallel on ml.g5.2xlarge (24GB VRAM)
- ⚠️ May cause OOM on smaller instances

**Optimization:**
```python
# In config/config.py (already optimized)
TEXT_MODEL = {
    'batch_size': 32,  # Moderate batch size
}
IMAGE_MODEL = {
    'batch_size': 128,  # Larger batch size (CLIP uses less VRAM)
}
```

---

## 🚀 Recommended Workflow

### For ml.g5.2xlarge (24GB VRAM):

**Use automated parallel execution:**
```bash
cd ~/Smart-Product-Pricing-Challenge/try4
./run_parallel.sh
```

This safely runs DeBERTa + CLIP in parallel and handles all dependencies.

---

### For ml.g5.xlarge (24GB VRAM):

**Option A: Parallel (may be tight on memory)**
```bash
./run_parallel.sh
```

**Option B: Sequential (safer)**
```bash
./run_fast.sh
```

---

## 📈 Expected Timeline (Parallel on ml.g5.2xlarge)

```
[0:00] Setup & optimization
[0:02] ━━━━━━━━━━━━━━━━━━━━━━━━
[0:02] Phase 1: Parallel Embeddings
[0:03]   ┌─ DeBERTa starts...
[0:03]   └─ CLIP starts...
[0:12]   ✓ CLIP finishes (8 min)
[0:32]   ✓ DeBERTa finishes (30 min)
[0:32] ━━━━━━━━━━━━━━━━━━━━━━━━
[0:32] Phase 2: Sequential Features
[0:35]   ✓ Tabular (3 min)
[0:38]   ✓ Outliers (3 min)
[0:41]   ✓ Treatment (3 min)
[0:41] ━━━━━━━━━━━━━━━━━━━━━━━━
[0:41] Phase 3: Training (3 folds)
[1:21]   ✓ Training (40 min)
[1:21] ━━━━━━━━━━━━━━━━━━━━━━━━
[1:21] ✅ Complete!
```

**Total: ~40 minutes** (vs 50 min sequential)

---

## ⚠️ Monitoring Parallel Jobs

### Check if jobs are running:
```bash
ps aux | grep python
```

### Monitor GPU usage:
```bash
watch -n 2 nvidia-smi
```

You should see:
```
| Process name                     | GPU Memory |
|----------------------------------|------------|
| python .../deberta_embeddings.py | 8000-10000 MB |
| python .../clip_embeddings.py    | 2000-3000 MB  |
```

### Check logs:
```bash
# DeBERTa progress
tail -f logs/deberta.log

# CLIP progress  
tail -f logs/clip.log

# Both at once
tail -f logs/*.log
```

---

## 🔧 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch sizes in config/config.py
TEXT_MODEL = {
    'batch_size': 16,  # From 32
}
IMAGE_MODEL = {
    'batch_size': 64,  # From 128
}
```

### One stage fails but other continues
```bash
# Check logs
cat logs/deberta.log
cat logs/clip.log

# Manually re-run failed stage
python feature_extraction/deberta_embeddings.py
# or
python feature_extraction/clip_embeddings.py
```

### Kill background jobs
```bash
# List jobs
jobs

# Kill specific job
kill %1  # Kill job 1
kill %2  # Kill job 2

# Or kill by PID
kill $DEBERTA_PID
kill $CLIP_PID
```

---

## 📊 Summary

| Execution Mode | Time | Complexity | GPU Safe? |
|----------------|------|------------|-----------|
| Sequential (run_fast.sh) | 30-50 min | Simple ✅ | Always ✅ |
| **Parallel (run_parallel.sh)** | **22-40 min** | **Medium** | **ml.g5.2xlarge ✅** |
| Manual 3-terminal | 22-40 min | Complex ⚠️ | ml.g5.2xlarge ✅ |

**Recommendation:** Use `./run_parallel.sh` on ml.g5.2xlarge for best speed!

---

## 🎯 Quick Commands

```bash
# Sequential (safe, simple)
./run_fast.sh

# Parallel (8-10 min faster)
./run_parallel.sh

# Manual monitoring
watch -n 2 nvidia-smi                    # GPU usage
watch -n 10 'ls -lh outputs/embeddings/' # Progress
tail -f logs/*.log                       # Real-time logs
```

🚀 **Choose parallel for maximum speed!** 🚀
