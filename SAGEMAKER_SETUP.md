# AWS SageMaker Setup Guide for DistilBERT Training

## 🚀 Quick Start on SageMaker Code Editor

### 1. Instance Configuration

**Recommended Instance:** `ml.g4dn.xlarge`
- **GPU:** NVIDIA T4 (16GB VRAM)
- **CPU:** 4 vCPUs
- **Memory:** 16GB RAM
- **Cost:** ~$0.736/hour
- **Storage:** 50-100GB EBS volume

**Alternative Options:**
- **Budget:** `ml.g4dn.large` (8GB VRAM, $0.471/hour) - reduce batch size to 16
- **Performance:** `ml.g5.xlarge` (24GB VRAM, $1.206/hour) - can increase batch size to 48

---

## 📦 Installation Steps

### Step 1: Create SageMaker Code Editor Instance

1. Go to AWS SageMaker Console
2. Navigate to "Code Editor" (formerly SageMaker Studio)
3. Create new space with instance type: `ml.g4dn.xlarge`
4. Storage: 50-100GB (recommended 75GB)

### Step 2: Setup Python Environment

```bash
# Open terminal in Code Editor

# Create conda environment
conda create -n distilbert python=3.10 -y
conda activate distilbert

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements_sagemaker.txt

# Verify GPU access
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Upload Your Data

```bash
# Option A: Upload via UI (drag & drop to Code Editor)

# Option B: Download from S3
aws s3 cp s3://your-bucket/dataset/ ./dataset/ --recursive

# Option C: Clone from Git
git clone https://github.com/your-repo/project.git
cd project/try2
```

---

## 🏃 Running the Training

### Basic Execution

```bash
# Activate environment
conda activate distilbert

# Run training
python finetune_distilbert_sagemaker.py
```

### Monitor Training Progress

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir modeling/distilbert_sagemaker/logs --port 6006

# Access via Code Editor's port forwarding
# Go to: Ports panel → Forward port 6006
```

---

## ⚙️ Configuration Adjustments

### For ml.g4dn.large (8GB VRAM)
Edit `finetune_distilbert_sagemaker.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
GRADIENT_ACCUMULATION_STEPS = 4  # Increase from 2
```

### For ml.g5.xlarge (24GB VRAM)
Edit `finetune_distilbert_sagemaker.py`:
```python
BATCH_SIZE = 48  # Increase from 32
MAX_LENGTH = 256  # Can increase from 128
```

### For Faster Experimentation
```python
NUM_EPOCHS = 3  # Reduce from 5
SAVE_STEPS = 500  # Increase from 200
```

---

## 💾 Checkpoint Management

### Resume from Checkpoint
Training automatically resumes from latest checkpoint if interrupted.

### Manual Resume
```python
# Training will detect and resume automatically
# Checkpoints stored in: modeling/distilbert_sagemaker/checkpoints/
```

### Clear Checkpoints
```bash
# Remove old checkpoints to free space
rm -rf modeling/distilbert_sagemaker/checkpoints/checkpoint-*
```

---

## 📊 Expected Training Time

**On ml.g4dn.xlarge:**
- **1 epoch:** ~30-45 minutes (depends on data size)
- **5 epochs:** ~2.5-4 hours
- **Total cost:** $2-3 per training run

**Tips to reduce time:**
- Reduce `NUM_EPOCHS` to 3
- Increase `GRADIENT_ACCUMULATION_STEPS` (slower per epoch, but more stable)
- Use smaller `MAX_LENGTH` (128 vs 256)

---

## 🔍 Monitoring & Debugging

### Check GPU Utilization
```bash
# Install nvidia tools
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

### Memory Issues?
If you see CUDA out of memory errors:
1. Reduce `BATCH_SIZE` (32 → 16 → 8)
2. Reduce `MAX_LENGTH` (128 → 96 → 64)
3. Disable `FP16` (set `FP16 = False`)
4. Reduce `DATALOADER_NUM_WORKERS` (4 → 2 → 0)

### Training Too Slow?
1. Increase `GRADIENT_ACCUMULATION_STEPS`
2. Reduce `EVAL_STEPS` and `SAVE_STEPS`
3. Use smaller validation set
4. Reduce `DATALOADER_NUM_WORKERS` if CPU bottleneck

---

## 📤 Saving & Exporting Results

### Model Files
Training saves to:
- **Model:** `modeling/distilbert_sagemaker/final_model/`
- **Predictions:** `modeling/test_out_distilbert_sagemaker.csv`
- **Results:** `modeling/distilbert_sagemaker/training_results.json`

### Download Results
```bash
# Zip results
zip -r results.zip modeling/distilbert_sagemaker/

# Download via UI or upload to S3
aws s3 cp results.zip s3://your-bucket/results/
```

---

## 💰 Cost Optimization Tips

### 1. Use Spot Instances (70% savings)
SageMaker Code Editor doesn't support spot, but for training jobs:
```python
# Use SageMaker Training Jobs with spot instances
# See: sagemaker_training_job.py (create separate script)
```

### 2. Stop Instance When Not Training
- Stop Code Editor instance when not using
- Restart and resume from checkpoint

### 3. Use Smaller Instance for Development
- `ml.t3.medium` (CPU-only) for data prep and testing
- Switch to GPU instance only for training

### 4. Batch Multiple Experiments
- Run hyperparameter sweeps in one session
- Test 3-5 configurations before stopping instance

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8
GRADIENT_ACCUMULATION_STEPS = 4  # or 8
```

### Issue: "Disk space full"
**Solution:**
```bash
# Clear checkpoints
rm -rf modeling/*/checkpoints/checkpoint-*

# Clear cache
rm -rf ~/.cache/huggingface/

# Add more storage to instance (stop → modify → restart)
```

### Issue: "ImportError: No module named 'transformers'"
**Solution:**
```bash
# Make sure conda environment is activated
conda activate distilbert
pip install -r requirements_sagemaker.txt
```

### Issue: Training hangs at "Loading checkpoint"
**Solution:**
```bash
# Corrupted checkpoint - remove and restart
rm -rf modeling/distilbert_sagemaker/checkpoints/
```

---

## 📈 Performance Benchmarks

**Expected Results (ml.g4dn.xlarge):**

| Metric | Value |
|--------|-------|
| Training time/epoch | 30-45 min |
| GPU utilization | 80-95% |
| GPU memory | 12-14GB / 16GB |
| Samples/second | ~50-70 |
| Expected SMAPE | 50-55% |

---

## 🔐 Security Best Practices

1. **Don't commit credentials:** Use AWS IAM roles
2. **Use S3 for data:** Don't store large datasets in Git
3. **Enable encryption:** For S3 buckets and EBS volumes
4. **Clean up:** Delete instances after training

---

## 📚 Additional Resources

- [SageMaker Code Editor Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## 🎯 Next Steps After Training

1. **Evaluate results:** Check `training_results.json`
2. **Compare with baseline:** Current baseline is 63.28% SMAPE
3. **Hyperparameter tuning:** Try different learning rates, batch sizes
4. **Ensemble models:** Combine DistilBERT with LightGBM features
5. **Generate submission:** Use `test_out_distilbert_sagemaker.csv`

---

## 💡 Pro Tips

1. **Use tmux/screen:** Prevents training interruption if connection drops
   ```bash
   tmux new -s training
   python finetune_distilbert_sagemaker.py
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

2. **Log everything:** Training script already logs to TensorBoard
   
3. **Test on small subset first:**
   ```python
   # In load_and_prepare_data(), add:
   train_df = train_df.head(1000)  # Quick test
   ```

4. **Save intermediate predictions:** For ensemble later

5. **Version your models:** Add git commit hash to model name

---

**Ready to train? Run:**
```bash
conda activate distilbert
python finetune_distilbert_sagemaker.py
```

Good luck! 🚀
