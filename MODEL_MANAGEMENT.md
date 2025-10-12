# Model Management Guide
**Managing Large Model Files for Amazon ML Hackathon**

## 📋 Summary

Your DistilBERT model files are **too large** for GitHub (268MB+). This guide shows you how to:
1. ✅ Exclude models from Git
2. 📦 Share models via cloud storage
3. 💾 Download models to your local machine

---

## 🚫 What's Excluded from Git (Already in .gitignore)

### Model Files (~300MB each)
```
try2/modeling/distilbert_sagemaker/
├── final_model/
│   ├── model.safetensors          # 268MB - Main model weights
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── tokenizer.json
├── checkpoints/                   # Training checkpoints (large)
├── logs/                          # TensorBoard logs
└── training_results.json          # ✅ This WILL be committed (small)
```

### Other Large Files
- Embeddings data (`.npy`, `.pkl`)
- Training checkpoints
- All prediction CSVs except reference ones
- CatBoost model files

---

## ✅ What WILL Be Committed to Git

### Code & Scripts
- `finetune_distilbert_sagemaker.py` ✅
- `requirements_sagemaker.txt` ✅
- All Python scripts

### Documentation
- `SAGEMAKER_SETUP.md` ✅
- `MODEL_MANAGEMENT.md` ✅
- Training results JSON ✅

### Small Reference Files
- `training_results.json` ✅
- Feature lists ✅
- Config files ✅

---

## 📤 Option 1: Share Models via Hugging Face Hub (Recommended)

### On SageMaker (After Training):

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Upload your model
cd ~/Smart-Product-Pricing-Challenge/try2/modeling/distilbert_sagemaker/final_model

# Create and upload to your HF account
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Upload model
api.upload_folder(
    folder_path='.',
    repo_id='YOUR_USERNAME/amazon-ml-distilbert-price-predictor',
    repo_type='model'
)
"
```

### On Your Local Machine:

```bash
# Download from Hugging Face
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='YOUR_USERNAME/amazon-ml-distilbert-price-predictor',
    local_dir='try2/modeling/distilbert_sagemaker/final_model'
)
"
```

---

## 📤 Option 2: Share Models via AWS S3

### On SageMaker (Upload):

```bash
# Create S3 bucket (one-time)
aws s3 mb s3://amazon-ml-hackathon-models-YOUR_NAME

# Upload model
cd ~/Smart-Product-Pricing-Challenge/try2

# Upload entire model directory
aws s3 sync \
    modeling/distilbert_sagemaker/final_model \
    s3://amazon-ml-hackathon-models-YOUR_NAME/distilbert_sagemaker/final_model/ \
    --exclude "checkpoints/*" \
    --exclude "logs/*"

# Upload predictions
aws s3 cp \
    modeling/test_out_distilbert_sagemaker.csv \
    s3://amazon-ml-hackathon-models-YOUR_NAME/predictions/
```

### On Your Local Machine (Download):

```powershell
# Install AWS CLI if not already installed
# Download from: https://aws.amazon.com/cli/

# Configure AWS credentials
aws configure

# Download model
cd C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\try2

# Download model files
aws s3 sync `
    s3://amazon-ml-hackathon-models-YOUR_NAME/distilbert_sagemaker/final_model/ `
    modeling/distilbert_sagemaker/final_model/

# Download predictions
aws s3 cp `
    s3://amazon-ml-hackathon-models-YOUR_NAME/predictions/test_out_distilbert_sagemaker.csv `
    modeling/
```

---

## 📤 Option 3: Share Models via Google Drive

### On SageMaker (Upload):

```bash
# Install gdrive CLI
wget -O gdrive https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
chmod +x gdrive
sudo mv gdrive /usr/local/bin/

# Authenticate
gdrive about

# Upload model folder (creates a zip)
cd ~/Smart-Product-Pricing-Challenge/try2/modeling
tar -czf distilbert_sagemaker.tar.gz distilbert_sagemaker/final_model/

# Upload to Google Drive
gdrive upload distilbert_sagemaker.tar.gz
# Copy the file ID from output
```

### On Your Local Machine (Download):

1. **Via Browser:**
   - Go to shared Google Drive link
   - Download `distilbert_sagemaker.tar.gz`
   - Extract to: `try2/modeling/`

2. **Via gdown:**
   ```powershell
   pip install gdown
   cd try2\modeling
   gdown FILE_ID  # Replace with your file ID
   tar -xzf distilbert_sagemaker.tar.gz
   ```

---

## 📤 Option 4: Direct SCP/SFTP (If SageMaker allows)

### From SageMaker to Local (via SCP):

```powershell
# On your local machine (Windows PowerShell)
scp -r `
    sagemaker-user@YOUR_SAGEMAKER_IP:~/Smart-Product-Pricing-Challenge/try2/modeling/distilbert_sagemaker/final_model `
    C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code\try2\modeling\distilbert_sagemaker\
```

---

## 🔄 Git Workflow After .gitignore Update

### On SageMaker:

```bash
cd ~/Smart-Product-Pricing-Challenge

# Check what will be committed
git status

# Add your changes (models will be automatically excluded)
git add .

# Commit
git commit -m "Add DistilBERT SageMaker training script and results (53.78% SMAPE)"

# Push to GitHub
git push origin feature_engineering
```

### On Your Local Machine:

```powershell
cd C:\Users\param\Core\Code\Hackathon\Amazon_ML_hackathon\code

# Pull latest changes
git pull origin feature_engineering

# Download model separately (use one of the options above)
# Models are NOT in Git, so download from S3/HF/Drive
```

---

## 📊 What's Already Saved on SageMaker

```
/home/sagemaker-user/Smart-Product-Pricing-Challenge/try2/modeling/
├── distilbert_sagemaker/
│   ├── final_model/                    # 📦 ~300MB - Upload to cloud
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── tokenizer.json
│   ├── training_results.json           # ✅ In Git (3KB)
│   └── logs/                           # 📊 TensorBoard logs
├── test_out_distilbert_sagemaker.csv   # 📦 Upload to cloud
└── [other model outputs]
```

---

## 🎯 Recommended Workflow

1. **Finish training on SageMaker** ✅ (Done!)

2. **Upload model to Hugging Face Hub:**
   ```bash
   # On SageMaker
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload YOUR_USERNAME/amazon-ml-distilbert \
       ~/Smart-Product-Pricing-Challenge/try2/modeling/distilbert_sagemaker/final_model
   ```

3. **Commit code to Git:**
   ```bash
   # On SageMaker
   git add .
   git commit -m "Add DistilBERT training: 53.78% SMAPE (9.5% improvement)"
   git push
   ```

4. **Download on local:**
   ```powershell
   # Pull code
   git pull

   # Download model from HF
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='YOUR_USERNAME/amazon-ml-distilbert', local_dir='try2/modeling/distilbert_sagemaker/final_model')"
   ```

---

## 📝 Create Model Card (Optional but Recommended)

Save this as `try2/modeling/distilbert_sagemaker/MODEL_CARD.md`:

```markdown
# Amazon ML Product Price Predictor - DistilBERT

## Model Description
Fine-tuned DistilBERT model for predicting product prices based on catalog content.

## Performance
- **SMAPE:** 53.78%
- **Baseline SMAPE:** 63.28%
- **Improvement:** 9.50%
- **Training Samples:** 45,000
- **Validation Samples:** 11,250
- **Test Samples:** 18,750

## Training Details
- **Base Model:** distilbert-base-uncased
- **Max Sequence Length:** 128
- **Batch Size:** 32 (effective: 64 with gradient accumulation)
- **Learning Rate:** 2e-5
- **Epochs:** 5
- **Training Time:** ~12 minutes on Tesla T4 GPU

## Usage
\`\`\`python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model")

text = "Your product description here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    price = model(**inputs).logits.item()
print(f"Predicted price: ${price:.2f}")
\`\`\`

## Model Files
- `model.safetensors`: 268MB
- `config.json`: Model configuration
- `tokenizer files`: Vocabulary and tokenizer config

## Download
- Hugging Face: `YOUR_USERNAME/amazon-ml-distilbert`
- AWS S3: `s3://amazon-ml-hackathon-models/distilbert_sagemaker/`
```

---

## 🔍 Check What Will Be Committed

```bash
# On SageMaker - before committing
cd ~/Smart-Product-Pricing-Challenge

# See what's staged (should NOT include model files)
git status

# See file sizes (models should not appear)
git ls-files | xargs du -sh

# If models appear, they're not in .gitignore properly
```

---

## ⚠️ If You Accidentally Committed Large Files

```bash
# Remove from Git history (dangerous!)
git filter-branch --tree-filter 'rm -rf try2/modeling/distilbert_sagemaker/final_model' HEAD

# Or use BFG Repo-Cleaner (safer)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
```

---

## 📞 Need Help?

**If model upload fails:**
1. Check file sizes: `du -sh try2/modeling/distilbert_sagemaker/final_model/`
2. Compress before upload: `tar -czf model.tar.gz final_model/`
3. Use AWS S3 (most reliable for large files)

**If Git complains about large files:**
1. Make sure `.gitignore` is up to date
2. Run: `git rm --cached -r try2/modeling/distilbert_sagemaker/`
3. Commit the removal, then re-add with proper .gitignore

---

## ✅ Quick Commands Summary

**Upload to Hugging Face:**
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/model-name ./final_model
```

**Upload to S3:**
```bash
aws s3 sync ./final_model s3://bucket-name/path/
```

**Download to Local:**
```powershell
# From HF
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('USER/model', 'local_dir')"

# From S3
aws s3 sync s3://bucket-name/path/ ./final_model/
```

**Commit Code Only:**
```bash
git add .
git commit -m "Add DistilBERT training (53.78% SMAPE)"
git push
```

Good luck! 🚀
