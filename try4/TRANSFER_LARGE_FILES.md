# 📦 Transferring Large Files Between SageMaker Instances

## 🎯 Problem

You have large files on old SageMaker instance:
- **Images:** ~25-30GB (downloaded product images)
- **Embeddings:** ~10GB (DeBERTa, CLIP features)
- **Models:** ~9GB (trained models)

**Total:** ~45-50GB that GitHub cannot handle (100MB file limit)

---

## ⚠️ **IMPORTANT: Different AWS Accounts?**

If your old and new SageMaker instances are in **different AWS accounts**, see [Option 3: Cross-Account Transfer](#option-3-google-drive--dropbox-cross-account) below.

**Same AWS Account?** Use [Option 1: AWS S3](#option-1-aws-s3-transfer-%EF%B8%8F%EF%B8%8F%EF%B8%8F-best-same-account)

---

## 🏆 **Recommended Solutions (By Scenario)**

### **Option 1: AWS S3 Transfer** ⭐⭐⭐ (BEST - SAME ACCOUNT)

**Fastest, most reliable, AWS-native (ONLY if same AWS account)**

⚠️ **This only works if both instances are in the SAME AWS account!**

#### **On Old SageMaker Instance:**

```bash
# Install AWS CLI (usually pre-installed on SageMaker)
aws --version

# Create S3 bucket (one-time)
aws s3 mb s3://your-hackathon-data-bucket

# Sync large files to S3
cd ~/Smart-Product-Pricing-Challenge

# Upload images (25-30GB)
aws s3 sync try2/images/ s3://your-hackathon-data-bucket/images/ \
    --exclude "*.gitkeep" \
    --storage-class STANDARD_IA

# Upload embeddings (10GB)
aws s3 sync try3/outputs/balanced_model/embeddings_cache/ \
    s3://your-hackathon-data-bucket/embeddings/ \
    --storage-class STANDARD_IA

# Upload image features (if already extracted)
aws s3 sync try3/outputs/image_features/ \
    s3://your-hackathon-data-bucket/image_features/ \
    --storage-class STANDARD_IA

# Upload trained models (optional)
aws s3 sync try3/outputs/balanced_model/ \
    s3://your-hackathon-data-bucket/models/ \
    --storage-class STANDARD_IA \
    --exclude "*.csv"
```

#### **On New SageMaker Instance:**

```bash
cd ~/Smart-Product-Pricing-Challenge

# Create directories
mkdir -p try2/images/train try2/images/test
mkdir -p try4/outputs/embeddings
mkdir -p try4/outputs/image_features

# Download images
aws s3 sync s3://your-hackathon-data-bucket/images/ try2/images/

# Download embeddings (optional - can regenerate)
aws s3 sync s3://your-hackathon-data-bucket/embeddings/ \
    try4/outputs/embeddings/

# Download image features (optional)
aws s3 sync s3://your-hackathon-data-bucket/image_features/ \
    try4/outputs/image_features/

# Verify
du -sh try2/images try4/outputs/*
```

**Pros:**
- ✅ Super fast (AWS backbone network)
- ✅ No file size limits
- ✅ Reliable with automatic retries
- ✅ Can resume interrupted transfers
- ✅ Free within same AWS region
- ✅ Data persists even if instances shut down

**Cons:**
- ⚠️ Storage costs (~$1-2/month for 50GB STANDARD_IA)
- ⚠️ Requires AWS CLI setup

**Cost Estimate:**
```
Storage: $0.0125/GB/month × 50GB = $0.625/month
Transfer: FREE (same region)
Total: <$1/month
```

---

### **Option 2: EFS (Elastic File System)** ⭐⭐ (GOOD)

**Shared filesystem between SageMaker instances**

#### **Setup (One-time):**

```bash
# Create EFS in AWS Console
# Attach to same VPC as SageMaker
# Mount on both old and new instances
```

#### **On Old Instance:**

```bash
# Mount EFS
sudo mkdir /mnt/efs
sudo mount -t nfs4 -o nfsvers=4.1 \
    fs-xxxxx.efs.us-east-1.amazonaws.com:/ /mnt/efs

# Copy data
cp -r ~/Smart-Product-Pricing-Challenge/try2/images /mnt/efs/
cp -r ~/Smart-Product-Pricing-Challenge/try3/outputs /mnt/efs/
```

#### **On New Instance:**

```bash
# Mount same EFS
sudo mkdir /mnt/efs
sudo mount -t nfs4 -o nfsvers=4.1 \
    fs-xxxxx.efs.us-east-1.amazonaws.com:/ /mnt/efs

# Copy data locally
cp -r /mnt/efs/images ~/Smart-Product-Pricing-Challenge/try2/
cp -r /mnt/efs/outputs ~/Smart-Product-Pricing-Challenge/try4/
```

**Pros:**
- ✅ Shared filesystem
- ✅ Fast access
- ✅ No manual upload/download

**Cons:**
- ⚠️ More complex setup
- ⚠️ Higher costs (~$0.30/GB/month)
- ⚠️ Requires VPC configuration

---

### **Option 3: Google Drive / Dropbox** ⭐⭐⭐ (BEST - CROSS-ACCOUNT)

**Best for different AWS accounts - free tier available**

#### **Method 3A: Google Drive (15GB Free)**

##### **On Old SageMaker Instance:**

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config
# Choose: n (new remote)
# Name: gdrive
# Storage: drive (Google Drive)
# Follow browser authentication

# Upload images to Google Drive
cd ~/Smart-Product-Pricing-Challenge
rclone copy try2/images/ gdrive:hackathon-images/ \
    --progress \
    --transfers 8 \
    --checkers 16

# For files >15GB, split upload:
rclone copy try2/images/train/ gdrive:hackathon-images/train/ --progress
rclone copy try2/images/test/ gdrive:hackathon-images/test/ --progress
```

##### **On New SageMaker Instance (Different Account):**

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive (use SAME account)
rclone config
# Follow same authentication

# Download images
cd ~/Smart-Product-Pricing-Challenge
mkdir -p try2/images
rclone copy gdrive:hackathon-images/ try2/images/ \
    --progress \
    --transfers 8

# Verify
du -sh try2/images
ls -lh try2/images/train/ | wc -l
```

**Pros:**
- ✅ **Works across different AWS accounts**
- ✅ 15GB free (enough for most images if compressed)
- ✅ No AWS charges
- ✅ Familiar interface
- ✅ Can access from anywhere
- ✅ Resume support

**Cons:**
- ⚠️ Slower than S3 (~1-2 hours for 30GB)
- ⚠️ Need to upgrade for full 30GB ($2/month for 100GB)
- ⚠️ Requires Google account

**Cost:**
```
0-15GB: FREE
15-100GB: $1.99/month (Google One)
100GB-2TB: $9.99/month
```

---

#### **Method 3B: Dropbox (2GB Free, 2TB Paid)**

```bash
# Install Dropbox Uploader
cd ~
git clone https://github.com/andreafabrizi/Dropbox-Uploader.git
cd Dropbox-Uploader
chmod +x dropbox_uploader.sh

# Configure (follow prompts for app token)
./dropbox_uploader.sh

# Upload
cd ~/Smart-Product-Pricing-Challenge
~/Dropbox-Uploader/dropbox_uploader.sh upload try2/images/ /hackathon-images/

# Download on new instance (same configuration)
~/Dropbox-Uploader/dropbox_uploader.sh download /hackathon-images/ try2/images/
```

**Pros:**
- ✅ Works across AWS accounts
- ✅ Simple script-based
- ✅ Resume support

**Cons:**
- ⚠️ Only 2GB free (not enough)
- ⚠️ Need paid plan ($11.99/month for 2TB)

---

#### **Method 3C: Mega.nz (20GB Free)**

```bash
# Install MEGAcmd
cd ~
wget https://mega.nz/linux/repo/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb
sudo dpkg -i megacmd-xUbuntu_22.04_amd64.deb
sudo apt-get install -f

# Login
mega-login your-email@example.com

# Upload
cd ~/Smart-Product-Pricing-Challenge
mega-put -c try2/images/ /hackathon-images/

# Download on new instance
mega-login your-email@example.com
mega-get /hackathon-images/ try2/images/
```

**Pros:**
- ✅ 20GB free (best free tier)
- ✅ Works across accounts
- ✅ Good for images
- ✅ Resume support

**Cons:**
- ⚠️ Slower downloads
- ⚠️ Less reliable than Google Drive
- ⚠️ May need paid for 30GB ($5.40/month for 400GB)

---

#### **Method 3D: WeTransfer (200GB Paid)**

**For quick one-time transfers:**

```bash
# Use web interface or API
# Upload: transfer.wetransfer.com
# Generate link, share with yourself
# Download on new instance
```

**Pros:**
- ✅ No signup for <2GB
- ✅ Very simple
- ✅ Works across accounts

**Cons:**
- ❌ Free: only 2GB
- ❌ Files expire after 7 days
- ⚠️ Need WeTransfer Pro ($12/month) for 200GB

---

### **Option 4: S3 Cross-Account Transfer** ⭐⭐ (ADVANCED)

**Use S3 with cross-account bucket policies**

#### **On Old Account (Account A):**

```bash
# Create bucket with cross-account access
aws s3 mb s3://hackathon-images-transfer

# Add bucket policy for Account B access
cat > policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_B_ID:root"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::hackathon-images-transfer",
                "arn:aws:s3:::hackathon-images-transfer/*"
            ]
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket hackathon-images-transfer --policy file://policy.json

# Upload images
cd ~/Smart-Product-Pricing-Challenge
aws s3 sync try2/images/ s3://hackathon-images-transfer/images/ \
    --storage-class STANDARD_IA
```

#### **On New Account (Account B):**

```bash
# Access bucket from Account A
cd ~/Smart-Product-Pricing-Challenge
mkdir -p try2/images

# Download with Account B credentials
aws s3 sync s3://hackathon-images-transfer/images/ try2/images/ \
    --request-payer requester

# Note: You'll pay for download bandwidth (~$0.09/GB if different region)
```

**Pros:**
- ✅ Fast AWS transfer
- ✅ Reliable

**Cons:**
- ⚠️ Complex setup (IAM policies)
- ⚠️ Requires admin access on Account A
- ⚠️ Data transfer costs if different regions
- ⚠️ Storage costs on Account A

**Cost:**
```
Storage (Account A): $0.40/month for 30GB
Transfer (if different region): $2.70 for 30GB
Total: ~$3
```

---

### **Option 5: Public HTTP Server (Temporary)** ⭐ (QUICK & DIRTY)

**Quick transfer without third-party services**

#### **On Old Instance:**

```bash
cd ~/Smart-Product-Pricing-Challenge/try2/images

# Compress images first
echo "Compressing images..."
tar -czf images.tar.gz train/ test/

# Start HTTP server (Python built-in)
python3 -m http.server 8000

# Get instance public IP
curl http://169.254.169.254/latest/meta-data/public-ipv4
# Copy this IP!

# Keep terminal open
```

#### **On New Instance:**

```bash
cd ~/Smart-Product-Pricing-Challenge/try2

# Download from old instance
wget http://OLD_INSTANCE_IP:8000/images.tar.gz

# Extract
tar -xzf images.tar.gz
rm images.tar.gz

# Verify
du -sh images/
```

#### **Security Note:**

```bash
# On old instance, open security group temporarily:
# AWS Console → EC2 → Security Groups
# Add inbound rule: TCP port 8000 from new instance IP only
# REMOVE RULE AFTER TRANSFER!
```

**Pros:**
- ✅ No third-party service
- ✅ Fast if same region
- ✅ Free
- ✅ Direct transfer

**Cons:**
- ❌ Requires security group changes
- ❌ Less secure
- ❌ Manual process
- ❌ Not recommended for production

**Time:** ~1-2 hours for 30GB

---

### **Option 6: Regenerate Everything (Just Re-download)** ⭐⭐⭐ (SIMPLEST - CROSS-ACCOUNT)

**Best option if time > cost**

**Use Git Large File Storage**

```bash
# On old instance
cd ~/Smart-Product-Pricing-Challenge
git lfs install
git lfs track "*.npy" "*.npz" "*.jpg" "*.png"
git add .gitattributes
git lfs push origin feature_engineering

# On new instance
git clone https://github.com/shramana263/Smart-Product-Pricing-Challenge.git
cd Smart-Product-Pricing-Challenge
git lfs pull
```

**Pros:**
- ✅ Version controlled
- ✅ Works with GitHub

**Cons:**
- ❌ GitHub LFS: 1GB free, then $5/50GB/month
- ❌ Slow for 50GB
- ❌ Can hit bandwidth limits

---

### **Option 4: Direct Instance-to-Instance Transfer** (ADVANCED)

**Using SSH/SCP between instances**

```bash
# On new instance, get IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# On old instance, transfer
scp -r ~/Smart-Product-Pricing-Challenge/try2/images/ \
    sagemaker-user@<new-instance-ip>:~/Smart-Product-Pricing-Challenge/try2/
```

**Pros:**
- ✅ Direct transfer
- ✅ No intermediate storage

**Cons:**
- ❌ Complex security group setup
- ❌ Requires SSH keys
- ❌ Not recommended for SageMaker

---

### **Option 5: Regenerate Everything** ⭐⭐⭐ (SIMPLEST!)

**Just re-download and re-process**

```bash
cd ~/Smart-Product-Pricing-Challenge/try4

# Let the pipeline handle it
python main_pipeline.py

# The pipeline will automatically:
# 1. Download images (~3-4 hours)
# 2. Generate DeBERTa embeddings (~1 hour)
# 3. Extract CLIP features (~30 min)
# 4. Train models (~2 hours)
```

**Pros:**
- ✅ **Simplest solution**
- ✅ No transfer needed
- ✅ No storage costs
- ✅ Always fresh data
- ✅ Good for reproducibility

**Cons:**
- ⚠️ Takes 6-8 hours total
- ⚠️ Uses compute time

**💡 This is actually recommended for Try4!**
- Try4 uses **different models** (DeBERTa-v3-large, CLIP) than Try3
- Can't reuse Try3 embeddings anyway
- Fresh start ensures no compatibility issues

---

## 🎯 **My Recommendation for Your Situation**

### **Scenario A: SAME AWS Account** ✅

Use **AWS S3** (Option 1) - fastest and cheapest

### **Scenario B: DIFFERENT AWS Accounts** ⚠️

**Best Choice: Google Drive + rclone** (Option 3A)

**Why?**
- ✅ Free for first 15GB
- ✅ Only $2/month for 100GB (enough for 30GB images)
- ✅ Works perfectly across AWS accounts
- ✅ Resume support
- ✅ Can access from anywhere
- ✅ Faster than regenerating (saves 3-4 hours)

**Alternative: Just Regenerate** (Option 6)
- ✅ Free
- ✅ No setup
- ⚠️ Takes 6-8 hours total

---

### **Cross-Account Transfer Strategy (RECOMMENDED)**

#### **If Images < 15GB (Compressed):**

```bash
# OLD INSTANCE: Compress and upload to Google Drive FREE
cd ~/Smart-Product-Pricing-Challenge/try2
tar -czf images.tar.gz images/
# This should compress 30GB → ~10-12GB

curl https://rclone.org/install.sh | sudo bash
rclone config  # Setup Google Drive
rclone copy images.tar.gz gdrive:hackathon/

# NEW INSTANCE: Download and extract
curl https://rclone.org/install.sh | sudo bash
rclone config  # Same Google account
rclone copy gdrive:hackathon/images.tar.gz ./
tar -xzf images.tar.gz
mv images try2/
```

**Time:** ~2 hours
**Cost:** FREE (fits in 15GB)

---

#### **If Images > 15GB (Full Size):**

**Option A: Upgrade Google Drive ($2/month)**

```bash
# Pay $1.99 for Google One (100GB)
# Then upload all images without compression
rclone copy try2/images/ gdrive:hackathon-images/ --progress
```

**Time:** ~2-3 hours
**Cost:** $2 (can cancel after transfer)

**Option B: Use Mega.nz (20GB free)**

```bash
# Install MEGAcmd on both instances
# Upload to Mega.nz (free 20GB)
mega-put try2/images/ /hackathon/
```

**Time:** ~3-4 hours
**Cost:** FREE if under 20GB

**Option C: Just Regenerate**

```bash
# On new instance, run pipeline
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py
# Will download images fresh
```

**Time:** ~6-8 hours
**Cost:** FREE

---

### **Hybrid Approach for Cross-Account (BEST VALUE)**

#### **1. Transfer Only Images (saves 3-4 hours):**

```bash
# OLD INSTANCE: Upload images to S3
aws s3 sync ~/Smart-Product-Pricing-Challenge/try2/images/ \
    s3://your-hackathon-images/images/ \
    --storage-class STANDARD_IA

# Takes ~30 minutes for 30GB
```

```bash
# NEW INSTANCE: Download images from S3
cd ~/Smart-Product-Pricing-Challenge
mkdir -p try2/images
aws s3 sync s3://your-hackathon-images/images/ try2/images/

# Takes ~20 minutes
```

#### **2. Regenerate Embeddings (different models anyway):**

```bash
cd ~/Smart-Product-Pricing-Challenge/try4
python main_pipeline.py

# Automatically generates:
# - DeBERTa-v3-large embeddings (different from Try3's DistilBERT)
# - CLIP ViT-Large features (different from Try3's ResNet50)
# - New models trained on new embeddings
```

**Time Saved:**
- Image download: 3-4 hours → 20 minutes = **3+ hours saved**
- Embedding generation: Must do anyway (different models)
- Total time: ~3 hours instead of 6-8 hours

**Cost:**
- S3 storage: ~$0.40/month for 30GB
- S3 transfer: FREE (same region)
- **Total: <$1**

---

## 📋 **Step-by-Step Guide (Recommended)**

### **Phase 1: On Old SageMaker Instance**

```bash
# 1. Create S3 bucket
aws s3 mb s3://hackathon-try4-data-$(date +%s)
# Note: Copy the bucket name!

# 2. Upload images only
cd ~/Smart-Product-Pricing-Challenge
aws s3 sync try2/images/ s3://hackathon-try4-data-XXXXX/images/ \
    --storage-class STANDARD_IA \
    --exclude "*.gitkeep"

# 3. Verify upload
aws s3 ls s3://hackathon-try4-data-XXXXX/images/ --recursive --human-readable

# 4. You can now shut down old instance!
```

### **Phase 2: On New SageMaker Instance**

```bash
# 1. Clone repo
cd ~
git clone https://github.com/shramana263/Smart-Product-Pricing-Challenge.git
cd Smart-Product-Pricing-Challenge

# 2. Setup environment
cd try4
conda activate multimodal  # or your env name
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# 3. Download images from S3
mkdir -p try2/images
aws s3 sync s3://hackathon-try4-data-XXXXX/images/ try2/images/

# 4. Verify
ls -lh try2/images/train/ | head
ls -lh try2/images/test/ | head

# 5. Run pipeline (will generate new embeddings)
python check_system.py
python main_pipeline.py
```

### **Phase 3: Cleanup (After Success)**

```bash
# Delete S3 bucket to stop charges
aws s3 rb s3://hackathon-try4-data-XXXXX --force
```

---

## 🔍 **What to Transfer vs Regenerate?**

| Data Type | Size | Transfer? | Reason |
|-----------|------|-----------|--------|
| **Product Images** | 25-30GB | ✅ **YES** | Saves 3-4 hours download time |
| Dataset CSVs | <1GB | ✅ YES | In git already |
| DistilBERT embeddings | ~4GB | ❌ NO | Try4 uses DeBERTa (different) |
| ResNet50 features | ~5GB | ❌ NO | Try4 uses CLIP (different) |
| Try3 models | ~5GB | ❌ NO | Not compatible with Try4 |
| DeBERTa embeddings | N/A | ❌ NO | Must generate for Try4 |
| CLIP features | N/A | ❌ NO | Must generate for Try4 |

**Summary:** Only transfer **images**, regenerate everything else!

---

## 💰 **Cost Comparison**

| Method | Cost | Time |
|--------|------|------|
| **S3 (images only)** | $0.40/month | Transfer: 50min, Pipeline: 3hrs |
| **S3 (everything)** | $1.50/month | Transfer: 2hrs, Pipeline: <1hr |
| **Regenerate all** | $0 | Pipeline: 6-8hrs |
| **Git LFS** | $5/month | Transfer: 8hrs+ |
| **EFS** | $15/month | Transfer: 1hr, Pipeline: <1hr |

**Recommendation:** S3 for images only = **best balance** ⭐

---

## 🚀 **Quick Start Commands**

### **Upload Images from Old Instance:**
```bash
aws s3 mb s3://hackathon-images-$(whoami)
aws s3 sync ~/Smart-Product-Pricing-Challenge/try2/images/ \
    s3://hackathon-images-$(whoami)/images/
echo "Bucket: hackathon-images-$(whoami)"
```

### **Download Images on New Instance:**
```bash
# Replace BUCKET_NAME with your bucket
aws s3 sync s3://BUCKET_NAME/images/ \
    ~/Smart-Product-Pricing-Challenge/try2/images/
```

### **Delete Bucket After Done:**
```bash
aws s3 rb s3://BUCKET_NAME --force
```

---

## ✅ **Verification Checklist**

After transfer, verify:

```bash
cd ~/Smart-Product-Pricing-Challenge

# Check images
echo "Train images: $(ls try2/images/train/ | wc -l)"
echo "Test images: $(ls try2/images/test/ | wc -l)"
# Should be ~75K each

# Check total size
du -sh try2/images
# Should be ~25-30GB

# Test image loading
python -c "
from PIL import Image
img = Image.open('try2/images/train/$(ls try2/images/train/ | head -1)')
print(f'✅ Image size: {img.size}')
"
```

---

## 🎓 **Best Practice Summary**

1. ✅ **Transfer images** via S3 (saves 3-4 hours)
2. ✅ **Regenerate embeddings** (different models anyway)
3. ✅ **Delete S3 bucket** after successful pipeline run
4. ✅ **Keep receipts** for cost tracking

**Expected Timeline:**
- Image upload (old instance): 30 min
- Image download (new instance): 20 min
- Try4 pipeline: 3-4 hours
- **Total: ~4-5 hours** (vs 6-8 hours regenerating all)

---

## 📞 **Troubleshooting**

### **"AWS CLI not found"**
```bash
pip install awscli
aws configure  # Enter your credentials
```

### **"Access Denied" on S3**
```bash
# Check IAM role has S3 permissions
aws sts get-caller-identity
# Add S3FullAccess policy to SageMaker execution role
```

### **"Slow S3 transfer"**
```bash
# Use parallel uploads
aws s3 sync --exclude "*" --include "*.jpg" --parallel-requests 10
```

### **"Out of disk space on new instance"**
```bash
# Ensure you have 100GB volume
df -h
# If needed, resize volume in AWS Console
```

---

**Ready to transfer? Start with S3 for images, then let Try4 pipeline handle the rest!** 🚀
