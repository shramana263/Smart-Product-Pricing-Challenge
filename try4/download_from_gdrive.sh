#!/bin/bash
# Download ALL required data from Google Drive for Try4
# This sets up complete environment on new AWS account

set -e

echo "📦 Downloading Try4 Data from Google Drive"
echo "==========================================="
echo ""

# Check if bucket name provided
if [ -z "$1" ]; then
    echo "❌ Error: No bucket path provided!"
    echo ""
    echo "Usage: ./download_from_gdrive.sh BUCKET_PATH"
    echo ""
    echo "Example:"
    echo "  ./download_from_gdrive.sh hackathon-try4-20251013"
    echo ""
    echo "The bucket path was shown when you uploaded."
    echo "Check ~/gdrive_bucket_name.txt on old instance."
    exit 1
fi

BUCKET_PATH=$1

# Check if rclone installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Check if rclone configured
if ! rclone listremotes | grep -q "gdrive"; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  First time setup required!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "You need to configure Google Drive access."
    echo "Use the SAME Google account as old instance!"
    echo "This will open a browser for authentication."
    echo ""
    read -p "Press Enter to continue..."
    echo ""
    
    rclone config
    
    echo ""
    echo "✅ Configuration complete!"
    echo ""
fi

# Verify bucket exists
echo "Verifying Google Drive bucket..."
if ! rclone lsd gdrive:$BUCKET_PATH/ >/dev/null 2>&1; then
    echo "❌ Error: Bucket not found: gdrive:$BUCKET_PATH/"
    echo ""
    echo "Please check:"
    echo "1. Bucket name is correct"
    echo "2. You're using the same Google account"
    echo "3. rclone is configured properly"
    echo ""
    echo "Available buckets:"
    rclone lsd gdrive: | grep hackathon || echo "None found"
    exit 1
fi

echo "✅ Bucket found: gdrive:$BUCKET_PATH/"
echo ""

# Show available directories
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Available directories:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rclone lsd gdrive:$BUCKET_PATH/
echo ""

cd ~/Smart-Product-Pricing-Challenge

# Create directory structure
echo "Creating directory structure..."
mkdir -p try2/images/train
mkdir -p try2/images/test
mkdir -p try2/dataset
mkdir -p try4/outputs/embeddings
mkdir -p try4/outputs/features
mkdir -p try4/outputs/models
mkdir -p try4/outputs/predictions
mkdir -p try4/outputs/analysis
echo "✅ Directories created"
echo ""

# Download images (required)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 [1/2] Downloading Images (REQUIRED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if rclone lsd gdrive:$BUCKET_PATH/images/ >/dev/null 2>&1; then
    echo "⏳ Downloading images..."
    echo "This will take ~1-2 hours for 30GB..."
    echo ""
    
    rclone sync gdrive:$BUCKET_PATH/images/ try2/images/ \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 1m
    
    echo ""
    echo "✅ Images downloaded"
    
    # Verify
    TRAIN_COUNT=$(ls try2/images/train/ 2>/dev/null | wc -l)
    TEST_COUNT=$(ls try2/images/test/ 2>/dev/null | wc -l)
    IMAGE_SIZE=$(du -sh try2/images/ 2>/dev/null | cut -f1)
    
    echo "   Train images: $TRAIN_COUNT"
    echo "   Test images: $TEST_COUNT"
    echo "   Total size: $IMAGE_SIZE"
else
    echo "⚠️  Images directory not found in bucket"
    echo "   You'll need to download images during Try4 pipeline"
fi

echo ""

# Download datasets (required)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "� [2/2] Downloading Datasets (REQUIRED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if rclone lsd gdrive:$BUCKET_PATH/dataset/ >/dev/null 2>&1; then
    echo "⏳ Downloading datasets..."
    
    rclone sync gdrive:$BUCKET_PATH/dataset/ try2/dataset/ \
        --progress \
        --transfers 4 \
        --include "*.csv"
    
    echo "✅ Datasets downloaded"
    
    # Verify
    echo "   Files:"
    ls -lh try2/dataset/*.csv 2>/dev/null | awk '{print "   - " $9 ": " $5}'
else
    echo "⚠️  Dataset directory not found in bucket"
    echo "   Make sure you have train1.csv, train2.csv, test1.csv, test2.csv"
fi

echo ""

# Download Try3 outputs (optional - for reference)
if rclone lsd gdrive:$BUCKET_PATH/try3_embeddings/ >/dev/null 2>&1 || \
   rclone lsd gdrive:$BUCKET_PATH/try3_models/ >/dev/null 2>&1; then
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📥 [Optional] Try3 Outputs Available"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Try3 embeddings and models are available."
    echo "These are for REFERENCE ONLY (Try4 uses different models)."
    echo ""
    read -p "Download Try3 outputs? (yes/no): " DOWNLOAD_TRY3
    
    if [ "$DOWNLOAD_TRY3" = "yes" ]; then
        mkdir -p try3/outputs/balanced_model/embeddings_cache
        
        if rclone lsd gdrive:$BUCKET_PATH/try3_embeddings/ >/dev/null 2>&1; then
            echo "⏳ Downloading Try3 embeddings..."
            rclone sync gdrive:$BUCKET_PATH/try3_embeddings/ \
                try3/outputs/balanced_model/embeddings_cache/ \
                --progress
            echo "✅ Try3 embeddings downloaded"
        fi
        
        if rclone lsd gdrive:$BUCKET_PATH/try3_models/ >/dev/null 2>&1; then
            echo "⏳ Downloading Try3 models..."
            rclone sync gdrive:$BUCKET_PATH/try3_models/ \
                try3/outputs/balanced_model/ \
                --progress
            echo "✅ Try3 models downloaded"
        fi
    else
        echo "⏭️  Skipping Try3 outputs"
    fi
    
    echo ""
fi

# Final verification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Download Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check images
if [ -d "try2/images/train" ] && [ "$(ls -A try2/images/train 2>/dev/null)" ]; then
    TRAIN_COUNT=$(ls try2/images/train/ | wc -l)
    echo "✅ Images (train): $TRAIN_COUNT files"
else
    echo "❌ Images (train): Not found"
fi

if [ -d "try2/images/test" ] && [ "$(ls -A try2/images/test 2>/dev/null)" ]; then
    TEST_COUNT=$(ls try2/images/test/ | wc -l)
    echo "✅ Images (test): $TEST_COUNT files"
else
    echo "❌ Images (test): Not found"
fi

# Check datasets
if [ -f "try2/dataset/train1.csv" ]; then
    echo "✅ Dataset: train1.csv"
else
    echo "❌ Dataset: train1.csv missing"
fi

if [ -f "try2/dataset/train2.csv" ]; then
    echo "✅ Dataset: train2.csv"
else
    echo "❌ Dataset: train2.csv missing"
fi

if [ -f "try2/dataset/test1.csv" ]; then
    echo "✅ Dataset: test1.csv"
else
    echo "❌ Dataset: test1.csv missing"
fi

if [ -f "try2/dataset/test2.csv" ]; then
    echo "✅ Dataset: test2.csv"
else
    echo "❌ Dataset: test2.csv missing"
fi

echo ""

# Test image loading
if command -v python3 &> /dev/null && [ -d "try2/images/train" ]; then
    python3 -c "
from PIL import Image
import os
train_dir = 'try2/images/train/'
if os.path.exists(train_dir) and os.listdir(train_dir):
    first_image = os.path.join(train_dir, os.listdir(train_dir)[0])
    img = Image.open(first_image)
    print(f'✅ Image test: {img.size} pixels, {img.mode} mode')
else:
    print('⚠️  No images found')
" 2>/dev/null || echo "⚠️  PIL not installed (run: pip install Pillow)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Download Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Install Try4 dependencies:"
echo "   cd ~/Smart-Product-Pricing-Challenge/try4"
echo "   pip install -r requirements.txt"
echo "   pip install git+https://github.com/openai/CLIP.git"
echo ""
echo "2. Verify system:"
echo "   python check_system.py"
echo ""
echo "3. Run Try4 pipeline:"
echo "   python main_pipeline.py"
echo ""
echo "💡 Tip: The pipeline will generate NEW embeddings for Try4"
echo "   (DeBERTa-v3-large and CLIP ViT-Large are different from Try3)"
echo ""
echo "💰 Optional: Delete Google Drive files to free space:"
echo "   rclone delete gdrive:$BUCKET_PATH/ --rmdirs"
echo ""
