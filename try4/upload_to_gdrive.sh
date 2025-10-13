#!/bin/bash
# Upload ALL required data to Google Drive for Try4 (1TB space available)
# This includes: images, datasets, embeddings, and models

set -e

echo "📦 Uploading Try4 Data to Google Drive (1TB Space)"
echo "==================================================="
echo ""

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
    echo "This will open a browser for authentication."
    echo ""
    read -p "Press Enter to continue..."
    echo ""
    
    rclone config
    
    echo ""
    echo "✅ Configuration complete!"
    echo ""
fi

cd ~/Smart-Product-Pricing-Challenge

# Calculate total size
echo "📊 Calculating data sizes..."
echo ""

if [ -d "try2/images" ]; then
    IMAGE_SIZE=$(du -sh try2/images 2>/dev/null | cut -f1)
    echo "Images:      $IMAGE_SIZE (required)"
else
    IMAGE_SIZE="N/A"
    echo "Images:      Not found (will need to download)"
fi

if [ -d "try2/dataset" ]; then
    DATASET_SIZE=$(du -sh try2/dataset 2>/dev/null | cut -f1)
    echo "Dataset:     $DATASET_SIZE (required)"
else
    DATASET_SIZE="N/A"
    echo "Dataset:     Not found"
fi

if [ -d "try3/outputs" ]; then
    OUTPUTS_SIZE=$(du -sh try3/outputs 2>/dev/null | cut -f1)
    echo "Try3 Output: $OUTPUTS_SIZE (optional - for reference)"
else
    OUTPUTS_SIZE="N/A"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Upload Options:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1) IMAGES ONLY (25-30GB) - Saves 3-4 hours download time"
echo "2) IMAGES + DATASETS (26-31GB) - Ready to start Try4"
echo "3) EVERYTHING (Try3 outputs + images + datasets) - Complete backup"
echo ""
read -p "Enter choice (1, 2, or 3): " CHOICE

BASE_PATH="hackathon-try4-$(date +%Y%m%d)"

case $CHOICE in
    1)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📦 Option 1: Uploading Images Only"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        if [ ! -d "try2/images" ]; then
            echo "❌ Error: Images directory not found at try2/images/"
            exit 1
        fi
        
        echo "⏳ Uploading images to Google Drive..."
        echo "This will take ~1-2 hours for 30GB..."
        
        rclone sync try2/images/ gdrive:$BASE_PATH/images/ \
            --progress \
            --transfers 8 \
            --checkers 16 \
            --stats 30s \
            --exclude "*.gitkeep" \
            --exclude "*.md"
        
        echo ""
        echo "✅ Images uploaded!"
        echo ""
        echo "📋 Bucket path: gdrive:$BASE_PATH/"
        echo ""
        ;;
    
    2)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "� Option 2: Uploading Images + Datasets"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Upload images
        if [ -d "try2/images" ]; then
            echo "⏳ [1/2] Uploading images..."
            rclone sync try2/images/ gdrive:$BASE_PATH/images/ \
                --progress \
                --transfers 8 \
                --stats 30s \
                --exclude "*.gitkeep"
            echo "✅ Images uploaded"
        else
            echo "⚠️  Images not found, skipping..."
        fi
        
        echo ""
        
        # Upload datasets
        if [ -d "try2/dataset" ]; then
            echo "⏳ [2/2] Uploading datasets..."
            rclone sync try2/dataset/ gdrive:$BASE_PATH/dataset/ \
                --progress \
                --transfers 4 \
                --stats 10s \
                --include "*.csv"
            echo "✅ Datasets uploaded"
        else
            echo "⚠️  Dataset not found, skipping..."
        fi
        
        echo ""
        echo "✅ Upload complete!"
        echo ""
        echo "📋 Bucket path: gdrive:$BASE_PATH/"
        echo ""
        ;;
    
    3)
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "� Option 3: Uploading EVERYTHING"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "This includes:"
        echo "- Images (30GB)"
        echo "- Datasets (1GB)"
        echo "- Try3 embeddings (10GB) - for reference"
        echo "- Try3 models (5GB) - for reference"
        echo ""
        read -p "Continue? This will take 2-3 hours (yes/no): " CONFIRM
        
        if [ "$CONFIRM" != "yes" ]; then
            echo "Upload cancelled"
            exit 0
        fi
        
        # Upload images
        if [ -d "try2/images" ]; then
            echo "⏳ [1/4] Uploading images..."
            rclone sync try2/images/ gdrive:$BASE_PATH/images/ \
                --progress \
                --transfers 8 \
                --stats 1m \
                --exclude "*.gitkeep"
            echo "✅ Images uploaded"
        fi
        
        echo ""
        
        # Upload datasets
        if [ -d "try2/dataset" ]; then
            echo "⏳ [2/4] Uploading datasets..."
            rclone sync try2/dataset/ gdrive:$BASE_PATH/dataset/ \
                --progress \
                --include "*.csv"
            echo "✅ Datasets uploaded"
        fi
        
        echo ""
        
        # Upload Try3 embeddings (optional)
        if [ -d "try3/outputs/balanced_model/embeddings_cache" ]; then
            echo "⏳ [3/4] Uploading Try3 embeddings (for reference)..."
            rclone sync try3/outputs/balanced_model/embeddings_cache/ \
                gdrive:$BASE_PATH/try3_embeddings/ \
                --progress \
                --transfers 4 \
                --stats 1m
            echo "✅ Try3 embeddings uploaded"
        fi
        
        echo ""
        
        # Upload Try3 models (optional)
        if [ -d "try3/outputs/balanced_model" ]; then
            echo "⏳ [4/4] Uploading Try3 models (for reference)..."
            rclone sync try3/outputs/balanced_model/ \
                gdrive:$BASE_PATH/try3_models/ \
                --progress \
                --exclude "embeddings_cache/*" \
                --include "*.pkl" \
                --include "*.txt"
            echo "✅ Try3 models uploaded"
        fi
        
        echo ""
        echo "✅ Complete backup uploaded!"
        echo ""
        echo "📋 Bucket path: gdrive:$BASE_PATH/"
        echo ""
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Verify upload
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Verifying upload..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if rclone ls gdrive:$BASE_PATH/ >/dev/null 2>&1; then
    echo ""
    echo "📊 Upload Summary:"
    rclone size gdrive:$BASE_PATH/ 2>/dev/null | grep -E "Total|objects"
    echo ""
    echo "✅ Verification successful!"
else
    echo "⚠️  Could not verify upload"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Upload Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 IMPORTANT - Save this information:"
echo ""
echo "   Google Drive Path: gdrive:$BASE_PATH/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📥 On your NEW AWS account, run:"
echo ""
echo "   cd ~/Smart-Product-Pricing-Challenge/try4"
echo "   ./download_from_gdrive.sh $BASE_PATH"
echo ""

# Save bucket name
echo $BASE_PATH > ~/gdrive_bucket_name.txt
echo "✅ Bucket name saved to ~/gdrive_bucket_name.txt"
