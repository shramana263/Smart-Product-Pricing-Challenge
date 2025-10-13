#!/bin/bash
# Download images from Google Drive (works across AWS accounts)

set -e

echo "📦 Downloading from Google Drive"
echo "================================="
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

# Check argument
if [ -z "$1" ]; then
    echo "Usage: ./download_from_gdrive.sh [compressed|uncompressed]"
    echo ""
    echo "Examples:"
    echo "  ./download_from_gdrive.sh compressed     # If uploaded as .tar.gz"
    echo "  ./download_from_gdrive.sh uncompressed   # If uploaded as raw files"
    echo ""
    exit 1
fi

MODE=$1

cd ~/Smart-Product-Pricing-Challenge

# Create directory
mkdir -p try2/images

if [ "$MODE" = "compressed" ]; then
    echo "⏳ Downloading compressed file..."
    echo "This will take ~30-60 minutes..."
    echo ""
    
    cd try2
    rclone copy gdrive:hackathon-images/images.tar.gz ./ \
        --progress \
        --transfers 4 \
        --stats 10s
    
    echo ""
    echo "📦 Extracting images..."
    echo "This will take ~5-10 minutes..."
    tar -xzf images.tar.gz
    
    echo ""
    echo "🗑️  Cleaning up compressed file..."
    rm images.tar.gz
    
else
    echo "⏳ Downloading images..."
    echo "This will take ~1-2 hours for 30GB..."
    echo ""
    
    rclone copy gdrive:hackathon-images/images/ try2/images/ \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 30s
fi

echo ""
echo "✅ Download complete!"
echo ""

# Verify download
echo "Verifying download..."
if [ -d "try2/images/train" ] && [ -d "try2/images/test" ]; then
    TRAIN_COUNT=$(ls try2/images/train/ 2>/dev/null | wc -l)
    TEST_COUNT=$(ls try2/images/test/ 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh try2/images/ 2>/dev/null | cut -f1)
    
    echo "✅ Train images: $TRAIN_COUNT"
    echo "✅ Test images: $TEST_COUNT"
    echo "✅ Total size: $TOTAL_SIZE"
    
    # Test image loading
    if command -v python3 &> /dev/null; then
        python3 -c "
from PIL import Image
import os
train_dir = 'try2/images/train/'
if os.path.exists(train_dir) and os.listdir(train_dir):
    first_image = os.path.join(train_dir, os.listdir(train_dir)[0])
    img = Image.open(first_image)
    print(f'✅ Image test: {img.size} pixels, {img.mode} mode')
else:
    print('⚠️  No images found in train directory')
" 2>/dev/null || echo "⚠️  Could not test image loading (PIL not installed)"
    fi
else
    echo "⚠️  Warning: Expected directory structure not found"
    echo "Please verify manually: ls -la try2/images/"
fi

echo ""
echo "🎉 Images ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. cd ~/Smart-Product-Pricing-Challenge/try4"
echo "2. python check_system.py"
echo "3. python main_pipeline.py"
echo ""
echo "💡 Optional: Delete from Google Drive to free space:"
echo "   rclone delete gdrive:hackathon-images/"
