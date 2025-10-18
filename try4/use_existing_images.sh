#!/bin/bash

# ⚡ Use Existing Images from Try3 - Auto Setup
# This script configures Try4 to use pre-downloaded images from Try3
# Saves 20-30 minutes of downloading time!

set -e  # Exit on error

echo "========================================"
echo "⚡ Try4 - Use Existing Images Setup"
echo "========================================"
echo ""

# Check if we're in try4 directory
if [ ! -f "config/config.py" ]; then
    echo "❌ Error: Must run from try4/ directory"
    echo "Usage: cd ~/Smart-Product-Pricing-Challenge/try4 && ./use_existing_images.sh"
    exit 1
fi

echo "Step 1: Checking for existing images in try3..."
if [ -d "../try3/outputs/images_efficient/train" ] && [ -d "../try3/outputs/images_efficient/test" ]; then
    TRAIN_COUNT=$(find ../try3/outputs/images_efficient/train -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l)
    TEST_COUNT=$(find ../try3/outputs/images_efficient/test -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l)
    echo "✅ Found images!"
    echo "   Train images: $TRAIN_COUNT"
    echo "   Test images:  $TEST_COUNT"
else
    echo "❌ Error: Images not found at ../try3/outputs/images_efficient/"
    echo "Please check the path or download images first."
    exit 1
fi

echo ""
echo "Step 2: Creating symlink to existing images..."
mkdir -p outputs

TARGET_REL="../try3/outputs/images_efficient"
LINK="outputs/images"

if [ ! -d "$TARGET_REL" ]; then
    echo "❌ Expected image cache not found at $TARGET_REL"
    exit 1
fi

# Resolve absolute path to avoid nested symlink issues
TARGET_ABS=$(python - "$TARGET_REL" <<'PY'
import os
import sys

if len(sys.argv) < 2:
    sys.exit(1)

print(os.path.realpath(sys.argv[1]))
PY
)

if [ -z "$TARGET_ABS" ]; then
    echo "❌ Could not resolve absolute path for $TARGET_REL"
    exit 1
fi

if [ -L "$LINK" ] || [ -d "$LINK" ]; then
    rm -rf "$LINK"
    echo "   Removed existing outputs/images"
fi

ln -s "$TARGET_ABS" "$LINK"
echo "✅ Symlink created: $LINK -> $TARGET_ABS"

if [ -d "$LINK/train" ] && [ -d "$LINK/test" ]; then
    echo "✅ Symlink verified - train/ and test/ folders accessible"
else
    echo "⚠️  Warning: train/ or test/ not found under $LINK (check if cache mounted locally)"
fi

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Your Try4 pipeline now loads images from the existing Try3 cache."
echo "No downloads required."
echo ""
