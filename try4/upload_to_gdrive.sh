#!/bin/bash
# Upload images to Google Drive (works across AWS accounts)

set -e

echo "📦 Uploading to Google Drive (Cross-Account)"
echo "============================================="
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

cd ~/Smart-Product-Pricing-Challenge/try2

# Option 1: Upload compressed (recommended for <15GB free tier)
echo "Choose upload method:"
echo "1) Compressed (recommended for free 15GB tier)"
echo "2) Uncompressed (requires 100GB Google One - $1.99/month)"
echo ""
read -p "Enter choice (1 or 2): " CHOICE

if [ "$CHOICE" = "1" ]; then
    echo ""
    echo "📦 Compressing images..."
    echo "This will take ~10-15 minutes..."
    
    if [ ! -f images.tar.gz ]; then
        tar -czf images.tar.gz images/
        echo "✅ Compressed: $(du -sh images.tar.gz | cut -f1)"
    else
        echo "⚠️  images.tar.gz already exists, skipping compression"
    fi
    
    echo ""
    echo "⏳ Uploading compressed file to Google Drive..."
    echo "This will take ~30-60 minutes..."
    
    rclone copy images.tar.gz gdrive:hackathon-images/ \
        --progress \
        --transfers 4 \
        --stats 10s
    
    echo ""
    echo "✅ Upload complete!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 On your NEW AWS account, run:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "cd ~/Smart-Product-Pricing-Challenge/try4"
    echo "./download_from_gdrive.sh compressed"
    echo ""
    
else
    echo ""
    echo "⏳ Uploading images to Google Drive..."
    echo "This will take ~1-2 hours for 30GB..."
    echo ""
    echo "⚠️  Note: Requires Google One (100GB) subscription"
    echo "   Sign up at: https://one.google.com/about/plans"
    echo ""
    read -p "Press Enter to continue..."
    
    rclone copy images/ gdrive:hackathon-images/images/ \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 30s
    
    echo ""
    echo "✅ Upload complete!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 On your NEW AWS account, run:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "cd ~/Smart-Product-Pricing-Challenge/try4"
    echo "./download_from_gdrive.sh uncompressed"
    echo ""
fi

# Verify upload
echo "Verifying upload..."
if rclone ls gdrive:hackathon-images/ >/dev/null 2>&1; then
    FILES=$(rclone ls gdrive:hackathon-images/ | wc -l)
    SIZE=$(rclone size gdrive:hackathon-images/ 2>/dev/null | grep "Total size:" | awk '{print $3, $4}')
    echo "✅ Files uploaded: $FILES"
    echo "✅ Total size: $SIZE"
else
    echo "⚠️  Could not verify upload (but it might still be there)"
fi

echo ""
echo "🎉 Ready to download on new instance!"
