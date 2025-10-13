#!/bin/bash

# ⚡⚡⚡ Try4 Ultra-Fast Complete Setup & Run ⚡⚡⚡
# This script configures ALL optimizations and runs the pipeline
# Expected time: 30-50 minutes on ml.g5.2xlarge

set -e  # Exit on error

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     ⚡ Try4 Ultra-Fast Pipeline Setup & Execution ⚡       ║"
echo "║                                                            ║"
echo "║  Target: 30-50 minutes on ml.g5.2xlarge                   ║"
echo "║  Cost: ~$1.00                                              ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check directory
if [ ! -f "config/config.py" ]; then
    echo "❌ Error: Must run from try4/ directory"
    echo "Usage: cd ~/Smart-Product-Pricing-Challenge/try4 && ./run_fast.sh"
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================================
# Step 1: Use Existing Images
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/4: Configuring Local Image Loading (saves 20-30 min)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "./use_existing_images.sh" ]; then
    chmod +x use_existing_images.sh
    ./use_existing_images.sh
    echo "✅ Local images configured!"
else
    echo "⚠️  use_existing_images.sh not found, skipping..."
fi

echo ""

# ============================================================================
# Step 2: Fix Imports & Dependencies
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/4: Fixing Imports & Installing Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "./fix_all.sh" ]; then
    chmod +x fix_all.sh
    ./fix_all.sh
    echo "✅ Setup fixed!"
else
    echo "Running manual fix..."
    # Create __init__.py files
    touch config/__init__.py
    touch feature_extraction/__init__.py
    touch preprocessing/__init__.py
    touch modeling/__init__.py
    
    # Install dependencies
    pip install -q tiktoken protobuf sentencepiece
    echo "✅ Manual fix complete!"
fi

echo ""

# ============================================================================
# Step 3: Optimize for Speed
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/4: Optimizing Configuration for Speed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "optimize_speed.py" ]; then
    python optimize_speed.py
    echo "✅ Config optimized!"
else
    echo "⚠️  optimize_speed.py not found, using defaults..."
fi

echo ""

# ============================================================================
# Verify Configuration
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check folds
FOLDS=$(grep "n_folds" config/config.py | grep -v "#" | head -1 | grep -o "[0-9]\+")
echo "✓ Cross-validation folds: $FOLDS (lower = faster)"

# Check images
if [ -L "outputs/images" ]; then
    TRAIN_IMAGES=$(ls outputs/images/train/*.jpg 2>/dev/null | wc -l)
    TEST_IMAGES=$(ls outputs/images/test/*.jpg 2>/dev/null | wc -l)
    echo "✓ Local images: $TRAIN_IMAGES train, $TEST_IMAGES test (fast loading!)"
else
    echo "⚠️  Local images not configured (will download, slower)"
fi

# Check script version
if grep -q "LOCAL VERSION" feature_extraction/clip_embeddings.py 2>/dev/null; then
    echo "✓ CLIP script: Local image loading (fast!)"
else
    echo "⚠️  CLIP script: URL downloading (slower)"
fi

echo ""

# ============================================================================
# Step 4: Run Pipeline
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/4: Running Try4 Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Expected time: 30-50 minutes on ml.g5.2xlarge"
echo "Expected cost: ~$0.75-1.25"
echo ""
echo "⏰ Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Press Ctrl+C to cancel, or wait for pipeline to start..."
sleep 3

# Run pipeline
python main_pipeline.py

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              ✅ Pipeline Complete! ✅                      ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 Outputs saved to:"
echo "   - Embeddings:  outputs/embeddings/"
echo "   - Models:      outputs/models/"
echo "   - Predictions: outputs/predictions/"
echo ""
echo "📊 Next steps:"
echo "   1. Check OOF SMAPE: head outputs/predictions/oof_predictions.csv"
echo "   2. Check final predictions: ls -lh outputs/predictions/test_predictions.csv"
echo "   3. Submit if SMAPE < 35%!"
echo ""
echo "🎉 Done! Good luck with your submission! 🎉"
echo ""
