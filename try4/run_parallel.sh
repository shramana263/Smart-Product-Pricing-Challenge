#!/bin/bash

# ⚡ Try4 Parallel Pipeline Execution
# Runs independent stages in parallel to save time
# 
# Dependency graph:
#   Stage 1 (DeBERTa) ─┐
#   Stage 2 (CLIP)    ─┼─→ Stage 3 (Tabular) ─→ Stage 4 (Outliers) ─→ Stage 5 (Treatment) ─→ Stage 6 (Training)
#
# Parallelizable:
#   - Stage 1 & 2 can run in parallel (no dependencies)
#   - Stages 3-6 must run sequentially (depend on each other)

set -e

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     ⚡ Try4 Parallel Pipeline Execution ⚡                 ║"
echo "║                                                            ║"
echo "║  Runs DeBERTa & CLIP in parallel for faster execution     ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check directory
if [ ! -f "config/config.py" ]; then
    echo "❌ Error: Must run from try4/ directory"
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================================
# Setup Phase
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup: Configuring optimizations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use existing images
if [ -f "./use_existing_images.sh" ]; then
    chmod +x use_existing_images.sh
    ./use_existing_images.sh
fi

# Fix imports
if [ -f "./fix_all.sh" ]; then
    chmod +x fix_all.sh
    ./fix_all.sh
fi

# Optimize config
if [ -f "optimize_speed.py" ]; then
    python optimize_speed.py
fi

echo ""

# ============================================================================
# Phase 1: Parallel Embedding Extraction (DeBERTa + CLIP)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 1: Parallel Embedding Extraction"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PHASE1_START=$(date +%s)

# Check if embeddings already exist
DEBERTA_EXISTS=false
CLIP_EXISTS=false

if [ -f "outputs/embeddings/deberta_train_embeddings.csv" ]; then
    echo "✓ DeBERTa embeddings already exist, skipping..."
    DEBERTA_EXISTS=true
fi

if [ -f "outputs/embeddings/clip_train_embeddings.csv" ]; then
    echo "✓ CLIP embeddings already exist, skipping..."
    CLIP_EXISTS=true
fi

# Run parallel extraction if needed
if [ "$DEBERTA_EXISTS" = false ] || [ "$CLIP_EXISTS" = false ]; then
    echo "Starting parallel extraction..."
    echo ""
    
    # Create log directory
    mkdir -p logs
    
    # Start DeBERTa in background if needed
    if [ "$DEBERTA_EXISTS" = false ]; then
        echo "🔤 Starting DeBERTa extraction (background)..."
        python feature_extraction/deberta_embeddings.py > logs/deberta.log 2>&1 &
        DEBERTA_PID=$!
        echo "   PID: $DEBERTA_PID"
    fi
    
    # Start CLIP in background if needed
    if [ "$CLIP_EXISTS" = false ]; then
        echo "🖼️  Starting CLIP extraction (background)..."
        python feature_extraction/clip_embeddings.py > logs/clip.log 2>&1 &
        CLIP_PID=$!
        echo "   PID: $CLIP_PID"
    fi
    
    echo ""
    echo "⏳ Waiting for both embeddings to complete..."
    echo "   (This typically takes 25-35 minutes on ml.g5.2xlarge)"
    echo ""
    echo "   Monitor progress in separate terminals:"
    echo "   - DeBERTa: tail -f logs/deberta.log"
    echo "   - CLIP:    tail -f logs/clip.log"
    echo ""
    
    # Wait for both processes
    DEBERTA_SUCCESS=true
    CLIP_SUCCESS=true
    
    if [ "$DEBERTA_EXISTS" = false ]; then
        if wait $DEBERTA_PID; then
            echo "✅ DeBERTa extraction complete!"
        else
            echo "❌ DeBERTa extraction failed! Check logs/deberta.log"
            DEBERTA_SUCCESS=false
        fi
    fi
    
    if [ "$CLIP_EXISTS" = false ]; then
        if wait $CLIP_PID; then
            echo "✅ CLIP extraction complete!"
        else
            echo "❌ CLIP extraction failed! Check logs/clip.log"
            CLIP_SUCCESS=false
        fi
    fi
    
    # Check if both succeeded
    if [ "$DEBERTA_SUCCESS" = false ] || [ "$CLIP_SUCCESS" = false ]; then
        echo ""
        echo "❌ Phase 1 failed! Check logs for details."
        exit 1
    fi
fi

PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
echo ""
echo "✓ Phase 1 completed in $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s"

# ============================================================================
# Phase 2: Sequential Feature Engineering
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 2: Feature Engineering (Sequential)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PHASE2_START=$(date +%s)

# Stage 3: Tabular features
if [ ! -f "outputs/features/tabular_train_features.csv" ]; then
    echo "3️⃣ Engineering tabular features..."
    python feature_extraction/tabular_features.py
    echo "✅ Tabular features complete!"
else
    echo "✓ Tabular features already exist, skipping..."
fi

# Stage 4: Outlier detection
if [ ! -f "outputs/analysis/outlier_detection_results.csv" ]; then
    echo ""
    echo "4️⃣ Detecting outliers..."
    python preprocessing/outlier_detection.py
    echo "✅ Outlier detection complete!"
else
    echo "✓ Outlier detection already done, skipping..."
fi

# Stage 5: Outlier treatment
if [ ! -f "outputs/features/train_features_treated.csv" ]; then
    echo ""
    echo "5️⃣ Treating outliers..."
    python preprocessing/outlier_treatment.py
    echo "✅ Outlier treatment complete!"
else
    echo "✓ Outlier treatment already done, skipping..."
fi

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))
echo ""
echo "✓ Phase 2 completed in $((PHASE2_DURATION / 60))m $((PHASE2_DURATION % 60))s"

# ============================================================================
# Phase 3: Model Training
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 3: Model Training (3 folds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PHASE3_START=$(date +%s)

echo "6️⃣ Training fusion model..."
python modeling/fusion_model.py
echo "✅ Model training complete!"

PHASE3_END=$(date +%s)
PHASE3_DURATION=$((PHASE3_END - PHASE3_START))
echo ""
echo "✓ Phase 3 completed in $((PHASE3_DURATION / 60))m $((PHASE3_DURATION % 60))s"

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              ✅ Pipeline Complete! ✅                      ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Timing Breakdown:"
echo "   Phase 1 (Parallel):   $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s  (DeBERTa + CLIP)"
echo "   Phase 2 (Sequential): $((PHASE2_DURATION / 60))m $((PHASE2_DURATION % 60))s  (Features + Outliers)"
echo "   Phase 3 (Training):   $((PHASE3_DURATION / 60))m $((PHASE3_DURATION % 60))s  (LightGBM 3-fold)"
echo "   ─────────────────────────────────────────"
echo "   Total:                $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "📁 Outputs:"
echo "   - Predictions: outputs/predictions/test_predictions.csv"
echo "   - OOF:         outputs/predictions/oof_predictions.csv"
echo "   - Models:      outputs/models/lightgbm_fold*.txt"
echo ""
echo "📊 Check OOF SMAPE:"
echo "   head outputs/predictions/oof_predictions.csv"
echo ""
echo "🎉 Ready for submission!"
echo ""
