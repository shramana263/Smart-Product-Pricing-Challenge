#!/bin/bash

# 🔥 Clean & Fresh Pipeline Execution
# Deletes all outputs and runs complete pipeline from scratch

set -e

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     🔥 CLEAN & FRESH PIPELINE EXECUTION 🔥                ║"
echo "║                                                            ║"
echo "║  Removes all outputs and recompiles from scratch          ║"
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
# Step 1: Clean All Outputs
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Cleaning All Previous Outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clean outputs directory
if [ -d "outputs" ]; then
    echo "🗑️  Removing outputs/embeddings/..."
    rm -rf outputs/embeddings/*.csv 2>/dev/null || true
    
    echo "🗑️  Removing outputs/features/..."
    rm -rf outputs/features/*.csv 2>/dev/null || true
    
    echo "🗑️  Removing outputs/analysis/..."
    rm -rf outputs/analysis/*.csv 2>/dev/null || true
    
    echo "🗑️  Removing outputs/models/..."
    rm -rf outputs/models/*.txt 2>/dev/null || true
    
    echo "🗑️  Removing outputs/predictions/..."
    rm -rf outputs/predictions/*.csv 2>/dev/null || true
    
    echo "✅ All outputs cleaned!"
else
    echo "⚠️  No outputs directory found"
fi

# Clean logs
if [ -d "logs" ]; then
    echo "🗑️  Removing logs/..."
    rm -rf logs/*.log 2>/dev/null || true
    echo "✅ Logs cleaned!"
fi

# Create fresh directories
echo ""
echo "📁 Creating fresh output directories..."
mkdir -p outputs/embeddings
mkdir -p outputs/features
mkdir -p outputs/analysis
mkdir -p outputs/models
mkdir -p outputs/predictions
mkdir -p logs
echo "✅ Directories created!"

echo ""

# ============================================================================
# Step 2: Verify Fixes
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Verifying Bug Fixes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check multi-GPU fix
if grep -q "DataParallel" feature_extraction/deberta_embeddings.py; then
    echo "✅ Multi-GPU fix: Applied"
else
    echo "⚠️  Multi-GPU fix: NOT found (may run into OOM)"
fi

# Check float conversion fix
FLOAT_FIXES=$(grep -c "astype('float32')" feature_extraction/tabular_features.py)
if [ "$FLOAT_FIXES" -ge 2 ]; then
    echo "✅ Float conversion fix: Applied ($FLOAT_FIXES occurrences)"
else
    echo "⚠️  Float conversion fix: Missing or incomplete"
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "✅ Available GPUs: $GPU_COUNT"

echo ""

# ============================================================================
# Step 3: Setup & Optimization
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Setup & Optimization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use existing images
if [ -f "./use_existing_images.sh" ]; then
    echo "⚡ Configuring local image loading..."
    chmod +x use_existing_images.sh
    ./use_existing_images.sh 2>&1 | grep -E "✅|⚠️|❌"
else
    echo "⚠️  use_existing_images.sh not found, will download images"
fi

echo ""

# Fix imports
if [ -f "./fix_all.sh" ]; then
    echo "🔧 Fixing imports & dependencies..."
    chmod +x fix_all.sh
    ./fix_all.sh 2>&1 | grep -E "✅|⚠️|❌"
else
    echo "🔧 Manually fixing imports..."
    touch config/__init__.py
    touch feature_extraction/__init__.py
    touch preprocessing/__init__.py
    touch modeling/__init__.py
    pip install -q tiktoken protobuf sentencepiece
    echo "✅ Basic setup complete"
fi

echo ""

# Optimize config
if [ -f "optimize_speed.py" ]; then
    echo "⚡ Optimizing configuration..."
    python optimize_speed.py 2>&1 | grep -E "✅|Instance|batch|workers|Expected"
else
    echo "⚠️  optimize_speed.py not found, using default config"
fi

echo ""

# ============================================================================
# Step 4: Choose Execution Mode
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Execution Mode Selection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if we have multiple GPUs
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "🚀 Multiple GPUs detected ($GPU_COUNT GPUs)"
    echo "   Using PARALLEL execution for maximum speed!"
    EXEC_MODE="parallel"
else
    echo "📊 Single GPU detected"
    echo "   Using SEQUENTIAL execution (safer)"
    EXEC_MODE="sequential"
fi

echo ""
echo "Execution mode: $EXEC_MODE"
echo ""

# ============================================================================
# Step 5: Run Pipeline
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Running Complete Pipeline (Fresh)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⏰ Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

PIPELINE_START=$(date +%s)

if [ "$EXEC_MODE" = "parallel" ]; then
    # ========================================================================
    # PARALLEL EXECUTION
    # ========================================================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 1: Parallel Embedding Extraction"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Start DeBERTa in background
    echo "🔤 Starting DeBERTa extraction (background)..."
    python feature_extraction/deberta_embeddings.py > logs/deberta.log 2>&1 &
    DEBERTA_PID=$!
    echo "   PID: $DEBERTA_PID"
    
    # Start CLIP in background
    echo "🖼️  Starting CLIP extraction (background)..."
    python feature_extraction/clip_embeddings.py > logs/clip.log 2>&1 &
    CLIP_PID=$!
    echo "   PID: $CLIP_PID"
    
    echo ""
    echo "⏳ Running in parallel... Monitor with:"
    echo "   tail -f logs/deberta.log"
    echo "   tail -f logs/clip.log"
    echo ""
    
    # Wait for both
    DEBERTA_SUCCESS=true
    CLIP_SUCCESS=true
    
    if wait $DEBERTA_PID; then
        echo "✅ DeBERTa extraction complete!"
    else
        echo "❌ DeBERTa extraction failed! Check logs/deberta.log"
        DEBERTA_SUCCESS=false
    fi
    
    if wait $CLIP_PID; then
        echo "✅ CLIP extraction complete!"
    else
        echo "❌ CLIP extraction failed! Check logs/clip.log"
        CLIP_SUCCESS=false
    fi
    
    if [ "$DEBERTA_SUCCESS" = false ] || [ "$CLIP_SUCCESS" = false ]; then
        echo ""
        echo "❌ Parallel phase failed! Check logs for details."
        exit 1
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 2: Sequential Feature Engineering"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "3️⃣ Engineering tabular features..."
    python feature_extraction/tabular_features.py
    echo "✅ Tabular features complete!"
    
    echo ""
    echo "4️⃣ Detecting outliers..."
    python preprocessing/outlier_detection.py
    echo "✅ Outlier detection complete!"
    
    echo ""
    echo "5️⃣ Treating outliers..."
    python preprocessing/outlier_treatment.py
    echo "✅ Outlier treatment complete!"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 3: Model Training"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "6️⃣ Training fusion model (3 folds)..."
    python modeling/fusion_model.py
    echo "✅ Model training complete!"
    
else
    # ========================================================================
    # SEQUENTIAL EXECUTION
    # ========================================================================
    echo "Running all stages sequentially..."
    echo ""
    
    echo "1️⃣ DeBERTa embeddings..."
    python feature_extraction/deberta_embeddings.py
    echo "✅ Stage 1 complete!"
    
    echo ""
    echo "2️⃣ CLIP embeddings..."
    python feature_extraction/clip_embeddings.py
    echo "✅ Stage 2 complete!"
    
    echo ""
    echo "3️⃣ Tabular features..."
    python feature_extraction/tabular_features.py
    echo "✅ Stage 3 complete!"
    
    echo ""
    echo "4️⃣ Outlier detection..."
    python preprocessing/outlier_detection.py
    echo "✅ Stage 4 complete!"
    
    echo ""
    echo "5️⃣ Outlier treatment..."
    python preprocessing/outlier_treatment.py
    echo "✅ Stage 5 complete!"
    
    echo ""
    echo "6️⃣ Model training (3 folds)..."
    python modeling/fusion_model.py
    echo "✅ Stage 6 complete!"
fi

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              ✅ PIPELINE COMPLETE! ✅                      ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Timing:"
echo "   Setup time:     $((START_TIME - START_TIME))s"
echo "   Pipeline time:  $((PIPELINE_DURATION / 60))m $((PIPELINE_DURATION % 60))s"
echo "   Total time:     $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "📁 Generated Files:"
echo ""

# Check outputs
echo "   Embeddings:"
ls -lh outputs/embeddings/*.csv 2>/dev/null | awk '{print "     "$9" ("$5")"}'

echo ""
echo "   Features:"
ls -lh outputs/features/*.csv 2>/dev/null | awk '{print "     "$9" ("$5")"}'

echo ""
echo "   Models:"
ls -lh outputs/models/*.txt 2>/dev/null | awk '{print "     "$9" ("$5")"}'

echo ""
echo "   Predictions:"
ls -lh outputs/predictions/*.csv 2>/dev/null | awk '{print "     "$9" ("$5")"}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Performance Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check OOF results
if [ -f "outputs/predictions/oof_predictions.csv" ]; then
    echo "🎯 Calculating OOF SMAPE..."
    python << 'PYEOF'
import pandas as pd
import numpy as np

try:
    oof = pd.read_csv('outputs/predictions/oof_predictions.csv')
    
    # Calculate SMAPE
    numerator = np.abs(oof['price_pred'] - oof['price_true'])
    denominator = (np.abs(oof['price_true']) + np.abs(oof['price_pred'])) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    
    print(f"   Out-of-Fold SMAPE: {smape:.3f}%")
    
    if smape < 35:
        print(f"   ✅ EXCELLENT! Better than target (<35%)")
    elif smape < 40:
        print(f"   ✅ GOOD! Close to target")
    elif smape < 47:
        print(f"   ⚠️  OK - Better than Try3 baseline (47.378%)")
    else:
        print(f"   ❌ Needs improvement (worse than Try3)")
except Exception as e:
    print(f"   ⚠️  Could not calculate SMAPE: {e}")
PYEOF
else
    echo "   ⚠️  OOF predictions file not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next Steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Review predictions:"
echo "   head outputs/predictions/test_predictions.csv"
echo ""
echo "2. Check feature importance:"
echo "   ls outputs/models/feature_importance*.csv"
echo ""
echo "3. Submit predictions if SMAPE < 35%"
echo ""
echo "🎉 Done!"
echo ""
