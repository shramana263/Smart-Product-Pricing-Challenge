#!/bin/bash
# ONE-COMMAND SETUP: Complete Try4 setup on new AWS account

set -e

echo "🚀 ONE-COMMAND TRY4 SETUP"
echo "========================="
echo ""

# Check if bucket name provided
if [ -z "$1" ]; then
    echo "❌ Error: No Google Drive bucket name provided!"
    echo ""
    echo "Usage: ./one_command_setup.sh BUCKET_NAME"
    echo ""
    echo "Example:"
    echo "  ./one_command_setup.sh hackathon-try4-20251013"
    echo ""
    echo "Get the bucket name from OLD instance after uploading."
    exit 1
fi

BUCKET_NAME=$1

echo "📋 This script will:"
echo "  1. Clone repository (if needed)"
echo "  2. Download data from Google Drive"
echo "  3. Install all dependencies"
echo "  4. Verify system"
echo "  5. Run Try4 pipeline"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Setup cancelled"
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Repository Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd ~

if [ ! -d "Smart-Product-Pricing-Challenge" ]; then
    echo "Cloning repository..."
    git clone https://github.com/shramana263/Smart-Product-Pricing-Challenge.git
    echo "✅ Repository cloned"
else
    echo "✅ Repository already exists"
    cd Smart-Product-Pricing-Challenge
    git pull origin feature_engineering || echo "⚠️  Could not pull latest changes"
    cd ~
fi

cd Smart-Product-Pricing-Challenge/try4

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Download Data from Google Drive"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

chmod +x download_from_gdrive.sh
./download_from_gdrive.sh $BUCKET_NAME

if [ ! -d "../try2/images/train" ] || [ ! -f "../try2/dataset/train1.csv" ]; then
    echo ""
    echo "❌ Error: Data download incomplete"
    echo "Please run download script manually and verify"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Install Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

chmod +x install.sh
./install.sh

echo ""
echo "Fixing Python import structure..."
python fix_imports.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: System Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python check_system.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: System check failed"
    echo "Please fix issues above before running pipeline"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Run Try4 Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⏳ This will take ~3-4 hours on GPU"
echo ""
read -p "Start pipeline now? (yes/no): " START_PIPELINE

if [ "$START_PIPELINE" = "yes" ]; then
    echo ""
    echo "🚀 Starting Try4 pipeline..."
    echo "   You can safely close terminal (process will continue)"
    echo ""
    
    # Run in background with nohup
    nohup python main_pipeline.py > pipeline_output.log 2>&1 &
    PIPELINE_PID=$!
    
    echo "✅ Pipeline started!"
    echo "   Process ID: $PIPELINE_PID"
    echo "   Log file: pipeline_output.log"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f pipeline_output.log"
    echo ""
    echo "Check if running:"
    echo "  ps aux | grep main_pipeline"
    echo ""
else
    echo ""
    echo "⏭️  Pipeline not started"
    echo ""
    echo "To start manually:"
    echo "  python main_pipeline.py"
    echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 SETUP COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Summary:"
echo "  ✅ Repository: ~/Smart-Product-Pricing-Challenge"
echo "  ✅ Data: Downloaded from Google Drive"
echo "  ✅ Dependencies: Installed"
echo "  ✅ System: Verified"

if [ "$START_PIPELINE" = "yes" ]; then
    echo "  🔄 Pipeline: Running in background"
else
    echo "  ⏸️  Pipeline: Ready to start"
fi

echo ""
echo "📁 Important directories:"
echo "  Images: ~/Smart-Product-Pricing-Challenge/try2/images/"
echo "  Datasets: ~/Smart-Product-Pricing-Challenge/try2/dataset/"
echo "  Outputs: ~/Smart-Product-Pricing-Challenge/try4/outputs/"
echo ""
echo "💡 After pipeline completes (~3-4 hours):"
echo "  Results: try4/outputs/predictions/test_predictions.csv"
echo "  SMAPE: Check try4/outputs/predictions/oof_predictions.csv"
echo ""
