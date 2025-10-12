#!/bin/bash
# Quick Model Upload Script for SageMaker
# Run this on SageMaker after training

echo "======================================================================"
echo "📤 Model Upload Helper"
echo "======================================================================"
echo ""

MODEL_DIR="$HOME/Smart-Product-Pricing-Challenge/try2/modeling/distilbert_sagemaker/final_model"
RESULTS_FILE="$HOME/Smart-Product-Pricing-Challenge/try2/modeling/distilbert_sagemaker/training_results.json"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found!"
    echo "   Looking for: $MODEL_DIR"
    exit 1
fi

echo "✅ Model found: $MODEL_DIR"
echo ""

# Show model size
echo "📊 Model Size:"
du -sh "$MODEL_DIR"
echo ""

echo "Choose upload method:"
echo "  1) Hugging Face Hub (Recommended)"
echo "  2) AWS S3"
echo "  3) Create compressed archive (manual upload)"
echo "  4) Skip (I'll do it manually)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📦 Uploading to Hugging Face Hub..."
        echo ""
        read -p "Enter your Hugging Face username: " hf_user
        read -p "Enter model repo name (e.g., amazon-ml-distilbert): " repo_name
        
        pip install -q huggingface_hub
        
        echo ""
        echo "Please login to Hugging Face (get token from https://huggingface.co/settings/tokens)"
        huggingface-cli login
        
        echo ""
        echo "Uploading model..."
        huggingface-cli upload "$hf_user/$repo_name" "$MODEL_DIR" --repo-type model
        
        echo ""
        echo "✅ Model uploaded!"
        echo "📍 Access at: https://huggingface.co/$hf_user/$repo_name"
        echo ""
        echo "💾 To download on local machine:"
        echo "   pip install huggingface_hub"
        echo "   python -c \"from huggingface_hub import snapshot_download; snapshot_download('$hf_user/$repo_name', 'try2/modeling/distilbert_sagemaker/final_model')\""
        ;;
        
    2)
        echo ""
        echo "📦 Uploading to AWS S3..."
        echo ""
        read -p "Enter S3 bucket name: " bucket_name
        read -p "Enter S3 path prefix (e.g., models/distilbert): " s3_path
        
        echo ""
        echo "Uploading model to s3://$bucket_name/$s3_path/"
        aws s3 sync "$MODEL_DIR" "s3://$bucket_name/$s3_path/" \
            --exclude "checkpoints/*" \
            --exclude "logs/*"
        
        echo ""
        echo "Uploading predictions..."
        aws s3 cp \
            "$HOME/Smart-Product-Pricing-Challenge/try2/modeling/test_out_distilbert_sagemaker.csv" \
            "s3://$bucket_name/${s3_path}_predictions/"
        
        echo ""
        echo "✅ Model uploaded!"
        echo "📍 S3 location: s3://$bucket_name/$s3_path/"
        echo ""
        echo "💾 To download on local machine:"
        echo "   aws s3 sync s3://$bucket_name/$s3_path/ try2/modeling/distilbert_sagemaker/final_model/"
        ;;
        
    3)
        echo ""
        echo "📦 Creating compressed archive..."
        echo ""
        
        cd "$HOME/Smart-Product-Pricing-Challenge/try2/modeling"
        tar -czf distilbert_sagemaker_model.tar.gz \
            distilbert_sagemaker/final_model \
            distilbert_sagemaker/training_results.json \
            test_out_distilbert_sagemaker.csv
        
        echo "✅ Archive created: distilbert_sagemaker_model.tar.gz"
        echo "📊 Size:"
        du -sh distilbert_sagemaker_model.tar.gz
        echo ""
        echo "Upload this file manually via:"
        echo "  - Google Drive"
        echo "  - Dropbox"
        echo "  - OneDrive"
        echo "  - Any file sharing service"
        echo ""
        echo "💾 To extract on local machine:"
        echo "   tar -xzf distilbert_sagemaker_model.tar.gz"
        ;;
        
    4)
        echo ""
        echo "⏭️  Skipping upload. Model remains on SageMaker."
        echo "   Location: $MODEL_DIR"
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "✅ Done!"
echo "======================================================================"
