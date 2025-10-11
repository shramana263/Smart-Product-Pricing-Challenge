# Text Embeddings Approach - Quick Start Guide

## 🎯 Overview
This approach uses **pre-trained sentence transformers** (BERT-based models) to create dense semantic embeddings from product catalog content, then trains gradient boosting models on these embeddings.

## 📦 Why This Works Better

### Current Approach Issues:
- Basic text features: word count, length, keyword matching
- Doesn't capture **semantic meaning**
- Example: "organic premium coffee" vs "coffee organic premium" → same features, but contextually different

### Embeddings Approach Benefits:
- **Semantic understanding**: Captures meaning, not just keywords
- **Context-aware**: "premium" in coffee vs "premium" in electronics
- **Dense representations**: 128-384 dimensions encode rich information
- **Pre-trained knowledge**: Model already knows product terminology

## 🚀 Installation

```powershell
# Install required packages
pip install sentence-transformers
pip install transformers
pip install torch
pip install scikit-learn
pip install pandas numpy
pip install lightgbm xgboost catboost
```

## 📋 Step-by-Step Usage

### Step 1: Extract Text Embeddings

```powershell
cd try2
python text_embeddings.py
```

**What it does:**
- Loads train1.csv and train2.csv
- Extracts 384-dim embeddings using `all-MiniLM-L6-v2`
- Applies PCA to reduce to 128 dimensions
- Saves embeddings to `embeddings_data/train_embeddings.csv`
- Processes test data with same transformations

**Expected output:**
```
train_embeddings.csv: (75000, 130) - 128 embedding dims + sample_id + price
test_embeddings.csv: (75000, 129) - 128 embedding dims + sample_id
```

**Time:** ~5-10 minutes (depends on CPU/GPU)

### Step 2: Train Models with Embeddings

```powershell
python train_with_embeddings.py
```

**What it does:**
- Loads embeddings
- Optionally combines with your existing engineered features
- Trains LightGBM, XGBoost, and CatBoost
- Evaluates on test set
- Saves predictions to `test_out_embeddings.csv`

**Two Options:**

**Option A: Embeddings Only** (in script, set `USE_EXISTING_FEATURES = False`)
- Pure semantic approach
- Faster training
- Good baseline

**Option B: Embeddings + Existing Features** (set `USE_EXISTING_FEATURES = True`)
- Best of both worlds
- Combines semantics with explicit features (brand, quantity, etc.)
- **RECOMMENDED** for best performance

## 📊 Expected Results

| Approach | Test SMAPE | Notes |
|----------|-----------|-------|
| Current (hand-crafted features) | 47.52% | Your baseline |
| Embeddings only | 44-46% | Expected improvement |
| Embeddings + Existing features | 42-45% | **Best expected** |
| Embeddings + Images (future) | 38-42% | With visual features |

## 🔧 Configuration Options

### Model Selection (in `text_embeddings.py`):

```python
# Fast, good quality (RECOMMENDED)
MODEL_NAME = 'all-MiniLM-L6-v2'  # 384 dims, fast

# Best quality (slower)
MODEL_NAME = 'all-mpnet-base-v2'  # 768 dims, slower

# Balanced
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'  # 384 dims
```

### PCA Components:

```python
USE_PCA = True
N_COMPONENTS = 128  # Reduce dimensionality

# Try these variations:
# N_COMPONENTS = 64   # Faster, may lose info
# N_COMPONENTS = 256  # Slower, more info
# USE_PCA = False     # Keep all 384/768 dims
```

## 📁 Files Created

```
try2/
├── embeddings_data/
│   ├── train_embeddings.csv      # Training embeddings
│   ├── test_embeddings.csv       # Test embeddings
│   ├── pca_model.pkl              # PCA transformer
│   └── embedding_info.txt         # Metadata
├── modeling/
│   ├── test_out_embeddings.csv    # Final predictions
│   └── embedding_model_results.csv # Model comparison
├── text_embeddings.py
└── train_with_embeddings.py
```

## 🎨 Customization Examples

### Example 1: Quick Test (No PCA, Fewer Samples)

```python
# In text_embeddings.py, modify:
USE_PCA = False  # Skip PCA
N_COMPONENTS = 128

# Test on smaller subset first
train_df = train_df.sample(10000)  # 10k samples only
```

### Example 2: Maximum Quality

```python
# Best quality model
MODEL_NAME = 'all-mpnet-base-v2'  # 768 dims
USE_PCA = True
N_COMPONENTS = 256  # Keep more info
```

### Example 3: Combine Multiple Embeddings

```python
# Extract embeddings from different models
extractor1 = TextEmbeddingExtractor('all-MiniLM-L6-v2')
extractor2 = TextEmbeddingExtractor('paraphrase-MiniLM-L6-v2')

emb1 = extractor1.extract_embeddings(texts)
emb2 = extractor2.extract_embeddings(texts)

# Concatenate
combined_emb = np.concatenate([emb1, emb2], axis=1)
```

## 🔍 Debugging

### Check Embeddings Quality:

```python
import pandas as pd

# Load embeddings
emb = pd.read_csv('embeddings_data/train_embeddings.csv')

# Check for NaN/Inf
print(emb.isnull().sum())
print(np.isinf(emb.select_dtypes(include=[np.number])).sum())

# Check correlation with price
print(emb.corr()['price'].sort_values(ascending=False).head(20))
```

### Visualize Embeddings (Optional):

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
emb_2d = pca_2d.fit_transform(emb.drop(['sample_id', 'price'], axis=1))

# Plot
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=emb['price'], alpha=0.5, cmap='viridis')
plt.colorbar(label='Price')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Product Embeddings (colored by price)')
plt.show()
```

## 🎯 Next Steps After Running

1. **Compare Results:**
   - Check `embedding_model_results.csv`
   - Compare SMAPE with your baseline (47.52%)

2. **If SMAPE Improved:**
   - Try different embedding models
   - Experiment with PCA dimensions
   - Add image embeddings next

3. **If SMAPE Same/Worse:**
   - Ensure embeddings loaded correctly
   - Check for NaN values
   - Try `USE_EXISTING_FEATURES = True`
   - Increase PCA components

4. **Final Optimization:**
   - Ensemble embeddings + baseline models
   - Hyperparameter tuning with Optuna
   - Add image features (next phase)

## 💡 Pro Tips

1. **GPU Acceleration**: If you have CUDA-enabled GPU, embeddings will be 10-20x faster
2. **Batch Size**: Increase batch_size in `extract_embeddings()` for faster processing (32 → 64)
3. **Normalization**: Embeddings are L2-normalized by default (unit vectors)
4. **Feature Importance**: Embedding dimensions have less interpretability than hand-crafted features
5. **Ensemble**: Best results come from ensembling embeddings + hand-crafted features

## 🐛 Common Issues

### Issue 1: "No module named 'sentence_transformers'"
```powershell
pip install sentence-transformers
```

### Issue 2: Out of Memory
```python
# Reduce batch size
batch_size=16  # Default is 32

# Or use smaller model
MODEL_NAME = 'all-MiniLM-L6-v2'  # Smaller footprint
```

### Issue 3: Slow Processing
```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME, device=device)
```

### Issue 4: Different Feature Count
```python
# Make sure train and test have same features
print(f"Train features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Check column alignment
print(set(train_cols) - set(test_cols))  # Missing in test
print(set(test_cols) - set(train_cols))  # Missing in train
```

## 📚 References

- **Sentence-BERT Paper**: https://arxiv.org/abs/1908.10084
- **Models**: https://www.sbert.net/docs/pretrained_models.html
- **Hugging Face**: https://huggingface.co/sentence-transformers

## 🤝 Support

If you encounter issues:
1. Check terminal output for error messages
2. Verify file paths are correct
3. Ensure all dependencies installed
4. Check data shapes match expected dimensions

---

**Ready to start?** 
```powershell
cd try2
python text_embeddings.py
```

Good luck! 🚀
