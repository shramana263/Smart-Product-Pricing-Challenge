"""
Text Embedding Extraction for Product Price Prediction
Uses pre-trained transformer models to create dense embeddings from catalog_content
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TextEmbeddingExtractor:
    """
    Extract embeddings from product catalog content using pre-trained models
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', use_pca=True, n_components=128):
        """
        Initialize the embedding extractor
        
        Args:
            model_name: Hugging Face model name
                - 'all-MiniLM-L6-v2': Fast, 384 dims (RECOMMENDED)
                - 'all-mpnet-base-v2': Best quality, 768 dims
                - 'paraphrase-MiniLM-L6-v2': Good balance
            use_pca: Whether to apply PCA for dimensionality reduction
            n_components: Number of PCA components (if use_pca=True)
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING TEXT EMBEDDING EXTRACTOR")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"PCA: {use_pca} (components={n_components if use_pca else 'N/A'})")
        
        self.model_name = model_name
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
        
        # Load model
        print("\nLoading pre-trained model...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def extract_embeddings(self, texts, batch_size=32, show_progress=True):
        """
        Extract embeddings from texts
        
        Args:
            texts: List or Series of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        print(f"\n{'='*60}")
        print("EXTRACTING EMBEDDINGS")
        print(f"{'='*60}")
        print(f"Total texts: {len(texts)}")
        print(f"Batch size: {batch_size}")
        
        # Convert to list and handle missing values
        texts_list = texts.fillna('').astype(str).tolist() if isinstance(texts, pd.Series) else list(texts)
        
        # Encode in batches
        print("\nEncoding texts...")
        embeddings = self.model.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        print(f"✓ Embeddings extracted: {embeddings.shape}")
        return embeddings
    
    def fit_pca(self, embeddings):
        """Fit PCA on embeddings"""
        if self.use_pca:
            print(f"\nFitting PCA (reducing to {self.n_components} components)...")
            self.pca = PCA(n_components=self.n_components, random_state=42)
            self.pca.fit(embeddings)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"✓ PCA fitted")
            print(f"✓ Explained variance: {explained_var*100:.2f}%")
    
    def transform_pca(self, embeddings):
        """Transform embeddings using fitted PCA"""
        if self.use_pca and self.pca is not None:
            embeddings_pca = self.pca.transform(embeddings)
            print(f"✓ PCA transformed: {embeddings_pca.shape}")
            return embeddings_pca
        return embeddings
    
    def save(self, filepath):
        """Save PCA model"""
        if self.pca is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'pca': self.pca,
                    'model_name': self.model_name,
                    'n_components': self.n_components
                }, f)
            print(f"\n✓ PCA model saved: {filepath}")
    
    def load(self, filepath):
        """Load PCA model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pca = data['pca']
            self.model_name = data['model_name']
            self.n_components = data['n_components']
        print(f"\n✓ PCA model loaded: {filepath}")


def process_training_data(train_path, output_dir='./embeddings_data', 
                          model_name='all-MiniLM-L6-v2', 
                          use_pca=True, n_components=128):
    """
    Process training data and extract embeddings
    
    Args:
        train_path: Path to training CSV (or list of paths)
        output_dir: Directory to save embeddings
        model_name: Pre-trained model name
        use_pca: Whether to use PCA
        n_components: Number of PCA components
    """
    print(f"\n{'='*70}")
    print("PROCESSING TRAINING DATA - TEXT EMBEDDINGS")
    print(f"{'='*70}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    if isinstance(train_path, list):
        print(f"\nLoading {len(train_path)} training files...")
        dfs = []
        for path in train_path:
            df = pd.read_csv(path)
            print(f"  ✓ Loaded: {path} ({len(df)} samples)")
            dfs.append(df)
        train_df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"\nLoading training file: {train_path}")
        train_df = pd.read_csv(train_path)
    
    print(f"\nTotal training samples: {len(train_df)}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Initialize extractor
    extractor = TextEmbeddingExtractor(
        model_name=model_name,
        use_pca=use_pca,
        n_components=n_components
    )
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(train_df['catalog_content'])
    
    # Fit and transform PCA
    if use_pca:
        extractor.fit_pca(embeddings)
        embeddings_final = extractor.transform_pca(embeddings)
    else:
        embeddings_final = embeddings
    
    # Create DataFrame with embeddings
    embedding_cols = [f'text_emb_{i}' for i in range(embeddings_final.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_final, columns=embedding_cols)
    embeddings_df['sample_id'] = train_df['sample_id'].values
    embeddings_df['price'] = train_df['price'].values
    
    # Save embeddings
    output_path = os.path.join(output_dir, 'train_embeddings.csv')
    embeddings_df.to_csv(output_path, index=False)
    print(f"\n✓ Embeddings saved: {output_path}")
    print(f"  Shape: {embeddings_df.shape}")
    
    # Save PCA model
    if use_pca:
        pca_path = os.path.join(output_dir, 'pca_model.pkl')
        extractor.save(pca_path)
    
    # Save embedding info
    info = {
        'model_name': model_name,
        'use_pca': use_pca,
        'n_components': n_components if use_pca else embeddings_final.shape[1],
        'n_samples': len(embeddings_df),
        'embedding_dim': embeddings_final.shape[1]
    }
    
    info_path = os.path.join(output_dir, 'embedding_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n{'='*70}")
    print("EMBEDDING EXTRACTION COMPLETED!")
    print(f"{'='*70}")
    print(f"Total samples: {len(embeddings_df)}")
    print(f"Embedding dimensions: {embeddings_final.shape[1]}")
    print(f"Files saved in: {output_dir}")
    
    return embeddings_df, extractor


def process_test_data(test_path, extractor, output_dir='./embeddings_data'):
    """
    Process test data using the same extractor
    
    Args:
        test_path: Path to test CSV (or list of paths)
        extractor: Fitted TextEmbeddingExtractor
        output_dir: Directory to save embeddings
    """
    print(f"\n{'='*70}")
    print("PROCESSING TEST DATA - TEXT EMBEDDINGS")
    print(f"{'='*70}")
    
    # Load test data
    if isinstance(test_path, list):
        print(f"\nLoading {len(test_path)} test files...")
        dfs = []
        for path in test_path:
            df = pd.read_csv(path)
            print(f"  ✓ Loaded: {path} ({len(df)} samples)")
            dfs.append(df)
        test_df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"\nLoading test file: {test_path}")
        test_df = pd.read_csv(test_path)
    
    print(f"\nTotal test samples: {len(test_df)}")
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings(test_df['catalog_content'])
    
    # Transform using PCA
    if extractor.use_pca:
        embeddings_final = extractor.transform_pca(embeddings)
    else:
        embeddings_final = embeddings
    
    # Create DataFrame
    embedding_cols = [f'text_emb_{i}' for i in range(embeddings_final.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_final, columns=embedding_cols)
    embeddings_df['sample_id'] = test_df['sample_id'].values
    
    # Save embeddings
    output_path = os.path.join(output_dir, 'test_embeddings.csv')
    embeddings_df.to_csv(output_path, index=False)
    print(f"\n✓ Test embeddings saved: {output_path}")
    print(f"  Shape: {embeddings_df.shape}")
    
    return embeddings_df


def combine_with_existing_features(embeddings_df, features_df, on='sample_id'):
    """
    Combine text embeddings with existing engineered features
    
    Args:
        embeddings_df: DataFrame with text embeddings
        features_df: DataFrame with existing features
        on: Column to merge on
    """
    print(f"\n{'='*70}")
    print("COMBINING EMBEDDINGS WITH EXISTING FEATURES")
    print(f"{'='*70}")
    
    print(f"\nEmbeddings shape: {embeddings_df.shape}")
    print(f"Existing features shape: {features_df.shape}")
    
    # Merge
    combined_df = embeddings_df.merge(features_df, on=on, how='inner')
    
    print(f"\nCombined shape: {combined_df.shape}")
    print(f"Total features: {combined_df.shape[1] - 2}")  # Exclude sample_id and price
    
    return combined_df


if __name__ == "__main__":
    """
    Example usage
    """
    print("\n" + "="*70)
    print("TEXT EMBEDDING EXTRACTION - MAIN SCRIPT")
    print("="*70)
    
    # Configuration
    TRAIN_PATHS = [
        './dataset/train1.csv',
        './dataset/train2.csv'
    ]
    
    TEST_PATHS = [
        './dataset/test1.csv',
        './dataset/test2.csv'
    ]
    
    OUTPUT_DIR = './embeddings_data'
    
    # Model options (choose one):
    # 1. 'all-MiniLM-L6-v2' - Fast, 384 dims (RECOMMENDED)
    # 2. 'all-mpnet-base-v2' - Best quality, 768 dims
    # 3. 'paraphrase-MiniLM-L6-v2' - Good balance, 384 dims
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    USE_PCA = True
    N_COMPONENTS = 128  # Reduce to 128 dims
    
    # Process training data
    train_embeddings, extractor = process_training_data(
        train_path=TRAIN_PATHS,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        use_pca=USE_PCA,
        n_components=N_COMPONENTS
    )
    
    # Process test data
    test_embeddings = process_test_data(
        test_path=TEST_PATHS,
        extractor=extractor,
        output_dir=OUTPUT_DIR
    )
    
    # Optional: Combine with existing features
    print("\n" + "="*70)
    print("OPTIONAL: COMBINING WITH EXISTING FEATURES")
    print("="*70)
    print("\nTo combine with your existing engineered features:")
    print("1. Load your features_clean.csv")
    print("2. Use combine_with_existing_features()")
    print("\nExample:")
    print("  existing_features = pd.read_csv('preparation/features_clean.csv')")
    print("  combined = combine_with_existing_features(train_embeddings, existing_features)")
    
    print("\n" + "="*70)
    print("✓ TEXT EMBEDDING EXTRACTION COMPLETED!")
    print("="*70)
    print(f"\nFiles created in '{OUTPUT_DIR}':")
    print("  - train_embeddings.csv")
    print("  - test_embeddings.csv")
    print("  - pca_model.pkl")
    print("  - embedding_info.txt")
    print("\nNext steps:")
    print("  1. Load embeddings")
    print("  2. Optionally combine with existing features")
    print("  3. Train XGBoost/LightGBM on combined features")
    print("  4. Compare SMAPE with previous approach")
