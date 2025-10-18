"""
CLIP ViT-Large Image Embedding Extraction (Local Image Cache)

Loads pre-downloaded images from try3/outputs/images_efficient/ and extracts
768-dimensional CLIP embeddings using all available GPUs.
"""

import sys
sys.path.insert(0, '..')

from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CLIP imports
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    import clip

from config.config import (
    DATA_DIR, EMBEDDINGS_DIR,
    IMAGE_MODEL, DEVICE, RANDOM_SEED, IMAGE_CACHE_DIR
)

NUM_AVAILABLE_GPUS = max(1, torch.cuda.device_count())
PIN_MEMORY = DEVICE.startswith('cuda')
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

CLIP_NAME_MAPPING = {
    'openai/clip-vit-base-patch32': 'ViT-B/32',
    'openai/clip-vit-base-patch16': 'ViT-B/16',
    'openai/clip-vit-large-patch14': 'ViT-L/14',
    'openai/clip-vit-large-patch14-336': 'ViT-L/14@336px',
}

print("="*80)
print("🖼️ CLIP ViT-Large Image Embedding Extraction")
print("="*80)
print(f"Model: {IMAGE_MODEL['name']}")
print(f"Device: {DEVICE}")
print(f"GPUs detected: {NUM_AVAILABLE_GPUS}")
print("="*80)


class LocalImageDataset(Dataset):
    """Dataset that loads images from local cache using sample_id"""

    def __init__(self, df: pd.DataFrame, split: str, preprocess, image_root: Path, image_size: int):
        self.sample_ids = df['sample_id'].astype(str).tolist()
        self.preprocess = preprocess
        self.image_dir = Path(image_root) / split
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                "Ensure use_existing_images.sh configured the symlink correctly."
            )

    def __len__(self):
        return len(self.sample_ids)

    def _resolve_path(self, sample_id: str):
        for ext in SUPPORTED_EXTENSIONS:
            candidate = self.image_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        path = self._resolve_path(sample_id)

        if path is None:
            tensor = torch.zeros(3, self.image_size, self.image_size)
            valid = torch.tensor(False, dtype=torch.bool)
        else:
            with Image.open(path) as img:
                tensor = self.preprocess(img.convert('RGB'))
            valid = torch.tensor(True, dtype=torch.bool)

        return tensor, valid


class ClipImageEncoder(nn.Module):
    """Wrapper to expose encode_image for DataParallel"""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images):
        return self.clip_model.encode_image(images)


class CLIPFeatureExtractor:
    """Extract image embeddings using CLIP with optional multi-GPU"""

    def __init__(self, model_name='ViT-L/14', device='cuda'):
        self.device = device
        resolved_name = CLIP_NAME_MAPPING.get(model_name, model_name)
        if resolved_name != model_name:
            print(f"\n🔄 Loading CLIP {resolved_name} (mapped from {model_name})...")
        else:
            print(f"\n🔄 Loading CLIP {model_name}...")

        self.clip_model, self.preprocess = clip.load(resolved_name, device=device)
        self.clip_model.eval()

        self.encoder = ClipImageEncoder(self.clip_model)
        if IMAGE_MODEL.get('scale_batch_by_gpu', True) and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs for CLIP embeddings")
            self.encoder = nn.DataParallel(self.encoder)
        self.encoder = self.encoder.to(device)
        print("✓ CLIP model ready")

    def extract_embeddings(self, dataloader):
        embeddings = []
        valid_total = 0
        total = 0

        for images, valid_mask in tqdm(dataloader, desc="Extracting", ncols=100):
            images = images.to(self.device, non_blocking=PIN_MEMORY)
            with torch.no_grad():
                features = self.encoder(images)
                features = features / features.norm(dim=-1, keepdim=True)

            features = features.cpu()
            if valid_mask is not None:
                mask = valid_mask.cpu().bool()
                invalid = ~mask
                if invalid.any():
                    features[invalid] = 0
                valid_total += mask.sum().item()
                total += mask.numel()
            else:
                total += features.shape[0]
                valid_total += features.shape[0]

            embeddings.append(features.numpy())

        if not embeddings:
            return np.empty((0, IMAGE_MODEL['embedding_dim'])), valid_total, total

        return np.concatenate(embeddings, axis=0), valid_total, total


def build_dataloader(df, split, extractor, image_root, batch_size, num_workers):
    dataset = LocalImageDataset(
        df=df,
        split=split,
        preprocess=extractor.preprocess,
        image_root=image_root,
        image_size=IMAGE_MODEL['image_size']
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n📂 Loading data...")
    train1 = pd.read_csv(DATA_DIR / 'train1.csv')
    train2 = pd.read_csv(DATA_DIR / 'train2.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)

    test1 = pd.read_csv(DATA_DIR / 'test1.csv')
    test2 = pd.read_csv(DATA_DIR / 'test2.csv')
    test_df = pd.concat([test1, test2], ignore_index=True)

    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")

    image_root = Path(IMAGE_MODEL.get('local_image_dir', IMAGE_CACHE_DIR)).resolve()
    print(f"✓ Using local image cache: {image_root}")

    per_device_batch = IMAGE_MODEL['batch_size']
    scale_batch = IMAGE_MODEL.get('scale_batch_by_gpu', True)
    effective_batch = max(1, per_device_batch * NUM_AVAILABLE_GPUS) if scale_batch else per_device_batch
    num_workers = max(2, IMAGE_MODEL.get('num_workers', 4))
    print(f"✓ Effective batch size: {effective_batch}")
    print(f"✓ DataLoader workers: {num_workers}")

    extractor = CLIPFeatureExtractor(
        model_name=IMAGE_MODEL['name'],
        device=DEVICE
    )

    print("\n" + "="*80)
    print("📸 LOADING TRAINING IMAGES FROM DISK")
    print("="*80)
    train_loader = build_dataloader(train_df, 'train', extractor, image_root, effective_batch, num_workers)
    train_embeddings, train_valid, train_total = extractor.extract_embeddings(train_loader)
    print(f"✓ Train embeddings shape: {train_embeddings.shape}")
    print(f"✓ Images found: {train_valid:,}/{train_total:,} ({(train_valid/max(1, train_total))*100:.1f}%)")

    print("\n" + "="*80)
    print("📸 LOADING TEST IMAGES FROM DISK")
    print("="*80)
    test_loader = build_dataloader(test_df, 'test', extractor, image_root, effective_batch, num_workers)
    test_embeddings, test_valid, test_total = extractor.extract_embeddings(test_loader)
    print(f"✓ Test embeddings shape: {test_embeddings.shape}")
    print(f"✓ Images found: {test_valid:,}/{test_total:,} ({(test_valid/max(1, test_total))*100:.1f}%)")

    print("\n💾 Saving embeddings...")
    train_emb_df = pd.DataFrame(
        train_embeddings,
        columns=[f'clip_{i}' for i in range(IMAGE_MODEL['embedding_dim'])]
    )
    train_emb_df.insert(0, 'sample_id', train_df['sample_id'].values)
    train_path = EMBEDDINGS_DIR / 'clip_train_embeddings.csv'
    train_emb_df.to_csv(train_path, index=False)
    print(f"✓ Saved train embeddings to: {train_path}")

    test_emb_df = pd.DataFrame(
        test_embeddings,
        columns=[f'clip_{i}' for i in range(IMAGE_MODEL['embedding_dim'])]
    )
    test_emb_df.insert(0, 'sample_id', test_df['sample_id'].values)
    test_path = EMBEDDINGS_DIR / 'clip_test_embeddings.csv'
    test_emb_df.to_csv(test_path, index=False)
    print(f"✓ Saved test embeddings to: {test_path}")

    print("\n" + "="*80)
    print("📊 EMBEDDING STATISTICS")
    print("="*80)
    print(f"Embedding dimension: {IMAGE_MODEL['embedding_dim']}")
    print(f"Train mean: {train_embeddings.mean():.6f} | std: {train_embeddings.std():.6f}")
    print(f"Test  mean: {test_embeddings.mean():.6f} | std: {test_embeddings.std():.6f}")
    print(f"Missing train embeddings (all zeros): {np.sum(np.all(train_embeddings == 0, axis=1)):,}")
    print(f"Missing test embeddings (all zeros):  {np.sum(np.all(test_embeddings == 0, axis=1)):,}")

    print("\n" + "="*80)
    print("✅ CLIP EMBEDDING EXTRACTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
