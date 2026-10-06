"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    images_dir = Path(images_dir)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_total = n_train + n_val + n_test
    
    print(f"Total: {n_total} (Train: {n_train}, Val: {n_val}, Test: {n_test})")
    assert n_total == 17509, f"Total images should be 17509, but got {n_total}"
    
    train_files = set(train_df['Filename'])
    val_files = set(val_df['Filename'])
    test_files = set(test_df['Filename'])
    
    assert len(train_files.intersection(val_files)) == 0, "Overlap between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Overlap between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Overlap between val and test!"
    
    all_files = train_files | val_files | test_files
    for f in all_files:
        assert (images_dir / f).exists(), f"File {f} not found in {images_dir}"
        
    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_total,
        "train_per_class": train_df['Label'].value_counts().to_dict(),
        "val_per_class": val_df['Label'].value_counts().to_dict(),
        "test_per_class": test_df['Label'].value_counts().to_dict(),
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if train:
        if aug == "basic":
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        elif aug == "color":
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        elif aug == "randaug":
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        else:
            raise ValueError(f"Unknown aug: {aug}")
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row['Filename']
        label = int(row['Label'])
        img_path = self.images_dir / filename
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        return img, label, filename

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    if train:
        if sampler == "balanced":
            class_counts = df['Label'].value_counts().sort_index().values
            class_weights = 1.0 / class_counts
            weights = [class_weights[label] for label in df['Label']]
            wrs = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=wrs, 
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=g)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=g)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True, drop_last=False,
                          worker_init_fn=seed_worker, generator=g)
