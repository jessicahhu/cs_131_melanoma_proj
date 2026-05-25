from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import JPEG_DIR, TRAIN_CSV, TrainConfig


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SIIMISICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_dir / f"{row['image_name']}.jpg"
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, target, row["image_name"]


def load_split(cfg: TrainConfig):
    df = pd.read_csv(TRAIN_CSV)
    if cfg.subsample_fraction is not None:
        df, _ = train_test_split(
            df,
            train_size=cfg.subsample_fraction,
            stratify=df["target"],
            random_state=cfg.seed,
        )
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.val_fraction,
        stratify=df["target"],
        random_state=cfg.seed,
    )
    return train_df, val_df


def make_sampler(df: pd.DataFrame, pos_oversample: float) -> WeightedRandomSampler:
    weights = np.where(df["target"].values == 1, pos_oversample, 1.0).astype(np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(cfg: TrainConfig):
    train_df, val_df = load_split(cfg)
    train_ds = SIIMISICDataset(train_df, JPEG_DIR, build_transforms(cfg.image_size, train=True))
    val_ds = SIIMISICDataset(val_df, JPEG_DIR, build_transforms(cfg.image_size, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=make_sampler(train_df, cfg.pos_oversample),
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
