import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from config import CHECKPOINT_DIR, TrainConfig
from dataset import build_dataloaders
from model import MelanomaResNet18
from utils import get_device, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses, all_logits, all_targets = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, targets, _ in tqdm(loader, desc="train" if train else "val", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    auc = roc_auc_score(targets, logits) if len(np.unique(targets)) > 1 else float("nan")
    return float(np.mean(losses)), auc


def main(cfg: TrainConfig, out: Path):
    set_seed(cfg.seed)
    device = get_device()
    train_loader, val_loader = build_dataloaders(cfg)
    model = MelanomaResNet18(pretrained=True, dropout=cfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_auc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(
            f"epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f}  "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "resnet18_best.pt")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        subsample_fraction=args.subsample,
    )
    main(cfg, args.out)
