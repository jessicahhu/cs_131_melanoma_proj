import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, EMBEDDING_DIR, JPEG_DIR, TrainConfig
from dataset import SIIMISICDataset, build_transforms, load_split
from model import MelanomaResNet18
from utils import get_device, set_seed


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    scores, targets = [], []
    for images, target, _ in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        scores.append(probs)
        targets.append(target.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def main(checkpoint: Path, out: Path):
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = get_device()

    model = MelanomaResNet18(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    _, val_df = load_split(cfg)
    ds = SIIMISICDataset(val_df, JPEG_DIR, build_transforms(cfg.image_size, train=False))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    scores, targets = collect_scores(model, loader, device)
    auc = roc_auc_score(targets, scores)
    np.savez(out, scores=scores, targets=targets, val_auc=auc)
    print(f"val AUC={auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "resnet18_best.pt")
    parser.add_argument("--out", type=Path, default=EMBEDDING_DIR / "val_scores.npz")
    args = parser.parse_args()
    main(args.checkpoint, args.out)
