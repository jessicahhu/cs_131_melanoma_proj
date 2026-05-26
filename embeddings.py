import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, EMBEDDING_DIR, FIGURE_DIR, JPEG_DIR, TrainConfig
from dataset import SIIMISICDataset, build_transforms, load_split
from model import MelanomaResNet18
from utils import get_device, set_seed


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, targets, names = [], [], []
    for images, target, name in tqdm(loader, desc="embed"):
        images = images.to(device, non_blocking=True)
        feats = model.embed(images).cpu().numpy()
        embeddings.append(feats)
        targets.append(target.numpy())
        names.extend(name)
    return np.concatenate(embeddings), np.concatenate(targets), np.array(names)


def save_embeddings(checkpoint: Path, cfg: TrainConfig, out: Path):
    set_seed(cfg.seed)
    device = get_device()
    model = MelanomaResNet18(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    _, val_df = load_split(cfg)
    ds = SIIMISICDataset(val_df, JPEG_DIR, build_transforms(cfg.image_size, train=False))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    embeds, targets, names = extract_embeddings(model, loader, device)
    np.savez(out, embeddings=embeds, targets=targets, names=names)
    print(f"saved {len(embeds)} embeddings to {out}")
    return embeds, targets, names


def nearest_neighbors(query: np.ndarray, gallery: np.ndarray, k: int = 5, metric: str = "cosine"):
    if metric == "cosine":
        sims = cosine_similarity(query, gallery)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
    elif metric == "euclidean":
        dists = euclidean_distances(query, gallery)
        idx = np.argsort(dists, axis=1)[:, :k]
        scores = np.take_along_axis(dists, idx, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return idx, scores


def plot_embedding_2d(embeds: np.ndarray, targets: np.ndarray, out: Path, method: str = "tsne"):
    import matplotlib.pyplot as plt

    if method == "tsne":
        reducer = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
        pre = PCA(n_components=min(50, embeds.shape[1])).fit_transform(embeds)
        xy = reducer.fit_transform(pre)
    else:
        xy = PCA(n_components=2).fit_transform(embeds)

    fig, ax = plt.subplots(figsize=(8, 8))
    for label, color, name in [(0, "tab:blue", "benign"), (1, "tab:red", "melanoma")]:
        mask = targets == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.5, c=color, label=name)
    ax.legend()
    ax.set_title(f"Lesion embedding ({method.upper()})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved embedding plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "resnet18_best.pt")
    parser.add_argument("--out", type=Path, default=EMBEDDING_DIR / "val_embeddings.npz")
    parser.add_argument("--plot", choices=["tsne", "pca", "none"], default="tsne")
    args = parser.parse_args()

    cfg = TrainConfig()
    embeds, targets, _ = save_embeddings(args.checkpoint, cfg, args.out)
    if args.plot != "none":
        plot_embedding_2d(embeds, targets, FIGURE_DIR / f"embedding_{args.plot}.png", method=args.plot)
