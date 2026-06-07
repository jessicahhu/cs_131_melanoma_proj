import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from plotnine import (
    aes, element_text, geom_density, geom_point, ggplot, labs,
    scale_color_manual, scale_fill_manual, theme, theme_bw,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_DIR, JPEG_DIR, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_NAMES = {0: "benign", 1: "melanoma"}
LABEL_COLORS = {"benign": "#2c7fb8", "melanoma": "#d95f02"}
PAIR_COLORS = {
    "benign–benign": "#2c7fb8",
    "melanoma–melanoma": "#d95f02",
    "benign–melanoma": "#7f7f7f",
}


def projection_plot(xy, labels, title, out):
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                       "diagnosis": [LABEL_NAMES[t] for t in labels]})
    df["diagnosis"] = pd.Categorical(df["diagnosis"], categories=["benign", "melanoma"])
    df = df.sort_values("diagnosis")
    p = (
        ggplot(df, aes("x", "y", color="diagnosis"))
        + geom_point(size=0.8, alpha=0.5)
        + scale_color_manual(values=LABEL_COLORS)
        + labs(title=title, x="dim 1", y="dim 2", color="")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"))
    )
    p.save(out, width=7, height=6, dpi=150, verbose=False)
    print(f"saved {out}")


def similarity_distribution(embeds, targets, out, n_pairs=60000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(embeds)
    norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    sims = np.sum(norm[i] * norm[j], axis=1)

    ti, tj = targets[i], targets[j]
    pair = np.where(
        (ti == 1) & (tj == 1), "melanoma–melanoma",
        np.where((ti == 0) & (tj == 0), "benign–benign", "benign–melanoma"),
    )
    df = pd.DataFrame({"cosine_similarity": sims, "pair": pair})
    p = (
        ggplot(df, aes("cosine_similarity", color="pair", fill="pair"))
        + geom_density(alpha=0.15)
        + scale_color_manual(values=PAIR_COLORS)
        + scale_fill_manual(values=PAIR_COLORS)
        + labs(title="Cosine similarity by lesion-pair type",
               x="Cosine similarity", y="Density", color="", fill="")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"), legend_position="bottom")
    )
    p.save(out, width=7, height=4.5, dpi=150, verbose=False)
    print(f"saved {out}")


def knn_stats(embeds, targets, k=5):
    sims = cosine_similarity(embeds)
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    nn_labels = targets[nn]

    base_mel = float(np.mean(targets == 1))
    mel_q = targets == 1
    ben_q = targets == 0
    prec_mel = float(np.mean(nn_labels[mel_q] == 1))
    prec_ben = float(np.mean(nn_labels[ben_q] == 0))

    sims_full = cosine_similarity(embeds)
    np.fill_diagonal(sims_full, np.nan)
    mm = np.nanmean(sims_full[np.ix_(mel_q, mel_q)])
    bb = np.nanmean(sims_full[np.ix_(ben_q, ben_q)])
    mb = np.nanmean(sims_full[np.ix_(mel_q, ben_q)])

    rows = [
        ("base_melanoma_rate", base_mel),
        (f"precision@{k}_melanoma_queries", prec_mel),
        (f"precision@{k}_melanoma_lift_vs_base", prec_mel / base_mel),
        (f"precision@{k}_benign_queries", prec_ben),
        ("mean_cos_melanoma_melanoma", mm),
        ("mean_cos_benign_benign", bb),
        ("mean_cos_melanoma_benign", mb),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def retrieval_grid(embeds, targets, names, out, n_queries=4, k=5, seed=0):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    sims = cosine_similarity(embeds)
    np.fill_diagonal(sims, -np.inf)
    mel_idx = np.where(targets == 1)[0]
    queries = rng.choice(mel_idx, size=min(n_queries, len(mel_idx)), replace=False)

    fig, axes = plt.subplots(len(queries), k + 1, figsize=(2 * (k + 1), 2 * len(queries)))
    for r, q in enumerate(queries):
        nbrs = np.argsort(-sims[q])[:k]
        cols = [q] + list(nbrs)
        for c, idx in enumerate(cols):
            img = Image.open(JPEG_DIR / f"{names[idx]}.jpg").convert("RGB")
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            lab = LABEL_NAMES[int(targets[idx])]
            if c == 0:
                ax.set_title(f"query\n{lab}", fontsize=9, color=LABEL_COLORS[lab])
            else:
                ax.set_title(f"nn{c} ({sims[q][idx]:.2f})\n{lab}", fontsize=9,
                             color=LABEL_COLORS[lab])
    fig.suptitle("Nearest-neighbor retrieval in embedding space (melanoma queries)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main(npz_path: Path, k: int):
    data = np.load(npz_path, allow_pickle=True)
    embeds, targets, names = data["embeddings"], data["targets"].astype(int), data["names"]
    print(f"loaded {len(embeds)} embeddings; {int(targets.sum())} melanoma / {len(targets)} total")

    pre = PCA(n_components=min(50, embeds.shape[1])).fit_transform(embeds)
    tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(pre)
    projection_plot(tsne, targets, "Lesion embeddings (t-SNE)", FIG_DIR / "embedding_tsne.png")
    pca2 = PCA(n_components=2).fit_transform(embeds)
    projection_plot(pca2, targets, "Lesion embeddings (PCA)", FIG_DIR / "embedding_pca.png")

    similarity_distribution(embeds, targets, FIG_DIR / "similarity_distribution.png")
    stats = knn_stats(embeds, targets, k=k)
    stats.to_csv(RESULTS_DIR / "similarity_stats.csv", index=False)
    print(stats.to_string(index=False))
    retrieval_grid(embeds, targets, names, FIG_DIR / "nearest_neighbors.png", k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=EMBEDDING_DIR / "val_embeddings.npz")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    main(args.npz, args.k)
