import argparse

import pandas as pd
import torch
from plotnine import (
    aes, element_text, geom_col, geom_errorbar, ggplot, labs, position_dodge,
    scale_fill_manual, theme, theme_bw,
)

from config import CHECKPOINT_DIR, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ARCH_LABELS = {
    "baseline": "ResNet18",
    "regularized": "ResNet18 (drop 0.4, wd 1e-3)",
    "effb0": "EfficientNet-B0",
    "rn50": "ResNet50",
}
ARCH_COLORS = {
    "ResNet18": "#2c7fb8",
    "ResNet18 (drop 0.4, wd 1e-3)": "#7fcdbb",
    "EfficientNet-B0": "#d95f02",
    "ResNet50": "#5f3dc4",
}


def collect(prefix, n_folds):
    rows = []
    for f in range(n_folds):
        ck = CHECKPOINT_DIR / "cv" / f"{prefix}_fold{f}.pt"
        if not ck.exists():
            continue
        state = torch.load(ck, map_location="cpu")
        rows.append({"arch": ARCH_LABELS[prefix], "fold": f,
                     "best_val_auc": float(state["val_auc"]),
                     "best_epoch": int(state["epoch"])})
    return rows


def main(prefixes, n_folds):
    rows = []
    for p in prefixes:
        rows.extend(collect(p, n_folds))
    best = pd.DataFrame(rows)
    if best.empty:
        print("no checkpoints found")
        return
    print(best.to_string(index=False))
    summ = (best.groupby("arch")["best_val_auc"]
            .agg(["mean", "std", "min", "max", "count"]).reset_index())
    print()
    print(summ.to_string(index=False))

    best.to_csv(RESULTS_DIR / "arch_metrics.csv", index=False)
    summ.to_csv(RESULTS_DIR / "arch_summary.csv", index=False)

    p_fold = (
        ggplot(best, aes("factor(fold)", "best_val_auc", fill="arch"))
        + geom_col(position=position_dodge(0.8), width=0.7)
        + scale_fill_manual(values=ARCH_COLORS)
        + labs(title="Per-fold best validation ROC-AUC by architecture",
               x="Fold", y="ROC-AUC", fill="")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"), legend_position="bottom")
    )
    p_fold.save(FIG_DIR / "arch_per_fold_auc.png", width=8, height=5, dpi=150, verbose=False)
    print(f"saved {FIG_DIR / 'arch_per_fold_auc.png'}")

    summ["lo"] = summ["mean"] - summ["std"]
    summ["hi"] = summ["mean"] + summ["std"]
    p_summ = (
        ggplot(summ, aes("arch", "mean", fill="arch"))
        + geom_col(width=0.6)
        + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.2)
        + scale_fill_manual(values=ARCH_COLORS)
        + labs(title="Cross-validated validation ROC-AUC by architecture (mean ± std)",
               x="", y="ROC-AUC")
        + theme_bw()
        + theme(plot_title=element_text(size=12, weight="bold"),
                legend_position="none",
                axis_text_x=element_text(angle=15, hjust=1))
    )
    p_summ.save(FIG_DIR / "arch_summary_auc.png", width=7.5, height=5, dpi=150, verbose=False)
    print(f"saved {FIG_DIR / 'arch_summary_auc.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", default="baseline,regularized,effb0,rn50")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.prefixes.split(","), args.folds)
