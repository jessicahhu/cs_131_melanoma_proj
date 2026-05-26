from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, coord_equal, element_text, geom_abline, geom_line, geom_point,
    ggplot, labs, scale_color_manual, scale_x_continuous, theme_bw, theme,
)
from sklearn.metrics import roc_curve

from config import EMBEDDING_DIR, FIGURE_DIR, PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "baseline_metrics.csv"
RESULTS_REG = PROJECT_ROOT / "results" / "regularized_metrics.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SPLIT_COLORS = {"Train": "#2c7fb8", "Validation": "#d95f02"}
MODEL_COLORS = {"Baseline": "#2c7fb8", "Regularized (dropout 0.4, wd 1e-3)": "#d95f02"}


def epoch_plot(df, value_cols, ylab, title, out):
    long = df.melt(id_vars="epoch", value_vars=list(value_cols),
                   var_name="split", value_name="value")
    long["split"] = long["split"].map({value_cols[0]: "Train", value_cols[1]: "Validation"})
    p = (
        ggplot(long, aes("epoch", "value", color="split"))
        + geom_line(size=1.0)
        + geom_point(size=3.0)
        + scale_color_manual(values=SPLIT_COLORS)
        + scale_x_continuous(breaks=sorted(df["epoch"].unique()))
        + labs(title=title, x="Epoch", y=ylab, color="")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"))
    )
    p.save(out, width=7, height=4.5, dpi=150, verbose=False)
    print(f"saved {out}")


def roc_plot(scores_path: Path, out: Path):
    d = np.load(scores_path)
    fpr, tpr, _ = roc_curve(d["targets"], d["scores"])
    val_auc = float(d["val_auc"])
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    p = (
        ggplot(roc_df, aes("fpr", "tpr"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="grey")
        + geom_line(size=1.1, color="#d95f02")
        + annotate("text", x=0.62, y=0.18, label=f"AUC = {val_auc:.3f}", size=12)
        + coord_equal()
        + labs(title="Validation ROC Curve",
               x="False positive rate", y="True positive rate")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"))
    )
    p.save(out, width=5.5, height=5.5, dpi=150, verbose=False)
    print(f"saved {out}")


def comparison_plot(base_df, reg_df, out: Path):
    base = base_df[["epoch", "val_auc"]].assign(model="Baseline")
    reg = reg_df[["epoch", "val_auc"]].assign(model="Regularized (dropout 0.4, wd 1e-3)")
    both = pd.concat([base, reg], ignore_index=True)
    p = (
        ggplot(both, aes("epoch", "val_auc", color="model"))
        + geom_line(size=1.0)
        + geom_point(size=3.0)
        + scale_color_manual(values=MODEL_COLORS)
        + scale_x_continuous(breaks=sorted(both["epoch"].unique()))
        + labs(title="Validation ROC-AUC by epoch",
               x="Epoch", y="Validation ROC-AUC", color="")
        + theme_bw()
        + theme(plot_title=element_text(size=13, weight="bold"), legend_position="bottom")
    )
    p.save(out, width=7, height=5, dpi=150, verbose=False)
    print(f"saved {out}")


def main():
    df = pd.read_csv(RESULTS)
    epoch_plot(df, ("train_auc", "val_auc"), "ROC-AUC",
               "ROC-AUC by epoch", FIG_DIR / "auc_by_epoch.png")
    epoch_plot(df, ("train_loss", "val_loss"), "Cross entropy loss",
               "Cross Entropy Loss by epoch", FIG_DIR / "loss_by_epoch.png")

    scores_path = EMBEDDING_DIR / "val_scores.npz"
    if scores_path.exists():
        roc_plot(scores_path, FIG_DIR / "val_roc_curve.png")
    else:
        print(f"{scores_path} not found")

    if RESULTS_REG.exists():
        comparison_plot(df, pd.read_csv(RESULTS_REG), FIG_DIR / "baseline_vs_regularized.png")


if __name__ == "__main__":
    main()
