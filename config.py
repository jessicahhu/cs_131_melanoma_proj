from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
JPEG_DIR = DATA_DIR / "train"
TRAIN_CSV = DATA_DIR / "train.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
EMBEDDING_DIR = ARTIFACTS_DIR / "embeddings"
FIGURE_DIR = ARTIFACTS_DIR / "figures"

for d in (CHECKPOINT_DIR, EMBEDDING_DIR, FIGURE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    image_size: int = 192
    batch_size: int = 96
    num_workers: int = 4
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.0
    val_fraction: float = 0.15
    seed: int = 42
    pos_oversample: float = 20.0
    subsample_fraction: float | None = None
