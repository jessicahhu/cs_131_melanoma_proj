import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from config import CHECKPOINT_DIR, FIGURE_DIR, JPEG_DIR, TrainConfig
from dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_split
from model import MelanomaResNet18
from utils import get_device


def load_model(checkpoint: Path, device: torch.device) -> MelanomaResNet18:
    model = MelanomaResNet18(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def gradcam_for_image(model, image_path: Path, image_size: int, device):
    pil = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    rgb = np.asarray(pil) / 255.0

    tfm = build_transforms(image_size, train=False)
    tensor = tfm(pil).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=[model.gradcam_target_layer])
    grayscale = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return rgb, overlay, prob


def main(checkpoint: Path, n_examples: int, out_dir: Path):
    device = get_device()
    cfg = TrainConfig()
    model = load_model(checkpoint, device)

    _, val_df = load_split(cfg)
    pos = val_df[val_df["target"] == 1].head(n_examples)
    neg = val_df[val_df["target"] == 0].head(n_examples)
    samples = list(pos["image_name"]) + list(neg["image_name"])
    labels = [1] * len(pos) + [0] * len(neg)

    fig, axes = plt.subplots(len(samples), 2, figsize=(6, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for i, (name, label) in enumerate(zip(samples, labels)):
        rgb, overlay, prob = gradcam_for_image(model, JPEG_DIR / f"{name}.jpg", cfg.image_size, device)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{name}  label={label}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM  p(mel)={prob:.3f}")
        axes[i, 1].axis("off")

    out = out_dir / "gradcam_examples.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved Grad-CAM grid to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "resnet18_best.pt")
    parser.add_argument("--n", type=int, default=4, help="examples per class")
    args = parser.parse_args()
    main(args.checkpoint, args.n, FIGURE_DIR)
