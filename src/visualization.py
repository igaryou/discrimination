from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CITYSCAPES_20_PALETTE, IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3,H,W], got {tuple(image.shape)}")
    mean_tensor = image.new_tensor(mean).view(-1, 1, 1)
    std_tensor = image.new_tensor(std).view(-1, 1, 1)
    return image * std_tensor + mean_tensor


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3,H,W], got {tuple(image.shape)}")
    array = image.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def colorize_mask(mask: torch.Tensor, palette: np.ndarray = CITYSCAPES_20_PALETTE) -> np.ndarray:
    mask_np = mask.detach().cpu().long().numpy()
    out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid = (mask_np >= 0) & (mask_np < len(palette))
    out[valid] = palette[mask_np[valid]]
    return out


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(image)).save(path)


def save_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_mask(mask)).save(path)


@torch.no_grad()
def save_visualization_triplets(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    save_dir: Path,
    start_index: int = 0,
    max_count: int = 16,
    images_normalized: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> int:
    save_dir = Path(save_dir)
    image_dir = save_dir / "image"
    gt_dir = save_dir / "gt"
    pred_dir = save_dir / "pred"
    num_save = min(max_count, images.shape[0])
    for i in range(num_save):
        idx = start_index + i
        image = images[i]
        if images_normalized:
            image = denormalize_image(image, mean=mean, std=std)
        save_image(image, image_dir / f"{idx:04d}.png")
        save_mask(masks[i], gt_dir / f"{idx:04d}.png")
        save_mask(preds[i], pred_dir / f"{idx:04d}.png")
    return num_save


def save_loss_curve(losses: Sequence[float], save_path: Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
