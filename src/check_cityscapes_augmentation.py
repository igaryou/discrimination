import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset import (
    CITYSCAPES_IGNORE_INDEX,
    Cityscapes20ClassDataset,
)
from mmseg_model_factory import build_mmseg_model, forward_logits
from visualization import (
    colorize_mask,
    denormalize_image,
    tensor_to_uint8_image,
)


TRAIN_PIPELINE = (
    "random_resize",
    "random_crop",
    "flip",
    "photometric_distortion",
    "to_tensor",
    "torchvision_normalise",
)
VAL_PIPELINE = ("resize", "to_tensor", "torchvision_normalise")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and visualize the Cityscapes training augmentation."
    )
    parser.add_argument("--cityscapes_root", default="~/datasets/cityscapes")
    parser.add_argument("--output_dir", default="result/augmentation_check")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="segformer")
    parser.add_argument("--backbone", default="mit_b0")
    parser.add_argument("--skip_forward", action="store_true")
    return parser.parse_args()


def make_datasets(root):
    train_dataset = Cityscapes20ClassDataset(
        root=root,
        split="train",
        mode="fine",
        image_size=(256, 512),
        train_base_size=(256, 512),
        train_crop_size=(256, 256),
        random_scale_range=(0.5, 2.0),
        cat_max_ratio=0.75,
        is_train=True,
        pipeline=TRAIN_PIPELINE,
        hflip_prob=0.5,
    )
    val_dataset = Cityscapes20ClassDataset(
        root=root,
        split="val",
        mode="fine",
        image_size=(256, 512),
        is_train=False,
        pipeline=VAL_PIPELINE,
    )
    return train_dataset, val_dataset


def class_ratio(mask):
    valid = mask[mask != CITYSCAPES_IGNORE_INDEX]
    if valid.numel() == 0:
        return None, 0
    _, counts = torch.unique(valid, return_counts=True)
    return float(counts.max().float() / counts.sum()), int(counts.numel())


def validate_ids(mask):
    observed = set(int(value) for value in torch.unique(mask).tolist())
    allowed = set(range(19)) | {CITYSCAPES_IGNORE_INDEX}
    invalid = sorted(observed - allowed)
    if invalid:
        raise AssertionError(f"Mask contains invalid class IDs: {invalid}")
    return sorted(observed)


def save_sample(image, mask, output_dir, index):
    image_uint8 = tensor_to_uint8_image(denormalize_image(image))
    mask_color = colorize_mask(mask)
    overlay = np.clip(
        image_uint8.astype(np.float32) * 0.55
        + mask_color.astype(np.float32) * 0.45,
        0.0,
        255.0,
    ).round().astype(np.uint8)

    for name, array in (
        ("image", image_uint8),
        ("mask", mask_color),
        ("overlay", overlay),
    ):
        directory = output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(directory / f"{index:04d}.png")


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

    torch.manual_seed(args.seed)
    root = os.path.expanduser(args.cityscapes_root)
    output_dir = Path(args.output_dir)
    train_dataset, val_dataset = make_datasets(root)

    samples = []
    for index in range(min(args.num_samples, len(train_dataset))):
        image, _, mask = train_dataset[index]
        if tuple(image.shape) != (3, 256, 256):
            raise AssertionError(f"Unexpected train image shape: {tuple(image.shape)}")
        if tuple(mask.shape) != (256, 256):
            raise AssertionError(f"Unexpected train mask shape: {tuple(mask.shape)}")
        observed_ids = validate_ids(mask)
        max_ratio, num_valid_classes = class_ratio(mask)
        samples.append(
            {
                "index": index,
                "observed_ids": observed_ids,
                "num_valid_classes": num_valid_classes,
                "max_class_ratio": max_ratio,
                "cat_max_ratio_satisfied": (
                    max_ratio is not None
                    and num_valid_classes > 1
                    and max_ratio < 0.75
                ),
                "note": (
                    None
                    if (
                        max_ratio is not None
                        and num_valid_classes > 1
                        and max_ratio < 0.75
                    )
                    else "Allowed fallback: no valid multi-class crop met the "
                    "constraint within 10 trials."
                ),
            }
        )
        save_sample(image, mask, output_dir, index)

    val_image, _, val_mask = val_dataset[0]
    if tuple(val_image.shape) != (3, 256, 512):
        raise AssertionError(f"Unexpected val image shape: {tuple(val_image.shape)}")
    if tuple(val_mask.shape) != (256, 512):
        raise AssertionError(f"Unexpected val mask shape: {tuple(val_mask.shape)}")
    val_ids = validate_ids(val_mask)

    forward_result = {"skipped": True}
    if not args.skip_forward:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        images, _, masks = next(iter(loader))
        images = images.to(device)
        masks = masks.to(device)
        model = build_mmseg_model(
            model=args.model,
            backbone=args.backbone,
            num_classes=20,
            pretrained=False,
        ).to(device)
        model.train()
        logits = forward_logits(model, images, masks)
        loss = F.cross_entropy(logits, masks)
        loss.backward()
        forward_result = {
            "skipped": False,
            "device": str(device),
            "batch_image_shape": list(images.shape),
            "batch_mask_shape": list(masks.shape),
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "backward": "passed",
        }

    result = {
        "seed": args.seed,
        "ignore_index": CITYSCAPES_IGNORE_INDEX,
        "train_pipeline": list(TRAIN_PIPELINE),
        "val_pipeline": list(VAL_PIPELINE),
        "train_image_shape": [3, 256, 256],
        "train_mask_shape": [256, 256],
        "val_image_shape": list(val_image.shape),
        "val_mask_shape": list(val_mask.shape),
        "val_observed_ids": val_ids,
        "samples": samples,
        "forward": forward_result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "report.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"visualizations and report saved to: {output_dir}")


if __name__ == "__main__":
    main()
