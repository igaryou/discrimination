import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    STOCHASTIC_DATASET_TRANSFORMS,
    Cityscapes20ClassDataset,
)
from metrics import SegmentationMetrics
from mmseg_model_factory import (
    SUPPORTED_BACKBONES,
    SUPPORTED_MODELS,
    build_mmseg_model,
    forward_logits,
)
from visualization import save_visualization_triplets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str, default="~/datasets/cityscapes")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--backbone", type=str, choices=SUPPORTED_BACKBONES, default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--eval_num_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 256])
    parser.add_argument("--train_pipeline", nargs="+", default=["resize"])
    parser.add_argument("--val_pipeline", nargs="+", default=["resize"])
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--color_jitter_prob", type=float, default=1.0)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.5)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.5)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.5)
    parser.add_argument("--color_jitter_hue", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_visualize", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.eval_num_classes is None:
        args.eval_num_classes = args.num_classes
    return args


def amp_settings(args: argparse.Namespace, device: torch.device) -> Tuple[bool, torch.dtype]:
    enabled = bool(args.amp and device.type == "cuda")
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return enabled, dtype


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = Cityscapes20ClassDataset(
        root=os.path.expanduser(args.cityscapes_root),
        split=args.split,
        mode="fine",
        image_size=tuple(args.image_size),
        is_train=False,
        pipeline=args.val_pipeline,
        hflip_prob=args.hflip_prob,
        color_jitter_prob=args.color_jitter_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    vis_dir: Optional[Path],
    num_visualize: int,
    images_normalized: bool = False,
    eval_num_classes: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    if eval_num_classes is None:
        eval_num_classes = num_classes
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        eval_num_classes=eval_num_classes,
        ignore_target_classes=tuple(range(eval_num_classes, num_classes)),
    )
    visualized = 0
    for images, _, masks in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = forward_logits(model, images, masks)
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        if vis_dir is not None and visualized < num_visualize:
            remaining = num_visualize - visualized
            saved = save_visualization_triplets(
                images=images.cpu(),
                masks=masks.cpu(),
                preds=preds.cpu(),
                save_dir=vis_dir,
                start_index=visualized,
                max_count=remaining,
                images_normalized=images_normalized,
            )
            visualized += saved
    return metrics.compute()


def main() -> None:
    args = parse_args()
    if args.eval_num_classes is None:
        args.eval_num_classes = args.num_classes
    if not 1 <= args.eval_num_classes <= args.num_classes:
        raise ValueError(
            "--eval_num_classes must be in [1, --num_classes], got "
            f"{args.eval_num_classes} for --num_classes {args.num_classes}"
        )
    stochastic = [
        transform
        for transform in args.val_pipeline
        if transform in STOCHASTIC_DATASET_TRANSFORMS
    ]
    if stochastic:
        raise ValueError(
            "Evaluation --val_pipeline must not contain stochastic transforms: "
            f"{stochastic}"
        )
    result_dir = Path(args.result_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else result_dir / "model.pth"
    save_dir = result_dir / f"eval_{args.split}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = amp_settings(args, device)
    loader = make_loader(args)
    model = build_mmseg_model(
        model=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    result = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        vis_dir=save_dir / "visualizations",
        num_visualize=args.num_visualize,
        images_normalized="torchvision_normalise" in args.val_pipeline,
        eval_num_classes=args.eval_num_classes,
    )
    result.update(
        {
            "split": args.split,
            "batch_size": args.batch_size,
            "image_size": list(args.image_size),
            "checkpoint": str(checkpoint),
            "model": args.model,
            "backbone": args.backbone,
            "val_pipeline": list(args.val_pipeline),
        }
    )

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"split      : {result['split']}\n")
        f.write(f"pixel_acc  : {result['pixel_acc']:.6f}\n")
        f.write(f"mIoU       : {result['mIoU']:.6f}\n")
        f.write(f"mAcc       : {result['mAcc']:.6f}\n")
        f.write(f"checkpoint : {result['checkpoint']}\n")

    print(f"pixel_acc: {result['pixel_acc']:.6f}")
    print(f"mIoU     : {result['mIoU']:.6f}")
    print(f"mAcc     : {result['mAcc']:.6f}")
    print(f"saved to : {save_dir}")


if __name__ == "__main__":
    main()
