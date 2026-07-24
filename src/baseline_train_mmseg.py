import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Cityscapes20ClassDataset
from metrics import SegmentationMetrics
from mmseg_model_factory import (
    SUPPORTED_BACKBONES,
    SUPPORTED_MODELS,
    build_mmseg_model,
    count_parameters,
    forward_logits,
)
from visualization import save_loss_curve, save_visualization_triplets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", type=str, default="~/datasets/cityscapes")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--backbone", type=str, choices=SUPPORTED_BACKBONES, default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--eval_num_classes", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 256])
    parser.add_argument("--train_base_size", type=int, nargs=2, default=[256, 512])
    parser.add_argument("--train_crop_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--random_scale_min", type=float, default=0.5)
    parser.add_argument("--random_scale_max", type=float, default=2.0)
    parser.add_argument("--cat_max_ratio", type=float, default=0.75)
    parser.add_argument("--train_pipeline", nargs="+", default=["resize"])
    parser.add_argument("--val_pipeline", nargs="+", default=["resize"])
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--color_jitter_prob", type=float, default=1.0)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.5)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.5)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.5)
    parser.add_argument("--color_jitter_hue", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=5e-7)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_visualize", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_mode", type=str, choices=("full", "model_only"), default="full")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cityscapes20-mmseg-baselines")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_epochs = max(0, min(warmup_epochs, epochs))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=eta_min,
        )
    if epochs <= warmup_epochs:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(epochs, 1),
        )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(epochs - warmup_epochs, 1),
                eta_min=eta_min,
            ),
        ],
        milestones=[warmup_epochs],
    )


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    root = os.path.expanduser(args.cityscapes_root)
    image_size = tuple(args.image_size)
    train_dataset = Cityscapes20ClassDataset(
        root=root,
        split="train",
        mode="fine",
        image_size=image_size,
        train_base_size=tuple(args.train_base_size),
        train_crop_size=tuple(args.train_crop_size),
        random_scale_range=(args.random_scale_min, args.random_scale_max),
        cat_max_ratio=args.cat_max_ratio,
        is_train=True,
        pipeline=args.train_pipeline,
        hflip_prob=args.hflip_prob,
        color_jitter_prob=args.color_jitter_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
    )
    val_dataset = Cityscapes20ClassDataset(
        root=root,
        split="val",
        mode="fine",
        image_size=image_size,
        is_train=False,
        pipeline=args.val_pipeline,
        hflip_prob=args.hflip_prob,
        color_jitter_prob=args.color_jitter_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def init_wandb(args: argparse.Namespace, config: Dict[str, object]):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb logging.")
        return None
    init_kwargs = dict(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
    )
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_miou: float,
    config: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mIoU": best_miou,
            "config": config,
        },
        path,
    )


def load_resume(
    resume_path: str,
    resume_mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    current_config: Optional[Dict[str, object]] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(resume_path, map_location=device)
    checkpoint_config = ckpt.get("config", {})
    if current_config is not None and isinstance(checkpoint_config, dict):
        mismatches = []
        for key in ("model", "backbone", "num_classes", "eval_num_classes"):
            if key not in checkpoint_config:
                continue
            checkpoint_value = checkpoint_config[key]
            current_value = current_config[key]
            if checkpoint_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint={checkpoint_value!r}, current={current_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Resume checkpoint is incompatible with the requested model: "
                + "; ".join(mismatches)
            )
    model.load_state_dict(ckpt["model"], strict=True)
    if resume_mode == "model_only":
        return 0, float("-inf")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", 0))
    best_miou = float(ckpt.get("best_mIoU", float("-inf")))
    return start_epoch, best_miou


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    vis_dir: Optional[Path] = None,
    num_visualize: int = 16,
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
    for images, _, masks in tqdm(val_loader, desc="val", leave=False):
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


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    epochs: int,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    grad_norm_sum = 0.0
    count = 0
    for images, _, masks in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = forward_logits(model, images, masks)
            loss = F.cross_entropy(logits, masks)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_sum += float(loss.detach().item())
        grad_norm_sum += float(grad_norm)
        count += 1

    return loss_sum / max(count, 1), grad_norm_sum / max(count, 1)


def main() -> None:
    args = parse_args()
    if args.eval_interval < 1:
        raise ValueError("--eval_interval must be >= 1")
    if args.eval_num_classes is None:
        args.eval_num_classes = args.num_classes
    if not 1 <= args.eval_num_classes <= args.num_classes:
        raise ValueError(
            "--eval_num_classes must be in [1, --num_classes], got "
            f"{args.eval_num_classes} for --num_classes {args.num_classes}"
        )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "train_log.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = amp_settings(args, device)

    train_loader, val_loader = make_loaders(args)
    model = build_mmseg_model(
        model=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    params = count_parameters(model)
    config = vars(args).copy()
    config.update(params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    start_epoch = 0
    best_miou = float("-inf")
    if args.resume is not None:
        start_epoch, best_miou = load_resume(
            resume_path=args.resume,
            resume_mode=args.resume_mode,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            current_config=config,
        )

    with open(result_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    wandb_run = init_wandb(args, config)
    losses = []

    for epoch_idx in range(start_epoch, args.epochs):
        epoch = epoch_idx + 1
        loss_avg, grad_norm_avg = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch,
            epochs=args.epochs,
        )
        losses.append(loss_avg)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        should_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        val_result: Optional[Dict[str, object]] = None
        if should_eval:
            val_result = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                vis_dir=result_dir / "infer_val" / f"epoch_{epoch:03d}",
                num_visualize=args.num_visualize,
                images_normalized="torchvision_normalise" in args.val_pipeline,
                eval_num_classes=args.eval_num_classes,
            )
            val_miou = float(val_result["mIoU"])
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(
                    result_dir / "model.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_miou=best_miou,
                    config=config,
                )

        log_line = (
            f"epoch:{epoch}  loss_avg:{loss_avg:.6f}  "
            f"lr:{current_lr:.8f}  grad_norm:{grad_norm_avg:.6f}"
        )
        if val_result is not None:
            log_line += (
                f"  val_pixel_acc:{float(val_result['pixel_acc']):.6f}"
                f"  val_mIoU:{float(val_result['mIoU']):.6f}"
                f"  val_mAcc:{float(val_result['mAcc']):.6f}"
                f"  best_mIoU:{best_miou:.6f}"
            )
        print(log_line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        save_loss_curve(losses, result_dir / "loss_curves" / f"loss_{epoch}.png")

        if wandb_run is not None:
            import wandb

            wandb_log = {
                "train/loss": loss_avg,
                "train/lr": current_lr,
                "train/grad_norm": grad_norm_avg,
                "epoch": epoch,
            }
            if val_result is not None:
                wandb_log.update(
                    {
                        "val/pixel_acc": float(val_result["pixel_acc"]),
                        "val/mIoU": float(val_result["mIoU"]),
                        "val/mAcc": float(val_result["mAcc"]),
                    }
                )
            wandb.log(wandb_log, step=epoch)

    save_checkpoint(
        result_dir / "model_final.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_miou=best_miou,
        config=config,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
