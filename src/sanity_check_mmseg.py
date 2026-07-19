import argparse
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn.functional as F

from mmseg_model_factory import (
    SUPPORTED_BACKBONES,
    SUPPORTED_MODELS,
    build_mmseg_model,
    count_parameters,
    forward_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--backbone", type=str, choices=SUPPORTED_BACKBONES, default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 256])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def amp_settings(args: argparse.Namespace, device: torch.device) -> Tuple[bool, torch.dtype]:
    enabled = bool(args.amp and device.type == "cuda")
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return enabled, dtype


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = amp_settings(args, device)

    model = build_mmseg_model(
        model=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    model.train()
    if args.batch_size == 1:
        # ASPP image pooling produces [1,C,1,1], which training BatchNorm
        # cannot normalize. Keep the rest of the model in training mode.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    height, width = args.image_size
    images = torch.rand(args.batch_size, 3, height, width, device=device)
    masks = torch.randint(
        low=0,
        high=args.num_classes,
        size=(args.batch_size, height, width),
        device=device,
        dtype=torch.long,
    )

    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_enabled, amp_dtype):
        logits = forward_logits(model, images, masks)
        loss = F.cross_entropy(logits, masks)

    expected_shape = (args.batch_size, args.num_classes, height, width)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected logits shape: got {tuple(logits.shape)}, "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError("NaN or Inf detected in logits")
    if not torch.isfinite(loss):
        raise RuntimeError("NaN or Inf detected in Cross Entropy loss")

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

    if not torch.isfinite(grad_norm):
        raise RuntimeError("NaN or Inf detected in gradients")
    if not all(
        torch.isfinite(parameter).all().item() for parameter in model.parameters()
    ):
        raise RuntimeError("NaN or Inf detected in parameters after optimizer.step")

    params = count_parameters(model)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu_name: {torch.cuda.get_device_name(device)}")
    print(f"model: {args.model}")
    print(f"backbone: {args.backbone}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss.detach().item()):.6f}")
    print(f"grad_norm: {float(grad_norm):.6f}")
    print(f"params/total: {params['params/total']}")
    print(f"params/trainable_total: {params['params/trainable_total']}")
    print("forward: passed")
    print("cross_entropy: passed")
    print("backward: passed")
    print("optimizer_step: passed")
    print("finite_check: passed")
    print("sanity check passed")


if __name__ == "__main__":
    main()
