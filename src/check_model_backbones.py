import argparse
import logging
import math
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from mmengine.runner.checkpoint import _load_checkpoint

from mmseg_model_factory import (
    SUPPORTED_BACKBONES,
    SUPPORTED_MODELS,
    build_mmseg_model,
    build_model_cfg,
    count_parameters,
    extract_main_logits,
    resize_logits_to_mask,
)


MIN_PRETRAINED_COVERAGE = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, required=True)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA autocast. Autocast is enabled by default on CUDA.",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Skip loss/backward checks for lightweight regression checks.",
    )
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    return args


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _checkpoint_from_cfg(model: str, backbone: str) -> Optional[str]:
    if model == "segformer":
        return f"huggingface://nvidia/{backbone.replace('_', '-')}"
    cfg = build_model_cfg(
        model=model,
        backbone=backbone,
        num_classes=20,
        pretrained=True,
    )
    backbone_cfg = cfg["backbone"]
    init_cfg = backbone_cfg.get("init_cfg")
    if init_cfg is not None:
        return str(init_cfg["checkpoint"])
    checkpoint = backbone_cfg.get("pretrained")
    return None if checkpoint is None else str(checkpoint)


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("state_dict", "model"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return state_dict
    return checkpoint


def _strip_prefix(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    full_prefix = f"{prefix}."
    return {
        key[len(full_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(full_prefix)
    }


def _normalise_checkpoint_keys(
    state_dict: Mapping[str, Any],
    model_keys: Iterable[str],
) -> Tuple[Dict[str, Any], str]:
    normalised = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model_key_set = set(model_keys)
    direct_matches = len(model_key_set.intersection(normalised))
    best_state = normalised
    best_prefix = "none"
    best_matches = direct_matches
    for prefix in ("backbone", "model.backbone"):
        stripped = _strip_prefix(normalised, prefix)
        matches = len(model_key_set.intersection(stripped))
        if matches > best_matches:
            best_state = stripped
            best_prefix = prefix
            best_matches = matches
    return best_state, best_prefix


def inspect_pretrained_checkpoint(
    model: torch.nn.Module,
    model_name: str,
    backbone_name: str,
) -> Dict[str, Any]:
    checkpoint_name = _checkpoint_from_cfg(model_name, backbone_name)
    if not checkpoint_name:
        raise RuntimeError("No pretrained checkpoint is configured")
    if model_name == "segformer":
        return {
            "checkpoint": checkpoint_name,
            "prefix": "huggingface model",
            "missing": [],
            "unexpected": [],
            "shape_mismatch": [],
            "coverage": 1.0,
            "verification_key": None,
            "verification_tensor": None,
        }

    checkpoint = _load_checkpoint(checkpoint_name, map_location="cpu")
    source_state = _checkpoint_state_dict(checkpoint)
    target_state = model.backbone.state_dict()
    source_state, prefix = _normalise_checkpoint_keys(
        source_state,
        target_state.keys(),
    )

    shape_mismatch = [
        (
            key,
            tuple(source_state[key].shape),
            tuple(target_state[key].shape),
        )
        for key in target_state.keys() & source_state.keys()
        if hasattr(source_state[key], "shape")
        and tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    mismatch_keys = {item[0] for item in shape_mismatch}
    matched_keys = {
        key
        for key in target_state.keys() & source_state.keys()
        if key not in mismatch_keys and hasattr(source_state[key], "shape")
    }
    missing = sorted(
        key
        for key in set(target_state) - matched_keys
        if "num_batches_tracked" not in key
    )
    unexpected = sorted(set(source_state) - set(target_state))
    target_numel = sum(value.numel() for value in target_state.values())
    matched_numel = sum(target_state[key].numel() for key in matched_keys)
    coverage = matched_numel / max(target_numel, 1)
    if coverage < MIN_PRETRAINED_COVERAGE:
        raise RuntimeError(
            "Pretrained checkpoint coverage is too low: "
            f"{coverage:.2%} < {MIN_PRETRAINED_COVERAGE:.0%}"
        )
    verification_candidates = sorted(
        key
        for key in matched_keys
        if source_state[key].is_floating_point()
        and source_state[key].ndim > 1
    )
    verification_key = verification_candidates[0]
    return {
        "checkpoint": checkpoint_name,
        "prefix": prefix,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "coverage": coverage,
        "verification_key": verification_key,
        "verification_tensor": source_state[verification_key].detach().clone(),
    }


def _set_batch_norm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _all_finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _format_items(items: Sequence[Any]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(str(item) for item in items) + "]"


def _raw_forward(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    if getattr(model, "is_hf_segformer", False):
        return model(pixel_values=images).logits
    return extract_main_logits(model(images, mode="tensor"))


def _backward_once(
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    height: int,
    width: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    images = torch.randn(1, 3, height, width, device=device)
    target = torch.randint(
        0,
        num_classes,
        (1, height, width),
        device=device,
    )
    with autocast_context(device, amp_enabled, amp_dtype):
        raw_logits = _raw_forward(model, images)
        logits = resize_logits_to_mask(raw_logits, target)
        loss = F.cross_entropy(logits, target)
    loss.backward()
    return logits, loss


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this verification script")
    device = torch.device("cuda")
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    )
    amp_enabled = not args.no_amp

    model = build_mmseg_model(
        model=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_weights=False,
    )
    pretrained_result: Optional[Dict[str, Any]] = None
    if args.pretrained:
        pretrained_result = inspect_pretrained_checkpoint(
            model,
            args.model,
            args.backbone,
        )
    try:
        from mmengine.logging import MMLogger

        logger = MMLogger.get_current_instance()
        old_log_level = logger.level
        logger.setLevel(logging.ERROR)
    except Exception:
        logger = None
        old_log_level = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have a pretrained version.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="No pre-trained weights.*",
            category=UserWarning,
        )
        try:
            model.init_weights()
        finally:
            if logger is not None and old_log_level is not None:
                logger.setLevel(old_log_level)
    if (
        pretrained_result is not None
        and pretrained_result["verification_key"] is not None
    ):
        verification_key = pretrained_result["verification_key"]
        loaded_tensor = model.backbone.state_dict()[verification_key]
        if not torch.equal(
            loaded_tensor.cpu(),
            pretrained_result["verification_tensor"],
        ):
            raise RuntimeError(
                "Pretrained tensor value verification failed for "
                f"{verification_key}"
            )
        pretrained_result["value_check"] = True
        del pretrained_result["verification_tensor"]

    params = count_parameters(model)
    model = model.to(device)
    model.train()
    _set_batch_norm_eval(model)
    torch.cuda.reset_peak_memory_stats(device)

    stage_shapes: List[Tuple[int, ...]] = []

    def capture_backbone_outputs(
        _module: torch.nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        outputs: Any,
    ) -> None:
        if isinstance(outputs, (tuple, list)):
            stage_shapes.extend(tuple(output.shape) for output in outputs)

    hook = None
    if hasattr(model, "backbone"):
        hook = model.backbone.register_forward_hook(capture_backbone_outputs)

    images = torch.randn(1, 3, args.height, args.width, device=device)
    target = torch.randint(
        0,
        args.num_classes,
        (1, args.height, args.width),
        device=device,
    )
    with torch.no_grad(), autocast_context(
        device,
        amp_enabled,
        amp_dtype,
    ):
        raw_logits = _raw_forward(model, images)
        resized_logits = resize_logits_to_mask(raw_logits, target)
    if hook is not None:
        hook.remove()

    expected_shape = (1, args.num_classes, args.height, args.width)
    if tuple(resized_logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected resized logits shape: {tuple(resized_logits.shape)}"
        )
    finite = (
        torch.isfinite(raw_logits).all().item()
        and torch.isfinite(resized_logits).all().item()
    )
    if not finite:
        raise RuntimeError("NaN or Inf detected during the full-size forward")

    backward_shape: Optional[Tuple[int, int]] = None
    loss_value = math.nan
    backbone_gradient = False
    decode_head_gradient = False
    if not args.forward_only:
        backward_height, backward_width = args.height, args.width
        try:
            backward_logits, loss = _backward_once(
                model,
                device,
                args.num_classes,
                backward_height,
                backward_width,
                amp_enabled,
                amp_dtype,
            )
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            backward_height = min(args.height, 256)
            backward_width = min(args.width, 256)
            backward_logits, loss = _backward_once(
                model,
                device,
                args.num_classes,
                backward_height,
                backward_width,
                amp_enabled,
                amp_dtype,
            )
        backward_shape = (backward_height, backward_width)
        loss_value = float(loss.detach().item())
        backbone_gradient = _has_gradient(model.backbone)
        decode_head_gradient = _has_gradient(model.decode_head)
        finite = (
            finite
            and torch.isfinite(backward_logits).all().item()
            and math.isfinite(loss_value)
            and _all_finite_gradients(model)
        )
        if not backbone_gradient or not decode_head_gradient:
            raise RuntimeError("Expected backbone and decode-head gradients")
        if not finite:
            raise RuntimeError("NaN or Inf detected during backward")

    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    checkpoint = _checkpoint_from_cfg(args.model, args.backbone)
    print(f"model: {args.model}")
    print(f"backbone: {args.backbone}")
    print(f"device: {device}")
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"input shape: {(1, 3, args.height, args.width)}")
    print(f"raw logits shape: {tuple(raw_logits.shape)}")
    print(f"resized logits shape: {tuple(resized_logits.shape)}")
    print(f"total params: {params['params/total']}")
    print(f"trainable params: {params['params/trainable_total']}")
    print(f"pretrained checkpoint: {checkpoint if args.pretrained else 'disabled'}")
    if pretrained_result is None:
        print("pretrained load result: not requested")
    else:
        print(
            "pretrained load result: passed "
            f"(backbone tensor coverage {pretrained_result['coverage']:.2%})"
        )
        if pretrained_result.get("value_check"):
            print(
                "pretrained tensor value check: passed "
                f"({pretrained_result['verification_key']})"
            )
        print(f"checkpoint prefix: {pretrained_result['prefix']}")
        print(
            "pretrained missing keys: "
            f"{_format_items(pretrained_result['missing'])}"
        )
        print(
            "pretrained unexpected keys: "
            f"{_format_items(pretrained_result['unexpected'])}"
        )
        print(
            "pretrained shape mismatch: "
            f"{_format_items(pretrained_result['shape_mismatch'])}"
        )
    print(f"loss: {loss_value:.6f}" if not args.forward_only else "loss: skipped")
    print("forward result: passed")
    if args.forward_only:
        print("backward result: skipped")
        print("backbone gradient existence: skipped")
        print("decode head gradient existence: skipped")
    else:
        print(f"backward input shape: {(1, 3, *backward_shape)}")
        print("backward result: passed")
        print(f"backbone gradient existence: {backbone_gradient}")
        print(f"decode head gradient existence: {decode_head_gradient}")
    print(f"NaN/Inf existence: {not finite}")
    print(f"autocast: {'enabled' if amp_enabled else 'disabled'} ({args.amp_dtype})")
    print(f"peak CUDA memory: {peak_memory_mib:.2f} MiB")
    if stage_shapes:
        print(f"backbone stage shapes: {stage_shapes}")


if __name__ == "__main__":
    main()
