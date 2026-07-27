import copy
import importlib
import logging
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.config import ConfigDict
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


SUPPORTED_MODELS = ("deeplabv3plus", "pspnet", "unet", "upernet", "segformer")
RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101")
CONVNEXT_BACKBONES = ("convnext_base", "convnext_large")
TIMM_CONVNEXTV2_BACKBONES = ("convnextv2_huge",)
SWIN_BACKBONES = ("swin_small", "swin_base", "swin_large")
SEGFORMER_BACKBONES = ("mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5")
SUPPORTED_BACKBONES = (
    RESNET_BACKBONES
    + CONVNEXT_BACKBONES
    + TIMM_CONVNEXTV2_BACKBONES
    + SWIN_BACKBONES
    + SEGFORMER_BACKBONES
)

CONVNEXTV2_HUGE_TIMM_NAME = "convnextv2_huge.fcmae_ft_in22k_in1k_512"
CONVNEXTV2_HUGE_HF_NAME = "hf_hub:timm/convnextv2_huge.fcmae_ft_in22k_in1k_512"

_MMSEG_REGISTERED = False


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for NCHW feature maps."""

    _abbr_ = "ln"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)


def _install_mmseg_utils_shim() -> None:
    if "mmseg.utils" in sys.modules and hasattr(sys.modules["mmseg.utils"], "ConfigType"):
        return

    utils_mod = types.ModuleType("mmseg.utils")
    config_type = Union[ConfigDict, dict]
    opt_config_type = Optional[config_type]
    multi_config = Union[config_type, Sequence[config_type]]
    opt_multi_config = Optional[multi_config]
    sample_list = Sequence[SegDataSample]
    opt_sample_list = Optional[sample_list]

    utils_mod.ConfigType = config_type
    utils_mod.OptConfigType = opt_config_type
    utils_mod.MultiConfig = multi_config
    utils_mod.OptMultiConfig = opt_multi_config
    utils_mod.SampleList = sample_list
    utils_mod.OptSampleList = opt_sample_list
    utils_mod.TensorDict = Dict[str, torch.Tensor]
    utils_mod.TensorList = Sequence[torch.Tensor]
    utils_mod.ForwardResults = Union[
        Dict[str, torch.Tensor],
        List[SegDataSample],
        Tuple[torch.Tensor],
        torch.Tensor,
    ]
    utils_mod.add_prefix = lambda inputs, prefix: {
        f"{prefix}.{key}": value for key, value in inputs.items()
    }
    sys.modules["mmseg.utils"] = utils_mod


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]
    return module


def _register_layer_norm2d() -> None:
    if MMENGINE_MODELS.get("LN2d") is None:
        MMENGINE_MODELS.register_module(name="LN2d", module=LayerNorm2d)


def _register_torchvision_convnext_backbone() -> None:
    if MODELS.get("TorchVisionConvNeXt") is not None:
        return

    from mmengine.model import BaseModule
    from torchvision.models import (
        ConvNeXt_Base_Weights,
        ConvNeXt_Large_Weights,
        convnext_base,
        convnext_large,
    )

    class TorchVisionConvNeXt(BaseModule):
        ARCHES = {
            "base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
            "large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT),
        }
        STAGE_MODULES = {1: 0, 3: 1, 5: 2, 7: 3}

        def __init__(
            self,
            arch: str,
            out_indices: Sequence[int] = (0, 1, 2, 3),
            pretrained: bool = False,
            init_cfg: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__(init_cfg=init_cfg)
            if arch not in self.ARCHES:
                raise ValueError(f"Unsupported ConvNeXt arch: {arch}")
            out_indices = tuple(out_indices)
            if any(index not in (0, 1, 2, 3) for index in out_indices):
                raise ValueError("ConvNeXt out_indices must be in [0, 3]")

            model_fn, weights_enum = self.ARCHES[arch]
            weights = weights_enum if pretrained else None
            model = model_fn(weights=weights)
            self.features = model.features
            self.out_indices = out_indices

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            outs = []
            for module_idx, layer in enumerate(self.features):
                x = layer(x)
                stage_idx = self.STAGE_MODULES.get(module_idx)
                if stage_idx in self.out_indices:
                    outs.append(x)
            return tuple(outs)

    MODELS.register_module(
        name="TorchVisionConvNeXt",
        module=TorchVisionConvNeXt,
    )


def _register_timm_convnextv2_backbone() -> None:
    if MODELS.get("TimmConvNeXtV2") is not None:
        return

    from mmengine.model import BaseModule

    class TimmConvNeXtV2(BaseModule):
        def __init__(
            self,
            model_name: str,
            fallback_model_name: Optional[str] = None,
            out_indices: Sequence[int] = (0, 1, 2, 3),
            pretrained: bool = False,
            init_cfg: Optional[Dict[str, Any]] = None,
        ) -> None:
            super().__init__(init_cfg=init_cfg)
            self.out_indices = tuple(out_indices)
            self.model = _create_timm_features_model(
                model_name=model_name,
                fallback_model_name=fallback_model_name,
                pretrained=pretrained,
                out_indices=self.out_indices,
            )
            self.out_channels = tuple(self.model.feature_info.channels())

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            return tuple(self.model(x))

    MODELS.register_module(
        name="TimmConvNeXtV2",
        module=TimmConvNeXtV2,
    )


def _register_swin_backbone() -> None:
    if MODELS.get("SwinTransformer") is not None:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Fail to import ``MultiScaleDeformableAttention``.*",
            category=UserWarning,
        )
        importlib.import_module("mmseg.models.backbones.swin")


def _register_minimal_mmseg_modules() -> None:
    """Register only modules needed here, avoiding mmcv.ops-only imports."""
    global _MMSEG_REGISTERED
    if _MMSEG_REGISTERED:
        return

    import mmseg

    _install_mmseg_utils_shim()
    _register_layer_norm2d()

    mmseg_root = Path(mmseg.__file__).resolve().parent
    models_root = mmseg_root / "models"
    _ensure_package("mmseg.models", models_root)
    utils_pkg = _ensure_package("mmseg.models.utils", models_root / "utils")
    losses_pkg = _ensure_package("mmseg.models.losses", models_root / "losses")
    _ensure_package("mmseg.models.decode_heads", models_root / "decode_heads")
    _ensure_package("mmseg.models.backbones", models_root / "backbones")
    _ensure_package("mmseg.models.segmentors", models_root / "segmentors")

    res_layer = importlib.import_module("mmseg.models.utils.res_layer")
    wrappers = importlib.import_module("mmseg.models.utils.wrappers")
    utils_pkg.ResLayer = res_layer.ResLayer
    utils_pkg.resize = wrappers.resize
    utils_pkg.Upsample = wrappers.Upsample

    importlib.import_module("mmseg.models.builder")
    importlib.import_module("mmseg.models.utils.embed")
    importlib.import_module("mmseg.models.losses.utils")
    accuracy_mod = importlib.import_module("mmseg.models.losses.accuracy")
    ce_mod = importlib.import_module("mmseg.models.losses.cross_entropy_loss")
    losses_pkg.accuracy = accuracy_mod.accuracy
    losses_pkg.Accuracy = accuracy_mod.Accuracy
    losses_pkg.CrossEntropyLoss = ce_mod.CrossEntropyLoss

    for module_name in (
        "mmseg.models.decode_heads.decode_head",
        "mmseg.models.decode_heads.fcn_head",
        "mmseg.models.decode_heads.aspp_head",
        "mmseg.models.decode_heads.sep_aspp_head",
        "mmseg.models.decode_heads.psp_head",
        "mmseg.models.decode_heads.fpn_head",
        "mmseg.models.decode_heads.uper_head",
        "mmseg.models.backbones.resnet",
        "mmseg.models.segmentors.base",
        "mmseg.models.segmentors.encoder_decoder",
    ):
        importlib.import_module(module_name)

    _register_torchvision_convnext_backbone()
    _register_timm_convnextv2_backbone()
    _register_resnet_unet_head()
    _MMSEG_REGISTERED = True


def _register_resnet_unet_head() -> None:
    if MODELS.get("ResNetUNetHead") is not None:
        return

    from mmseg.models.decode_heads.decode_head import BaseDecodeHead
    from mmseg.models.utils import resize

    class ResNetUNetHead(BaseDecodeHead):
        """Top-down fusion head for four-stage multi-scale backbones.

        The registry name is retained for checkpoint/config compatibility, but
        the implementation is backbone-independent and supports ResNet and Swin
        feature pyramids through ``in_channels`` and ``norm_cfg``.
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(input_transform="multiple_select", **kwargs)
            self.lateral_convs = nn.ModuleList(
                [
                    ConvModule(
                        in_channels,
                        self.channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                    for in_channels in self.in_channels
                ]
            )
            self.fuse_convs = nn.ModuleList()
            for _ in range(len(self.in_channels) - 1):
                self.fuse_convs.append(
                    nn.Sequential(
                        ConvModule(
                            self.channels * 2,
                            self.channels,
                            kernel_size=3,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                        ),
                        ConvModule(
                            self.channels,
                            self.channels,
                            kernel_size=3,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                        ),
                    )
                )

        def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
            feats = self._transform_inputs(inputs)
            laterals = [
                conv(feat) for conv, feat in zip(self.lateral_convs, feats)
            ]
            x = laterals[-1]
            fuse_idx = 0
            for level in range(len(laterals) - 2, -1, -1):
                x = resize(
                    x,
                    size=laterals[level].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                x = torch.cat([x, laterals[level]], dim=1)
                x = self.fuse_convs[fuse_idx](x)
                fuse_idx += 1
            return self.cls_seg(x)

    MODELS.register_module(name="ResNetUNetHead", module=ResNetUNetHead)


def _is_resnet(backbone: str) -> bool:
    return backbone in RESNET_BACKBONES


def _is_convnext(backbone: str) -> bool:
    return backbone in CONVNEXT_BACKBONES


def _is_timm_convnextv2(backbone: str) -> bool:
    return backbone in TIMM_CONVNEXTV2_BACKBONES


def _is_swin(backbone: str) -> bool:
    return backbone in SWIN_BACKBONES


def validate_model_backbone(model: str, backbone: str) -> Tuple[str, str]:
    model = model.lower()
    backbone = backbone.lower()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if model == "segformer" and backbone not in SEGFORMER_BACKBONES:
        raise ValueError(
            "--model segformer requires a MiT backbone; expected one of "
            f"{SEGFORMER_BACKBONES}, got {backbone!r}"
        )
    if model != "segformer" and backbone in SEGFORMER_BACKBONES:
        raise ValueError(
            f"--backbone {backbone} is supported only with --model segformer"
        )
    if _is_swin(backbone) and model not in ("upernet", "unet"):
        raise ValueError(
            f"{backbone} is currently supported only with "
            "--model upernet or --model unet"
        )
    return model, backbone


def _backbone_depth(backbone: str) -> int:
    if not _is_resnet(backbone):
        raise ValueError(f"Unsupported backbone: {backbone}")
    return int(backbone.replace("resnet", ""))


def _backbone_channels(backbone: str) -> Tuple[int, int, int, int]:
    if _is_timm_convnextv2(backbone):
        return _timm_convnextv2_channels(backbone)
    if _is_convnext(backbone):
        if backbone == "convnext_base":
            return (128, 256, 512, 1024)
        return (192, 384, 768, 1536)
    if _is_swin(backbone):
        if backbone == "swin_small":
            return (96, 192, 384, 768)
        if backbone == "swin_base":
            return (128, 256, 512, 1024)
        if backbone == "swin_large":
            return (192, 384, 768, 1536)
        raise ValueError(f"Unsupported Swin backbone: {backbone}")

    depth = _backbone_depth(backbone)
    if depth in (18, 34):
        return (64, 128, 256, 512)
    return (256, 512, 1024, 2048)


def _head_channels(backbone: str, high_capacity: bool = True) -> int:
    if _is_convnext(backbone) or _is_timm_convnextv2(backbone) or _is_swin(backbone):
        return 512

    depth = _backbone_depth(backbone)
    if depth in (18, 34):
        return 256
    return 512 if high_capacity else 256


def _decode_norm_cfg(backbone: str) -> Dict[str, Any]:
    if _is_resnet(backbone):
        return dict(type="BN", requires_grad=True)
    return dict(type="LN2d", requires_grad=True, eps=1e-6)


def _create_timm_features_model(
    model_name: str,
    fallback_model_name: Optional[str],
    pretrained: bool,
    out_indices: Sequence[int],
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "convnextv2_huge requires timm. Install it in this environment "
            "before using --backbone convnextv2_huge."
        ) from exc

    kwargs = dict(
        pretrained=pretrained,
        features_only=True,
        out_indices=tuple(out_indices),
    )
    try:
        return timm.create_model(model_name, **kwargs)
    except Exception:
        if fallback_model_name is None:
            raise
        return timm.create_model(fallback_model_name, **kwargs)


@lru_cache(maxsize=None)
def _timm_convnextv2_channels(backbone: str) -> Tuple[int, int, int, int]:
    if backbone != "convnextv2_huge":
        raise ValueError(f"Unsupported timm ConvNeXt V2 backbone: {backbone}")
    model = _create_timm_features_model(
        model_name=CONVNEXTV2_HUGE_TIMM_NAME,
        fallback_model_name=CONVNEXTV2_HUGE_HF_NAME,
        pretrained=False,
        out_indices=(0, 1, 2, 3),
    )
    channels = tuple(int(channel) for channel in model.feature_info.channels())
    if len(channels) != 4:
        raise RuntimeError(
            f"Expected 4 feature channels for {backbone}, got {channels}"
        )
    return channels


def _resnet_backbone_cfg(
    backbone: str,
    pretrained: bool,
    output_stride_8: bool,
) -> Dict[str, Any]:
    depth = _backbone_depth(backbone)
    cfg: Dict[str, Any] = dict(
        type="ResNet",
        depth=depth,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        contract_dilation=output_stride_8,
    )
    if output_stride_8:
        cfg.update(dilations=(1, 1, 2, 4), strides=(1, 2, 1, 1))
    else:
        cfg.update(dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2))
    if pretrained:
        cfg["pretrained"] = f"torchvision://{backbone}"
    return cfg


def _convnext_backbone_cfg(backbone: str, pretrained: bool) -> Dict[str, Any]:
    return dict(
        type="TorchVisionConvNeXt",
        arch=backbone.replace("convnext_", ""),
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained,
    )


def _timm_convnextv2_backbone_cfg(backbone: str, pretrained: bool) -> Dict[str, Any]:
    if backbone != "convnextv2_huge":
        raise ValueError(f"Unsupported timm ConvNeXt V2 backbone: {backbone}")
    return dict(
        type="TimmConvNeXtV2",
        model_name=CONVNEXTV2_HUGE_TIMM_NAME,
        fallback_model_name=CONVNEXTV2_HUGE_HF_NAME,
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained,
    )


def _swin_pretrained_url(backbone: str) -> str:
    if backbone == "swin_small":
        return (
            "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
            "swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth"
        )
    if backbone == "swin_base":
        return (
            "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
            "swin/swin_base_patch4_window7_224_20220317-e9b98025.pth"
        )
    if backbone == "swin_large":
        return (
            "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
            "swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth"
        )
    raise ValueError(f"Unsupported Swin backbone: {backbone}")


def _swin_backbone_cfg(backbone: str, pretrained: bool) -> Dict[str, Any]:
    if backbone == "swin_small":
        embed_dims = 96
        num_heads = (3, 6, 12, 24)
    elif backbone == "swin_base":
        embed_dims = 128
        num_heads = (4, 8, 16, 32)
    elif backbone == "swin_large":
        embed_dims = 192
        num_heads = (6, 12, 24, 48)
    else:
        raise ValueError(f"Unsupported Swin backbone: {backbone}")

    cfg: Dict[str, Any] = dict(
        type="SwinTransformer",
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=embed_dims,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=(2, 2, 18, 2),
        num_heads=num_heads,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        frozen_stages=-1,
    )
    if pretrained:
        cfg["init_cfg"] = dict(type="Pretrained", checkpoint=_swin_pretrained_url(backbone))
    return cfg


def _backbone_cfg(
    backbone: str,
    pretrained: bool,
    output_stride_8: bool,
) -> Dict[str, Any]:
    if _is_resnet(backbone):
        return _resnet_backbone_cfg(
            backbone=backbone,
            pretrained=pretrained,
            output_stride_8=output_stride_8,
        )
    if _is_convnext(backbone):
        return _convnext_backbone_cfg(backbone=backbone, pretrained=pretrained)
    if _is_timm_convnextv2(backbone):
        return _timm_convnextv2_backbone_cfg(backbone=backbone, pretrained=pretrained)
    if _is_swin(backbone):
        return _swin_backbone_cfg(backbone=backbone, pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_model_cfg(
    model: str,
    backbone: str = "resnet50",
    num_classes: int = 20,
    pretrained: bool = False,
) -> Dict[str, Any]:
    model, backbone = validate_model_backbone(model, backbone)
    if model == "segformer":
        raise ValueError(
            "SegFormer uses a Hugging Face config rather than an MMSegmentation "
            "model config; call build_mmseg_model() to construct it"
        )

    channels = _backbone_channels(backbone)
    norm_cfg = _decode_norm_cfg(backbone)
    loss_decode = dict(
        type="CrossEntropyLoss",
        use_sigmoid=False,
        loss_weight=1.0,
        avg_non_ignore=True,
    )
    output_stride_8 = model in ("deeplabv3plus", "pspnet")

    if model == "deeplabv3plus":
        decode_head = dict(
            type="DepthwiseSeparableASPPHead",
            in_channels=channels[3],
            in_index=3,
            channels=_head_channels(backbone),
            dilations=(1, 12, 24, 36),
            c1_in_channels=channels[0],
            c1_channels=48,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=loss_decode,
        )
    elif model == "pspnet":
        decode_head = dict(
            type="PSPHead",
            in_channels=channels[3],
            in_index=3,
            channels=_head_channels(backbone),
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=loss_decode,
        )
    elif model == "upernet":
        decode_head = dict(
            type="UPerHead",
            in_channels=list(channels),
            in_index=[0, 1, 2, 3],
            channels=_head_channels(backbone, high_capacity=False),
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=loss_decode,
        )
    else:
        decode_head = dict(
            type="ResNetUNetHead",
            in_channels=list(channels),
            in_index=[0, 1, 2, 3],
            channels=_head_channels(backbone, high_capacity=False),
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=loss_decode,
        )

    return dict(
        type="EncoderDecoder",
        data_preprocessor=None,
        pretrained=None,
        backbone=_backbone_cfg(
            backbone=backbone,
            pretrained=pretrained,
            output_stride_8=output_stride_8,
        ),
        decode_head=decode_head,
        auxiliary_head=None,
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
    )


def build_mmseg_model(
    model: str,
    backbone: str = "resnet50",
    num_classes: int = 20,
    pretrained: bool = False,
    init_weights: bool = True,
) -> nn.Module:
    model, backbone = validate_model_backbone(model, backbone)
    if model == "segformer":
        from segformer_model import build_segformer_model

        return build_segformer_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    _register_minimal_mmseg_modules()
    if _is_swin(backbone):
        _register_swin_backbone()
    model_cfg = build_model_cfg(
        model=model,
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="``build_loss`` would be deprecated soon.*",
            category=UserWarning,
        )
        segmentor = MODELS.build(copy.deepcopy(model_cfg))
    if init_weights:
        try:
            from mmengine.logging import MMLogger

            logger = MMLogger.get_current_instance()
            old_level = logger.level
            logger.setLevel(logging.WARNING)
        except Exception:
            logger = None
            old_level = None
        try:
            segmentor.init_weights()
        finally:
            if logger is not None and old_level is not None:
                logger.setLevel(old_level)
    return segmentor


def extract_main_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "seg_logits", "out", "main"):
            if key in output:
                try:
                    return extract_main_logits(output[key])
                except TypeError:
                    pass
        for value in output.values():
            try:
                return extract_main_logits(value)
            except TypeError:
                continue
    if isinstance(output, (list, tuple)):
        for value in output:
            try:
                return extract_main_logits(value)
            except TypeError:
                continue
    if hasattr(output, "data") and isinstance(output.data, torch.Tensor):
        return output.data
    raise TypeError(f"Could not extract logits from output type {type(output)!r}")


def resize_logits_to_mask(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


def forward_logits(model: nn.Module, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if getattr(model, "is_hf_segformer", False):
        output = model(pixel_values=images)
        logits = output.logits
    else:
        output = model(images, mode="tensor")
        logits = extract_main_logits(output)
    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits shape [B,C,H,W], got {tuple(logits.shape)}"
        )
    return resize_logits_to_mask(logits, masks)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params/total": total, "params/trainable_total": trainable}
