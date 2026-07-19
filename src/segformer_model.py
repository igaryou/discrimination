from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerModel,
)


SEGFORMER_VARIANTS: Dict[str, Dict[str, Any]] = {
    "mit_b0": {
        "hidden_sizes": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 256,
    },
    "mit_b1": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 256,
    },
    "mit_b2": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 768,
    },
    "mit_b3": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 768,
    },
    "mit_b4": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 768,
    },
    "mit_b5": {
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "num_attention_heads": [1, 2, 5, 8],
        "decoder_hidden_size": 768,
    },
}


class CityscapesSegformerForSemanticSegmentation(
    SegformerForSemanticSegmentation
):
    """HF SegFormer with an explicit marker for the shared forward adapter."""

    is_hf_segformer = True


def build_segformer_config(backbone: str, num_classes: int) -> SegformerConfig:
    if backbone not in SEGFORMER_VARIANTS:
        raise ValueError(
            f"Unsupported SegFormer backbone: {backbone}. "
            f"Expected one of {tuple(SEGFORMER_VARIANTS)}"
        )
    if num_classes != 20:
        raise ValueError(
            "SegFormer is configured for the existing Cityscapes 20-class "
            f"definition; got num_classes={num_classes}"
        )

    variant = SEGFORMER_VARIANTS[backbone]
    return SegformerConfig(
        num_labels=num_classes,
        num_channels=3,
        hidden_sizes=variant["hidden_sizes"],
        depths=variant["depths"],
        num_attention_heads=variant["num_attention_heads"],
        decoder_hidden_size=variant["decoder_hidden_size"],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        classifier_dropout_prob=0.1,
        drop_path_rate=0.1,
        semantic_loss_ignore_index=255,
    )


def build_segformer_model(
    backbone: str,
    num_classes: int = 20,
    pretrained: bool = False,
) -> nn.Module:
    config = build_segformer_config(backbone=backbone, num_classes=num_classes)
    model = CityscapesSegformerForSemanticSegmentation(config)

    if pretrained:
        hf_backbone = backbone.replace("mit_", "mit-")
        repo_id = f"nvidia/{hf_backbone}"
        try:
            encoder = SegformerModel.from_pretrained(repo_id)
        except ValueError as exc:
            if "upgrade torch to at least v2.6" not in str(exc):
                raise

            # Transformers 4.57 rejects legacy .bin checkpoints with torch<2.6.
            # These official NVIDIA repos do not publish safetensors, so load the
            # trusted checkpoint with weights_only=True and give from_pretrained
            # an encoder-only state dict explicitly.
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(repo_id, "pytorch_model.bin")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            encoder_state_dict = {
                key.removeprefix("segformer."): value
                for key, value in checkpoint.items()
                if key.startswith("segformer.")
            }
            expected_keys = set(model.segformer.state_dict())
            loaded_keys = set(encoder_state_dict)
            if loaded_keys != expected_keys:
                missing = sorted(expected_keys - loaded_keys)
                unexpected = sorted(loaded_keys - expected_keys)
                raise RuntimeError(
                    f"Invalid encoder state_dict for {repo_id}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            encoder = SegformerModel.from_pretrained(
                None,
                config=config,
                state_dict=encoder_state_dict,
            )
        model.segformer.load_state_dict(encoder.state_dict(), strict=True)

    return model
