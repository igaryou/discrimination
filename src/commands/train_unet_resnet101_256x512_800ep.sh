#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"

cd /home/igarashi_25/playground_2/discrimination || exit 1

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
uv run python src/baseline_train_mmseg.py \
  --cityscapes_root /home/igarashi_25/datasets/cityscapes \
  --model unet \
  --backbone resnet101 \
  --pretrained \
  --result_dir /home/igarashi_25/playground_2/discrimination/result/unet_resnet101_aug_256x512_800ep \
  --num_classes 20 \
  --eval_num_classes 19 \
  --image_size 256 512 \
  --train_pipeline flip resize colorjitter torchvision_normalise \
  --val_pipeline resize torchvision_normalise \
  --hflip_prob 0.5 \
  --color_jitter_prob 1.0 \
  --color_jitter_brightness 0.5 \
  --color_jitter_contrast 0.5 \
  --color_jitter_saturation 0.5 \
  --color_jitter_hue 0.0 \
  --epochs 800 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --eta_min 5e-7 \
  --warmup_epochs 10 \
  --eval_interval 200 \
  --num_visualize 16 \
  --amp \
  --amp_dtype bf16 \
  --wandb \
  --wandb_project cityscapes20-unet \
  --wandb_run_name unet_resnet101_aug_256x512_800ep
