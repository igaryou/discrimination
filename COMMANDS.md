# 実行コマンド集

`~/datasets/cityscapes` は環境に合わせて変更してください。以下はすべて `/home/igarashi_25/playground_2/discrimination` で実行する想定です。

```bash
cd /home/igarashi_25/playground_2/discrimination
```

## Sanity Check

```bash
for m in deeplabv3plus pspnet unet upernet; do
  CUDA_VISIBLE_DEVICES=0 uv run python src/sanity_check_mmseg.py \
    --model $m \
    --backbone resnet50 \
    --num_classes 20 \
    --image_size 128 256 \
    --batch_size 2 \
    --amp \
    --amp_dtype bf16
done
```

## deeplabv3plus_r50 debug

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/deeplabv3plus_r50_debug \
  --model deeplabv3plus \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 4 \
  --eval_interval 1 \
  --num_visualize 4 \
  --amp \
  --amp_dtype bf16
```

## pspnet_r50 debug

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/pspnet_r50_debug \
  --model pspnet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 4 \
  --eval_interval 1 \
  --num_visualize 4 \
  --amp \
  --amp_dtype bf16
```

## unet_r50 debug

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/unet_r50_debug \
  --model unet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 4 \
  --eval_interval 1 \
  --num_visualize 4 \
  --amp \
  --amp_dtype bf16
```

## upernet_r50 debug

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/upernet_r50_debug \
  --model upernet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 4 \
  --eval_interval 1 \
  --num_visualize 4 \
  --amp \
  --amp_dtype bf16
```

## deeplabv3plus_r50 150epoch

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/deeplabv3plus_r50_150e \
  --model deeplabv3plus \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --eta_min 1e-6 \
  --warmup_epochs 10 \
  --eval_interval 1 \
  --num_visualize 16 \
  --amp \
  --amp_dtype bf16
```

## pspnet_r50 150epoch

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/pspnet_r50_150e \
  --model pspnet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --eta_min 1e-6 \
  --warmup_epochs 10 \
  --eval_interval 1 \
  --num_visualize 16 \
  --amp \
  --amp_dtype bf16
```

## unet_r50 150epoch

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/unet_r50_150e \
  --model unet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --eta_min 1e-6 \
  --warmup_epochs 10 \
  --eval_interval 1 \
  --num_visualize 16 \
  --amp \
  --amp_dtype bf16
```

## upernet_r50 150epoch

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/upernet_r50_150e \
  --model upernet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 3e-5 \
  --weight_decay 1e-4 \
  --eta_min 1e-6 \
  --warmup_epochs 10 \
  --eval_interval 1 \
  --num_visualize 16 \
  --amp \
  --amp_dtype bf16
```

## Eval: deeplabv3plus_r50

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_eval_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/deeplabv3plus_r50_150e \
  --checkpoint results/mmseg/deeplabv3plus_r50_150e/model.pth \
  --model deeplabv3plus \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --split val \
  --batch_size 4 \
  --num_workers 4 \
  --num_visualize 32 \
  --amp \
  --amp_dtype bf16
```

## Eval: pspnet_r50

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_eval_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/pspnet_r50_150e \
  --checkpoint results/mmseg/pspnet_r50_150e/model.pth \
  --model pspnet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --split val \
  --batch_size 4 \
  --num_workers 4 \
  --num_visualize 32 \
  --amp \
  --amp_dtype bf16
```

## Eval: unet_r50

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_eval_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/unet_r50_150e \
  --checkpoint results/mmseg/unet_r50_150e/model.pth \
  --model unet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --split val \
  --batch_size 4 \
  --num_workers 4 \
  --num_visualize 32 \
  --amp \
  --amp_dtype bf16
```

## Eval: upernet_r50

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_eval_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/upernet_r50_150e \
  --checkpoint results/mmseg/upernet_r50_150e/model.pth \
  --model upernet \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --split val \
  --batch_size 4 \
  --num_workers 4 \
  --num_visualize 32 \
  --amp \
  --amp_dtype bf16
```

`model_final.pth` を評価する場合は、各evalコマンドの `--checkpoint` を `results/mmseg/<run_name>/model_final.pth` に変更してください。
