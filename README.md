# Cityscapes 20-class MMSegmentation Baselines

このディレクトリは、DFM/CSFM系モデルとの比較用に、Cityscapes 20クラスセグメンテーションの識別モデルbaselineを学習・評価するためのものです。

対象モデルは MMSegmentation 1.2.2 + mmcv-lite 上で構成した DeepLabV3+, PSPNet, UNet, UPerNet です。backboneはResNet系に加えて、UPerNet用にConvNeXt Base/LargeとSwin Transformer Base/Largeを選択できます。MMSegmentation の Runner や Dataset pipeline には移行せず、既存の自作training loopから `image` と `mask` のみを使って学習します。

## 目的

- Cityscapes 20クラス設定で、DFM/CSFMとの比較対象になる識別モデルbaselineを再現可能にする。
- 4種類のMMSegmentationモデルを、同じ `Cityscapes20ClassDataset`、同じ入力解像度、同じoptimizer/scheduler/eval設定で比較する。
- mmcv-lite環境で動かすため、SyncBN、DeformConv、DCN、custom CUDA ops は使わない。
- ResNet50 baselineを基準にしつつ、モデル規模をDFM/CSFMに近づける候補としてUPerNet + ConvNeXt/Swinも同じ自作training loopで動かせるようにする。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `src/sanity_check_mmseg.py` | ランダム入力で forward、CE loss、backward、optimizer step を確認する最小テスト。 |
| `src/baseline_train_mmseg.py` | 自作training loop。Cityscapes train/valを読み込み、学習、val評価、checkpoint保存、loss curve保存、wandb loggingを行う。 |
| `src/baseline_eval_mmseg.py` | 学習済みcheckpointを読み込み、指定splitで評価し、metricsと可視化画像を保存する。 |
| `src/mmseg_model_factory.py` | mmcv-liteで必要なMMSegmentationモジュールだけを登録し、DeepLabV3+、PSPNet、UNet、UPerNetを構築する。ResNetはBN、ConvNeXt/SwinはUPerNet優先でLN系normを使用する。 |
| `src/dataset.py` | `Cityscapes20ClassDataset`。返り値は `(image, onehot, mask)`。識別モデルでは `onehot` は使わず、`image` と `mask` のみを使う。 |
| `src/metrics.py` | confusion matrix、pixel accuracy、mIoU、mAcc、class別IoU/Accを計算する。 |
| `src/visualization.py` | 入力画像、GT mask、予測mask、loss curveをPNGとして保存する。 |

## 現状の点検メモ

- 学習・評価コードは `for images, _, masks in loader` の形で `onehot` を無視しており、識別モデル側の入力は `image` と `mask` のみです。
- `Cityscapes20ClassDataset` の返り値 `(image, onehot, mask)` は維持されています。
- `norm_cfg` はResNetでは通常BN、ConvNeXt/SwinではLN系です。SyncBN、DeformConv、DCN、custom CUDA ops は使っていません。
- ConvNeXt/Swinは現時点では `--model upernet` のみ対応です。
- MMSegmentationの Runner/Dataset pipeline は使っていません。
- `baseline_eval_mmseg.py` は、`--checkpoint` 未指定の場合に `result_dir/model.pth`、つまりbest mIoU checkpointを読みます。最終epochを評価したい場合は `--checkpoint result_dir/model_final.pth` を明示してください。
- 現状ではseed固定引数はありません。厳密な再現性が必要な比較では、今後seed制御を追加する余地があります。
- `mIoU` は20クラス平均で、class 19 の `void` も1クラスとして含みます。DFM/CSFM側と同じclass mapping・評価定義で比較してください。

## 環境構築

想定環境:

- Python 3.10
- `torch==2.5.0+cu121`
- `torchvision==0.20.0+cu121`
- `mmsegmentation==1.2.2`
- `mmengine==0.10.7`
- `mmcv-lite`
- `timm`

例:

```bash
cd /home/igarashi_25/playground_2/discrimination

uv venv --python 3.10
source .venv/bin/activate

uv pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install mmsegmentation==1.2.2 mmengine==0.10.7 mmcv-lite timm
uv pip install numpy pillow matplotlib tqdm wandb
```

注意:

- import名は `mmcv` ですが、インストールするのはfull版ではなく `mmcv-lite` です。
- `mmcv` full版、`mmcv-full`、SyncBN、DeformConv/DCNに依存する設定は使いません。
- `--pretrained` は任意です。現在の再現コマンドでは指定していないため、torchvision pretrained重みは使いません。

## 対応backbone

| backbone | 主な用途 | stage出力channel |
| --- | --- | --- |
| `resnet18`, `resnet34`, `resnet50`, `resnet101` | 4モデル共通baseline | ResNet18/34: `[64, 128, 256, 512]`、ResNet50/101: `[256, 512, 1024, 2048]` |
| `convnext_base`, `convnext_large` | UPerNet large baseline候補 | Base: `[128, 256, 512, 1024]`、Large: `[192, 384, 768, 1536]` |
| `convnextv2_huge` | UPerNet large baseline候補 | `[352, 704, 1408, 2816]` |
| `swin_base`, `swin_large` | UPerNet large baseline候補 | Base: `[128, 256, 512, 1024]`、Large: `[192, 384, 768, 1536]` |

## パラメータ数

ResNet50構成:

| モデル | パラメータ数 |
| --- | ---: |
| DeepLabV3+ + ResNet50 | 約41.2M |
| PSPNet + ResNet50 | 約46.6M |
| UNet + ResNet50 | 約29.8M |
| UPerNet + ResNet50 | 約37.3M |

追加したlarge backboneのsanity check時点の値:

| モデル | パラメータ数 |
| --- | ---: |
| UPerNet + ConvNeXt-Large | 約233.1M |
| UPerNet + Swin-Large | 約231.9M |

DFM/CSFM側は約4億パラメータ規模であるため、ResNet50識別モデルbaselineはかなり小さいモデルです。ConvNeXt-Large/Swin-LargeでもDFM/CSFMよりは小さいため、モデル容量を揃えた比較ではない点に注意してください。

## Sanity Check

4モデルすべてを確認する例:

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

期待されるlogits shapeは、各モデルで `logits_shape: (2, 20, 128, 256)` です。

UPerNet + large backboneを確認する例:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/sanity_check_mmseg.py \
  --model upernet \
  --backbone convnext_large \
  --num_classes 20 \
  --image_size 128 256 \
  --batch_size 1 \
  --amp \
  --amp_dtype bf16

CUDA_VISIBLE_DEVICES=0 uv run python src/sanity_check_mmseg.py \
  --model upernet \
  --backbone swin_large \
  --num_classes 20 \
  --image_size 128 256 \
  --batch_size 1 \
  --amp \
  --amp_dtype bf16
```

## Debug学習

debug学習は、dataset読み込み、学習loop、val評価、checkpoint保存、可視化保存が一通り動くかを見るための短時間実行です。性能比較には使いません。

例:

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

全モデル分のコピー用コマンドは `COMMANDS.md` にまとめています。

## 150epoch本番学習

例:

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

同じ条件で4モデルを回すコマンドは `COMMANDS.md` を参照してください。

## 評価

best checkpointを評価する例:

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

`model.pth` はval mIoUが最良のcheckpoint、`model_final.pth` は最終epochのcheckpointです。

## 出力ファイル

学習時:

```text
results/mmseg/<run_name>/
  config.json
  train_log.txt
  model.pth
  model_final.pth
  loss_curves/
    loss_<epoch>.png
  infer_val/
    epoch_001/
      image/
      gt/
      pred/
    ...
```

評価時:

```text
results/mmseg/<run_name>/
  eval_val/
    metrics.json
    metrics.txt
    visualizations/
      image/
      gt/
      pred/
```

## wandbを使う場合

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/baseline_train_mmseg.py \
  --cityscapes_root ~/datasets/cityscapes \
  --result_dir results/mmseg/deeplabv3plus_r50_150e_wandb \
  --model deeplabv3plus \
  --backbone resnet50 \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4 \
  --eval_interval 1 \
  --amp \
  --amp_dtype bf16 \
  --wandb \
  --wandb_project cityscapes20-mmseg-baselines \
  --wandb_run_name deeplabv3plus_r50_150e
```

オフライン記録にしたい場合は `--wandb_mode offline` を追加します。

## SegFormer Cityscapes augmentation

SegFormer公式設定を1/4解像度へ縮小した学習パイプラインは、次の順序で指定します。

```text
random_resize -> random_crop -> flip -> photometric_distortion
-> to_tensor -> torchvision_normalise
```

- RandomResize: 基準 `[H, W] = [256, 512]`、倍率 `[0.5, 2.0]`
- RandomCrop: `[H, W] = [256, 256]`、`cat_max_ratio=0.75`、最大10回試行
- validation/test: `resize -> to_tensor -> torchvision_normalise` のみ
- void/ignore index: `19`
- Normalize: ImageNet mean/stdをデータセット内で1回だけ適用

既存実験のデフォルトpipelineは変更せず、SegFormer用コマンドで明示的に有効化します。学習コマンド全体は `src/commandv2.txt` を参照してください。

shape、mask ID、cat ratio、overlay、SegFormerのforward/backwardを確認するには次を実行します。

```bash
uv run python src/check_cityscapes_augmentation.py \
  --cityscapes_root ~/datasets/cityscapes \
  --output_dir result/augmentation_check
```

## DFM/CSFMとの比較時の注意点

- DFM/CSFM側の `/home/igarashi_25/playground_2/DSDFM/dfm/src` は、このbaseline整理では変更しません。
- 比較時は、train/val split、入力解像度、class mapping、評価指標、checkpoint選択を揃えてください。
- この識別baselineでは `onehot` は使いません。`Cityscapes20ClassDataset` の互換性維持のため返り値には残しています。
- ResNet50 baselineは約29.8Mから46.6Mパラメータで、約4億パラメータ規模のDFM/CSFMよりかなり小さいです。
- UPerNet + ConvNeXt-Large/Swin-Largeは約232Mから233M規模です。ResNet50より大きいものの、DFM/CSFMよりはまだ小さいため、結果表ではモデル規模を明記してください。
