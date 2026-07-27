# Cityscapes 20-class MMSegmentation Baselines

このディレクトリは、DFM/CSFM系モデルとの比較用に、Cityscapes 20クラスセグメンテーションの識別モデルbaselineを学習・評価するためのものです。

対象モデルは MMSegmentation 1.2.2 + mmcv-lite 上で構成した DeepLabV3+, PSPNet, U-Net-style model, UPerNet です。backboneはResNet系に加えて、UPerNetではConvNeXt Base/LargeとSwin Transformer Small/Base/Large、U-Net-style modelではSwin Transformer Small/Base/Largeを選択できます。MMSegmentation の Runner や Dataset pipeline には移行せず、既存の自作training loopから `image` と `mask` のみを使って学習します。

## 目的

- Cityscapes 20クラス設定で、DFM/CSFMとの比較対象になる識別モデルbaselineを再現可能にする。
- 4種類のMMSegmentationモデルを、同じ `Cityscapes20ClassDataset`、同じ入力解像度、同じoptimizer/scheduler/eval設定で比較する。
- mmcv-lite環境で動かすため、SyncBN、DeformConv、DCN、custom CUDA ops は使わない。
- ResNet50 baselineを基準にしつつ、モデル規模をDFM/CSFMに近づける候補としてUPerNet + ConvNeXt/SwinとU-Net-style model + Swinも同じ自作training loopで動かせるようにする。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `src/sanity_check_mmseg.py` | ランダム入力で forward、CE loss、backward、optimizer step を確認する最小テスト。 |
| `src/check_model_backbones.py` | 指定したmodel/backboneについて、pretrained key照合、raw/resized logits、forward/backward、勾配、finite、stage shape、CUDAメモリを確認する詳細テスト。 |
| `src/baseline_train_mmseg.py` | 自作training loop。Cityscapes train/valを読み込み、学習、val評価、checkpoint保存、loss curve保存、wandb loggingを行う。 |
| `src/baseline_eval_mmseg.py` | 学習済みcheckpointを読み込み、指定splitで評価し、metricsと可視化画像を保存する。 |
| `src/mmseg_model_factory.py` | mmcv-liteで必要なMMSegmentationモジュールだけを登録し、DeepLabV3+、PSPNet、U-Net-style model、UPerNetを構築する。ResNetはBN、ConvNeXt/SwinはLN系normを使用する。SwinはUPerNetとU-Net-style modelに対応する。 |
| `src/dataset.py` | `Cityscapes20ClassDataset`。返り値は `(image, onehot, mask)`。識別モデルでは `onehot` は使わず、`image` と `mask` のみを使う。 |
| `src/metrics.py` | confusion matrix、pixel accuracy、mIoU、mAcc、class別IoU/Accを計算する。 |
| `src/visualization.py` | 入力画像、GT mask、予測mask、loss curveをPNGとして保存する。 |

## 現状の点検メモ

- 学習・評価コードは `for images, _, masks in loader` の形で `onehot` を無視しており、識別モデル側の入力は `image` と `mask` のみです。
- `Cityscapes20ClassDataset` の返り値 `(image, onehot, mask)` は維持されています。
- `norm_cfg` はResNetでは通常BN、ConvNeXt/SwinではLN系です。SyncBN、DeformConv、DCN、custom CUDA ops は使っていません。
- Swinは `--model upernet` と `--model unet` に対応します。DeepLabV3+とPSPNetでは未対応です。
- MMSegmentationの Runner/Dataset pipeline は使っていません。
- `baseline_eval_mmseg.py` は、`--checkpoint` 未指定の場合に `result_dir/model.pth`、つまりbest mIoU checkpointを読みます。最終epochを評価したい場合は `--checkpoint result_dir/model_final.pth` を明示してください。
- 現状ではseed固定引数はありません。厳密な再現性が必要な比較では、今後seed制御を追加する余地があります。
- `--eval_num_classes 19` の場合、学習出力は20クラスのまま、評価平均からclass 19の `void` を除外します。DFM/CSFM側と同じclass mapping・評価定義で比較してください。

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
- `--pretrained` は任意です。このREADMEで追加した800 epoch比較コマンドでは、Swin-SmallはOpenMMLab公式重み、ResNet-101はtorchvision ImageNet重みを使用します。

## 対応backbone

| backbone | 主な用途 | stage出力channel |
| --- | --- | --- |
| `resnet18`, `resnet34`, `resnet50`, `resnet101` | 4モデル共通baseline | ResNet18/34: `[64, 128, 256, 512]`、ResNet50/101: `[256, 512, 1024, 2048]` |
| `convnext_base`, `convnext_large` | UPerNet large baseline候補 | Base: `[128, 256, 512, 1024]`、Large: `[192, 384, 768, 1536]` |
| `convnextv2_huge` | UPerNet large baseline候補 | `[352, 704, 1408, 2816]` |
| `swin_small`, `swin_base`, `swin_large` | UPerNet / U-Net-style baseline候補 | Small: `[96, 192, 384, 768]`、Base: `[128, 256, 512, 1024]`、Large: `[192, 384, 768, 1536]` |

## 比較対象の構成

以下は比較対象として選択した標準的な高性能backbone構成です。全構成で `auxiliary_head=None`、Cross Entropy loss、`align_corners=False` を使用します。

| CLI | backbone設定 | decode head設定 |
| --- | --- | --- |
| `--model upernet --backbone swin_small` | Swin-Small、stage channels `[96, 192, 384, 768]`、通常の1/4・1/8・1/16・1/32出力 | `UPerHead`、`in_index=[0,1,2,3]`、`channels=512`、`pool_scales=(1,2,3,6)`、LN2d |
| `--model pspnet --backbone resnet101` | ResNet-101、`strides=(1,2,1,1)`、`dilations=(1,1,2,4)`、output stride 8 | `PSPHead`、`in_channels=2048`、`in_index=3`、`channels=512`、`pool_scales=(1,2,3,6)`、BN |
| `--model deeplabv3plus --backbone resnet101` | ResNet-101、`strides=(1,2,1,1)`、`dilations=(1,1,2,4)`、output stride 8 | `DepthwiseSeparableASPPHead`、`in_channels=2048`、`channels=512`、`dilations=(1,12,24,36)`、`c1_in_channels=256`、`c1_channels=48`、BN。MMSegmentation 1.2.2実装はlow-level featureとして明示的に `inputs[0]` を使用 |
| `--model unet --backbone resnet101` | ResNet-101、通常の1/4・1/8・1/16・1/32出力、stage channels `[256, 512, 1024, 2048]` | 独自 `ResNetUNetHead`、`in_index=[0,1,2,3]`、`channels=256`、BN |

CLI名は `unet` ですが、4段階のResNet-101 encoder featureを独自 `ResNetUNetHead` でtop-down fusionするU-Net-style decoder構成です。標準U-Netそのものではありません。

### U-Net-style decoder + Swin-Small

`--model unet --backbone swin_small` は、Swin-Smallの4段階featureを既存の
`ResNetUNetHead`でtop-down fusionする構成です。registry名は既存checkpointと
configの互換性のため維持していますが、head実装は`in_channels`と`norm_cfg`を
受け取るbackbone非依存の実装です。Swin専用decoderや標準的な対称U-Netでは
ありません。

```text
stage channels: [96, 192, 384, 768]
decode head in_index: [0, 1, 2, 3]
decode head channels: 512
decode head norm: LN2d (eps=1e-6)
num_classes: 20
align_corners: False
```

256×512入力での実測shape:

```text
stage 0: [1, 96, 64, 128]
stage 1: [1, 192, 32, 64]
stage 2: [1, 384, 16, 32]
stage 3: [1, 768, 8, 16]
raw logits: [1, 20, 64, 128]
resized logits: [1, 20, 256, 512]
```

### Swin-Small architectureと事前学習重み

Swin-Smallは公式の `swin_small_patch4_window7_224` 相当です。

```text
embed_dims=96
depths=(2, 2, 18, 2)
num_heads=(3, 6, 12, 24)
patch_size=4
window_size=7
mlp_ratio=4
strides=(4, 2, 2, 2)
out_indices=(0, 1, 2, 3)
qkv_bias=True
patch_norm=True
use_abs_pos_embed=False
drop_rate=0.0
attn_drop_rate=0.0
drop_path_rate=0.3
with_cp=False
frozen_stages=-1
```

`--pretrained` 時のcheckpoint:

```text
https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth
```

## パラメータ数

今回の比較構成の実測値です。全parameterが学習可能です。差は `model - 84,608,724`、差分率は指定どおり絶対差を用いています。

| モデル | total | trainable | SegFormer + MiT-B5との差 | 差分率 |
| --- | ---: | ---: | ---: | ---: |
| UPerNet + Swin-Small | 80,269,278 | 80,269,278 | -4,339,446 | 5.13% |
| U-Net-style decoder + Swin-Small | 70,830,046 | 70,830,046 | -13,778,678 | 16.29% |
| PSPNet + ResNet-101 | 65,584,212 | 65,584,212 | -19,024,512 | 22.49% |
| DeepLabV3+ + ResNet-101 | 60,198,596 | 60,198,596 | -24,410,128 | 28.85% |
| U-Net-style decoder + ResNet-101 | 48,801,876 | 48,801,876 | -35,806,848 | 42.32% |
| SegFormer + MiT-B5（比較基準） | 84,608,724 | 84,608,724 | 0 | 0.00% |

ResNet50構成:

| モデル | パラメータ数 |
| --- | ---: |
| DeepLabV3+ + ResNet50 | 約41.2M |
| PSPNet + ResNet50 | 約46.6M |
| U-Net-style decoder + ResNet50 | 約29.8M |
| UPerNet + ResNet50 | 約37.3M |

追加したlarge backboneのsanity check時点の値:

| モデル | パラメータ数 |
| --- | ---: |
| UPerNet + ConvNeXt-Large | 約233.1M |
| UPerNet + Swin-Large | 約231.9M |

DFM/CSFM側は約4億パラメータ規模であるため、ResNet50識別モデルbaselineはかなり小さいモデルです。ConvNeXt-Large/Swin-LargeでもDFM/CSFMよりは小さいため、モデル容量を揃えた比較ではない点に注意してください。

## Sanity Check

詳細確認スクリプトは、デフォルトでCUDA bf16 autocastを有効にし、256×512 forward/backwardを行います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/check_model_backbones.py \
  --model upernet \
  --backbone swin_small \
  --num_classes 20 \
  --height 256 \
  --width 512 \
  --pretrained
```

U-Net-style decoder + Swin-Smallのpretrained coverage、stage shape、raw/resized
logits、backbone/headのgradientをGPU 3で確認するコマンド:

```bash
CUDA_VISIBLE_DEVICES=3 uv run python src/check_model_backbones.py \
  --model unet \
  --backbone swin_small \
  --num_classes 20 \
  --height 256 \
  --width 512 \
  --pretrained \
  --amp_dtype bf16
```

optimizer stepまで確認するコマンド:

```bash
CUDA_VISIBLE_DEVICES=3 uv run python src/sanity_check_mmseg.py \
  --model unet \
  --backbone swin_small \
  --pretrained \
  --num_classes 20 \
  --image_size 256 512 \
  --batch_size 1 \
  --amp \
  --amp_dtype bf16
```

実測では公式Swin-Small checkpointのbackbone tensor coverageは99.99%で、
`patch_embed.projection.weight`のtensor値一致、256×512でのforward/backward、
backboneとdecode headのgradient、optimizer step、finite checkに合格しました。

`--model` と `--backbone` をそれぞれ `pspnet/resnet101`、`deeplabv3plus/resnet101`、`unet/resnet101` に変更して同じ検証ができます。軽量な回帰確認だけを行う場合は `--forward_only`、autocastを無効化する場合は `--no_amp` を指定します。

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

## 256×512・800 epoch比較学習

5構成の実行スクリプト:

```text
src/commands/train_upernet_swin_small_256x512_800ep.sh
src/commands/train_pspnet_resnet101_256x512_800ep.sh
src/commands/train_deeplabv3plus_resnet101_256x512_800ep.sh
src/commands/train_unet_resnet101_256x512_800ep.sh
src/commands/train_unet_swin_small_256x512_800ep.sh
```

いずれも学習pipelineは `flip -> resize -> colorjitter -> torchvision_normalise`、評価pipelineは `resize -> torchvision_normalise` です。共通条件は20出力クラス、19評価クラス、800 epochs、batch size 4、AdamW、`lr=5e-5`、`weight_decay=1e-4`、10 epoch warmup、cosine scheduler、`eta_min=5e-7`、200 epochごとの評価、bf16 AMPです。

実行例:

```bash
GPU_ID=0 bash src/commands/train_upernet_swin_small_256x512_800ep.sh
GPU_ID=0 bash src/commands/train_pspnet_resnet101_256x512_800ep.sh
GPU_ID=0 bash src/commands/train_deeplabv3plus_resnet101_256x512_800ep.sh
GPU_ID=0 bash src/commands/train_unet_resnet101_256x512_800ep.sh
GPU_ID=3 bash src/commands/train_unet_swin_small_256x512_800ep.sh
```

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
-> to_tensor -> torchvision_normalise -> pad
```

- RandomResize: 基準 `[H, W] = [256, 512]`、倍率 `[0.5, 2.0]`
- RandomCrop: `[H, W] = [256, 256]`、`cat_max_ratio=0.75`、最大10回試行
- Pad: Normalize後に右端・下端のみ。画像は`0.0`、maskはignore index `19`
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
