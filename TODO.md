# TODO

## ResNet101 baseline

- DeepLabV3+、PSPNet、UNet、UPerNetの `--backbone resnet101` 実験を追加する。
- ResNet50と同じ学習条件、入力解像度、評価指標で回し、backbone容量だけの影響を確認する。
- 実行後、ResNet101構成のパラメータ数と結果をREADMEまたは別の結果表に追記する。

## Large backbone baseline

- DFM/CSFMが約4億パラメータ規模であるため、比較用にモデル規模を近づけたlarge baselineの学習実験を追加する。
- まずは追加済みのUPerNet + ConvNeXt/Swinを候補にする。
- mmcv-lite環境を維持し、SyncBN、DeformConv、DCN、custom CUDA opsに依存する構成は避ける。

## UPerNet + ConvNeXt-Large

- UPerNet + ConvNeXt-Largeのdebug学習と150epoch学習を実行する。
- 事前学習重みを使う場合は、ResNet baselineとは別条件として記録する。
- ConvNeXt backboneはtorchvision実装をMMSeg backboneとしてwrapしているため、mmpretrain依存はない。

## UPerNet + Swin-Large

- UPerNet + Swin-Largeのdebug学習と150epoch学習を実行する。
- window attentionはmmcv-lite環境でsanity check済み。今後は長時間学習時の安定性を確認する。
- 入力解像度、batch size、AMP dtype、GPU memory使用量をResNet系とは別に記録する。

## Foundation model baseline

- DINOv2/DINOv3系のfoundation model baselineは、通常のMMSeg識別モデルbaselineとは別枠で扱う。
- fine-tuning、linear probe、frozen encoder + segmentation headなど、比較条件が大きく変わるため、ResNet/ConvNeXt/Swinのlarge baselineと同じ表に単純混在させない。
- 使う事前学習データ、freeze方針、head構成、学習率を明示する。

## Reproducibility improvements

- seed固定引数を追加し、Python、NumPy、PyTorch、DataLoader workerのseedを統一する。
- 学習時の `metrics.jsonl` 保存を追加し、wandbなしでもepochごとの数値を機械的に集計できるようにする。
- READMEに実験結果表を追加し、best checkpointとfinal checkpointを区別して記録する。
