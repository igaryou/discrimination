from typing import Dict, Optional, Sequence

import torch


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        eval_num_classes: Optional[int] = None,
        ignore_target_classes: Optional[Sequence[int]] = None,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if eval_num_classes is None:
            eval_num_classes = num_classes
        if not 1 <= eval_num_classes <= num_classes:
            raise ValueError(
                "eval_num_classes must be in [1, num_classes], got "
                f"{eval_num_classes} for num_classes={num_classes}"
            )

        if ignore_target_classes is None:
            ignore_target_classes = tuple(range(eval_num_classes, num_classes))
        ignored = tuple(sorted(set(int(index) for index in ignore_target_classes)))
        if any(index < 0 or index >= num_classes for index in ignored):
            raise ValueError(
                f"ignore_target_classes must be in [0, {num_classes - 1}], "
                f"got {ignored}"
            )
        required_ignored = set(range(eval_num_classes, num_classes))
        missing_ignored = sorted(required_ignored - set(ignored))
        if missing_ignored:
            raise ValueError(
                "Target classes outside the evaluated class rows must be ignored; "
                f"missing {missing_ignored} from ignore_target_classes"
            )

        self.num_classes = num_classes
        self.eval_num_classes = eval_num_classes
        self.ignore_target_classes = ignored
        self.confmat = torch.zeros(
            eval_num_classes,
            num_classes,
            dtype=torch.int64,
        )

    def reset(self) -> None:
        self.confmat.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().view(-1).cpu()
        target = target.detach().view(-1).cpu()
        if pred.numel() != target.numel():
            raise ValueError(
                f"pred and target must contain the same number of elements, "
                f"got {pred.numel()} and {target.numel()}"
            )

        valid = (target >= 0) & (target < self.eval_num_classes)
        for ignored_class in self.ignore_target_classes:
            valid &= target != ignored_class
        pred = pred[valid]
        target = target[valid]
        if pred.numel() == 0:
            return
        if not ((pred >= 0) & (pred < self.num_classes)).all():
            invalid = pred[(pred < 0) | (pred >= self.num_classes)].unique().tolist()
            raise ValueError(
                f"Prediction class indices must be in [0, {self.num_classes - 1}], "
                f"got {invalid}"
            )
        idx = target * self.num_classes + pred
        bins = torch.bincount(
            idx,
            minlength=self.eval_num_classes * self.num_classes,
        )
        self.confmat += bins.reshape(self.eval_num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        conf = self.confmat.float()
        tp = conf.diag()
        gt = conf.sum(dim=1)
        pred = conf[:, :self.eval_num_classes].sum(dim=0)
        union = gt + pred - tp
        iou = tp / union.clamp_min(1.0)
        acc_cls = tp / gt.clamp_min(1.0)
        pixel_acc = tp.sum() / conf.sum().clamp_min(1.0)
        return {
            "train_num_classes": self.num_classes,
            "eval_num_classes": self.eval_num_classes,
            "ignored_eval_class_indices": list(self.ignore_target_classes),
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(iou.mean().item()),
            "mAcc": float(acc_cls.mean().item()),
            "IoU_per_class": [float(x.item()) for x in iou],
            "Acc_per_class": [float(x.item()) for x in acc_cls],
            "confusion_matrix_shape": [
                self.eval_num_classes,
                self.num_classes,
            ],
            "confusion_matrix_target_class_indices": list(
                range(self.eval_num_classes)
            ),
            "confusion_matrix_prediction_class_indices": list(
                range(self.num_classes)
            ),
            "confusion_matrix": conf.to(torch.int64).tolist(),
        }
