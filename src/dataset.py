import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import ColorJitter, Normalize
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_DATASET_TRANSFORMS = (
    "flip",
    "resize",
    "colorjitter",
    "torchvision_normalise",
)

CITYSCAPES_20_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "void",
)

CITYSCAPES_20_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

# まず全部を background/void=19 にする
ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)

mapping = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}
for k, v in mapping.items():
    ID_TO_20CLASS[k] = v


class Cityscapes20ClassDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        mode="fine",
        image_size=None,
        is_train=None,
        pipeline=None,
        hflip_prob=0.5,
        color_jitter_prob=1.0,
        color_jitter_brightness=0.5,
        color_jitter_contrast=0.5,
        color_jitter_saturation=0.5,
        color_jitter_hue=0.0,
    ):
        self.image_size = tuple(image_size) if image_size is not None else None
        self.num_classes = 20
        self.is_train = split == "train" if is_train is None else bool(is_train)
        self.pipeline = tuple(pipeline) if pipeline is not None else ("resize",)
        unknown = [
            transform
            for transform in self.pipeline
            if transform not in SUPPORTED_DATASET_TRANSFORMS
        ]
        if unknown:
            raise ValueError(
                f"Unknown dataset transform(s): {unknown}. Supported transforms: "
                f"{SUPPORTED_DATASET_TRANSFORMS}"
            )
        if len(set(self.pipeline)) != len(self.pipeline):
            raise ValueError(
                f"Dataset pipeline must not contain duplicate transforms: {self.pipeline}"
            )
        if not self.is_train:
            stochastic = [
                transform
                for transform in self.pipeline
                if transform in ("flip", "colorjitter")
            ]
            if stochastic:
                raise ValueError(
                    "Validation/evaluation pipeline must not contain stochastic "
                    f"transforms: {stochastic}"
                )
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError(f"hflip_prob must be in [0, 1], got {hflip_prob}")
        if not 0.0 <= color_jitter_prob <= 1.0:
            raise ValueError(
                "color_jitter_prob must be in [0, 1], got "
                f"{color_jitter_prob}"
            )
        self.hflip_prob = float(hflip_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.color_jitter = ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )
        self.normalise = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.ds = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic",
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.ds[idx]

        target_np = np.array(target, dtype=np.uint8)
        mask = torch.from_numpy(ID_TO_20CLASS[target_np]).long()   # (H, W), 0..19

        for transform in self.pipeline:
            if transform == "flip":
                if torch.rand(()) < self.hflip_prob:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
            elif transform == "resize":
                if self.image_size is not None:
                    image = TF.resize(
                        image,
                        self.image_size,
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                    mask = TF.resize(
                        mask.unsqueeze(0),
                        self.image_size,
                        interpolation=TF.InterpolationMode.NEAREST,
                    ).squeeze(0).long()
            elif transform == "colorjitter":
                if torch.rand(()) < self.color_jitter_prob:
                    image = self.color_jitter(image)
            elif transform == "torchvision_normalise":
                if not isinstance(image, torch.Tensor):
                    image = TF.pil_to_tensor(image).float() / 255.0
                image = self.normalise(image)

        if not isinstance(image, torch.Tensor):
            image = TF.pil_to_tensor(image).float() / 255.0
        else:
            image = image.float()
        mask = mask.long()

        onehot = F.one_hot(mask, num_classes=self.num_classes)      # (H, W, 20)
        onehot = onehot.permute(2, 0, 1).float()                   # (20, H, W)

        return image, onehot, mask
