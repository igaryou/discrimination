import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import ColorJitter, Normalize
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CITYSCAPES_IGNORE_INDEX = 19
SUPPORTED_DATASET_TRANSFORMS = (
    "flip",
    "resize",
    "random_resize",
    "random_crop",
    "colorjitter",
    "photometric_distortion",
    "to_tensor",
    "torchvision_normalise",
)
STOCHASTIC_DATASET_TRANSFORMS = (
    "flip",
    "random_resize",
    "random_crop",
    "colorjitter",
    "photometric_distortion",
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
        train_base_size=(256, 512),
        train_crop_size=(256, 256),
        random_scale_range=(0.5, 2.0),
        cat_max_ratio=0.75,
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
        self.train_base_size = tuple(train_base_size)
        self.train_crop_size = tuple(train_crop_size)
        self.random_scale_range = tuple(float(value) for value in random_scale_range)
        self.cat_max_ratio = float(cat_max_ratio)
        self.ignore_index = CITYSCAPES_IGNORE_INDEX
        self.num_classes = 20
        self.is_train = split == "train" if is_train is None else bool(is_train)
        self.pipeline = tuple(pipeline) if pipeline is not None else ("resize",)
        for name, size in (
            ("image_size", self.image_size),
            ("train_base_size", self.train_base_size),
            ("train_crop_size", self.train_crop_size),
        ):
            if size is not None and (
                len(size) != 2 or any(int(value) <= 0 for value in size)
            ):
                raise ValueError(f"{name} must contain two positive integers, got {size}")
        scale_min, scale_max = self.random_scale_range
        if scale_min <= 0.0 or scale_min > scale_max:
            raise ValueError(
                "random_scale_range must satisfy 0 < min <= max, got "
                f"{self.random_scale_range}"
            )
        if not 0.0 < self.cat_max_ratio <= 1.0:
            raise ValueError(
                f"cat_max_ratio must be in (0, 1], got {self.cat_max_ratio}"
            )
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
        if (
            "colorjitter" in self.pipeline
            and "photometric_distortion" in self.pipeline
        ):
            raise ValueError(
                "colorjitter and photometric_distortion must not be used together"
            )
        if not self.is_train:
            stochastic = [
                transform
                for transform in self.pipeline
                if transform in STOCHASTIC_DATASET_TRANSFORMS
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

    @staticmethod
    def _to_float_tensor(image):
        if not isinstance(image, torch.Tensor):
            return TF.pil_to_tensor(image).float() / 255.0
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        return image.float()

    @staticmethod
    def _uniform(low, high):
        return float(torch.empty(()).uniform_(low, high).item())

    @staticmethod
    def _rgb_to_hsv(image):
        rgb = np.clip(image, 0.0, 255.0).astype(np.float32) / 255.0
        red, green, blue = np.moveaxis(rgb, -1, 0)
        value = rgb.max(axis=-1)
        minimum = rgb.min(axis=-1)
        delta = value - minimum

        saturation = np.zeros_like(value)
        nonzero_value = value > 0.0
        saturation[nonzero_value] = delta[nonzero_value] / value[nonzero_value]

        hue = np.zeros_like(value)
        chromatic = delta > 0.0
        red_max = chromatic & (value == red)
        green_max = chromatic & ~red_max & (value == green)
        blue_max = chromatic & ~red_max & ~green_max
        hue[red_max] = (
            (green[red_max] - blue[red_max]) / delta[red_max]
        ) % 6.0
        hue[green_max] = (
            (blue[green_max] - red[green_max]) / delta[green_max]
        ) + 2.0
        hue[blue_max] = (
            (red[blue_max] - green[blue_max]) / delta[blue_max]
        ) + 4.0
        hue *= 60.0
        return np.stack((hue, saturation, value * 255.0), axis=-1)

    @staticmethod
    def _hsv_to_rgb(image):
        hue = (image[..., 0] % 360.0) / 60.0
        saturation = np.clip(image[..., 1], 0.0, 1.0)
        value = np.clip(image[..., 2], 0.0, 255.0)

        sector = np.floor(hue).astype(np.int64) % 6
        fraction = hue - np.floor(hue)
        p = value * (1.0 - saturation)
        q = value * (1.0 - saturation * fraction)
        t = value * (1.0 - saturation * (1.0 - fraction))

        red = np.choose(sector, (value, q, p, p, t, value))
        green = np.choose(sector, (t, value, value, q, p, p))
        blue = np.choose(sector, (p, p, t, value, value, q))
        return np.stack((red, green, blue), axis=-1).astype(np.float32)

    def _random_resize(self, image, mask):
        ratio = self._uniform(*self.random_scale_range)
        base_height, base_width = self.train_base_size
        new_size = (
            max(1, round(base_height * ratio)),
            max(1, round(base_width * ratio)),
        )
        image = TF.resize(
            image,
            new_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask.unsqueeze(0),
            new_size,
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(0).long()
        return image, mask

    def _random_crop(self, image, mask):
        crop_height, crop_width = self.train_crop_size
        height, width = mask.shape[-2:]
        pad_bottom = max(crop_height - height, 0)
        pad_right = max(crop_width - width, 0)
        if pad_bottom or pad_right:
            padding = [0, 0, pad_right, pad_bottom]
            image = TF.pad(image, padding, fill=0)
            mask = F.pad(
                mask,
                (0, pad_right, 0, pad_bottom),
                value=self.ignore_index,
            )
            height, width = mask.shape[-2:]

        cropped_image = None
        cropped_mask = None
        for _ in range(10):
            top = int(torch.randint(height - crop_height + 1, ()).item())
            left = int(torch.randint(width - crop_width + 1, ()).item())
            cropped_image = TF.crop(
                image,
                top,
                left,
                crop_height,
                crop_width,
            )
            cropped_mask = mask[
                top : top + crop_height,
                left : left + crop_width,
            ]
            valid_mask = cropped_mask[cropped_mask != self.ignore_index]
            if valid_mask.numel() == 0:
                continue
            _, counts = torch.unique(valid_mask, return_counts=True)
            if (
                counts.numel() > 1
                and counts.max().float() / counts.sum() < self.cat_max_ratio
            ):
                break
        return cropped_image, cropped_mask

    def _photometric_distortion(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "photometric_distortion must run before to_tensor/normalise"
            )
        image_np = np.asarray(image, dtype=np.float32).copy()

        if torch.rand(()) < 0.5:
            image_np += self._uniform(-32.0, 32.0)

        contrast_first = bool(torch.randint(2, ()).item())
        if contrast_first and torch.rand(()) < 0.5:
            image_np *= self._uniform(0.5, 1.5)

        hsv = self._rgb_to_hsv(image_np)
        if torch.rand(()) < 0.5:
            hsv[..., 1] *= self._uniform(0.5, 1.5)
            hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 1.0)
        if torch.rand(()) < 0.5:
            hsv[..., 0] = (
                hsv[..., 0] + self._uniform(-18.0, 18.0)
            ) % 360.0
        image_np = self._hsv_to_rgb(hsv)

        if not contrast_first and torch.rand(()) < 0.5:
            image_np *= self._uniform(0.5, 1.5)
        if torch.rand(()) < 0.5:
            permutation = torch.randperm(3).tolist()
            image_np = image_np[..., permutation]

        image_np = np.clip(np.rint(image_np), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(image_np, mode="RGB")

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
            elif transform == "random_resize":
                image, mask = self._random_resize(image, mask)
            elif transform == "random_crop":
                image, mask = self._random_crop(image, mask)
            elif transform == "colorjitter":
                if torch.rand(()) < self.color_jitter_prob:
                    image = self.color_jitter(image)
            elif transform == "photometric_distortion":
                image = self._photometric_distortion(image)
            elif transform == "to_tensor":
                image = self._to_float_tensor(image)
            elif transform == "torchvision_normalise":
                image = self.normalise(self._to_float_tensor(image))

        if not isinstance(image, torch.Tensor):
            image = self._to_float_tensor(image)
        else:
            image = image.float()
        mask = mask.long()

        onehot = F.one_hot(mask, num_classes=self.num_classes)      # (H, W, 20)
        onehot = onehot.permute(2, 0, 1).float()                   # (20, H, W)

        return image, onehot, mask
