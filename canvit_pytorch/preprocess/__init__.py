"""ImageNet preprocessing utilities."""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import transforms


def imagenet_normalize(img: Tensor) -> Tensor:
    """Normalize [C, H, W] or [B, C, H, W] tensor from [0, 1] to ImageNet distribution."""
    mean = img.new_tensor(IMAGENET_DEFAULT_MEAN)
    std = img.new_tensor(IMAGENET_DEFAULT_STD)
    if img.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    return (img - mean) / std


def imagenet_denormalize(img: Tensor) -> Tensor:
    """Denormalize ImageNet-normalized tensor back to [0, 1]."""
    mean = img.new_tensor(IMAGENET_DEFAULT_MEAN)
    std = img.new_tensor(IMAGENET_DEFAULT_STD)
    if img.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def preprocess(size: int) -> transforms.Compose:
    """Standard preprocessing: resize shortest edge, center crop, normalize."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
