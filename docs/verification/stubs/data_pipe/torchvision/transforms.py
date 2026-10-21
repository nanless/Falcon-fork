"""Faithful-enough stubs: Grayscale / Compose / Normalize / RandomCrop / CenterCrop."""
import torch


class Grayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        if img.shape[-3] == 1:
            return img
        if img.shape[-3] != 3:
            raise TypeError("Grayscale expects 1 or 3 channels")
        r, g, b = img.unbind(-3)
        gray = 0.2989 * r + 0.587 * g + 0.114 * b  # torchvision ITU-R 601-2 luma
        if self.num_output_channels == 1:
            return gray.unsqueeze(-3)
        return torch.stack([gray] * self.num_output_channels, -3)


class Compose:
    def __init__(self, ts):
        self.transforms = ts

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return (x - self.mean) / self.std


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        import random
        th, tw = self.size
        h, w = x.shape[-2:]
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return x[..., i:i + th, j:j + tw]


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        th, tw = self.size
        h, w = x.shape[-2:]
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[..., i:i + th, j:j + tw]
