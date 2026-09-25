"""P1: normalized RGB sampling with clamped, pixel-center bilinear filtering."""
from pathlib import Path
import numpy as np
from PIL import Image


def load_rgb(texture):
    if isinstance(texture, (str, Path)):
        texture = np.asarray(Image.open(texture).convert("RGB"))
    a = np.asarray(texture)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 RGB texture")
    a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Texture values must be finite and normalized to [0, 1]")
    return a


def texel_centers(height, width):
    u, v = np.meshgrid((np.arange(width, dtype=np.float32) + .5) / width,
                       (np.arange(height, dtype=np.float32) + .5) / height)
    return np.stack((u, v), axis=-1).reshape(-1, 2)


def bilinear(fetch, width, height, u, v):
    """Fetch four texels independently, including across compression-block edges."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32))
    x = np.clip(u * width - .5, 0, width - 1)
    y = np.clip(v * height - .5, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    s, t = (x - x0)[..., None], (y - y0)[..., None]
    top = (1 - s) * fetch(x0, y0) + s * fetch(x1, y0)
    bottom = (1 - s) * fetch(x0, y1) + s * fetch(x1, y1)
    return ((1 - t) * top + t * bottom).astype(np.float32)


class TextureSampler:
    def __init__(self, texture):
        self.texels = load_rgb(texture)
        self.height, self.width = self.texels.shape[:2]

    @classmethod
    def compress(cls, texture):
        return cls(texture)

    @property
    def stored_bytes(self):
        return self.height * self.width * 3

    def sample(self, u, v):
        return bilinear(lambda x, y: self.texels[y, x], self.width, self.height, u, v)


def psnr(reconstruction, target):
    mse = np.mean((np.asarray(reconstruction, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2)
    return float(-10 * np.log10(mse)) if mse > 0 else float("inf")


def save_rgb(path, a):
    Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)
