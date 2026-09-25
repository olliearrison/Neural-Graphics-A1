"""P2: opaque BC1, packed into exactly 8 little-endian bytes per 4x4 block.

Endpoint expansion uses q/(31,63,31), as permitted in the assignment.
This is an approximation to integer hardware expansion/interpolation.
"""
import json
from pathlib import Path
import numpy as np
from .sampling import load_rgb, bilinear

BLOCK_DTYPE = np.dtype([("c0", "<u2"), ("c1", "<u2"), ("indices", "<u4")])
LEVELS = np.array([31, 63, 31], dtype=np.float32)


def pack565(c):
    q = np.round(np.clip(c, 0, 1) * LEVELS).astype(np.uint16)
    return (q[..., 0] << 11) | (q[..., 1] << 5) | q[..., 2]


def unpack565(code):
    return np.stack(((code >> 11) & 31, (code >> 5) & 63, code & 31), axis=-1).astype(np.float32) / LEVELS


def palette(c0, c1):
    a, b = unpack565(c0), unpack565(c1)
    return np.stack((a, b, (2 * a + b) / 3, (a + 2 * b) / 3), axis=-2)


def encode_candidate(blocks, a, b):
    a, b = pack565(a), pack565(b)
    c0, c1 = np.maximum(a, b), np.minimum(a, b)
    # Strict ordering selects BC1's four-color opaque mode, even for a flat block.
    same = c0 == c1
    c0 = np.where(same & (c0 < 65535), c0 + 1, c0).astype(np.uint16)
    c1 = np.where(same & (c0 == c1), c1 - 1, c1).astype(np.uint16)
    p = palette(c0, c1)
    d = ((blocks[:, :, None, :] - p[:, None, :, :]) ** 2).sum(axis=-1)
    indices = d.argmin(axis=-1)
    error = np.take_along_axis(d, indices[..., None], axis=-1).sum(axis=(1, 2))
    return c0, c1, indices, error


class BC1Texture:
    def __init__(self, width, height, blocks):
        self.width, self.height = int(width), int(height)
        self.blocks_w = (self.width + 3) // 4
        self.blocks = np.asarray(blocks, dtype=BLOCK_DTYPE)
        if len(self.blocks) != self.blocks_w * ((self.height + 3) // 4):
            raise ValueError("Wrong number of BC1 blocks")

    @classmethod
    def compress(cls, texture, iterations=8):
        im = load_rgb(texture)
        h, w = im.shape[:2]
        padded = np.pad(im, ((0, (-h) % 4), (0, (-w) % 4), (0, 0)), mode="edge")
        blocks = padded.reshape((h + 3) // 4, 4, (w + 3) // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3)
        mean = blocks.mean(axis=1)
        centered = blocks - mean[:, None, :]
        _, axes = np.linalg.eigh(np.einsum("bnc,bnd->bcd", centered, centered))
        axis = axes[:, :, -1]
        proj = np.einsum("bnc,bc->bn", centered, axis)
        a = mean + proj.max(axis=1)[:, None] * axis
        b = mean + proj.min(axis=1)[:, None] * axis
        best = encode_candidate(blocks, a, b)
        other = encode_candidate(blocks, blocks.max(axis=1), blocks.min(axis=1))
        better = other[-1] < best[-1]
        best = tuple(np.where(better[:, None], y, x) if x.ndim == 2 else np.where(better, y, x) for x, y in zip(best, other))
        current = best
        weights = np.array([1, 0, 2 / 3, 1 / 3], dtype=np.float32)
        for _ in range(iterations):
            alpha = weights[current[2]]
            beta = 1 - alpha
            aa, bb, ab = (alpha * alpha).sum(1), (beta * beta).sum(1), (alpha * beta).sum(1)
            ax = np.einsum("bn,bnc->bc", alpha, blocks)
            bx = np.einsum("bn,bnc->bc", beta, blocks)
            det = aa * bb - ab * ab
            valid = det > 1e-8
            safe_det = np.where(valid, det, 1)[:, None]
            a = np.where(valid[:, None], (bb[:, None] * ax - ab[:, None] * bx) / safe_det, unpack565(current[0]))
            b = np.where(valid[:, None], (aa[:, None] * bx - ab[:, None] * ax) / safe_det, unpack565(current[1]))
            current = encode_candidate(blocks, a, b)
            better = current[-1] < best[-1]
            best = tuple(np.where(better[:, None], y, x) if x.ndim == 2 else np.where(better, y, x) for x, y in zip(best, current))
        packed = np.empty(len(blocks), dtype=BLOCK_DTYPE)
        packed["c0"], packed["c1"] = best[:2]
        shifts = (2 * np.arange(16)).astype(np.uint32)
        packed["indices"] = np.bitwise_or.reduce(best[2].astype(np.uint32) << shifts, axis=1)
        return cls(w, h, packed)

    @property
    def stored_bytes(self):
        return self.blocks.nbytes

    def fetch(self, x, y):
        block = self.blocks[(y // 4) * self.blocks_w + x // 4]
        code = (block["indices"] >> (2 * ((y % 4) * 4 + x % 4)).astype(np.uint32)) & 3
        colors = palette(block["c0"], block["c1"])
        return np.take_along_axis(colors, code[..., None, None].astype(np.int64), axis=-2)[..., 0, :]

    def sample(self, u, v):
        return bilinear(self.fetch, self.width, self.height, u, v)

    def reconstruct(self):
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return self.fetch(x, y)

    def save(self, path):
        path = Path(path)
        path.write_bytes(self.blocks.tobytes())
        path.with_suffix(path.suffix + ".json").write_text(json.dumps({"width": self.width, "height": self.height, "format": "BC1 RGB; normalized RGB565 endpoint expansion"}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        return cls(metadata["width"], metadata["height"], np.frombuffer(path.read_bytes(), dtype=BLOCK_DTYPE).copy())
