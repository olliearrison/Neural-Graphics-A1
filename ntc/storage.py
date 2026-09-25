"""P7: actual uint8 storage with one float32 (lo, scale) pair per array."""
import copy
import json
from pathlib import Path
import struct
import numpy as np
import torch
from .model import NeuralTexture


def quantize_uint8(x):
    a = x.detach().cpu().numpy().astype(np.float32)
    lo, hi = np.float32(a.min()), np.float32(a.max())
    scale = np.float32((hi - lo) / 255) if hi != lo else np.float32(0)
    q = np.zeros_like(a, dtype=np.uint8) if scale == 0 else np.round((a - lo) / scale).clip(0, 255).astype(np.uint8)
    restored = lo + q.astype(np.float32) * scale
    return q, lo, scale, torch.from_numpy(restored).to(x.device)


def quantize_model(model, quantize_mlp=False):
    result = copy.deepcopy(model)
    arrays = {}
    with torch.no_grad():
        for name, p in result.named_parameters():
            if name.startswith("grid.") or quantize_mlp:
                q, lo, scale, restored = quantize_uint8(p)
                p.copy_(restored)
                arrays[name] = (q, lo, scale)
    return result, arrays


def save_model(model, path, width, height, quantized_arrays=None):
    arrays = quantized_arrays or {}
    metadata = {"version": 1, "architecture": model.architecture, "width": width, "height": height, "quantized": list(arrays)}
    header = json.dumps(metadata, separators=(",", ":")).encode()
    payload = bytearray()
    for name, p in model.named_parameters():
        if name in arrays:
            q, lo, scale = arrays[name]
            payload.extend(struct.pack("<ff", lo, scale))
            payload.extend(q.tobytes())
        else:
            payload.extend(p.detach().cpu().numpy().astype("<f4").tobytes())
    Path(path).write_bytes(b"NTC1" + struct.pack("<I", len(header)) + header + payload)
    return {"stored_bytes": len(payload), "file_bytes": Path(path).stat().st_size, "header_bytes": len(header) + 8}


def load_model(path, device="cpu"):
    data = Path(path).read_bytes()
    if data[:4] != b"NTC1":
        raise ValueError("Not an NTC1 model")
    size, = struct.unpack_from("<I", data, 4)
    metadata = json.loads(data[8:8 + size])
    model = NeuralTexture(metadata["architecture"])
    offset = 8 + size
    with torch.no_grad():
        for name, p in model.named_parameters():
            n = p.numel()
            if name in metadata["quantized"]:
                lo, scale = struct.unpack_from("<ff", data, offset)
                offset += 8
                q = np.frombuffer(data, dtype=np.uint8, count=n, offset=offset)
                a = np.float32(lo) + q.astype(np.float32) * np.float32(scale)
                offset += n
            else:
                a = np.frombuffer(data, dtype="<f4", count=n, offset=offset).copy()
                offset += n * 4
            p.copy_(torch.from_numpy(a.reshape(tuple(p.shape))))
    if offset != len(data):
        raise ValueError("Unexpected model payload length")
    return model.to(device).eval(), metadata
