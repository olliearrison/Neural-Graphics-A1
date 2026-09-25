"""P3-P5: multi-resolution grids, a fixed decoder, and minibatch fitting."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sampling import load_rgb, texel_centers, psnr

ARCHITECTURES = {
    "small": {"resolutions": [64], "feat_dim": 2},
    "medium": {"resolutions": [16, 32, 64], "feat_dim": 2},
    "large": {"resolutions": [16, 32, 64, 128], "feat_dim": 4},
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.grids = nn.ParameterList([nn.Parameter(torch.randn(1, feat_dim, r, r) * .01) for r in resolutions])
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):
        coords = (2 * uv - 1).reshape(1, -1, 1, 2)
        features = [F.grid_sample(g, coords, mode="bilinear", padding_mode="border", align_corners=False).reshape(g.shape[1], -1).T for g in self.grids]
        return torch.cat(features, dim=-1)


class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class NeuralTexture(nn.Module):
    def __init__(self, architecture="medium"):
        super().__init__()
        self.architecture = architecture
        self.grid = FeatureGrid(**ARCHITECTURES[architecture])
        self.mlp = ColorMLP(self.grid.out_dim)

    def forward(self, uv):
        return self.mlp(self.grid(uv))

    @property
    def stored_bytes(self):
        return sum(p.numel() * 4 for p in self.parameters())

    @torch.no_grad()
    def sample(self, u, v):
        device = next(self.parameters()).device
        u, v = torch.broadcast_tensors(torch.as_tensor(u, dtype=torch.float32, device=device), torch.as_tensor(v, dtype=torch.float32, device=device))
        return self(torch.stack((u, v), dim=-1).reshape(-1, 2)).reshape(*u.shape, 3)

    @classmethod
    def compress(cls, texture, architecture="medium", **kwargs):
        return train_texture(texture, architecture=architecture, **kwargs)[0]


@torch.no_grad()
def reconstruct(model, coords, height, width, chunk=65536):
    return torch.cat([model(c).cpu() for c in coords.split(chunk)]).numpy().reshape(height, width, 3)


def train_texture(texture, architecture="medium", device=None, steps=2000, batch_size=16384, lr=1e-2, seed=674, eval_every=100, verbose=True):
    im = load_rgb(texture)
    h, w = im.shape[:2]
    device = device or get_device()
    torch.manual_seed(seed)
    # The fixed training set is every texel center, built once before training.
    coords = torch.from_numpy(texel_centers(h, w)).to(device)
    target = torch.from_numpy(im.reshape(-1, 3)).to(device)
    model = NeuralTexture(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    losses = []
    start = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            model.eval()
            result = reconstruct(model, coords, h, w)
            score = psnr(result, im)
            history.append({"step": step, "psnr": score, "elapsed_seconds": time.perf_counter() - start})
            if verbose:
                print(f"{architecture:6s} step={step:4d} full-image PSNR={score:.3f} dB elapsed={time.perf_counter()-start:.1f}s", flush=True)
        if step == steps:
            break
        model.train()
        indices = torch.randint(len(coords), (batch_size,), device=device)
        prediction = model(coords[indices])
        loss = F.mse_loss(prediction, target[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            losses.append({"step": step + 1, "minibatch_mse": float(loss.detach().cpu())})
    model.eval()
    return model, {"history": history, "minibatch_losses": losses, "seed": seed, "steps": steps, "batch_size": batch_size, "lr": lr, "device": device, "seconds": time.perf_counter() - start}, result
