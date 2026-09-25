"""Reproducible command-line entry point for all A1 experiments."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from ntc import ARCHITECTURES, BC1Texture, NeuralTexture, load_rgb, psnr, texel_centers
from ntc.model import get_device, reconstruct, train_texture
from ntc.sampling import save_rgb
from ntc.storage import quantize_model, save_model, load_model

ROOT = Path(__file__).resolve().parent
TEXTURES = {name: ROOT / 'data' / f'{name}.png' for name in ('gradient', 'bricks', 'clouds')}
TEXTURES.update({name: ROOT / 'data' / 'own' / f'{name}.png' for name in ('brick_photo', 'grass', 'gravel')})


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    torch.set_num_threads(args.threads)
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(), 'device': device, 'threads': args.threads}
    (out / 'environment.json').write_text(json.dumps(env, indent=2))
    print(env, flush=True)
    for texture_index, (name, path) in enumerate(TEXTURES.items()):
        if args.textures and name not in args.textures:
            continue
        im = load_rgb(path)
        h, w = im.shape[:2]
        raw_bytes = h * w * 3
        folder = out / name
        folder.mkdir(exist_ok=True)
        save_rgb(folder / 'original.png', im)
        baseline = BC1Texture.compress(im)
        baseline.save(folder / 'baseline.bc1')
        decoded_baseline = BC1Texture.load(folder / 'baseline.bc1')
        baseline_im = decoded_baseline.reconstruct()
        save_rgb(folder / 's3tc.png', baseline_im)
        baseline_result = {'texture': name, 'method': 's3tc', 'psnr': psnr(baseline_im, im), 'stored_bytes': baseline.stored_bytes, 'raw_bytes': raw_bytes, 'ratio': baseline.stored_bytes/raw_bytes, 'factor': raw_bytes/baseline.stored_bytes}
        (folder / 'baseline.json').write_text(json.dumps(baseline_result, indent=2))
        for ai, architecture in enumerate(ARCHITECTURES):
            if args.architectures and architecture not in args.architectures:
                continue
            run_dir = folder / architecture
            run_dir.mkdir(exist_ok=True)
            seed = args.seed + texture_index * 10 + ai
            if args.resume and (run_dir / 'metrics.json').exists():
                old = json.loads((run_dir / 'metrics.json').read_text())
                if old['training']['steps'] == args.steps and old['training']['seed'] == seed and old['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest():
                    print(f'Skipping completed {name}/{architecture}', flush=True)
                    continue
            print(f'\nTraining {name}/{architecture}', flush=True)
            model, training, result = train_texture(im, architecture, device, steps=args.steps, seed=seed)
            float_size = save_model(model, run_dir / 'float32.ntc', w, h)
            loaded, _ = load_model(run_dir / 'float32.ntc', device)
            coords = torch.from_numpy(texel_centers(h, w)).to(device)
            from_disk = reconstruct(loaded, coords, h, w)
            np.testing.assert_allclose(from_disk, result, atol=1e-6, rtol=0)
            save_rgb(run_dir / 'float32.png', from_disk)
            qmodel, arrays = quantize_model(model)
            qsize = save_model(qmodel, run_dir / 'uint8.ntc', w, h, arrays)
            qloaded, _ = load_model(run_dir / 'uint8.ntc', device)
            qresult = reconstruct(qloaded, coords, h, w)
            save_rgb(run_dir / 'uint8.png', qresult)
            metrics = {'texture': name, 'architecture': architecture, 'width': w, 'height': h, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'raw_bytes': raw_bytes, 'training': training, 'float32': {**float_size, 'psnr': psnr(from_disk, im)}, 'uint8': {**qsize, 'psnr': psnr(qresult, im)}}
            for mode in ('float32', 'uint8'):
                metrics[mode]['ratio'] = metrics[mode]['stored_bytes'] / raw_bytes
                metrics[mode]['factor'] = raw_bytes / metrics[mode]['stored_bytes']
            (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
            print(f"FINISHED {name}/{architecture}: FP32 {metrics['float32']['psnr']:.3f}, uint8 {metrics['uint8']['psnr']:.3f}", flush=True)
            del model, loaded, qmodel, qloaded, coords
            if device == 'mps':
                torch.mps.empty_cache()
    summarize(out)


def summarize(out):
    rows = []
    for name in TEXTURES:
        folder = out / name
        if (folder / 'baseline.json').exists():
            rows.append(json.loads((folder / 'baseline.json').read_text()))
        for architecture in ARCHITECTURES:
            path = folder / architecture / 'metrics.json'
            if path.exists():
                m = json.loads(path.read_text())
                for mode in ('float32', 'uint8'):
                    rows.append({'texture': name, 'method': f'{architecture}_{mode}', 'raw_bytes': m['raw_bytes'], **m[mode]})
    (out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    keys = ['texture', 'method', 'psnr', 'stored_bytes', 'raw_bytes', 'ratio', 'factor', 'file_bytes', 'header_bytes']
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def smoke(args):
    u, v = np.meshgrid(np.linspace(.05, .95, 32), np.linspace(.05, .95, 32))
    texture = np.stack((u, v, (u + v)/2), axis=-1).astype(np.float32)
    torch.set_num_threads(args.threads)
    _, training, _ = train_texture(texture, 'small', args.device or get_device(), steps=300, eval_every=100)
    before, after = training['history'][0]['psnr'], training['history'][-1]['psnr']
    assert after > 30 and after - before > 15, (before, after)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'smoke.json').write_text(json.dumps(training, indent=2))
    print(f'Smoke test passed: {before:.2f} -> {after:.2f} dB', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'smoke', 'summarize'])
    p.add_argument('--output', default=str(ROOT / 'results'))
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=674)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--textures', nargs='+', choices=list(TEXTURES))
    p.add_argument('--architectures', nargs='+', choices=list(ARCHITECTURES))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.command == 'smoke':
        smoke(args)
    elif args.command == 'summarize':
        summarize(Path(args.output))
    else:
        run(args)


if __name__ == '__main__':
    main()
