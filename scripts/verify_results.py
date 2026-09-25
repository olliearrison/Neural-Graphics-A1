"""Audit completed artifacts independently of the training loop."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import TEXTURES
from ntc import BC1Texture, load_rgb, psnr, texel_centers, ARCHITECTURES
from ntc.model import reconstruct
from ntc.storage import load_model


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1] / 'results'
    checked = []
    for name, source in TEXTURES.items():
        target = load_rgb(source)
        h, w = target.shape[:2]
        coords = torch.from_numpy(texel_centers(h, w))
        baseline = BC1Texture.load(root / name / 'baseline.bc1')
        b = json.loads((root / name / 'baseline.json').read_text())
        assert abs(psnr(baseline.reconstruct(), target) - b['psnr']) < 1e-7
        assert baseline.stored_bytes == b['stored_bytes'] == ((w+3)//4)*((h+3)//4)*8
        for architecture in ARCHITECTURES:
            folder = root / name / architecture
            m = json.loads((folder / 'metrics.json').read_text())
            assert m['training']['steps'] == 2000
            assert m['training']['batch_size'] == 16384
            assert m['training']['lr'] == .01
            assert [v['step'] for v in m['training']['history']] == list(range(0, 2001, 100))
            for mode in ('float32', 'uint8'):
                model, meta = load_model(folder / f'{mode}.ntc')
                score = psnr(reconstruct(model, coords, h, w), target)
                assert abs(score - m[mode]['psnr']) < 1e-7, (name, architecture, mode, score)
                assert m[mode]['stored_bytes'] + m[mode]['header_bytes'] == m[mode]['file_bytes'] == (folder/f'{mode}.ntc').stat().st_size
                expected_bytes = sum(p.numel() + 8 if key in meta['quantized'] else p.numel()*4 for key,p in model.named_parameters())
                assert expected_bytes == m[mode]['stored_bytes']
                assert np.isfinite(score)
                checked.append({'texture': name, 'architecture': architecture, 'mode': mode, 'psnr': score})
        print(f'Verified {name}: BC1 and all six neural files.', flush=True)
    (root / 'verification.json').write_text(json.dumps({'status': 'passed', 'baseline_count': 6, 'neural_files': len(checked), 'checked': checked}, indent=2) + '\n')
    print(f'PASS: 6 baselines, {len(checked)} neural files, all 18 training schedules and storage counts.')


if __name__ == '__main__':
    main()
