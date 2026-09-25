# A1: Neural Texture Compression

Repository: [https://github.com/olliearrison/Neural-Graphics-A1.git](https://github.com/olliearrison/Neural-Graphics-A1.git)

Completed implementation and measured results for P1-P8. The writeup is [writeup.pdf](writeup.pdf). The centerpiece plot is [size versus quality](results/figures/size_quality_provided.png), and all numeric results are in [summary.csv](results/summary.csv).

## Setup

Python 3.12 is the recorded environment. Install the exact versions used for the submitted results:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements.txt` lists less restrictive direct dependencies for other platforms. The implementation uses PyTorch, NumPy, Pillow, Matplotlib, ReportLab and pypdf; torchvision is unnecessary. CPU, CUDA and Apple MPS are supported through PyTorch. The submitted experiments used CPU with four threads; GPU numerical differences can change final PSNR.

## Reproduce everything

All six textures are included, so training needs no network access. To re-download the three CC0 samples if needed, run `python scripts/source_textures.py`.

```sh
python -m unittest discover -s tests -v
python main.py smoke
python main.py run --device cpu --threads 4
python scripts/verify_results.py
python scripts/build_report.py
```

The complete experiment is **18 separate fits**, one per texture and architecture. Each uses **2,000 Adam updates, learning rate 0.01 and 16,384 random texels per update**, with all 512 x 512 texel-center/color pairs prepared once beforehand. Every fit initializes from scratch. The report uses the final step, not a selected earlier checkpoint. Full-image PSNR is evaluated every 100 steps, including step 0.

For a new output directory or a subset:

```sh
python main.py run --textures gradient --architectures small --output work/one_fit
python main.py run --resume
```

`--resume` skips completed runs whose source hash, step count and seed match. It does not resume a partially completed optimizer. To ensure a full fresh rerun, omit it. `--steps` is available for debugging; the assignment results use the default 2000. `python main.py summarize` regenerates CSV and JSON summaries from completed metrics. Report and verification scripts use the default `results/` directory.

## Files and assignment mapping

| Problem | Implementation / evidence |
|---|---|
| P1 | `ntc/sampling.py`: normalized RGB, clamped texel-center bilinear sampler |
| P2 | `ntc/bc1.py`: PCA initialization, least-squares endpoint refinement, RGB565 and actual 8-byte block packing |
| P3 | `ntc/model.py`: learnable multi-resolution grids; bilinear `grid_sample`, border padding, `align_corners=False` |
| P4 | `ntc/model.py`: two 64-wide ReLU hidden layers and RGB sigmoid |
| P5 | `ntc/model.py`: fixed texel-center dataset, random minibatches and Adam |
| P6 | `main.py`, `results/`, `scripts/build_report.py`: nine supplied-texture fits, training curves and comparisons |
| P7 | `ntc/storage.py`: per-grid uint8 indices plus float32 min/scale; actual serialized payloads and reload |
| P8 | `data/own/`, `sources.json`: three attributed CC0 photographic textures and nine fresh fits |

Model configurations:

| Model | Grid resolutions | Features/level | Parameters | Float32 bytes | 8-bit-grid bytes |
|---|---|---:|---:|---:|---:|
| Small | 64 | 2 | 12,739 | 50,956 | 26,388 |
| Medium | 16,32,64 | 2 | 15,555 | 62,220 | 29,988 |
| Large | 16,32,64,128 | 4 | 92,483 | 369,932 | 108,844 |

The MLP stays float32 for all reported 8-bit models. An optional `quantize_mlp=True` code path is implemented and tested but is not part of the reported experiments.

## Common sampling interface

Coordinates use u horizontally, v vertically, with texel centers `((x+.5)/W, (y+.5)/H)`. Arrays are addressed `[y,x]` and colors are normalized to `[0,1]`.

```python
from ntc import TextureSampler, BC1Texture, NeuralTexture
from ntc.storage import load_model

reference = TextureSampler.compress('data/bricks.png')
bc1 = BC1Texture.compress('data/bricks.png')
print(reference.sample(0.4, 0.6))
print(bc1.sample(0.4, 0.6))

# Loading a stored representation requires no source image or training.
neural, metadata = load_model('results/bricks/medium/uint8.ntc')
print(neural.sample(0.4, 0.6))

# Alternatively fit a new texture with the same compress/sample interface.
# neural = NeuralTexture.compress('data/bricks.png', architecture='medium')
```

`sample` accepts scalar coordinates or broadcastable arrays and clamps border lookups. Neural sampling returns a PyTorch tensor. It interpolates **features and then decodes**, rather than interpolating four already-decoded RGB colors. That continuous reconstruction is the neural representation. Quantized models dequantize on load for evaluation; their storage on disk is still uint8. The code does not claim an optimized hardware inference implementation.

## Measurement and file format

- PSNR is computed over the entire normalized RGB texture from floating-point decoder output **before PNG rounding**. RGB values remain in the provided image's encoded space. MSE accumulation uses float64 for measurement.
- Raw size is `W*H*3` bytes: **786,432 bytes / 768 KiB** for these 512 x 512 textures. Size ratio means **compressed/raw**; compression factor means **raw/compressed**. `1 KiB=1024 bytes`.
- Float32 size is all grid and MLP parameter values times four bytes. Quantized size is one byte per grid value plus **two float32 constants (8 bytes) per grid**, plus the float32 MLP.
- The `NTC1` binary format is a four-byte magic, four-byte JSON-header length, a UTF-8 architecture/dimension header, then arrays in model parameter order. A quantized array stores little-endian float32 `lo,scale` followed by raw uint8 indices; other arrays store little-endian float32 values. Headers are excluded from the assignment's parameter-payload comparison, but `file_bytes` and `header_bytes` are also reported explicitly.
- BC1 stores exactly **8 bytes per 4x4 block**: two little-endian RGB565 codes and a 32-bit word of raster-order 2-bit indices. Dimensions are a separate JSON sidecar. Endpoints always satisfy `c0>c1` for opaque four-color decoding. The assignment-permitted `q/(31,63,31)` endpoint approximation and float palette interpolation are used; hardware decoders may differ slightly in rounding.
- PNG/container sizes, full decoded working images, optimizer state and temporary minibatches are not compressed-payload sizes. This is a storage comparison, not an inference-memory or performance benchmark.
- The three P8 photographs are grayscale source samples duplicated into RGB, without resizing. The denominator remains 24-bit RGB for assignment consistency; a grayscale-only raw comparison would reduce their factors by three.

## Results, validation, and provenance

Each `results/<texture>/<architecture>/` directory contains both `.ntc` representations, both rendered PNGs and `metrics.json` with training settings, histories, final metrics, source hash and exact sizes. Each texture also has `baseline.bc1`, its dimensions, `baseline.json`, `s3tc.png` and `original.png`.

`results/environment.json` and `requirements-lock.txt` record the runtime. Seeds start at 674 and add `10*texture_index + architecture_index`. Texture order is gradient, bricks, clouds, brick_photo, grass, gravel; model order is Small, Medium, Large. Repeated executions under the same recorded CPU environment are seeded; results across different devices or versions need not be identical.

Eight unit tests cover non-square sampling, borders, grid interpolation and gradients, BC1 bit layout and cross-block sampling, constant blocks, architecture parameter counts, constant quantization, error bounds and saved-model round trips. `scripts/verify_results.py` independently reopens all six baselines and 36 neural files, checks all 18 training schedules and byte counts, and recalculates every reported PSNR. See `results/verification.json`.

The writeup and analysis are generated from recorded results. Spectral analysis uses mean-subtracted grayscale, a Hann window and the fraction of 2D Fourier power above radial frequency 0.125 cycles/pixel. The source PNGs and exact attribution/license evidence are in [data/own/sources.json](data/own/sources.json).

## AI disclosure

Starting from the supplied assignment textures and an initial sampler draft, OpenAI Codex generated and revised the implementation, tests, experiment scripts and report-generation code; selected the additional CC0 textures; ran the experiments; and drafted the README, figures and written analysis. The reported metrics come from executed training and evaluation code. Codex also wrote and ran the automated verification checks.
