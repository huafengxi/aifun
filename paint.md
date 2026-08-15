# paint.py — Text-to-image generation

Generates images from text prompts using **Stable Diffusion** (1.x/2.x/SDXL/Flux)
via the diffusers library. Default model is **GuoFeng4_XL** (国风4) — a Chinese
art style SDXL model. Models are downloaded from **ModelScope** by default
(preferred for China), with HuggingFace fallback.

## Usage

```bash
# Basic generation
./paint.py "a cat sitting on a cloud" -o cat.png

# With SDXL
./paint.py "cyberpunk city at night, neon lights, rain" \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --width 1024 --height 1024 --steps 30

# With Flux
./paint.py "a beautiful landscape" \
  --model black-forest-labs/FLUX.1-dev \
  --steps 28 --cfg 3.5

# Batch generation
./paint.py "abstract art" --batch-size 4 -o abstract.png
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `prompt` | *(required)* | Text prompt describing the image |
| `-o`, `--output` | `output.png` | Output image file |
| `--model` | `xiaolxl/GuoFeng4_XL` | Model ID (SD, SDXL, Flux) |
| `--negative-prompt` | `""` | Negative prompt — what to avoid |
| `--steps` | `25` | Inference steps |
| `--cfg` | `7.5` | Classifier-free guidance scale |
| `--width` | auto | Image width: auto (512 SD, 1024 SDXL/Flux) |
| `--height` | auto | Image height: auto (512 SD, 1024 SDXL/Flux) |
| `--seed` | random | Random seed for reproducibility |
| `--device` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `--dtype` | `float16` | Compute dtype: `float16`, `float32`, `bfloat16` |
| `--batch-size` | `1` | Number of images to generate |

## Supported models

| Model ID | Pipeline | Default size |
|----------|----------|-------------|
| `xiaolxl/GuoFeng4_XL` ⭐ | StableDiffusionXLPipeline | 1024×1024 |
| `runwayml/stable-diffusion-v1-5` | StableDiffusionPipeline | 512×512 |
| `stabilityai/stable-diffusion-2-1` | StableDiffusionPipeline | 512×512 |
| `stabilityai/stable-diffusion-xl-base-1.0` | StableDiffusionXLPipeline | 1024×1024 |
| `black-forest-labs/FLUX.1-dev` | FluxPipeline | auto |

## Requirements

`diffusers`, `transformers`, `accelerate` — install via `requirements.txt`:

```bash
pip install diffusers transformers accelerate safetensors
```

For GPU memory optimization, optionally install `xformers`:

```bash
pip install xformers
```