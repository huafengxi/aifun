# aifun

AI tools for fun — Japanese ASMR voice analysis, funscript generation, and AI image generation.

## Tools

| Tool | Description |
|------|-------------|
| [siko.py](siko.md) | Detect 西口 (siko) instructions in Japanese ASMR audio using Qwen3-ASR-1.7B with forced alignment timestamps |
| [paint.py](paint.md) | Generate images from text prompts using Stable Diffusion / SDXL / Flux |
| [funscript.py](#funscriptpy) | Convert siko timestamps into funscript format for interactive toys |
| [download_model.py](#download_modelpy) | Download models from ModelScope (preferred for China) or HuggingFace |

### funscript.py

Converts siko timestamps into funscript format for interactive toys.

```bash
# Pipe siko output to funscript generator
./siko.py a.mp3 | ./funscript.py -o output.funscript

# With custom parameters
./siko.py a.mp3 | ./funscript.py --freq 6.0 --range 80
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o` | `output.funscript` | Output file |
| `--freq` | `5.0` | Oscillation frequency (Hz) |
| `--range` | `80` | Amplitude range 0-100 |
| `--fps` | `60` | Frames per second |
| `--attack` | `0.1` | Attack time in seconds |
| `--release` | `0.15` | Release time in seconds |

### download_model.py

Downloads models from **ModelScope** (preferred for China) or HuggingFace.

```bash
# Download from ModelScope (default)
python3 download_model.py Qwen/Qwen3-ASR-1.7B

# Download from HuggingFace
python3 download_model.py --source huggingface Qwen/Qwen3-ASR-1.7B

# Auto: try ModelScope first, fallback to HuggingFace
python3 download_model.py --source auto Qwen/Qwen3-ASR-1.7B
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch with CUDA for GPU acceleration.

## Model Download

Models are downloaded from **ModelScope** by default (preferred for better accessibility in China).
Set `MODEL_DOWNLOAD_SOURCE=huggingface` to use HuggingFace instead.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DOWNLOAD_SOURCE` | `auto` | Model source: `modelscope`, `huggingface`, or `auto` |
| `MODELSCOPE_CACHE` | `/model-cache` | ModelScope cache directory |
| `HUGGINGFACE_HUB_CACHE` | `/model-cache` | HuggingFace cache directory |
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Qwen3-ASR model name |
| `QWEN_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner model |
| `QWEN_DEVICE` | `auto` | Device: `cuda` or `cpu` |
| `QWEN_DTYPE` | `bfloat16` | Compute dtype |
| `QWEN_USE_ALIGNER` | `true` | Enable forced aligner |

## How it works

1. `siko.py` loads a **Qwen3-ASR-1.7B** model and transcribes the audio
2. Uses **Qwen3-ForcedAligner-0.6B** for word-level timestamp alignment
3. Detects siko-related patterns (しこ, シコ, 西口, siko, shico, etc.)
4. Outputs timestamps with duration for each detected instruction
5. `funscript.py` generates oscillating position patterns from the timestamps
6. Models are downloaded from **ModelScope** by default (preferred for China)