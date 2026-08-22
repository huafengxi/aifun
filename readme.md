# aifun

AI tools for fun — Japanese ASMR voice analysis, funscript generation, and AI image generation.

## Tools

| Tool | Description |
|------|-------------|
| [siko.py](siko.md) | Detect 西口 (siko) instructions in Japanese ASMR audio using Qwen3-ASR-1.7B with forced alignment timestamps |
| [paint.py](paint.md) | Generate images from text prompts with local Krea 2 (FP8 transformer + FP8 Qwen3-VL text encoder) diffusers pipeline; `expand.py` optionally expands brief prompts via qwen3-a |
| [funscript.py](#funscriptpy) | Convert siko timestamps into funscript format for interactive toys |
| [download_model.py](#download_modelpy) | Download models from ModelScope (preferred for China) or HuggingFace |
| [qwen3_serve.py](#qwen3_servepy) | Serve a Qwen3 LLM with vLLM (OpenAI API) in Docker |
| [qwen3_bench.py](#qwen3_benchpy) | Single-stream decode perf test for the qwen3 server |
| [video-enhance.py](#video-enhancepy) | Optimize video quality (denoise/sharpen/upscale) and re-encode to H.265 |

### qwen3_serve.py

Serves a Qwen3 LLM as an OpenAI-compatible API using vLLM in Docker.
Models are downloaded from ModelScope (preferred) into `~/m/run/models` first.

```bash
make -C ~/m qwen3.start      # download (if needed) + start vLLM on :8000
make -C ~/m qwen3.status     # container + API health
make -C ~/m qwen3.stop       # stop/remove the container
make -C ~/m qwen3.logs       # tail container logs

# override the model (default: Qwen/Qwen3.8-27B-FP8)
make -C ~/m qwen3.start QWEN3_MODEL=Qwen/Qwen3.8-27B-FP8
```

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_MODEL` | `Qwen/Qwen3.8-27B-FP8` | Model ID on ModelScope |
| `QWEN3_PORT` | `8000` | Host port for the OpenAI API |
| `QWEN3_MODELS_DIR` | `/data/yuanqi.xhf/models` | Local model directory |
| `QWEN3_IMAGE` | `mass-runner:cuda13.0-vllm0.22.1` | vLLM image (must support Qwen3.5 arch) |
| `QWEN3_TP` | `2` | Tensor-parallel size (27B uses both GPUs) |
| `QWEN3_MAX_MODEL_LEN` | `32768` | Max context length |
| `QWEN3_SPECULATIVE` | `{"method":"qwen3_5_mtp","num_speculative_tokens":1}` | MTP speculative decoding (speeds up single-stream decode; set empty to disable) |
| `QWEN3_TOOL_CALL_PARSER` | `qwen3_xml` | Tool call parser (enables `tool_choice: auto` / native function calling; set empty to disable) |
| `QWEN3_REASONING_PARSER` | `qwen3` | Reasoning parser (splits ` thinking`/` response` into `reasoning_content` vs `content`; set empty to disable) |

### qwen3_bench.py

Single-stream (one request at a time) decode performance test for the server
started by `qwen3_serve.py`. Reports throughput (tok/s) and, by default,
time-to-first-token (TTFT) via streaming.

```bash
make -C ~/m qwen3.bench                              # default: 512 tokens, 3 runs
make -C ~/m qwen3.bench ARGS="--max-tokens 1024 --runs 5"
./qwen3_bench.py --no-stream                         # throughput only
```

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_BENCH_URL` | `http://127.0.0.1:8000` | OpenAI base URL |
| `QWEN3_BENCH_MODEL` | `qwen3` | Model name |

### video-enhance.py

Optimizes video quality with an ffmpeg filter pipeline — denoise (`hqdn3d`) →
sharpen (`cas`) → optional lanczos upscale — and re-encodes to **H.265**.
Uses `hevc_nvenc` (GPU) by default, falls back to `libx265` (CPU) automatically.

```bash
./video-enhance.py input.mp4                          # -> input.enhanced.mp4 (H.265)
./video-enhance.py input.mp4 --scale 2 --cq 20        # 2x upscale, higher quality
./video-enhance.py input.mp4 --encoder cpu --dry-run  # preview command, CPU encoder
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o` | `<input>.enhanced.mp4` | Output file |
| `--scale` | `1.0` | Upscale factor (e.g. `2` for 2x, lanczos) |
| `--denoise` | `0.3` | Denoise strength 0.0–1.0 (0 = off) |
| `--sharpen` | `0.4` | Sharpen amount 0.0–1.0 (0 = off) |
| `--cq` | `22` | Quality target (NVENC `-cq` / x265 `-crf`), lower = better |
| `--encoder` | `auto` | `auto` / `gpu` (hevc_nvenc) / `cpu` (libx265) |
| `--force` | off | Overwrite existing output |

Audio is copied unchanged; output gets `hvc1` tag + faststart for compatibility.

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