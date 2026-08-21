# paint.py — Text-to-image via ideogram4 MaaS or local Krea 2

Two backends:

1. **ideogram4 MaaS** (default, `make ideogram4.start`, port 9114).
   Plain-text prompts are automatically expanded into Ideogram 4's
   structured **JSON caption** format via **qwen3-a** (`make qwen3-a.start`,
   port 9113), using Ideogram's official open-source magic-prompt system
   prompt (`ideogram4_magic_prompt_v1.txt`).
2. **Local Krea 2 diffusers pipelines** — select with a model alias as the
   first positional arg (or `--model`). Weights are downloaded on first use
   from ModelScope (preferred) or HuggingFace.

## Usage

```bash
# Plain-text prompt (auto-expanded to JSON by qwen3-a)
./paint.py "a cat sitting on a cloud" -o cat.png

# Read prompt from stdin
echo "cyberpunk city at night" | ./paint.py -o city.png
./paint.py - --width 1536 --height 864 < prompt.txt

# Prompt is already an Ideogram 4 JSON caption → passed through as-is
./paint.py '{"high_level_description": "...", "compositional_deconstruction": {...}}'

# Skip qwen3-a expansion entirely
./paint.py --no-magic "raw prompt"

# Local Krea 2 (first arg = model alias; prompt sent as-is, no magic expansion)
./paint.py krea2 "a fox walking in the snow" -o fox.png
./paint.py krea2 "a cat" --width 1024 --height 1024 --seed 42 -o cat.png
./paint.py krea2_raw "a fox" --steps 52 --cfg 3.5 -o fox.png
```

## Local models (Krea 2)

| Alias | Model | Notes |
|-------|-------|-------|
| `krea2` | `sakamakismile/Krea-2-Turbo-FP8` | **FP8 (W8A8)** transformer quantized with NVIDIA TensorRT Model Optimizer; ~12.8 GB vs ~25 GB bf16, near-bf16 quality |
| `krea2_bf16` | `krea/Krea-2-Turbo` | Original bf16 Turbo — 8-step distilled, no CFG |
| `krea2_raw` | `krea/Krea-2-Raw` | Original bf16 Raw — base model, full sampler |

Defaults follow the official krea-ai/krea-2 README:

- **Turbo** (`krea2`, `krea2_bf16`): 8 steps, `--cfg 0.0` (distilled, no CFG),
  2048×2048, timestep-shift `mu=1.15`.
- **Raw** (`krea2_raw`): 52 steps, `--cfg 3.5`, 1024×1024.

### How FP8 loading works

`sakamakismile/Krea-2-Turbo-FP8` is a **transformer-only** W8A8 quantization
(`mtq.quantize` + `mtq.compress`). Following its model card, paint.py
instantiates the transformer from the bf16 base repo (`krea/Krea-2-Turbo`,
which also provides the Qwen3-VL text encoder, VAE, tokenizer and scheduler)
and restores the quantization state with
`modelopt.torch.opt.restore(transformer, modelopt_state.pth)`.

- Requires `pip install nvidia-modelopt` and diffusers ≥ 0.39
  (`Krea2Pipeline`).
- If the whole pipeline does not fit in VRAM, paint.py automatically falls
  back to `enable_model_cpu_offload()` (on load OOM or generation OOM).

### Downloads & caches

Models resolve from local caches first (ModelScope then HuggingFace),
otherwise download from ModelScope, falling back to HuggingFace (use
`HF_ENDPOINT=https://hf-cdn.sufy.com` when HF is blocked; the FP8 repo is
HF-only, no ModelScope mirror). Caches checked: `$MODELSCOPE_CACHE`,
`/data/yuanqi.xhf/cache/modelscope`, `~/.cache/modelscope`,
`$HF_HOME`, `/data/yuanqi.xhf/cache/huggingface`, `~/.cache/huggingface`.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `prompt` | stdin | Text prompt; `-` or omitted reads from stdin. A leading model alias (`krea2`, `krea2_bf16`, `krea2_raw`) selects a local pipeline instead |
| `--model` | ideogram4 | Local model to run (alias or repo id); bypasses ideogram4 |
| `-o`, `--output` | `output.png` | Output image file |
| `--width` | 1024 (ideogram4); 2048 turbo / 1024 raw (krea2) | Image width (divisible by 16) |
| `--height` | 1024 (ideogram4); 2048 turbo / 1024 raw (krea2) | Image height (divisible by 16) |
| `--steps` | 10 (ideogram4); 8 turbo / 52 raw (krea2) | Inference steps |
| `--cfg` | server schedule (ideogram4); 0.0 turbo / 3.5 raw (krea2) | Guidance scale; `1.0` = ideogram4 fast mode (no CFG) |
| `--seed` | random | Random seed |
| `-n`, `--num-images` | `1` | Number of images |
| `--no-magic` | off | Skip JSON expansion, send prompt as-is (implicit for local models) |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `IDEOGRAM_API` | `http://localhost:9114` | ideogram4 service URL |
| `QWEN3_API` | `http://localhost:9113` | qwen3-a service URL |
| `QWEN3_MODEL` | `qwen3.8-a` | qwen3-a model name |

## Notes

- If qwen3-a is down, expansion falls back to a minimal JSON wrapper so
  generation still works (lower quality).
- JSON captions follow the Ideogram 4 schema: `high_level_description` +
  `compositional_deconstruction` (`background` + `elements[]`), optionally
  `style_description` with a `color_palette` of uppercase hex colors.
