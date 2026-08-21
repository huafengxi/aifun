# paint.py — Text-to-image via local Krea 2 (FP8)

Single backend: the **local Krea 2 diffusers pipeline**. Bare invocations
run the `krea2` alias (Krea 2 Turbo FP8); `--model` also accepts a Krea 2
repo id directly. Weights are downloaded on first use from ModelScope
(preferred) or HuggingFace.

Optional prompt expansion lives in a separate filter, **expand.py**: it
turns a brief idea into a richly detailed plain-text prompt via qwen3-a
(`make qwen3-a.start`, port 9113) and never blocks generation (passthrough
if the LLM is down).

## Usage

```bash
# Bare invocation = krea2 (Turbo FP8)
./paint.py "a cat sitting on a cloud" -o cat.png

# Read prompt from stdin
echo "cyberpunk city at night" | ./paint.py -o city.png
./paint.py - --width 1536 --height 864 < prompt.txt

# Explicit alias / repo id
./paint.py krea2 "a fox walking in the snow" -o fox.png
./paint.py --model sakamakismile/Krea-2-Turbo-FP8 "a fox" -o fox.png

# Prompt expansion via qwen3-a (plain text, no JSON), piped into paint.py
echo "a cat" | ./expand.py | ./paint.py -o cat.png
./expand.py "cyberpunk city at night, 电影感"   # inspect the expansion

./paint.py "a cat" --width 1024 --height 1024 --seed 42 -o cat.png
```

## Local models (Krea 2)

| Alias | Model | Notes |
|-------|-------|-------|
| `krea2` | `sakamakismile/Krea-2-Turbo-FP8` | **FP8 (W8A8)** transformer quantized with NVIDIA TensorRT Model Optimizer; ~12.8 GB vs ~25 GB bf16, near-bf16 quality. **Text encoder (Qwen3-VL) also runs FP8** (quantized in-process with modelopt, state cached) and the unused vision tower is dropped — ~4.3 GB less VRAM |

Removed aliases fail fast with a clear error (exit 2): `krea2_bf16`,
`krea2_raw` (both superseded by the FP8 `krea2`).

Defaults follow the official krea-ai/krea-2 README:

- **Turbo** (`krea2`): 8 steps, `--cfg 0.0` (distilled, no CFG),
  2048×2048, timestep-shift `mu=1.15`.

### Experimental dual-GPU component split

`--dual-gpu` (default **off**) places the transformer on the
freest GPU and the Qwen3-VL text encoder + VAE on the other. Measured on
2× RTX 5880 Ada (48 GB, ~25 GB free each while a vLLM worker runs):
1024² generation 27.4 s. Since the text encoder went FP8 (+ vision-tower
drop + cuDNN attention, see below) single-GPU 1024² now fits in ~25 GB
free as well and is faster (13.6 s), so dual-GPU mostly remains an option
for larger canvases / tighter GPUs. No denoise speedup from the split
(transformer stays on one GPU), and 2048² still OOMs (attention needs
~51 GiB on the transformer's GPU). Real tensor/sequence parallelism for
Krea 2 exists only in SGLang-diffusion (`--tp-size` / `--ulysses-degree`),
not in diffusers. See `runs/2026-08-21-17-12-57-24ge/report.md` for the
full evaluation.

### How FP8 loading works

`sakamakismile/Krea-2-Turbo-FP8` is a **transformer-only** W8A8 quantization
(`mtq.quantize` + `mtq.compress`). Following its model card, paint.py
instantiates the transformer from the bf16 base repo (`krea/Krea-2-Turbo`,
which also provides the VAE, tokenizer and scheduler)
and restores the quantization state with
`modelopt.torch.opt.restore(transformer, modelopt_state.pth)`.

The FP8 repo ships **no text encoder**, so paint.py also quantizes the base
repo's Qwen3-VL text encoder to FP8 in-process with the same toolchain
(`mtq.quantize` + `mtq.compress` with `FP8_DEFAULT_CFG`: linear
weights+activations → FP8 e4m3; embeddings and norms stay bf16). The
modelopt state is cached next to the base weights
(`text_encoder_fp8_modelopt_state.pth`) after a one-time calibration on a
few prompts, so later runs restore it instantly. (Community full-FP8 packs
like AlperKTS/Krea2_FP8 only offer the text encoder in ComfyUI single-file
format, which diffusers can't load — hence in-process quantization.)

Two more VRAM measures for the `krea2` path:

- The Qwen3-VL **vision tower is dropped** (~0.9 GiB): Krea 2 conditions on
  text only, `pixel_values` is never passed.
- `DIFFUSERS_ATTN_BACKEND=_native_cudnn` is set before diffusers is
  imported: the Krea 2 joint attention (GQA + mask) otherwise falls back to
  SDPA's MATH kernel, which materializes the seq² score matrix (~3.8 GiB at
  1024²) and OOMs on partially-free GPUs. cuDNN handles it with near-zero
  extra VRAM and the same numerics. Override via `DIFFUSERS_ATTN_BACKEND`.

Measured on an RTX 5880 Ada with ~25 GB free (vLLM occupying the rest):
whole-pipeline resident ~16.3 GiB, 1024² generation peak **~21.3 GiB**
(nvidia-smi) vs **23.2 GiB** for the old bf16-text-encoder build (which
also needed the CPU-offload fallback to reach 1024²), denoise 13.6 s vs
36 s. No CPU-offload fallback anymore — on OOM paint.py only halves the
resolution.

- Requires `pip install nvidia-modelopt` and diffusers ≥ 0.39
  (`Krea2Pipeline`).
- On generation OOM paint.py **halves the resolution** (down to 512²) and
  retries — by design there is **no CPU-offload fallback**.

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
| `prompt` | stdin | Text prompt; `-` or omitted reads from stdin. A leading model alias (`krea2`) selects the pipeline |
| `--model` | `krea2` | Model to run: alias or a Krea 2 repo id directly |
| `-o`, `--output` | `output.png` | Output image file |
| `--width` | 2048 (krea2) | Image width (divisible by 16) |
| `--height` | 2048 (krea2) | Image height (divisible by 16) |
| `--steps` | 8 (krea2) | Inference steps |
| `--cfg` | 0.0 (krea2) | Guidance scale (distilled Turbo runs without CFG) |
| `--seed` | random | Random seed |
| `-n`, `--num-images` | `1` | Number of images |
| `--dual-gpu` | off | Experimental component split across two GPUs |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_API` | `http://localhost:9113` | qwen3-a service URL (expand.py only) |
| `QWEN3_MODEL` | `qwen3.8-a` | qwen3-a model name (expand.py only) |

## Notes

- expand.py is a plain pipeline filter: if qwen3-a is down it passes the
  prompt through unchanged (warning on stderr), so generation is never
  blocked.
- The ideogram4 MaaS backend, its JSON-caption "magic prompt" expansion and
  the `krea2_raw`/`krea2_bf16` aliases were removed; everything runs
  locally on Krea 2 FP8 now.
