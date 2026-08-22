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

### Resident server mode (skip the ~17 s load per call)

One-shot runs spend ~17 s loading the pipeline and ~13.5 s generating. For
repeated generation, keep the pipeline resident:

```bash
./paint.py serve --port 8097            # load once, serve until Ctrl-C
./paint.py serve --idle-exit 1800       # auto-exit after 30 min idle

# thin client (same options as one-shot; also $PAINT_SERVER)
./paint.py --server http://127.0.0.1:8097 "a cat" -o cat.png
echo "a cat" | ./expand.py | ./paint.py --server http://127.0.0.1:8097 -o cat.png

# raw HTTP: POST /generate (JSON) → PNG bytes; GET /health
curl -s -X POST http://127.0.0.1:8097/generate \
  -d '{"prompt": "a cat", "width": 1024, "height": 1024, "seed": 42}' -o cat.png
```

Same single file, stdlib HTTP only, no new dependencies; requests are
serialized; outputs are bit-identical to one-shot mode for the same seed.
Default behavior (no `serve`, no `--server`) is unchanged.

#### Service management

> **Note (2026-08):** the `make paint.start/.stop/.status` targets and the
> `paint` entry in `env/services.yml` were removed — the diffusers resident
> server (8097) has been superseded by the SGLang backend (`make krea2.start`,
> :8098). To run a resident diffusers server manually:

```bash
nohup ./paint.py serve --host 127.0.0.1 --port 8097 >> ../logs/paint-serve.log 2>&1 &
curl -s -m 5 http://127.0.0.1:8097/health   # probe
pkill -f '[p]aint.py serve'                  # stop
```

## 产物格式约定

生成/保存图像一律用 **JPEG（quality=90）**：w/ 画廊（8080 expo 视图）
按原图全量加载，PNG 生成产物 3-4 MB/张导致浏览慢；JPEG q90 实测体积
约为 PNG 的 **1/10**（zhishi-xuebao 1024×1792 图集：10 张 35.7 MB →
3.3 MB，9.2%），肉眼画质无差。

- **paint.py 直接出 JPEG**：`-o x.jpg` 即可，PIL 按扩展名推格式
  （`./paint.py krea2 "a red fox" -o fox.jpg` 已实测合法）；
  `-n >1` 的多图 `base_NN<ext>` 命名同样跟随扩展名。
- **SGLang b64_json 解码**：`/v1/images/generations` 返回 PNG 编码的
  b64_json，落盘时用 `aifun/imgsave.py`，不要手写解码存 .png：

  ```bash
  jq -r .data[0].b64_json resp.json > b64.txt   # 或整段 b64 文本
  ./imgsave.py b64.txt out.jpg            # 也接受 .png 输入，默认 q90
  ./imgsave.py out.png out.jpg 95         # 可选 quality
  ```

- **存量 PNG 目录批量转**：

  ```bash
  ./imgsave.py --dir <gen-dir> --delete-src
  # *.png → 同名 .jpg，校验（重开 jpg + 尺寸一致）后才删源，
  # 并同步把 index.md 中的 .png 文件名重写为 .jpg
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

### SGLang-diffusion backend (2×GPU, faster + 2048²)

A separate, Docker-based serving route for Krea-2 using SGLang-diffusion
(native pipeline, `--tp-size 2` + online `--quantization fp8`). Managed by
`make krea2.start` / `krea2.stop` / `krea2.status` (registered in
`~/m/env/services.yml`, desired **offline**, binds `127.0.0.1:8098`).
paint.py stays the diffusers single-GPU fallback and is not involved.

```bash
make krea2.start    # docker run; ready in ~60s (docker logs -f krea2-sglang)
make krea2.status   # probes GET /health
make krea2.stop     # docker rm -f (releases both GPUs)

# OpenAI-compatible images API (seed/size/steps are request-time)
curl -s -X POST http://127.0.0.1:8098/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"krea2","prompt":"a cat","size":"1024x1024","seed":42,
       "num_inference_steps":8,"response_format":"b64_json"}'
```

Measured on 2× RTX 5880 Ada (vLLM resident, ~24.7 GiB free/card), same
prompt/seed as the paint.py baseline (1024², 8 steps, seed 42):

| Backend | 1024² steady | 2048² | Peak VRAM/card |
|---------|--------------|-------|----------------|
| paint.py serve (FP8, 1 GPU) | 13.5 s | OOM | ~20.5 G |
| SGLang tp2 bf16 | 6.8 s | 29 s | 45.5 / 47.1 G |
| SGLang tp2 + online fp8 | **4.7 s** | **21 s** | 46.4 G |

`--ulysses-degree 2` (bitwise-identical SP) does not fit here: bf16 keeps
the full 25 GB DiT per card (23.8 G vLLM + 23 G DiT > 47.4 G → OOM), and
online fp8 is not supported under SP (loader error). Use tp2. The mount
path must contain `Krea-2` (e.g. `/models/Krea-2-Turbo`) so the native
pipeline registry matches; `--component-residency
text_encoder=component-offload` keeps the Qwen3-VL TE on CPU between uses.
Details: `runs/2026-08-21-22-35-26-kwv2/report.md`.

#### Cache-DiT (per-request DiT cache acceleration)

SGLang-diffusion integrates [Cache-DiT](https://docs.sglang.io/docs/sglang-diffusion/cache_dit)
(DBCache block cache + TaylorSeer correction + SCM step masking). It is a
**per-request** switch on the images API — no restart needed:

```bash
curl -s -X POST http://127.0.0.1:8098/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"krea2","prompt":"a cat","size":"1024x1024","seed":42,
       "num_inference_steps":8,"response_format":"b64_json",
       "enable_cache_dit":true,
       "cache_dit_params":{"enable_taylorseer":true,"taylorseer_order":1}}'
```

`enable_cache_dit`: `true`/`false` per request, or unset to follow the
server default. Cache-DiT is **on by default**: `make krea2.start`
injects `SGLANG_CACHE_DIT_ENABLED=1` plus TaylorSeer
(`SGLANG_CACHE_DIT_TAYLORSEER=1`, order from `KREA2_CACHE_DIT_TAYLORSEER`,
default 1). To disable server-side: `KREA2_CACHE_DIT=0 make krea2.start`
(`0` or empty both mean off).
`cache_dit_params` accepts DBCache knobs (`Fn_compute_blocks`,
`Bn_compute_blocks`, `max_warmup_steps`, `residual_diff_threshold`,
`max_continuous_cached_steps`, `enable_taylorseer`, `taylorseer_order`)
and SCM knobs (`scm_preset`, ...).

Bench (same prompt/seed, 3 steady runs after warmup, vLLM resident;
runs/2026-08-21-23-51-55-7qfl):

| Config | 1024² | 2048² | Peak VRAM/card |
|--------|-------|-------|----------------|
| A baseline (cache off) | 4.67 s | 21.17 s | 39.3 / 46.4 G |
| B DBCache defaults | 3.92 s (1.19x) | 17.61 s (1.20x) | 39.4 / 46.9 G |
| C DBCache + TaylorSeer o1 | 3.94 s (1.19x) | 17.42 s (1.22x) | 39.6 / 46.0 G |
| D C + SCM medium | 3.99 s (1.17x) | 17.34 s (1.22x) | 39.6 / 47.5 G |

Recommended: **C** (DBCache + TaylorSeer order 1) — ~1.2x at both sizes,
visually lossless on photoreal / gongbi / anime portraits (side-by-side in
`temp/krea2-cachedit/`). SCM adds nothing at Turbo's 8 steps (docs: SCM
needs >= 8 steps and is marginal at the floor). Cache-DiT adds ~0.1-1 G
peak VRAM per card — still coexists with vLLM.

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
| `--server` | off | Use a running `paint.py serve` (thin client); also `$PAINT_SERVER` |
| `-n`, `--num-images` | `1` | Number of images |
| `--dual-gpu` | off | Experimental component split across two GPUs |

`serve` subcommand: `--host` (127.0.0.1), `--port` (8097),
`--idle-exit SEC` (0 = never), `--dual-gpu`.

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
