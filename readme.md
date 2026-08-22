# aifun

AI tools for fun — Krea 2 image generation, video demosaic & enhancement,
WeChat article image scraping, and assorted utilities.

## Tools

| Tool | Description |
|------|-------------|
| [paint.py](#paintpy) | Text-to-image with the local Krea 2 diffusers pipeline (Krea 2 Turbo FP8, single-GPU); the faster dual-GPU serving route is Krea 2 via SGLang (`make -C ~/m krea2.start`); `expand.py` optionally expands brief prompts via qwen3-a |
| [expand.py](#expandpy) | Plain-text prompt expansion filter for paint.py via the qwen3-a service |
| [mixgen.py](#mixgenpy) | Reference-image-conditioned generation (manual img2img) on the local Krea 2 pipeline |
| [imgsave.py](#imgsavepy) | Save/convert generated images as JPEG (aifun output-format convention, default q90) |
| [funscript.py](#funscriptpy) | Convert timing lines (TSV on stdin) into funscript format for interactive toys |
| [demosaic.py](#demosaicpy) | Video demosaic (mosaic removal) using LADA, single-file mode |
| [video-enhance.py](#video-enhancepy) | Optimize video quality (denoise/sharpen/upscale) and re-encode to H.265 |
| [wximg.py](#wximgpy) | WeChat article image scraper (standard library only) |

### paint.py

Text-to-image with the local Krea 2 diffusers pipeline (`krea2` = Krea 2
Turbo FP8, single GPU). Weights download on first use from ModelScope,
falling back to HuggingFace.

```bash
./paint.py "a cat" -o cat.png                    # bare invocation = krea2 alias
echo "a cat" | ./expand.py | ./paint.py -o cat.png
```

Key operational facts:

- One-shot runs spend ~17 s loading the pipeline each time; for a
  resident pipeline use the SGLang backend below. On generation OOM
  paint.py halves the resolution and retries — no CPU-offload fallback.
  Defaults (krea2 Turbo): 2048×2048, 8 steps, `--cfg 0.0`.
- **SGLang backend (faster, supports 2048²)**: `make -C ~/m krea2.start` —
  SGLang-diffusion Docker, 2×GPU tp2 + online FP8, binds `127.0.0.1:8098`,
  ready in ~60 s, desired=offline (managed via `maas/maas.py`). Exposes the
  OpenAI-compatible `POST /v1/images/generations` (`b64_json` → save with
  `imgsave.py`). 1024² ~4.7 s vs paint.py's ~13.5 s; 2048² (~21 s) fits
  only on SGLang (paint.py OOMs).
- **Cache-DiT** (SGLang only): on by default (~1.2× speedup, visually
  lossless); toggle per request via `enable_cache_dit` in the JSON body, or
  disable server-side with `KREA2_CACHE_DIT=0 make -C ~/m krea2.start`.
- Output convention: JPEG q90 (see imgsave.py below).
- Requires `nvidia-modelopt` + diffusers ≥ 0.39.

### expand.py

Pipeline filter that turns a brief prompt sketch into a richly detailed
plain-text scene description via the qwen3-a chat service (no JSON —
Krea 2 takes plain text). Reads the prompt from argv or stdin, writes the
expanded text to stdout (diagnostics on stderr). If the qwen3-a service is
unreachable, the original prompt is passed through unchanged so generation
is never blocked.

```bash
./expand.py "a cat"
echo "cyberpunk city at night" | ./expand.py
echo "a cat" | ./expand.py | ./paint.py -o cat.png
```

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_API` | `http://localhost:9113` | qwen3-a service URL |
| `QWEN3_MODEL` | `qwen3` | qwen3-a model name |

The qwen3-a server is managed from `~/m` via the maas convention
(`make -C ~/m qwen3.start` → `./maas/maas.py serve qwen3.8-a`).

### mixgen.py

Reference-image-conditioned generation (manual img2img) on the local Krea 2
diffusers pipeline. Krea-2 is T2I-only, so this implements the
community-standard manual img2img trick: VAE-encode the reference image,
normalize with the VAE's latents mean/std, mix with noise at the strength
ratio, then run the pipeline with a truncated sigma schedule. The pipeline
context is borrowed from `paint.load_krea2_pipeline` (dual-GPU aware);
LoRA aliases are not supported.

```bash
# Single shot: needs PROMPT, --ref and -o
./mixgen.py --ref ref.jpg --strength 0.55 --seed 42 \
    --width 1024 --height 1792 -o out.jpg "a prompt"

# Batch: JSON spec, loads the model once and renders all jobs
./mixgen.py --spec jobs.json
```

`--spec` JSON format: list of `{name, ref, prompt, seed, strength
[, out, width, height, steps]}` (per-job fields fall back to the CLI
defaults).

| Option | Default | Description |
|--------|---------|-------------|
| `prompt` | *(required in single-shot)* | Text prompt |
| `--ref` | *(required in single-shot)* | Reference image |
| `-o`, `--output` | *(required in single-shot)* | Output file |
| `--spec` | — | JSON batch spec (mutually exclusive with single-shot args) |
| `--width` | `1024` | Output width |
| `--height` | `1792` | Output height |
| `--steps` | `8` | Inference steps |
| `--strength` | `0.55` | img2img strength (0–1; fraction of the schedule denoised from noise) |
| `--seed` | `42` | Random seed |
| `--quality` | `90` | JPEG quality |
| `--dual-gpu` | off | Split components across two GPUs (transformer on one, text encoder + VAE on the other) |
| `--model` | `krea2` | Model alias or repo id (base models only) |

### imgsave.py

Saves generated images as JPEG (the aifun output-format convention:
quality 90, ~1/10 the size of PNG, gallery-friendly). Single-file mode accepts either an image file or a
base64 text file (the `b64_json` returned by SGLang
`/v1/images/generations`); batch mode converts every `*.png` in a
directory.

```bash
./imgsave.py out.png out.jpg
jq -r .data[0].b64_json resp.json > b64.txt && ./imgsave.py b64.txt out.jpg 95
./imgsave.py --dir ~/m/run/temp/some-gen --delete-src
```

| Option | Default | Description |
|--------|---------|-------------|
| `IN OUT [quality]` | — | Single-file mode: image or base64 text → output file (format by extension) |
| `--dir DIR` | — | Batch mode: convert every `*.png` in DIR to same-name `.jpg` |
| `--quality` | `90` | JPEG quality |
| `--delete-src` | off | Batch mode: delete source PNGs after converting & verifying, and rewrite `index.md` references |

Stdlib + PIL only.

### funscript.py

Converts timing lines read from **stdin** into funscript format for
interactive toys. Input format: whitespace-separated lines
`start_seconds duration_seconds` (blank lines and `#` comments ignored) —
any tool that emits such TSV timings can be piped in.

```bash
# Pipe timing TSV (start_ts <TAB> duration) into the generator
generate-timings | ./funscript.py -o output.funscript

# With custom parameters
generate-timings | ./funscript.py --freq 6.0 --range 80
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o` | `output.funscript` | Output file |
| `--freq` | `5.0` | Oscillation frequency (Hz) |
| `--range` | `80` | Amplitude range 0-100 |
| `--fps` | `60` | Frames per second |
| `--attack` | `0.1` | Attack time in seconds |
| `--release` | `0.15` | Release time in seconds |

### demosaic.py

Video demosaic (mosaic removal) using **LADA**. Runs LADA locally (`LADA_HOME`, default `/data/yuanqi.xhf/nano`)
and verifies each output carries both video and audio streams before
publishing it.

```bash
./demosaic.py input.mp4 -o output.mp4      # single file (bare filename also auto-detected)
./demosaic.py demosaic input.mp4           # same, explicit subcommand
```

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
| `--sharpen` | `0` | Sharpen amount 0.0–1.0 (0 = off, default; `0.4` restores legacy default) |
| `--cq` | `22` | Quality target (NVENC `-cq` / x265 `-crf`), lower = better |
| `--encoder` | `auto` | `auto` / `gpu` (hevc_nvenc) / `cpu` (libx265) |
| `--force` | off | Overwrite existing output |
| `--dry-run` | off | Print the ffmpeg command without running it |

Audio: MP4-compatible codecs (aac/mp3/ac3/eac3/…) are stream-copied; anything
else (wma*/wmapro, vorbis, opus, flac, pcm, …) is transcoded to AAC 192k so
the mp4 muxer never rejects the stream. Output gets `hvc1` tag + faststart
for compatibility.

Performance notes (benchmarked 2026-08-22 on a 45 s 1080p wmv3 clip,
RTX 5880 Ada, task 2026-08-22-20-08-07-koi3):

- The bottleneck is the **CPU filter chain**, not decode/encode: the same
  pipeline with no filters runs ~265 fps (8.8x) vs ~66 fps (2.2x) with
  denoise+sharpen. `-hwaccel cuda` and `hevc_nvenc -preset p1`
  therefore buy almost nothing (p1 also adds ~22% file size).
- `cas` sharpening costs ~1/3 of total runtime (2.2x → 3.4x, +52% when
  dropped), so **sharpening is off by default** since 2026-08-22
  (task 2026-08-22-21-07-22-16e9); use `--sharpen 0.4` to restore it.
  hqdn3d denoise is kept.
- VC-1/wmv3 cannot be hardware-decoded on Ada GPUs (`vc1_cuvid` fails with
  `CUDA_ERROR_NOT_SUPPORTED`; `-hwaccel cuda` silently falls back to
  software decode), so GPU decode is only worth considering for
  h264/hevc/av1 sources.
- One conversion uses ~13 of 64 CPU cores, so the watcher
  (`watcher/convert-file.py`) now processes files in parallel with
  `--workers` (default 2), roughly doubling total throughput.

### wximg.py

WeChat (微信公众号) article image scraper — standard library only
(urllib/re/json/hashlib/threading). Handles both image-carousel articles
(`picture_page_info_list` JS variable) and regular illustrated articles
(`<img data-src>` from `mmbiz.qpic.cn`); downloads concurrently with a
mp.weixin.qq.com Referer, verifies file magic bytes, and records
title/author/publish-time/tags/cover metadata.

```bash
./wximg.py https://mp.weixin.qq.com/s/XXXX -o ./wximg-out/
./wximg.py --list urls.txt -o ./wximg-out/
```

Output layout: `OUT_DIR/<title-slug>-<urlhash6>/` with sequential
`01.<ext>` images, an optional `cover.<ext>` and `meta.json`.

| Option | Default | Description |
|--------|---------|-------------|
| `URL [URL...]` | — | Article links (positional, multiple allowed) |
| `--list FILE` | — | Batch file, one URL per line (`#` comments allowed) |
| `-o`, `--out` | `./wximg-out/` | Output directory |
| `--workers` | `8` | Download concurrency (capped at 8) |
| `--min-delay` | `0.5` | Min seconds between articles |
| `--max-delay` | `1.0` | Max seconds between articles |

Known limits: account-level article lists are not publicly obtainable
(feed it article links); video-only articles have no images; scraping too
fast triggers WeChat's 「当前环境异常」 verification page — lower the
rate if that happens.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch with CUDA for GPU acceleration.
Some tools are standard-library only (`wximg.py`, `expand.py`,
`funscript.py`, `imgsave.py` needs PIL only).
`paint.py` additionally requires `nvidia-modelopt`.

