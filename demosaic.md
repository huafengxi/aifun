# demosaic.py — Video demosaic (mosaic removal) using LADA

Removes mosaics (pixelation) from video files using the **LADA** model
([ladaapp/lada](https://github.com/ladaapp/lada)). LADA runs **locally**
as a Python checkout (no Docker); the tool processes a single video file
at a time. The former watch-loop/daemon mode and the WebDAV pipeline have
been retired — use the single-file command directly.

## How it works

- LADA is invoked via `LADA_HOME/.venv/bin/python3 -m lada.cli.main` with
  the `hevc-nvidia-gpu-hq` encoding preset.
- LADA writes to a staging file first. The output is only published to the
  final path after `ffprobe` verifies it carries both a video and an audio
  stream, so a half-processed result is never exposed.

## Usage

```bash
./demosaic.py input.mp4 -o output.mp4   # single file
./demosaic.py input.mp4                 # -> input_demosaic.mp4
./demosaic.py demosaic input.mp4        # same, explicit subcommand
```

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | Input video file path |
| `-o`, `--output` | `{name}_demosaic.{ext}` | Output video file |
| `--device` | `auto` | Device: `auto` (→ `cuda:0`), `cuda:0`, `cpu` |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LADA_HOME` | `/data/yuanqi.xhf/nano` | Local LADA checkout (venv + sources) |
| `FFPROBE` | *(auto-detect on PATH)* | ffprobe binary used for output verification |

## Requirements

- **GPU** with NVIDIA NVENC (the default encoding preset is
  `hevc-nvidia-gpu-hq`); prefer nv1 or the k8s L20 node (`$WS/env/kube`)
  for GPU workloads
- A working LADA checkout at `LADA_HOME` with its venv
- `ffprobe` (falls back to skipping verification if unavailable)
