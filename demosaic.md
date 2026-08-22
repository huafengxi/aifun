# demosaic.py — Video demosaic (mosaic removal) using LADA

Removes mosaics (pixelation) from video files using the **LADA** model
([ladaapp/lada](https://github.com/ladaapp/lada)). LADA runs via Docker
(`ladaapp/lada:latest`). Supports a `loop` command that monitors the
PikPak WebDAV `shared/` directory for new videos and processes them
automatically.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      demosaic.py loop                        │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  WebDAV   │───▶│  Docker LADA │───▶│  WebDAV      │       │
│  │  shared/  │    │  (GPU)       │    │  shared/     │       │
│  │  (input)  │    │              │    │  (output)    │       │
│  └──────────┘    └──────────────┘    └──────────────┘       │
│       │               │                     │                │
│  poll every     download to           upload result         │
│  30s            /tmp/demosaic/        with _demosaic        │
│                 run LADA              suffix                │
└──────────────────────────────────────────────────────────────┘
```

### Pipeline

1. **Monitor** — poll `shared/` via WebDAV every 30s for new `.mp4`/`.mkv`/`.avi` files
2. **Download** — fetch the video from WebDAV to `/tmp/demosaic/`
3. **Demosaic** — run `docker run ladaapp/lada:latest` with GPU
4. **Upload** — write the processed video back to WebDAV `shared/` with `_demosaic` suffix
5. **Cleanup** — delete local temp files, update `demosaic_state.json`

## Usage

```bash
# Start the monitoring loop (runs forever)
./demosaic.py loop

# Custom poll interval and watch dir
./demosaic.py loop --watch-dir shared --interval 60

# Process a single local video
./demosaic.py input.mp4 -o output.mp4

# Same as above (auto-detected if arg is a file)
./demosaic.py input.mp4
```

## Options

### loop command

| Option | Default | Description |
|--------|---------|-------------|
| `--watch-dir` | `shared` | WebDAV directory to monitor |
| `--output-dir` | `shared` | WebDAV directory for output |
| `--interval` | `30` | Poll interval in seconds |
| `--state-file` | `demosaic_state.json` | File to track processed videos |
| `--temp-dir` | `/tmp/demosaic` | Local temp directory for processing |
| `--cleanup` | `true` | Delete local temp files after upload |
| `--device` | `auto` | Device: `auto`, `cuda:0`, `cpu` |

### demosaic command

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | Input video file path |
| `-o`, `--output` | `{name}_demosaic.{ext}` | Output video file |
| `--device` | `auto` | Device: `auto`, `cuda:0`, `cpu` |

## Requirements

### System

- **GPU** with >= 4-6GB VRAM (NVIDIA Turing or newer: RTX 20xx+)
- **Docker** with `nvidia-container-toolkit`
- **Python** >= 3.10 (use `~/miniconda3`)

### Docker

```bash
# Pull LADA image (auto-pulled on first run)
docker pull ladaapp/lada:latest
```

### Python packages

```bash
pip install webdav4
```

### WebDAV

Credentials are read from `$WS/env/webdav.env` (or `$WS/dav.env`).

## State file

`demosaic_state.json` tracks processed files to avoid re-processing:

```json
{
  "processed": {
    "video1.mp4": {"status": "done", "output": "video1_demosaic.mp4", "ts": "2026-08-15T12:00:00"},
    "video2.mp4": {"status": "failed", "error": "LADA processing failed", "ts": "2026-08-15T12:30:00"}
  },
  "in_progress": {}
}
```

## Running on GPU hosts

For GPU workloads, prefer nv1 or k8s L20 node (`$WS/env/kube`).

```bash
# On a GPU host:
cd ~/m/aifun
./demosaic.py loop --temp-dir /data/yuanqi.xhf/demosaic-tmp
```

## Notes

- LADA runs via Docker (`ladaapp/lada:latest`) — the image bundles the model weights
- The loop is designed to run 24/7 as a daemon
- WebDAV access uses direct credentials for file transfers
- Skips files whose output already exists on WebDAV
- Network errors are caught per-file; the loop continues