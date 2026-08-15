# demosaic.py — Video demosaic with LADA model

Removes mosaics (pixelation) from video files using the **LADA** (Latent
Aware Demosaic Architecture) model. Supports a `loop` command that monitors
the PikPak WebDAV `shared/` directory for new videos and processes them
automatically.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    demosaic.py loop                  │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │  WebDAV   │───▶│  LADA    │───▶│  WebDAV       │  │
│  │  shared/  │    │  Model   │    │  shared/     │  │
│  │  (input)  │    │  (GPU)   │    │  (output)     │  │
│  └──────────┘    └──────────┘    └───────────────┘  │
│       │               │                  │           │
│  poll every     download +           upload         │
│  30s            demosaic             result         │
└─────────────────────────────────────────────────────┘
```

### Pipeline

1. **Monitor** — poll `dav/shared/` via WebDAV every 30s for new `.mp4`/`.mkv` files
2. **Download** — fetch the video from WebDAV to local temp storage
3. **Demosaic** — run LADA model frame-by-frame (GPU required)
4. **Upload** — write the processed video back to `dav/shared/` with `_demosaic` suffix
5. **Cleanup** — delete local temp files, track processed files in a state file

## Usage

```bash
# Start the monitoring loop (runs forever)
./demosaic.py loop

# Demosaic a single video file
./demosaic.py input.mp4 -o output.mp4

# Demosaic with custom model path
./demosaic.py input.mp4 --model /path/to/lada --device cuda:0
```

## Options

### loop command

| Option | Default | Description |
|--------|---------|-------------|
| `--watch-dir` | `dav/shared/` | WebDAV directory to monitor |
| `--output-dir` | `dav/shared/` | WebDAV directory for output |
| `--interval` | `30` | Poll interval in seconds |
| `--state-file` | `demosaic_state.json` | File to track processed videos |
| `--temp-dir` | `/tmp/demosaic` | Local temp directory for processing |
| `--cleanup` | `true` | Delete local temp files after upload |

### Model options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `lada-demosaic` | LADA model ID or local path |
| `--device` | `auto` | Device: `auto`, `cuda:0`, `cpu` |
| `--dtype` | `float16` | Compute dtype: `float16`, `float32` |
| `--tile-size` | `512` | Tile size for large video processing |
| `--batch-size` | `1` | Number of frames per batch |

### Video options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `{name}_demosaic.mp4` | Output file path |
| `--crf` | `18` | Output video quality (lower = better) |
| `--fps` | source | Output frame rate (default: keep source) |
| `--start-time` | `0` | Process from this timestamp (seconds) |
| `--duration` | all | Process duration (seconds, default: entire video) |

## Requirements

### System

- **GPU** with >= 8GB VRAM (preferred: nv1 L20, or k8s L20 node from `$WS/env/kube`)
- **Python** >= 3.10 (use `~/miniconda3`)
- **FFmpeg** for video encoding/decoding

### Python packages

```bash
pip install torch torchvision
pip install opencv-python
pip install webdav4
pip install tqdm
```

### Model

The LADA model is downloaded from **ModelScope** by default:

```bash
# Auto-download on first run, or manually:
python download_model.py --model lada-demosaic
```

Model download path: `$WS/models/lada-demosaic`

## State file

`demosaic_state.json` tracks processed files to avoid re-processing:

```json
{
  "processed": {
    "video1.mp4": {"status": "done", "output": "video1_demosaic.mp4", "ts": "2026-08-15T12:00:00"},
    "video2.mp4": {"status": "failed", "error": "OOM", "ts": "2026-08-15T12:30:00"}
  },
  "in_progress": {}
}
```

## Docker / K8s

For GPU workloads, prefer running as a Docker container or Kubernetes pod on an
L20 GPU node (`$WS/env/kube`). The container needs:

- WebDAV credentials mounted from `$WS/env/webdav.env`
- Model volume from `$WS/models`
- GPU access (`nvidia.com/gpu: 1`)

## Notes

- Large videos are processed in tiles to stay within GPU memory limits
- The loop is designed to run 24/7 as a daemon
- WebDAV access goes through the local alist proxy for faster metadata operations
- Network issues are retried with exponential backoff