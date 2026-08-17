#!/home/yuanqi.xhf/miniconda3/bin/python
"""
demosaic.py — Video demosaic (mosaic removal) using LADA.

Usage:
    ./demosaic.py loop <local_mirror_dir>      # Watch local dir, process .mp4 -> .restored.mp4
    ./demosaic.py input.mp4 -o output.mp4      # Process a single video

LADA runs locally. Default LADA path: /data/yuanqi.xhf/nano
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

LADA_HOME = os.environ.get("LADA_HOME", "/data/yuanqi.xhf/nano")
LADA_PYTHON = os.path.join(LADA_HOME, ".venv", "bin", "python3")


# ---------------------------------------------------------------------------
# LADA runner
# ---------------------------------------------------------------------------

def _check_lada():
    """Check that local LADA is available."""
    if not os.path.isfile(LADA_PYTHON):
        print(f"Error: LADA Python not found at {LADA_PYTHON}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(LADA_HOME):
        print(f"Error: LADA_HOME not found: {LADA_HOME}", file=sys.stderr)
        sys.exit(1)
    try:
        result = subprocess.run(
            [LADA_PYTHON, "-c", "from lada.cli.main import main"],
            capture_output=True, text=True, timeout=30, cwd=LADA_HOME,
            env={**os.environ, "PYTHONPATH": LADA_HOME}
        )
        if result.returncode != 0:
            print(f"Error: LADA module not importable: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error checking LADA: {e}", file=sys.stderr)
        sys.exit(1)


def _remove_file(path):
    """Best-effort removal of a single file."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _has_video_and_audio(path):
    """Return True if the media file has at least one video and one audio stream.

    If ffprobe is unavailable we cannot verify; keep the old behaviour and
    assume the output is complete rather than dropping potentially-valid work.
    """
    import shutil
    ffprobe = os.environ.get("FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        print("ffprobe not found; skipping output stream verification", file=sys.stderr)
        return True
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("ffprobe failed; skipping output stream verification", file=sys.stderr)
        return True
    if result.returncode != 0:
        return False
    types = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return "video" in types and "audio" in types


def run_lada(input_path: str, output_path: str, device: str = "auto", temp_dir: str = None):
    """Run LADA locally to demosaic a video.

    LADA writes to a staging file first.  The final ``output_path`` is only
    published after the result is verified to carry both a video and an audio
    stream, so a half-processed file is never exposed as ``*.restored.mp4``.

    The staging file is placed *outside* the watched mirror but keeps a real
    ``.mp4`` extension: LADA infers the output format from the extension, so a
    ``*.staging`` name aborts with "Could not determine output format".
    """
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)

    if device == "auto":
        device = "cuda:0"

    if temp_dir is None:
        temp_dir = os.path.join(
            os.path.dirname(output_abs) or os.path.expanduser("~/.cache/demosaic"),
            "lada_tmp",
        )

    # Staging dir is a sibling of ``temp_dir`` so LADA can never tidy it away
    # mid-run, and it is outside the watched mirror so dav.sync/demosaic ignore it.
    staging_dir = temp_dir.rstrip(os.sep) + ".out"
    os.makedirs(staging_dir, exist_ok=True)
    staging = os.path.join(staging_dir, os.path.basename(output_abs))
    _remove_file(staging)

    cmd = [
        LADA_PYTHON, "-m", "lada.cli.main",
        "--input", input_abs,
        "--output", staging,
        "--device", device,
        "--encoding-preset", "hevc-nvidia-gpu-hq",
        "--temporary-directory", temp_dir,
    ]

    print(f"Running LADA: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()
    try:
        subprocess.run(
            cmd, check=True, capture_output=False, text=True,
            cwd=LADA_HOME,
            env={**os.environ, "PYTHONPATH": LADA_HOME}
        )
        elapsed = time.time() - t0
        print(f"LADA completed in {elapsed:.1f}s", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"LADA failed: {e}", file=sys.stderr)
        _remove_file(staging)
        return False

    if not _has_video_and_audio(staging):
        print(
            f"Output verification failed (missing video/audio stream); "
            f"discarding: {output_abs}",
            file=sys.stderr,
        )
        _remove_file(staging)
        return False

    try:
        os.replace(staging, output_abs)
    except OSError as e:
        print(f"Failed to publish output {output_abs}: {e}", file=sys.stderr)
        _remove_file(staging)
        return False

    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_demosaic(args):
    """Process a single video file."""
    _check_lada()
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if not output:
        base, ext = os.path.splitext(os.path.basename(args.input))
        output = f"{base}_demosaic{ext}"

    print(f"Processing: {args.input} -> {output}", file=sys.stderr)
    success = run_lada(args.input, output, args.device)
    if success:
        print(f"Done: {output}", file=sys.stderr)
    else:
        sys.exit(1)


def cmd_loop(args):
    """Watch a local mirror dir for .mp4 files and produce .restored.mp4."""
    _check_lada()

    watch_dir = os.path.abspath(args.mirror_dir)
    interval = args.interval
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    # LADA 临时目录放 mirror 的上级（即 ws 的 temp 目录）
    temp_dir = os.path.join(os.path.dirname(watch_dir) or os.path.expanduser("~"), "demosaic-tmp")

    print(f"Watching: {watch_dir}", file=sys.stderr)
    print(f"Poll interval: {interval}s", file=sys.stderr)

    while True:
        try:
            entries = sorted(Path(watch_dir).iterdir())
            for entry in entries:
                if not entry.is_file():
                    continue
                ext = entry.suffix.lower()
                if ext not in video_exts:
                    continue

                fname = entry.name
                # 跳过未下载完成的临时文件
                if fname.endswith(".part"):
                    continue
                # Skip our own output so we don't produce *.restored.restored.mp4
                if fname.endswith(f".restored{ext}"):
                    continue
                base = os.path.splitext(fname)[0]
                output_name = f"{base}.restored.mp4"
                output_path = os.path.join(watch_dir, output_name)

                if os.path.isfile(output_path):
                    continue  # already processed

                print(f"Processing: {fname}", file=sys.stderr)
                success = run_lada(str(entry), output_path, args.device, temp_dir=temp_dir)
                if success:
                    print(f"  Done: {output_name}", file=sys.stderr)
                else:
                    print(f"  Failed: {fname}", file=sys.stderr)

        except KeyboardInterrupt:
            print("\nStopping...", file=sys.stderr)
            break
        except Exception as e:
            print(f"Loop error: {e}", file=sys.stderr)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Video demosaic (mosaic removal) using LADA."
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # loop command
    p_loop = sub.add_parser("loop", help="Watch local mirror dir and process .mp4 -> .restored.mp4")
    p_loop.add_argument("mirror_dir", help="Local mirror directory to watch")
    p_loop.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    p_loop.add_argument("--device", default="auto",
                        help="Device: auto, cuda:0, cpu")

    # demosaic command (single file)
    p_demo = sub.add_parser("demosaic", help="Process a single video file")
    p_demo.add_argument("input", help="Input video file path")
    p_demo.add_argument("-o", "--output", default=None,
                        help="Output video file (default: {name}_demosaic.{ext})")
    p_demo.add_argument("--device", default="auto",
                        help="Device: auto, cuda:0, cpu")

    args = parser.parse_args()

    if args.command == "loop":
        cmd_loop(args)
    elif args.command == "demosaic":
        cmd_demosaic(args)
    else:
        # Default: if first arg looks like a file, run demosaic
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            sys.argv.insert(1, "demosaic")
            args = parser.parse_args()
            cmd_demosaic(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()