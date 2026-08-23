#!/home/yuanqi.xhf/miniconda3/bin/python
"""
demosaic.py — Video demosaic (mosaic removal) using Jasna.

Usage:
    ./demosaic.py input.mp4 -o output.mp4      # Process a single video
    ./demosaic.py demosaic input.mp4           # Same, explicit subcommand

Jasna runs locally (GPU-only pipeline, NVIDIA). Default Jasna path:
/data/yuanqi.xhf/jasna (override with JASNA_HOME).

Note: the first run on a new GPU compiles TensorRT engines (15-60 min,
one-time; cached in $JASNA_HOME/model_weights).
"""

import argparse
import os
import subprocess
import sys
import time

JASNA_HOME = os.environ.get("JASNA_HOME", "/data/yuanqi.xhf/jasna")
JASNA_BIN = os.path.join(JASNA_HOME, "jasna")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_jasna():
    """Check that local Jasna is available."""
    if not os.path.isdir(JASNA_HOME):
        print(f"Error: JASNA_HOME not found: {JASNA_HOME}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(JASNA_BIN) or not os.access(JASNA_BIN, os.X_OK):
        print(f"Error: Jasna binary not found at {JASNA_BIN}", file=sys.stderr)
        sys.exit(1)


def _remove_file(path):
    """Best-effort removal of a single file."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _stream_types(path):
    """Return the set of stream types (e.g. {'video', 'audio'}) of a media
    file, or None when ffprobe is unavailable/failed."""
    import shutil
    ffprobe = os.environ.get("FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        print("ffprobe not found; skipping output stream verification", file=sys.stderr)
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("ffprobe failed; skipping output stream verification", file=sys.stderr)
        return None
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _verify_output(staging, input_path):
    """Verify the staged output carries video (and audio when the input has
    audio). Returns True when the output is publishable."""
    out_types = _stream_types(staging)
    if out_types is None:
        return True
    if "video" not in out_types:
        return False
    in_types = _stream_types(input_path)
    if in_types and "audio" in in_types and "audio" not in out_types:
        return False
    return True


def run_jasna(input_path: str, output_path: str, device: str = "auto"):
    """Run Jasna locally to demosaic a video.

    Jasna writes to a staging file first. The final ``output_path`` is only
    published after the result is verified to carry the expected streams, so
    a half-processed file is never exposed as the final output.
    """
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)

    if device not in ("auto", "") and device.startswith("cpu"):
        print("Error: Jasna is a GPU-only pipeline; CPU is not supported", file=sys.stderr)
        return False

    # Staging sibling file keeps a real extension (Jasna infers the output
    # format from it), so a half-processed file is never exposed as the
    # final output.
    root, ext = os.path.splitext(output_abs)
    staging = f"{root}.staging{ext or '.mp4'}"
    _remove_file(staging)

    cmd = [JASNA_BIN, "--input", input_abs, "--output", staging, "--no-progress"]
    if device not in ("auto", ""):
        cmd += ["--device", device]

    print(f"Running Jasna: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()
    try:
        subprocess.run(
            cmd, check=True, capture_output=False, text=True,
            cwd=JASNA_HOME,
        )
        elapsed = time.time() - t0
        print(f"Jasna completed in {elapsed:.1f}s", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Jasna failed: {e}", file=sys.stderr)
        _remove_file(staging)
        return False

    if not _verify_output(staging, input_abs):
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
    _check_jasna()
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if not output:
        base, ext = os.path.splitext(os.path.basename(args.input))
        output = f"{base}_demosaic{ext}"

    print(f"Processing: {args.input} -> {output}", file=sys.stderr)
    success = run_jasna(args.input, output, args.device)
    if success:
        print(f"Done: {output}", file=sys.stderr)
    else:
        sys.exit(1)


def main():
    # Bare filename auto-detection: rewrite `demosaic.py input.mp4` into
    # `demosaic.py demosaic input.mp4` before parsing.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        sys.argv.insert(1, "demosaic")

    parser = argparse.ArgumentParser(
        description="Video demosaic (mosaic removal) using Jasna."
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # demosaic command (single file)
    p_demo = sub.add_parser("demosaic", help="Process a single video file")
    p_demo.add_argument("input", help="Input video file path")
    p_demo.add_argument("-o", "--output", default=None,
                        help="Output video file (default: {name}_demosaic.{ext})")
    p_demo.add_argument("--device", default="auto",
                        help="Device: auto or cuda:N (Jasna is GPU-only)")

    args = parser.parse_args()

    if args.command == "demosaic":
        cmd_demosaic(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
