#!/home/yuanqi.xhf/miniconda3/bin/python
"""
demosaic.py — Video demosaic (mosaic removal) using LADA.

Monitors a WebDAV shared directory for new videos and processes them
with the LADA depixelization model.

Usage:
    ./demosaic.py loop                         # Monitor and process new videos
    ./demosaic.py input.mp4 -o output.mp4      # Process a single video

LADA is run via Docker: ladaapp/lada:latest
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# WebDAV helpers (using webdav4, same as w/stores/webdav_store.py)
# ---------------------------------------------------------------------------

def _load_webdav_env():
    """Load WebDAV credentials from env files."""
    env = {}
    # Check env files in order of preference
    env_files = [
        os.path.join(os.path.dirname(__file__), "..", "env", "webdav.env"),
        os.path.join(os.path.dirname(__file__), "..", "dav.env"),
        os.path.expanduser("~/m/env/webdav.env"),
        os.path.expanduser("~/m/dav.env"),
    ]
    for f in env_files:
        if os.path.isfile(f):
            with open(f) as fh:
                for line in fh:
                    m = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$", line)
                    if not m:
                        continue
                    val = m.group(2).strip()
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    env[m.group(1)] = val
            break

    url = env.get("WEBDAV_ENDPOINT_URL") or env.get("WEBDAV_URL")
    if url:
        p = urllib.parse.urlsplit(url)
        return {
            "hostname": f"{p.scheme}://{p.hostname}" + (f":{p.port}" if p.port else ""),
            "username": p.username or env.get("WEBDAV_USERNAME", ""),
            "password": p.password or env.get("WEBDAV_PASSWORD", ""),
            "root": env.get("WEBDAV_ROOT", "/"),
        }
    return {
        "hostname": env.get("WEBDAV_HOSTNAME", ""),
        "username": env.get("WEBDAV_USERNAME", ""),
        "password": env.get("WEBDAV_PASSWORD", ""),
        "root": env.get("WEBDAV_ROOT", "/"),
    }


def _get_webdav_client():
    """Create a webdav4 Client from env config."""
    from webdav4.client import Client
    creds = _load_webdav_env()
    hostname = creds["hostname"].rstrip("/")
    root = "/" + creds["root"].strip("/")
    base_url = f"{hostname}{root}"
    return Client(
        base_url,
        auth=(creds["username"], creds["password"]),
        verify=False,
        follow_redirects=True,
        timeout=120.0,
    )


def _list_webdav_dir(dav, path):
    """List a WebDAV directory, returning list of file names."""
    import os as _os
    try:
        items = dav.ls(path, detail=True)
    except Exception as e:
        print(f"Error listing {path}: {e}", file=sys.stderr)
        return []

    names = []
    for item in items:
        name = item.get("name", "")
        name = _os.path.split(name)[1]
        # Skip dirs
        is_dir = (
            item.get("isdir") or
            item.get("href", "").endswith("/") or
            item.get("content_type") in ("httpd/unix-directory", "directory")
        )
        if not is_dir and name:
            names.append(name)
    return names


def _download_from_webdav(dav, remote_path, local_path):
    """Download a file from WebDAV to local path."""
    import io
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    try:
        buffer = io.BytesIO()
        dav.download_fileobj(remote_path, buffer)
        with open(local_path, "wb") as f:
            f.write(buffer.getvalue())
        return True
    except Exception as e:
        print(f"Download error {remote_path}: {e}", file=sys.stderr)
        return False


def _upload_to_webdav(dav, local_path, remote_path):
    """Upload a local file to WebDAV."""
    import io
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        buffer = io.BytesIO(data)
        dav.upload_fileobj(buffer, remote_path, overwrite=True)
        return True
    except Exception as e:
        print(f"Upload error {remote_path}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# LADA runner (via Docker)
# ---------------------------------------------------------------------------

def _check_docker():
    """Check if Docker is available and GPU-capable."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Docker is not available.", file=sys.stderr)
        sys.exit(1)

    # Check GPU support
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.4.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print("Warning: Docker GPU support may not be available.", file=sys.stderr)
            print("Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",
                  file=sys.stderr)
    except Exception:
        print("Warning: Could not verify Docker GPU support.", file=sys.stderr)


def _ensure_lada_image():
    """Pull LADA Docker image if not present."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "ladaapp/lada:latest"],
            capture_output=True, text=True, timeout=10
        )
        if not result.stdout.strip():
            print("Pulling ladaapp/lada:latest Docker image...", file=sys.stderr)
            subprocess.run(["docker", "pull", "ladaapp/lada:latest"], check=True)
            print("Done.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error pulling LADA image: {e}", file=sys.stderr)
        sys.exit(1)


def run_lada(input_path: str, output_path: str, device: str = "auto"):
    """Run LADA via Docker to demosaic a video."""
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    input_dir = os.path.dirname(input_abs)
    input_name = os.path.basename(input_abs)
    output_name = os.path.basename(output_abs)

    # Create output dir if needed
    os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)

    # Always use /mnt as mount point inside container
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--mount", f"type=bind,src={input_dir},dst=/mnt",
    ]

    # If output is in a different dir, mount it too
    output_dir = os.path.dirname(output_abs)
    if output_dir != input_dir:
        cmd.extend(["--mount", f"type=bind,src={output_dir},dst=/out"])

    cmd.extend([
        "ladaapp/lada:latest",
        "--input", f"/mnt/{input_name}",
    ])

    if output_dir != input_dir:
        cmd.extend(["--output", f"/out/{output_name}"])
    else:
        cmd.extend(["--output", f"/mnt/{output_name}"])

    print(f"Running LADA: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - t0
        print(f"LADA completed in {elapsed:.1f}s", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"LADA failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# State file management
# ---------------------------------------------------------------------------

def _load_state(state_file: str) -> dict:
    """Load the processing state file."""
    if os.path.isfile(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"processed": {}, "in_progress": {}}


def _save_state(state_file: str, state: dict):
    """Save the processing state file."""
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _file_hash(local_path: str) -> str:
    """Compute MD5 hash of a file for deduplication."""
    h = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------

def cmd_demosaic(args):
    """Process a single video file."""
    _check_docker()
    _ensure_lada_image()

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
    """Monitor WebDAV shared dir and process new videos."""
    _check_docker()
    _ensure_lada_image()

    watch_dir = args.watch_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    state_file = args.state_file
    interval = args.interval
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}

    print(f"Monitoring WebDAV: {watch_dir}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)
    print(f"Temp dir: {temp_dir}", file=sys.stderr)
    print(f"State file: {state_file}", file=sys.stderr)
    print(f"Poll interval: {interval}s", file=sys.stderr)

    os.makedirs(temp_dir, exist_ok=True)

    dav = _get_webdav_client()
    state = _load_state(state_file)

    while True:
        try:
            # List remote files
            remote_files = _list_webdav_dir(dav, watch_dir)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(remote_files)} files in {watch_dir}",
                  file=sys.stderr)

            for fname in remote_files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in video_exts:
                    continue

                # Skip already processed
                if fname in state.get("processed", {}):
                    entry = state["processed"][fname]
                    if entry.get("status") == "done":
                        continue

                # Skip in-progress
                if fname in state.get("in_progress", {}):
                    continue

                # Generate output name
                base, ext = os.path.splitext(fname)
                output_name = f"{base}_demosaic{ext}"

                # Check if output already exists remotely
                existing = _list_webdav_dir(dav, output_dir)
                if output_name in existing:
                    print(f"Skipping {fname}: output {output_name} already exists", file=sys.stderr)
                    state.setdefault("processed", {})[fname] = {
                        "status": "done",
                        "output": output_name,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_state(state_file, state)
                    continue

                # Download, process, upload
                print(f"Processing: {fname}", file=sys.stderr)
                state.setdefault("in_progress", {})[fname] = {
                    "started": datetime.now(timezone.utc).isoformat(),
                }
                _save_state(state_file, state)

                local_input = os.path.join(temp_dir, fname)
                local_output = os.path.join(temp_dir, output_name)

                # Download
                remote_path = f"{watch_dir}/{fname}" if not watch_dir.endswith("/") else f"{watch_dir}{fname}"
                print(f"  Downloading: {remote_path}", file=sys.stderr)
                if not _download_from_webdav(dav, remote_path, local_input):
                    state["in_progress"].pop(fname, None)
                    state.setdefault("processed", {})[fname] = {
                        "status": "failed",
                        "error": "download failed",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_state(state_file, state)
                    continue

                # Process with LADA
                print(f"  Running LADA...", file=sys.stderr)
                success = run_lada(local_input, local_output, args.device)

                if not success:
                    state["in_progress"].pop(fname, None)
                    state.setdefault("processed", {})[fname] = {
                        "status": "failed",
                        "error": "LADA processing failed",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_state(state_file, state)
                    # Clean up local files
                    if os.path.isfile(local_input):
                        os.remove(local_input)
                    continue

                # Upload result
                remote_output = f"{output_dir}/{output_name}" if not output_dir.endswith("/") else f"{output_dir}{output_name}"
                print(f"  Uploading: {remote_output}", file=sys.stderr)
                if not _upload_to_webdav(dav, local_output, remote_output):
                    state["in_progress"].pop(fname, None)
                    state.setdefault("processed", {})[fname] = {
                        "status": "failed",
                        "error": "upload failed",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_state(state_file, state)
                else:
                    state["in_progress"].pop(fname, None)
                    state.setdefault("processed", {})[fname] = {
                        "status": "done",
                        "output": output_name,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    _save_state(state_file, state)
                    print(f"  Done: {fname} -> {output_name}", file=sys.stderr)

                # Cleanup temp files
                if args.cleanup:
                    if os.path.isfile(local_input):
                        os.remove(local_input)
                    if os.path.isfile(local_output):
                        os.remove(local_output)

        except KeyboardInterrupt:
            print("\nStopping...", file=sys.stderr)
            break
        except Exception as e:
            print(f"Loop error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Video demosaic (mosaic removal) using LADA."
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # loop command
    p_loop = sub.add_parser("loop", help="Monitor WebDAV shared dir and process new videos")
    p_loop.add_argument("--watch-dir", default="shared",
                        help="WebDAV directory to monitor (default: shared)")
    p_loop.add_argument("--output-dir", default="shared",
                        help="WebDAV directory for output (default: shared)")
    p_loop.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    p_loop.add_argument("--state-file", default="demosaic_state.json",
                        help="State file to track processed videos")
    p_loop.add_argument("--temp-dir", default="/tmp/demosaic",
                        help="Local temp directory for processing")
    p_loop.add_argument("--cleanup", action="store_true", default=True,
                        help="Delete local temp files after upload")
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