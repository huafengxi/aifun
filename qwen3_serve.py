#!/usr/bin/env python3
"""
qwen3_serve.py — Serve a Qwen3 LLM with vLLM (OpenAI-compatible API) in Docker.

Model files are fetched from ModelScope first (preferred for China) into a local
``MODELS_DIR``, then a vLLM container is started with that directory mounted.

Usage:
    ./qwen3_serve.py download     # download model to MODELS_DIR
    ./qwen3_serve.py start        # download (if needed) + start container
    ./qwen3_serve.py stop         # stop + remove container
    ./qwen3_serve.py status       # container status + API health
    ./qwen3_serve.py logs         # tail container logs

Environment variables (all optional):

    QWEN3_MODEL          Model ID / ModelScope path   (default: Qwen/Qwen3.8-27B-FP8)
    QWEN3_PORT           Host port for the OpenAI API (default: 8000)
    QWEN3_IMAGE          vLLM OpenAI server image      (default: mass-runner:cuda13.0-vllm0.22.1)
    QWEN3_MODELS_DIR     Local model directory         (default: ~/m/models)
    QWEN3_CONTAINER      Container name                (default: qwen3-serve)
    QWEN3_SERVED_NAME    Served model name             (default: qwen3.8)
    QWEN3_TP             Tensor-parallel size          (default: 2)
    QWEN3_MAX_MODEL_LEN  Max context length            (default: 32768)
    QWEN3_SPECULATIVE    Extra --speculative-config      (default: MTP on)
    QWEN3_TOOL_CALL_PARSER  Tool call parser            (default: qwen3_xml; enables tool_choice=auto)
    QWEN3_REASONING_PARSER  Reasoning parser            (default: qwen3; splits thinking from answer)
    QWEN3_GPU            Docker --gpus value           (default: all)
    QWEN3_EXTRA_ARGS     Extra space-separated vLLM args

    MODEL_DOWNLOAD_SOURCE  auto | modelscope | huggingface  (default: auto)
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DOCKER = os.environ.get("QWEN3_DOCKER", "sudo").split() + ["docker"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def cfg():
    return {
        "model": os.environ.get("QWEN3_MODEL", "Qwen/Qwen3.8-27B-FP8"),
        "port": os.environ.get("QWEN3_PORT", "8000"),
        "image": os.environ.get(
            "QWEN3_IMAGE", "mass-runner:cuda13.0-vllm0.22.1"
        ),
        "models_dir": os.environ.get(
            "QWEN3_MODELS_DIR", os.path.expanduser("~/m/models")
        ),
        "container": os.environ.get("QWEN3_CONTAINER", "qwen3-serve"),
        "served_name": os.environ.get("QWEN3_SERVED_NAME", "qwen3.8"),
        "tp": os.environ.get("QWEN3_TP", "2"),
        "max_model_len": os.environ.get("QWEN3_MAX_MODEL_LEN", "32768"),
        "speculative": os.environ.get(
            "QWEN3_SPECULATIVE", '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
        ),
        "tool_call_parser": os.environ.get("QWEN3_TOOL_CALL_PARSER", "qwen3_xml"),
        "reasoning_parser": os.environ.get("QWEN3_REASONING_PARSER", "qwen3"),
        "gpu": os.environ.get("QWEN3_GPU", "all"),
        "extra_args": os.environ.get("QWEN3_EXTRA_ARGS", ""),
    }


def model_basename(model_id: str) -> str:
    return model_id.rstrip("/").rsplit("/", 1)[-1]


def local_model_dir(model_id: str, models_dir: str) -> str:
    return os.path.join(os.path.realpath(models_dir), model_basename(model_id))


def docker(*args, check=True, capture=True):
    return subprocess.run(DOCKER + list(args), check=check, text=True,
                          capture_output=capture)


# ---------------------------------------------------------------------------
# Model download (ModelScope preferred, HuggingFace fallback)
# ---------------------------------------------------------------------------

def download_model(model_id: str, models_dir: str) -> str:
    target = local_model_dir(model_id, models_dir)
    if (Path(target) / "config.json").is_file():
        print(f"Model already present: {target}", file=sys.stderr)
        return target

    os.makedirs(target, exist_ok=True)
    source = os.environ.get("MODEL_DOWNLOAD_SOURCE", "auto").lower()

    if source in ("modelscope", "auto"):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            print(f"Downloading {model_id} from ModelScope -> {target}",
                  file=sys.stderr)
            snapshot_download(model_id, local_dir=target)
            return target
        except Exception as e:  # noqa: BLE001
            if source == "modelscope":
                raise
            print(f"ModelScope download failed: {e}", file=sys.stderr)
            print("Falling back to HuggingFace...", file=sys.stderr)

    from huggingface_hub import snapshot_download as hf_snapshot_download
    print(f"Downloading {model_id} from HuggingFace -> {target}", file=sys.stderr)
    hf_snapshot_download(model_id, local_dir=target)
    return target


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

def container_exists(name: str) -> bool:
    r = docker("ps", "-a", "--filter", f"name=^{name}$",
               "--format", "{{.Names}}", check=False)
    return name in (r.stdout or "").splitlines()


def container_running(name: str) -> bool:
    r = docker("ps", "--filter", f"name=^{name}$",
               "--format", "{{.Names}}", check=False)
    return name in (r.stdout or "").splitlines()


def pull_image_if_needed(image: str) -> None:
    r = docker("image", "inspect", image, check=False)
    if r.returncode == 0:
        print(f"Image present: {image}", file=sys.stderr)
        return
    print(f"Pulling image {image} ...", file=sys.stderr)
    try:
        docker("pull", image, capture=False)
    except subprocess.CalledProcessError as e:
        print("Error: docker pull failed. The docker daemon may need a proxy:", file=sys.stderr)
        print("  see `./qwen3_serve.py docker-proxy-setup --help`", file=sys.stderr)
        raise e


def run_args(c):
    args = [
        "run", "-d",
        "--name", c["container"],
        "--gpus", c["gpu"],
        "--ipc", "host",
        "-p", f"{c['port']}:8000",
        "-v", f"{os.path.realpath(c['models_dir'])}:/models:ro",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        c["image"],
        "vllm", "serve",
        "--model", f"/models/{model_basename(c['model'])}",
        "--served-model-name", c["served_name"],
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", c["tp"],
        "--gpu-memory-utilization", "0.92",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
    ]
    if c.get("max_model_len"):
        args.extend(["--max-model-len", str(c["max_model_len"])])
    if c.get("speculative", "").strip():
        args.extend(["--speculative-config", c["speculative"].strip()])
    if c.get("tool_call_parser", "").strip():
        args.extend(["--tool-call-parser", c["tool_call_parser"].strip()])
    if c.get("reasoning_parser", "").strip():
        args.extend(["--reasoning-parser", c["reasoning_parser"].strip()])
    if c["extra_args"].strip():
        args.extend(shlex.split(c["extra_args"].strip()))
    return args


def do_start(c):
    model_path = download_model(c["model"], c["models_dir"])
    print(f"Model dir: {model_path}", file=sys.stderr)

    if container_running(c["container"]):
        print(f"Container '{c['container']}' already running — nothing to do.",
              file=sys.stderr)
        return 0

    if container_exists(c["container"]):
        print(f"Removing stopped container '{c['container']}' ...", file=sys.stderr)
        docker("rm", "-f", c["container"], check=False)

    pull_image_if_needed(c["image"])

    args = run_args(c)
    print(f"Starting container: {' '.join(DOCKER + args)}", file=sys.stderr)
    docker(*args, capture=False)


def do_stop(c):
    if container_exists(c["container"]):
        docker("rm", "-f", c["container"], capture=False)
        print(f"Container '{c['container']}' stopped/removed.", file=sys.stderr)
    else:
        print(f"Container '{c['container']}' not found.", file=sys.stderr)


def do_status(c):
    name = c["container"]
    r = docker("ps", "-a", "--filter", f"name=^{name}$",
               "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}", check=False)
    lines = (r.stdout or "").splitlines()
    if not lines:
        print(f"Container '{name}': NOT created")
        return
    for line in lines:
        print(f"Container: {line}")

    # API health
    try:
        import urllib.request
        with urllib.request.urlopen(
            f"http://127.0.0.1:{c['port']}/health", timeout=3
        ) as resp:
            print(f"API /health: HTTP {resp.status}")
    except Exception as e:  # noqa: BLE001
        print(f"API /health: not ready ({e})")

    print("--- recent container logs ---")
    logs = docker("logs", "--tail", "20", name, check=False)
    sys.stdout.write(logs.stdout or "(no logs)")


def do_logs(c, tail="50"):
    r = docker("logs", "--tail", str(tail), c["container"], check=False)
    sys.stdout.write(r.stdout or "")
    if r.stderr:
        sys.stderr.write(r.stderr)


def do_download(c):
    model_path = download_model(c["model"], c["models_dir"])
    print(model_path)


def do_docker_proxy_setup(args):
    """Configure the docker daemon to use a proxy for `docker pull` (China)."""
    proxy = args.proxy
    if not proxy and not args.remove:
        proxy = os.environ.get("CLASH_PROXY", "http://127.0.0.1:7897")

    dropin_dir = Path("/etc/systemd/system/docker.service.d")
    dropin = dropin_dir / "http-proxy.conf"

    if args.remove:
        if dropin.exists():
            subprocess.run(["sudo", "rm", "-f", str(dropin)], check=True)
            print(f"Removed {dropin}")
    else:
        dropin_dir.mkdir(parents=True, exist_ok=True)
        content = (
            "[Service]\n"
            f'Environment="HTTP_PROXY={proxy}"\n'
            f'Environment="HTTPS_PROXY={proxy}"\n'
            'Environment="NO_PROXY=localhost,127.0.0.1"\n'
        )
        # write via sudo tee so the root-owned dir is writable
        subprocess.run(
            ["sudo", "tee", str(dropin)],
            input=content, text=True, check=True, capture_output=True,
        )
        print(f"Wrote {dropin}")
        print(f"  HTTP_PROXY/HTTPS_PROXY -> {proxy}")

    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    if args.restart:
        print("Restarting docker daemon (WARNING: restarts running containers)...")
        subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
    else:
        print("Run `sudo systemctl restart docker` to apply.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Serve a Qwen3 LLM with vLLM in Docker.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="Download the model to MODELS_DIR")
    sub.add_parser("start", help="Download (if needed) and start the container")
    sub.add_parser("stop", help="Stop and remove the container")
    sub.add_parser("status", help="Show container status and API health")

    logs_p = sub.add_parser("logs", help="Tail container logs")
    logs_p.add_argument("--tail", default="50", help="Number of lines (default 50)")

    proxy_p = sub.add_parser(
        "docker-proxy-setup",
        help="Configure docker daemon proxy (for docker pull in China)",
    )
    proxy_p.add_argument("--proxy", default=None, help="Proxy URL (default http://127.0.0.1:7897)")
    proxy_p.add_argument("--restart", action="store_true", help="Restart docker daemon")
    proxy_p.add_argument("--remove", action="store_true", help="Remove the proxy config")

    args = p.parse_args()
    c = cfg()

    if args.cmd == "download":
        do_download(c)
    elif args.cmd == "start":
        do_start(c)
    elif args.cmd == "stop":
        do_stop(c)
    elif args.cmd == "status":
        do_status(c)
    elif args.cmd == "logs":
        do_logs(c, args.tail)
    elif args.cmd == "docker-proxy-setup":
        do_docker_proxy_setup(args)
    else:
        p.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()