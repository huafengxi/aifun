#!/usr/bin/env python3
"""
download_model.py — Download models from ModelScope or HuggingFace.

Supports downloading from ModelScope (preferred for China) or HuggingFace.

Usage:
    python3 download_model.py Qwen/Qwen3-ASR-1.7B
    python3 download_model.py --source huggingface Qwen/Qwen3-ASR-1.7B
    python3 download_model.py --source modelscope Qwen/Qwen3-ASR-1.7B --cache-dir /model-cache
"""

import argparse
import os
import sys


def download_from_modelscope(model_id: str, cache_dir: str = None) -> str:
    """Download a model from ModelScope and return the local path."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("Error: modelscope package not installed. Install with: pip install modelscope",
              file=sys.stderr)
        sys.exit(1)

    # Map HuggingFace model IDs to ModelScope format
    # ModelScope uses the same org/name format but may need different paths
    # Convert "Qwen/Qwen3-ASR-1.7B" to modelscope path
    ms_model_id = model_id

    print(f"Downloading model '{ms_model_id}' from ModelScope...", file=sys.stderr)
    print(f"Cache directory: {cache_dir or 'default'}", file=sys.stderr)

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    local_path = snapshot_download(ms_model_id, **kwargs)
    print(f"Model downloaded to: {local_path}", file=sys.stderr)
    return local_path


def download_from_huggingface(model_id: str, cache_dir: str = None) -> str:
    """Download a model from HuggingFace and return the local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface-hub not installed.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Downloading model '{model_id}' from HuggingFace...", file=sys.stderr)
    print(f"Cache directory: {cache_dir or 'default'}", file=sys.stderr)

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    local_path = snapshot_download(model_id, **kwargs)
    print(f"Model downloaded to: {local_path}", file=sys.stderr)
    return local_path


def resolve_model_path(model_id: str, source: str = "auto", cache_dir: str = None) -> str:
    """
    Resolve a model ID to a local path by downloading if needed.

    Args:
        model_id: Model ID (e.g., "Qwen/Qwen3-ASR-1.7B")
        source: "modelscope", "huggingface", or "auto" (try modelscope first, fallback to HF)
        cache_dir: Cache directory for model files

    Returns:
        Local path to the downloaded model
    """
    if source == "auto":
        # Try ModelScope first (preferred for China), fallback to HuggingFace
        try:
            return download_from_modelscope(model_id, cache_dir)
        except Exception as e:
            print(f"ModelScope download failed: {e}", file=sys.stderr)
            print("Falling back to HuggingFace...", file=sys.stderr)
            return download_from_huggingface(model_id, cache_dir)
    elif source == "modelscope":
        return download_from_modelscope(model_id, cache_dir)
    elif source == "huggingface":
        return download_from_huggingface(model_id, cache_dir)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'modelscope', 'huggingface', or 'auto'.")


def main():
    parser = argparse.ArgumentParser(
        description="Download models from ModelScope or HuggingFace."
    )
    parser.add_argument(
        "model_id",
        help="Model ID (e.g., Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--source", choices=["modelscope", "huggingface", "auto"],
        default="auto",
        help="Download source (default: auto — try modelscope first, then huggingface)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Model cache directory (default: $MODELSCOPE_CACHE or $HUGGINGFACE_HUB_CACHE)"
    )

    args = parser.parse_args()

    # Determine cache dir from env if not specified
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.environ.get("MODELSCOPE_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")

    local_path = resolve_model_path(args.model_id, args.source, cache_dir)
    print(local_path)


if __name__ == "__main__":
    main()