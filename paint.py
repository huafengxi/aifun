#!/home/yuanqi.xhf/miniconda3/bin/python
"""
paint.py — Text-to-image generation using Stable Diffusion / Flux.

Generates images from text prompts using diffusers pipelines.

Usage:
    ./paint.py "a cat sitting on a cloud" -o cat.png
    ./paint.py "cyberpunk city at night" --model runwayml/stable-diffusion-v1-5 --steps 30
    ./paint.py raemu "1girl, cherry blossoms"

Model aliases (set via first positional arg):
    raemu    Raemu-XL-V5 (anime-style SDXL model)
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import torch

# Model aliases — first positional arg can be an alias to select the model
MODEL_ALIASES = {
    "raemu": "Raelina/Raemu-XL-V5",
    "flux2": "black-forest-labs/FLUX.2-klein-9B",
    "flux2_fp8": "black-forest-labs/FLUX.2-klein-9b-fp8",
    "flux2_nsfw": "black-forest-labs/FLUX.2-klein-9B",
    "flux2_nsfw_fp8": "black-forest-labs/FLUX.2-klein-9b-fp8",
}

# LoRA aliases — auto-load LoRA for certain model aliases
LORA_ALIASES = {
    "flux2_nsfw": os.path.join(os.path.expanduser("~"), ".cache", "lora", "Flux_Klein_NSFW_v2.safetensors"),
    "flux2_nsfw_fp8": os.path.join(os.path.expanduser("~"), ".cache", "lora", "Flux_Klein_NSFW_v2.safetensors"),
}


def _detect_pipeline(model_path: str) -> str:
    """Detect pipeline class from model config.json."""
    config_path = os.path.join(model_path, "model_index.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("_class_name", "")
        except Exception:
            pass
    return ""


def _find_checkpoint_file(model_path: str) -> str | None:
    """Find a single .safetensors or .ckpt checkpoint file in the model directory."""
    if os.path.isfile(model_path):
        return model_path
    safetensors_files = sorted(
        [f for f in os.listdir(model_path) if f.endswith(".safetensors")],
        key=lambda f: os.path.getsize(os.path.join(model_path, f)),
        reverse=True,
    )
    if safetensors_files:
        return os.path.join(model_path, safetensors_files[0])
    ckpt_files = [f for f in os.listdir(model_path) if f.endswith(".ckpt")]
    if ckpt_files:
        return os.path.join(model_path, ckpt_files[0])
    return None


def _is_single_file_model(model_path: str) -> bool:
    """Check if the model path is a single-file checkpoint rather than diffusers format."""
    return not os.path.isfile(os.path.join(model_path, "model_index.json"))


def _find_original_config(model_id: str, is_xl: bool, is_flux: bool) -> str | None:
    """Find the original config files for a single-file checkpoint.

    Single-file checkpoints need the original model config (e.g., from SDXL base).
    Checks ModelScope and HuggingFace caches for the required config files.
    """
    if is_flux:
        original_model = "black-forest-labs/FLUX.1-dev"
    elif is_xl:
        original_model = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        original_model = "runwayml/stable-diffusion-v1-5"

    safe_id = original_model.replace("/", "--")

    # Check ModelScope cache
    ms_cache_dirs = [
        os.environ.get("MODELSCOPE_CACHE", ""),
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
    ]
    for ms_cache in ms_cache_dirs:
        if not ms_cache:
            continue
        ms_dir = os.path.join(ms_cache, "models", safe_id, "snapshots")
        if os.path.isdir(ms_dir):
            snapshots = sorted(os.listdir(ms_dir), reverse=True)
            for snap in snapshots:
                snap_path = os.path.join(ms_dir, snap)
                if os.path.isfile(os.path.join(snap_path, "model_index.json")):
                    return snap_path

    # Check HuggingFace cache
    hf_cache_dirs = [
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.environ.get("HF_HOME", ""),
        "/data/yuanqi.xhf/cache/huggingface/hub",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    for hf_cache in hf_cache_dirs:
        if not hf_cache:
            continue
        hf_dir = os.path.join(hf_cache, f"models--{safe_id}", "snapshots")
        if os.path.isdir(hf_dir):
            snapshots = sorted(glob.glob(os.path.join(hf_dir, "*")),
                              key=os.path.getmtime, reverse=True)
            for snap in snapshots:
                if os.path.isfile(os.path.join(snap, "model_index.json")):
                    return snap

    return None


def resolve_model_path(model_id: str, cache_dir: str = None) -> str:
    """Resolve model to local path, downloading from ModelScope or HuggingFace."""
    # Check local cache first
    source = os.environ.get("MODEL_DOWNLOAD_SOURCE", "auto").lower()

    local_path = _check_local_cache(model_id, cache_dir)
    if local_path:
        print(f"Using cached model: {local_path}", file=sys.stderr)
        return local_path

    # If user explicitly chose HuggingFace, use it directly
    if source == "huggingface":
        try:
            from download_model import download_from_huggingface
            return download_from_huggingface(model_id, cache_dir)
        except Exception:
            pass
        return model_id

    # Try ModelScope first, fallback to HuggingFace
    try:
        from download_model import resolve_model_path as rmp
        return rmp(model_id, source=source, cache_dir=cache_dir)
    except Exception:
        pass
    return model_id


def _check_local_cache(model_id: str, cache_dir: str = None) -> str | None:
    """Check if model exists in local cache (HF or ModelScope)."""
    safe_id = model_id.replace("/", "--")

    # Check ModelScope cache first (preferred, more reliable in China)
    ms_cache_dirs = []
    if cache_dir:
        ms_cache_dirs.append(cache_dir)
    ms_cache_dirs.extend([
        os.environ.get("MODELSCOPE_CACHE", ""),
        "/data/yuanqi.xhf/cache/modelscope",
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
    ])
    for ms_cache in ms_cache_dirs:
        if not ms_cache:
            continue
        ms_model_dir = os.path.join(ms_cache, "models", safe_id, "snapshots")
        if os.path.isdir(ms_model_dir):
            snapshots = sorted(os.listdir(ms_model_dir), reverse=True)
            for snap in snapshots:
                snap_path = os.path.join(ms_model_dir, snap)
                if os.path.isfile(os.path.join(snap_path, "model_index.json")):
                    return snap_path

    # Check HuggingFace cache
    hf_cache_dirs = []
    if cache_dir:
        hf_cache_dirs.append(cache_dir)
    hf_cache_dirs.extend([
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.environ.get("HF_HOME", ""),
        "/data/yuanqi.xhf/cache/huggingface/hub",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ])
    for hf_cache in hf_cache_dirs:
        if not hf_cache:
            continue
        hf_model_dir = os.path.join(hf_cache, f"models--{safe_id}")
        if os.path.isdir(hf_model_dir):
            snapshots = sorted(glob.glob(os.path.join(hf_model_dir, "snapshots", "*")),
                              key=os.path.getmtime, reverse=True)
            for snap in snapshots:
                if os.path.isfile(os.path.join(snap, "model_index.json")):
                    return snap

    return None


LORA_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lora")


def _resolve_lora(lora_spec: str, model_id: str = "") -> str:
    """Resolve LoRA from local path, URL, or CivitAI model URL.

    Supports:
      - Local .safetensors path
      - CivitAI model URL: https://civitai.com/models/<id>
      - CivitAI download URL: https://civitai.com/api/download/models/<version>
    """
    # Already a local file
    if os.path.isfile(lora_spec):
        return lora_spec

    # CivitAI URL handling
    if "civitai.com" in lora_spec:
        return _download_civitai_lora(lora_spec)

    # Try as local path even if not found
    if os.path.isfile(lora_spec):
        return lora_spec

    print(f"Error: LoRA not found: {lora_spec}", file=sys.stderr)
    sys.exit(1)


def _download_civitai_lora(url: str) -> str:
    """Download a LoRA from CivitAI.

    Requires CIVITAI_API_TOKEN environment variable.
    """
    api_token = os.environ.get("CIVITAI_API_TOKEN", "")
    if not api_token:
        # Try reading from ~/.pi/env/civitai
        token_file = os.path.join(os.path.expanduser("~"), ".pi", "env", "civitai")
        if os.path.isfile(token_file):
            with open(token_file, "r") as f:
                api_token = f.read().strip()
    if not api_token:
        print("Error: CIVITAI_API_TOKEN env var not set.", file=sys.stderr)
        print("Get your API token from https://civitai.com/user/account", file=sys.stderr)
        print("Then: export CIVITAI_API_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)

    # Extract model ID from URL
    # Patterns: /models/<id>, /models/<id>/slug, /api/download/models/<version>
    model_id = None
    version_id = None

    # Match /models/<id>
    m = re.search(r'/models/(\d+)', url)
    if m:
        model_id = m.group(1)

    # Match /api/download/models/<version>
    m = re.search(r'/api/download/models/(\d+)', url)
    if m:
        version_id = m.group(1)

    if not model_id:
        print(f"Error: Could not parse model ID from URL: {url}", file=sys.stderr)
        sys.exit(1)

    # Fetch model info from CivitAI API
    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    print(f"Fetching model info from CivitAI...", file=sys.stderr)

    req = urllib.request.Request(api_url)
    req.add_header("Authorization", f"Bearer {api_token}")
    req.add_header("User-Agent", "paint.py/1.0")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching model info: {e}", file=sys.stderr)
        sys.exit(1)

    # Find the model version
    versions = info.get("modelVersions", [])
    if not versions:
        print("Error: No model versions found.", file=sys.stderr)
        sys.exit(1)

    # Use specified version or latest
    target = versions[0]
    if version_id:
        for v in versions:
            if str(v["id"]) == version_id:
                target = v
                break

    # Find the primary .safetensors file
    files = target.get("files", [])
    safetensor_file = None
    for f in files:
        if f.get("name", "").endswith(".safetensors"):
            safetensor_file = f
            break

    if not safetensor_file:
        print("Error: No .safetensors file found in LoRA.", file=sys.stderr)
        sys.exit(1)

    file_name = safetensor_file["name"]
    download_url = safetensor_file.get("downloadUrl", "")
    if not download_url:
        download_url = f"https://civitai.com/api/download/models/{target['id']}?fileId={safetensor_file['id']}"

    # Download to cache
    os.makedirs(LORA_CACHE_DIR, exist_ok=True)
    dest = os.path.join(LORA_CACHE_DIR, file_name)

    if os.path.isfile(dest):
        print(f"Using cached LoRA: {dest}", file=sys.stderr)
        return dest

    print(f"Downloading {file_name} ({safetensor_file.get('sizeKB', 0):.0f} KB)...", file=sys.stderr)

    dl_req = urllib.request.Request(download_url)
    dl_req.add_header("Authorization", f"Bearer {api_token}")
    dl_req.add_header("User-Agent", "paint.py/1.0")

    try:
        with urllib.request.urlopen(dl_req, timeout=300) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"Error downloading LoRA: {e}", file=sys.stderr)
        if os.path.isfile(dest):
            os.remove(dest)
        sys.exit(1)

    print(f"Downloaded to {dest}", file=sys.stderr)
    return dest


def _load_lora_safe(pipe, lora_path: str, lora_scale: float = 1.0):
    """Load LoRA weights with compatibility handling for alpha keys."""
    import safetensors.torch as st

    state_dict = st.load_file(lora_path)

    # Handle alpha keys: apply scaling to lora_down/lora_A weights, then remove alpha keys.
    # Keep the "diffusion_model." prefix so diffusers can auto-convert the format.
    alpha_keys = [k for k in state_dict if k.endswith(".alpha")]
    if alpha_keys:
        for alpha_key in alpha_keys:
            base = alpha_key[:-len(".alpha")]
            lora_a_key = base + ".lora_A.weight"
            lora_down_key = base + ".lora_down.weight"
            alpha_val = state_dict.pop(alpha_key).item()

            # Apply alpha scaling to the down weight
            if lora_a_key in state_dict:
                state_dict[lora_a_key] = state_dict[lora_a_key] * (alpha_val / state_dict[lora_a_key].shape[0])
            elif lora_down_key in state_dict:
                state_dict[lora_down_key] = state_dict[lora_down_key] * (alpha_val / state_dict[lora_down_key].shape[0])

        print(f"Applied alpha scaling from {len(alpha_keys)} keys", file=sys.stderr)

    pipe.load_lora_weights(state_dict)

    if lora_scale != 1.0:
        pipe.fuse_lora(lora_scale=lora_scale)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion / Flux."
    )
    parser.add_argument("prompt", help="Text prompt describing the image to generate")
    parser.add_argument(
        "-o", "--output", default="output.png",
        help="Output image file (default: output.png)"
    )
    parser.add_argument(
        "--model", default="xiaolxl/GuoFeng4_XL",
        help="Model ID: stable-diffusion, SDXL, or Flux (default: xiaolxl/GuoFeng4_XL)"
    )
    parser.add_argument(
        "--negative-prompt", default="",
        help="Negative prompt — what to avoid in the image"
    )
    parser.add_argument(
        "--steps", type=int, default=25,
        help="Inference steps (default: 25)"
    )
    parser.add_argument(
        "--cfg", type=float, default=7.5,
        help="Classifier-free guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Image width (default: 512 for SD, 1024 for SDXL/Flux)"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Image height (default: 512 for SD, 1024 for SDXL/Flux)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: 'auto', 'cuda', 'cuda:0', 'cpu'"
    )
    parser.add_argument(
        "--dtype", default="float16",
        help="Compute dtype: 'float16', 'float32', 'bfloat16' (CUDA default: bfloat16)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of images to generate (default: 1)"
    )
    parser.add_argument(
        "--lora", default=None,
        help="LoRA weights: local .safetensors path, or CivitAI URL (e.g. https://civitai.com/models/123)"
    )
    parser.add_argument(
        "--lora-scale", type=float, default=1.0,
        help="LoRA strength multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile on transformer for faster inference (first run is slower)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="(Deprecated) CUDA optimizations are now enabled by default"
    )

    # Handle model aliases: if first positional arg is a known alias, shift it
    _used_alias = None
    if len(sys.argv) > 1 and sys.argv[1] in MODEL_ALIASES:
        _used_alias = sys.argv[1]
        sys.argv[1:2] = ["--model", MODEL_ALIASES[_used_alias]]

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = getattr(torch, args.dtype, torch.float16)
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32
        print("Info: CPU detected, using dtype=float32", file=sys.stderr)

    # CUDA optimizations (default on)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        # Use bfloat16 by default on CUDA (much faster on modern GPUs like RTX 40xx/50xx)
        if args.dtype == "float16":
            dtype = torch.bfloat16

    # Seed
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # Resolve model path
    cache_dir = os.environ.get("MODELSCOPE_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    model_path = resolve_model_path(args.model, cache_dir)

    print(f"Loading model '{model_path}' on {device} (dtype={dtype})...", file=sys.stderr)
    t0 = time.time()

    # Determine pipeline type and whether it's a single-file checkpoint
    model_lower = args.model.lower()
    pipeline_cls = _detect_pipeline(model_path)
    single_file = _is_single_file_model(model_path)

    is_flux = pipeline_cls == "FluxPipeline" or "flux" in model_lower
    is_flux2 = pipeline_cls == "Flux2KleinPipeline" or "flux2" in model_lower or "flux.2" in model_lower
    is_xl = pipeline_cls == "StableDiffusionXLPipeline" or "sdxl" in model_lower or ("xl" in model_lower and not is_flux2)

    if single_file:
        checkpoint_path = _find_checkpoint_file(model_path)
        if checkpoint_path is None:
            print("Error: No .safetensors or .ckpt file found in model directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Using checkpoint: {os.path.basename(checkpoint_path)}", file=sys.stderr)

        # Find original config for single-file checkpoints
        original_config = _find_original_config(args.model, is_xl, is_flux)
        if original_config:
            # original_config should be a YAML file path, not a directory
            if os.path.isdir(original_config):
                # Look for YAML config in the directory
                for yaml_name in ["sd_xl_base.yaml", "v1-inference.yaml", "sd_xl_base.yml"]:
                    yaml_path = os.path.join(original_config, yaml_name)
                    if os.path.isfile(yaml_path):
                        original_config = yaml_path
                        break
                else:
                    # Bundled config
                    bundled = os.path.join(os.path.dirname(__file__), "sdxl_original_config.yaml")
                    if os.path.isfile(bundled):
                        original_config = bundled
            print(f"Using original config: {original_config}", file=sys.stderr)

        # Find diffusers config directory for component loading
        diffusers_config = _find_original_config(args.model, is_xl, is_flux)

    if is_flux2:
        from diffusers import Flux2KleinPipeline
        if single_file:
            pipe = Flux2KleinPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype,
                original_config=original_config,
                config=diffusers_config,
                local_files_only=True,
            )
        else:
            pipe = Flux2KleinPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
            )
    elif is_flux:
        from diffusers import StableDiffusionXLPipeline
        if single_file:
            pipe = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype,
                use_safetensors=True,
                original_config=original_config,
                config=diffusers_config,
                local_files_only=True,
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
    else:
        from diffusers import StableDiffusionPipeline
        if single_file:
            pipe = StableDiffusionPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype,
                use_safetensors=True,
                original_config=original_config,
                config=diffusers_config,
                local_files_only=True,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )

    # Auto-detect width/height for SDXL/Flux (they use target_size internally)
    if args.width is None:
        args.width = 1024 if (is_flux or is_flux2 or is_xl) else 512
    if args.height is None:
        args.height = 1024 if (is_flux or is_flux2 or is_xl) else 512

    # Load LoRA if specified, or if model alias has an auto LoRA
    lora_path = None
    if args.lora:
        lora_path = _resolve_lora(args.lora, args.model)
    elif _used_alias and _used_alias in LORA_ALIASES:
        lora_path = LORA_ALIASES[_used_alias]

    if lora_path:
        print(f"Loading LoRA: {lora_path}", file=sys.stderr)
        _load_lora_safe(pipe, lora_path, args.lora_scale)

    # Enable torch.compile on transformer for faster inference
    if args.compile and device.startswith("cuda"):
        print("Compiling transformer (first run will be slower)...", file=sys.stderr)
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=False,
        )

    # Enable memory optimizations (xformers OR cpu_offload, not both)
    if device.startswith("cuda"):
        # Try xformers first; if not available, move to GPU and use SDPA
        try:
            pipe.enable_xformers_memory_efficient_attention()
            pipe = pipe.to(device)
        except Exception:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    print(f"Model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Generate
    print(f"Generating: \"{args.prompt}\"", file=sys.stderr)
    t1 = time.time()

    kwargs = dict(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        generator=generator,
    )

    # Flux2 is distilled, no CFG, no negative prompt
    if not is_flux2:
        kwargs["guidance_scale"] = args.cfg
        if args.negative_prompt:
            kwargs["negative_prompt"] = args.negative_prompt
    else:
        kwargs["guidance_scale"] = 1.0  # suppress "ignored" warning for distilled models

    if not is_xl and not is_flux and not is_flux2:
        kwargs["width"] = args.width
        kwargs["height"] = args.height

    # Generate requested number of images
    images = []
    for i in range(args.batch_size):
        if args.batch_size > 1:
            if args.seed is not None:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(args.seed + i)
            else:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(
                    torch.randint(0, 2**32 - 1, (1,)).item()
                )
        result = pipe(**kwargs)
        img = result.images[0] if hasattr(result, "images") else result[0]
        images.append(img)

    elapsed = time.time() - t1
    print(f"Generated {len(images)} image(s) in {elapsed:.1f}s ({elapsed/len(images):.1f}s each)", file=sys.stderr)

    # Save
    if len(images) == 1:
        images[0].save(args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        base, ext = os.path.splitext(args.output)
        for i, img in enumerate(images):
            path = f"{base}_{i:02d}{ext}"
            img.save(path)
            print(f"Saved to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()