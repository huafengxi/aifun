#!/home/yuanqi.xhf/miniconda3/bin/python
"""
paint.py — Text-to-image generation with the local Krea 2 diffusers pipeline.

Bare invocations run the `krea2` alias (Krea 2 Turbo FP8). `--model` also
accepts a repo id directly (any Krea 2 diffusers repo; weights are
downloaded from ModelScope (preferred) or HuggingFace on first use).

Usage:
    ./paint.py "a cat sitting on a cloud" -o cat.png
    echo "cyberpunk city at night" | ./paint.py -o city.png
    ./paint.py - --width 1536 --height 864 < prompt.txt
    ./paint.py krea2 "a fox walking in the snow" -o fox.png
    echo "a cat" | ./expand.py | ./paint.py -o cat.png   # LLM prompt expansion
    ./paint.py serve --port 8097                          # resident server
    ./paint.py --server http://127.0.0.1:8097 "a cat" -o cat.png  # thin client

Model aliases:
    krea2       Krea 2 Turbo FP8 (sakamakismile/Krea-2-Turbo-FP8) — W8A8
                quantized transformer (~12.8 GB vs ~25 GB bf16) + FP8
                Qwen3-VL text encoder (quantized in-process with modelopt
                on first use, state cached for later runs), requires
                `pip install nvidia-modelopt`
"""

import argparse
import glob
import inspect
import json
import os
import sys
import time

# Model aliases — the first positional arg can be an alias; a bare
# invocation (no alias, no --model) defaults to `krea2`.
MODEL_ALIASES = {
    "krea2": "sakamakismile/Krea-2-Turbo-FP8",
}

# Aliases that were removed — fail with a helpful message instead of
# silently treating the word as a prompt.
REMOVED_ALIASES = {
    "krea2_bf16": (
        "the bf16 Turbo alias was removed; use `krea2` (FP8, near-bf16 "
        "quality at ~half the VRAM)"
    ),
    "krea2_raw": (
        "the bf16 Raw alias was removed; use `krea2` (Turbo FP8, near-bf16 "
        "quality at ~half the VRAM)"
    ),
}

# FP8 (modelopt) repos are transformer-only; the rest of the pipeline
# (vae, tokenizer, scheduler) comes from the bf16 base. The Qwen3-VL
# text_encoder is also taken from the bf16 base but quantized to FP8
# in-process with modelopt (the FP8 repo ships no text encoder).
FP8_BASE = {
    "sakamakismile/Krea-2-Turbo-FP8": "krea/Krea-2-Turbo",
}

# Where the quantized text encoder's modelopt state is cached (next to the
# bf16 base weights it was quantized from; re-created if missing/corrupt).
KREA2_TE_FP8_STATE = "text_encoder_fp8_modelopt_state.pth"

# Diverse calibration prompts for the text encoder's FP8 amax calibration
# (same idea as the transformer repo's "small diverse prompt set").
KREA2_TE_CALIB_PROMPTS = [
    "a red fox walking through fresh snow at golden hour, photorealistic",
    "cyberpunk city street at night, neon signs reflecting in wet asphalt",
    "watercolor illustration of a cozy reading nook with warm lamplight",
    "macro photo of a honeybee on a lavender flower, shallow depth of field",
]

# Official recommended sampler settings (krea-ai/krea-2 README):
# Turbo — distilled, 8 steps, CFG disabled, mu=1.15, 1k~2k resolution.
KREA2_PRESETS = {
    "turbo": {"steps": 8, "cfg": 0.0, "width": 2048, "height": 2048, "mu": 1.15},
}


# ---------------------------------------------------------------------------
# Local model resolution (ModelScope preferred, HuggingFace fallback)
# ---------------------------------------------------------------------------

def _check_local_cache(model_id: str, cache_dir: str = None) -> str | None:
    """Check if a diffusers model exists in a local cache (ModelScope or HF)."""
    safe_id = model_id.replace("/", "--")

    # model_index.json for full pipelines; modelopt_state.pth for
    # transformer-only FP8 (modelopt) repos such as sakamakismile/Krea-2-Turbo-FP8.
    markers = ("model_index.json", "modelopt_state.pth")

    def _find_snapshot(model_dir: str) -> str | None:
        """Return the dir containing a model marker (snapshots/ layout or direct)."""
        for marker in markers:
            if os.path.isfile(os.path.join(model_dir, marker)):
                return model_dir
        snap_root = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snap_root):
            for snap in sorted(os.listdir(snap_root), reverse=True):
                snap_path = os.path.join(snap_root, snap)
                for marker in markers:
                    if os.path.isfile(os.path.join(snap_path, marker)):
                        return snap_path
        return None

    # ModelScope caches first (preferred, more reliable in China)
    ms_bases = []
    if cache_dir:
        ms_bases.append(cache_dir)
    ms_bases.extend([
        os.environ.get("MODELSCOPE_CACHE", ""),
        "/data/yuanqi.xhf/cache/modelscope",
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope"),
    ])
    for base in ms_bases:
        if not base:
            continue
        # Some caches have a hub/ subdir (which may be empty) — check both
        # hub/ and the cache root so models under <root>/models are found.
        hubs = []
        if os.path.isdir(os.path.join(base, "hub")):
            hubs.append(os.path.join(base, "hub"))
        hubs.append(base)
        # layouts: hub/models/org/name, hub/org/name, hub/models/org--name
        for hub in hubs:
            for sub in ("models", ""):
                for layout in (os.path.join(*model_id.split("/")), safe_id):
                    model_dir = os.path.join(hub, sub, layout) if sub else os.path.join(hub, layout)
                    found = _find_snapshot(model_dir)
                    if found:
                        return found

    # HuggingFace cache (models--org--name/snapshots/<rev>)
    hf_bases = []
    if cache_dir:
        hf_bases.append(cache_dir)
    hf_bases.extend([
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.environ.get("HF_HOME", ""),
        "/data/yuanqi.xhf/cache/huggingface/hub",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ])
    for base in hf_bases:
        if not base:
            continue
        hf_model_dir = os.path.join(base, f"models--{safe_id}")
        if os.path.isdir(hf_model_dir):
            snapshots = sorted(glob.glob(os.path.join(hf_model_dir, "snapshots", "*")),
                               key=os.path.getmtime, reverse=True)
            for snap in snapshots:
                for marker in markers:
                    if os.path.isfile(os.path.join(snap, marker)):
                        return snap

    return None


# HuggingFace-only repos (no ModelScope mirror) — skip the ModelScope probe.
HF_ONLY_ORGS = {"sakamakismile"}

# Redundant/non-essential files in Krea repos (single-file checkpoints,
# docs, gallery images); the diffusers-format sub-directories are enough.
KREA_IGNORE_PATTERNS = [
    "turbo.safetensors", "raw.safetensors", "*.pdf", "gallery/*", "images/*",
]


def resolve_model_path(model_id: str, cache_dir: str = None) -> str:
    """Resolve model to a local path, downloading from ModelScope or HuggingFace."""
    local_path = _check_local_cache(model_id, cache_dir)
    if local_path:
        print(f"Using cached model: {local_path}", file=sys.stderr)
        return local_path

    # ModelScope first (skip the probe for HF-only repos)
    if model_id.split("/")[0].lower() in HF_ONLY_ORGS:
        print(f"{model_id} is HuggingFace-only; skipping ModelScope",
              file=sys.stderr)
    else:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            kwargs = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            # Krea 2 repos also ship a redundant single-file checkpoint
            # (turbo.safetensors / raw.safetensors); the diffusers-format
            # sub-directories are enough — skip them to save ~26 GB.
            if "krea" in model_id.lower():
                kwargs["ignore_file_pattern"] = KREA_IGNORE_PATTERNS
            print(f"Downloading {model_id} from ModelScope...", file=sys.stderr)
            try:
                return snapshot_download(model_id, **kwargs)
            except TypeError:
                # Older modelscope without ignore_file_pattern
                kwargs.pop("ignore_file_pattern", None)
                return snapshot_download(model_id, **kwargs)
        except ImportError:
            print("modelscope not installed; trying HuggingFace...", file=sys.stderr)
        except Exception as e:
            print(f"ModelScope download failed ({e}); trying HuggingFace...", file=sys.stderr)

    # HuggingFace fallback (use HF_ENDPOINT=https://hf-cdn.sufy.com if blocked)
    try:
        # Keep downloads on the big data disk when the default cache is small.
        os.environ.setdefault("HF_HOME", "/data/yuanqi.xhf/cache/huggingface")
        from huggingface_hub import snapshot_download as hf_snapshot_download
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if "krea" in model_id.lower():
            kwargs["ignore_patterns"] = KREA_IGNORE_PATTERNS
        print(f"Downloading {model_id} from HuggingFace...", file=sys.stderr)
        return hf_snapshot_download(model_id, **kwargs)
    except ImportError:
        print("huggingface_hub not installed; trying download_model...",
              file=sys.stderr)
    except Exception as e:
        print(f"HuggingFace download failed ({e}); trying download_model...",
              file=sys.stderr)

    try:
        from download_model import download_from_huggingface
        return download_from_huggingface(model_id, cache_dir)
    except Exception as e:
        print(f"Error: could not download {model_id}: {e}", file=sys.stderr)
        sys.exit(1)


def _detect_pipeline(model_path: str) -> str:
    """Detect pipeline class from model_index.json."""
    config_path = os.path.join(model_path, "model_index.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("_class_name", "")
        except Exception:
            pass
    return ""


def _pick_cuda_device() -> str:
    """Pick the CUDA device with the most free memory."""
    import torch
    if torch.cuda.device_count() == 0:
        print("Error: no CUDA device available", file=sys.stderr)
        sys.exit(1)
    best, best_free = "cuda:0", -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = f"cuda:{i}", free
    print(f"Using device {best} ({best_free / 2**30:.1f} GiB free)", file=sys.stderr)
    return best


def _load_krea2_fp8_transformer(base_path: str, fp8_path: str):
    """Load the FP8 (W8A8) Krea 2 transformer per the
    sakamakismile/Krea-2-Turbo-FP8 model card: instantiate the base
    transformer structure, then restore the NVIDIA TensorRT Model Optimizer
    (modelopt) quantization state on top of it."""
    import torch
    from diffusers import Krea2Transformer2DModel

    try:
        import modelopt.torch.opt as mto
    except ImportError:
        print(
            "Error: FP8 Krea 2 weights (sakamakismile/Krea-2-Turbo-FP8) "
            "require nvidia-modelopt:\n"
            "    pip install nvidia-modelopt",
            file=sys.stderr,
        )
        sys.exit(1)

    state = os.path.join(fp8_path, "modelopt_state.pth")
    if not os.path.isfile(state):
        print(f"Error: {state} not found", file=sys.stderr)
        sys.exit(1)

    print(
        "Building transformer from bf16 base, then restoring FP8 modelopt state...",
        file=sys.stderr,
    )
    t0 = time.time()
    transformer = Krea2Transformer2DModel.from_pretrained(
        base_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    mto.restore(transformer, state)
    print(f"FP8 transformer ready in {time.time() - t0:.1f}s", file=sys.stderr)
    return transformer


def _drop_unused_vision_tower(te):
    """Krea 2 conditions on text only (no pixel_values is ever passed), so
    the Qwen3-VL vision tower (~0.9 GiB bf16) is dead weight — drop it.
    Must run after modelopt save/restore (the state covers the full model)."""
    if getattr(te, "visual", None) is not None:
        del te.visual


def _load_krea2_fp8_text_encoder(base_path: str, device: str):
    """Load the Qwen3-VL text encoder in FP8.

    The FP8 repo (sakamakismile/Krea-2-Turbo-FP8) is transformer-only, and
    the community full-FP8 packs (AlperKTS/Krea2_FP8, szwagros) ship the
    text encoder in ComfyUI single-file format, not diffusers-compatible.
    So we quantize the bf16 base text encoder in-process with modelopt —
    the same toolchain as the transformer (FP8_DEFAULT_CFG quantizes only
    nn.Linear weights+activations; embeddings/norms/vision stay bf16).

    First run calibrates amax on a few prompts, compresses the weights to
    FP8 (mtq.quantize + mtq.compress, the transformer repo's exact recipe)
    and caches the modelopt state next to the base weights; later runs
    restore from that cache.
    """
    import torch
    from transformers import Qwen3VLModel

    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except ImportError:
        print(
            "Error: FP8 Krea 2 (sakamakismile/Krea-2-Turbo-FP8) requires "
            "nvidia-modelopt for the transformer and text encoder:\n"
            "    pip install nvidia-modelopt",
            file=sys.stderr,
        )
        sys.exit(1)

    t0 = time.time()
    te = Qwen3VLModel.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )

    state = os.path.join(base_path, KREA2_TE_FP8_STATE)
    if os.path.isfile(state):
        try:
            mto.restore(te, state)
            _drop_unused_vision_tower(te)
            print(
                f"FP8 text encoder ready in {time.time() - t0:.1f}s "
                f"(restored cached modelopt state: {state})",
                file=sys.stderr,
            )
            return te
        except Exception as e:
            print(
                f"Warning: restoring cached FP8 text encoder state failed "
                f"({e}); re-quantizing", file=sys.stderr,
            )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    te = te.to(device)

    def forward_loop(model):
        for prompt in KREA2_TE_CALIB_PROMPTS:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128
            ).to(device)
            with torch.inference_mode():
                model(**inputs)

    print(
        "FP8 text encoder state not cached; quantizing with modelopt "
        "(one-time calibration)...", file=sys.stderr,
    )
    mtq.quantize(te, mtq.FP8_DEFAULT_CFG, forward_loop)
    # compress = fold quantization into the weights (fp8 storage), same as
    # the transformer repo's recipe (mtq.quantize + mtq.compress); without
    # it the weights stay bf16 and nothing is saved in VRAM.
    mtq.compress(te)
    torch.cuda.empty_cache()
    try:
        mto.save(te, state)
        print(f"Cached FP8 text encoder state at {state}", file=sys.stderr)
    except OSError as e:
        print(f"Warning: could not cache FP8 text encoder state ({e}); "
              "will re-quantize next run", file=sys.stderr)
    te = te.to("cpu")
    _drop_unused_vision_tower(te)
    print(f"FP8 text encoder ready in {time.time() - t0:.1f}s", file=sys.stderr)
    return te


def generate_krea2(
    model_id: str,
    prompt: str,
    width: int = None,
    height: int = None,
    steps: int = None,
    cfg: float = None,
    seed: int = None,
    n: int = 1,
    dual_gpu: bool = False,
) -> list:
    """Generate images locally with the Krea 2 diffusers pipeline.

    Returns a list of PIL Images. Defaults follow the official krea-ai/krea-2
    README recommendations (Turbo: 8 steps, cfg 0.0, 2048x2048).
    """
    import torch
    ctx = load_krea2_pipeline(model_id, dual_gpu=dual_gpu)
    preset = ctx["preset"]
    width = width or preset["width"]
    height = height or preset["height"]
    steps = steps or preset["steps"]
    if cfg is None:
        cfg = preset["cfg"]

    images = []
    i = 0
    while i < n:
        try:
            out = _render(ctx, prompt, width, height, steps, cfg,
                          seed + i if seed is not None else None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if width > 512 or height > 512:
                # Attention activations grow with seq_len^2 — halve the canvas.
                # (Deliberate policy: no CPU-offload fallback, downscale only.)
                width = max(512, (width // 2) // 64 * 64)
                height = max(512, (height // 2) // 64 * 64)
                print(f"OOM during generation; retrying at {width}x{height} "
                      "(pass a smaller --width/--height to skip this)",
                      file=sys.stderr)
                continue  # redo this image
            raise
        images.extend(out)
        i += 1
    return images


def load_krea2_pipeline(model_id: str, dual_gpu: bool = False) -> dict:
    """Load the Krea 2 pipeline once and return a render context (used by
    both the one-shot CLI path and `paint.py serve`)."""
    import torch  # noqa: F401  (also ensures CUDA init happens here)
    _prepare_krea2_env()

    try:
        from diffusers import Krea2Pipeline
    except ImportError as e:
        print(
            "Error: Krea2Pipeline is not available in the installed diffusers.\n"
            "Krea 2 requires diffusers >= 0.40.0:\n"
            "    pip install -U diffusers\n"
            "or install diffusers from source:\n"
            "    pip install git+https://github.com/huggingface/diffusers.git\n"
            f"(import error: {e})",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = resolve_model_path(model_id)

    # Transformer-only FP8 repo → rebuild on top of the bf16 base pipeline.
    fp8 = os.path.isfile(os.path.join(model_path, "modelopt_state.pth"))
    base_id = FP8_BASE.get(model_id)
    if fp8 and not base_id:
        print(
            f"Error: {model_id} looks like an FP8 (modelopt) transformer repo "
            f"but has no known bf16 base; add it to FP8_BASE",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_cls = _detect_pipeline(model_path)
    model_lower = model_id.lower()
    if not (pipeline_cls == "Krea2Pipeline" or "krea" in model_lower):
        print(
            f"Error: {model_id} is not a Krea 2 model "
            f"(detected pipeline: {pipeline_cls or 'unknown'})",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    dual = dual_gpu and torch.cuda.device_count() >= 2
    if dual:
        # Experimental component split (see runs/2026-08-21-17-12-57-24ge):
        # transformer on the freest GPU, text encoder + VAE on the other.
        # Adds headroom on shared GPUs; no denoise speedup, no 2048² fix.
        frees = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        order = sorted(range(len(frees)), key=lambda i: -frees[i])
        dev0, dev1 = f"cuda:{order[0]}", f"cuda:{order[1]}"
        device = dev0
        print(f"dual-GPU experimental mode: transformer@{dev0}, "
              f"text_encoder+vae@{dev1}", file=sys.stderr)
    else:
        device = _pick_cuda_device()
    print(f"Loading {model_id} from {model_path}...", file=sys.stderr)
    t0 = time.time()
    if fp8:
        base_path = resolve_model_path(base_id)
        transformer = _load_krea2_fp8_transformer(base_path, model_path)
        text_encoder = _load_krea2_fp8_text_encoder(base_path, device)
        pipe = Krea2Pipeline.from_pretrained(
            base_path, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16
        )
    else:
        pipe = Krea2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if dual:
        pipe.transformer.to(dev0)
        pipe.text_encoder.to(dev1)
        pipe.vae.to(dev1)
        _vae_dev = torch.device(dev1)
        _orig_decode = pipe.vae.decode
        pipe.vae.decode = lambda z, **kw: _orig_decode(z.to(_vae_dev), **kw)
        # pipeline derives _execution_device from its first component (vae)
        type(pipe)._execution_device = property(
            lambda self: torch.device(dev0))
    else:
        dev0 = dev1 = None
        try:
            pipe.to(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                "Error: not enough VRAM to hold the pipeline on one GPU "
                "(no CPU-offload fallback by design). Free VRAM on this "
                "GPU or try --dual-gpu.",
                file=sys.stderr,
            )
            sys.exit(1)
    print(f"Model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Optional timestep-shift mu (official README recommends 1.15 for Turbo);
    # only pass it if this diffusers version supports it.
    preset = KREA2_PRESETS["turbo"]
    extra = {}
    if "mu" in preset:
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        if "mu" in params:
            extra["mu"] = preset["mu"]

    return {"model_id": model_id, "pipe": pipe, "device": device,
            "dual": dual, "dev0": dev0, "dev1": dev1,
            "preset": preset, "extra": extra}


def _prepare_krea2_env():
    """Environment tweaks that must be set before diffusers/torch use."""
    # Reduce fragmentation on shared/partially-free GPUs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Krea 2's joint attention is GQA + attention-mask, which makes SDPA's
    # default backend selection fall back to the MATH kernel that
    # materializes the seq² score matrix (~3.8 GiB at 1024²) — instant OOM
    # on partially-free GPUs. The cuDNN kernel handles GQA+mask with
    # near-zero extra VRAM (same numerics), so prefer it. diffusers reads
    # DIFFUSERS_ATTN_BACKEND at import time → set it before importing;
    # respect an explicit user override.
    if "DIFFUSERS_ATTN_BACKEND" not in os.environ:
        try:
            from torch.nn.attention import SDPBackend
            if hasattr(SDPBackend, "CUDNN_ATTENTION"):
                os.environ["DIFFUSERS_ATTN_BACKEND"] = "_native_cudnn"
        except ImportError:
            pass


def _render(ctx: dict, prompt: str, width: int, height: int,
            steps: int, cfg: float, seed: int = None) -> list:
    """One generation call against a loaded pipeline context (no OOM retry)."""
    import torch
    pipe, device, extra = ctx["pipe"], ctx["device"], ctx["extra"]
    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    print(f"Generating {width}x{height}, {steps} steps, cfg {cfg}...", file=sys.stderr)
    t0 = time.time()
    call_args, call_kwargs = (prompt,), {}
    if ctx["dual"]:
        # encode on the text encoder's GPU, denoise on the transformer's
        emb_dev = torch.device(ctx["dev1"])
        pe, pem = pipe.encode_prompt(prompt=prompt, device=emb_dev,
                                     num_images_per_prompt=1)
        call_kwargs["prompt_embeds"] = pe.to(ctx["dev0"])
        call_kwargs["prompt_embeds_mask"] = pem.to(ctx["dev0"])
        if cfg > 1:
            npe, npem = pipe.encode_prompt(prompt="", device=emb_dev,
                                           num_images_per_prompt=1)
            call_kwargs["negative_prompt_embeds"] = npe.to(ctx["dev0"])
            call_kwargs["negative_prompt_embeds_mask"] = npem.to(ctx["dev0"])
        call_args = ()
    out = pipe(
        *call_args,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        **extra,
        **call_kwargs,
    ).images
    print(f"Done in {time.time() - t0:.1f}s", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# serve mode — hold the pipeline resident; thin HTTP client via --server
# ---------------------------------------------------------------------------

DEFAULT_SERVE_PORT = 8097


def serve_krea2(host: str, port: int, idle_exit: float = 0,
                dual_gpu: bool = False):
    """Load the pipeline once and serve POST /generate (PNG response).

    Single-file, stdlib-only. Requests are serialized (the pipeline is not
    thread-safe). GET /health reports readiness.
    """
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    ctx = load_krea2_pipeline(MODEL_ALIASES["krea2"], dual_gpu=dual_gpu)
    import torch
    state = {"last": time.time(), "ready": True}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logs
            print("[serve] " + fmt % args, file=sys.stderr)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, msg: str):
            self._send(code, json.dumps({"error": msg}).encode(),
                       "application/json")

        def do_GET(self):
            if self.path == "/health":
                self._send(200, json.dumps({
                    "status": "ok", "model": ctx["model_id"],
                }).encode(), "application/json")
            else:
                self._error(404, "not found (try GET /health or POST /generate)")

        def do_POST(self):
            if self.path != "/generate":
                return self._error(404, "POST /generate only")
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._error(400, "invalid JSON body")
            prompt = (req.get("prompt") or "").strip()
            if not prompt:
                return self._error(400, "missing 'prompt'")
            if req.get("model") and req["model"] != ctx["model_id"]:
                return self._error(
                    400, f"server holds {ctx['model_id']}, not {req['model']}")
            preset = ctx["preset"]
            width = int(req.get("width") or preset["width"])
            height = int(req.get("height") or preset["height"])
            steps = int(req.get("steps") or preset["steps"])
            cfg = float(req["cfg"]) if req.get("cfg") is not None else preset["cfg"]
            seed = req.get("seed")
            seed = int(seed) if seed is not None else None
            if int(req.get("n") or 1) != 1:
                return self._error(400, "serve mode supports n=1 per request")
            state["last"] = time.time()
            try:
                images = _render(ctx, prompt, width, height, steps, cfg, seed)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return self._error(507, f"OOM at {width}x{height}; retry smaller")
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            self._send(200, buf.getvalue(), "image/png")

    server = HTTPServer((host, port), Handler)
    print(f"[serve] ready on http://{host}:{port} "
          f"(POST /generate, GET /health)", file=sys.stderr)

    if idle_exit > 0:
        def _watchdog():
            while True:
                time.sleep(5)
                if time.time() - state["last"] > idle_exit:
                    print(f"[serve] idle for {idle_exit:.0f}s; shutting down",
                          file=sys.stderr)
                    server.shutdown()
                    return
        threading.Thread(target=_watchdog, daemon=True).start()

    try:
        server.serve_forever(poll_interval=1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def client_generate(server_url: str, prompt: str, output: str,
                    width: int = None, height: int = None, steps: int = None,
                    cfg: float = None, seed: int = None, model: str = None):
    """Thin client for `paint.py serve`: POST the request, save the PNG."""
    import urllib.error
    import urllib.request

    payload = {"prompt": prompt}
    for key, val in (("width", width), ("height", height), ("steps", steps),
                     ("cfg", cfg), ("seed", seed), ("model", model)):
        if val is not None:
            payload[key] = val
    req = urllib.request.Request(
        server_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        print(f"Error: server rejected the request: {msg}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: cannot reach {server_url} ({e.reason}); start it "
              f"with `paint.py serve` or drop --server for local mode",
              file=sys.stderr)
        sys.exit(1)
    with open(output, "wb") as f:
        f.write(body)
    print(f"Saved to {output} (server round-trip {time.time() - t0:.1f}s)",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # `paint.py serve` — resident pipeline + HTTP endpoint (own argparse).
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sp = argparse.ArgumentParser(
            description="Hold the krea2 pipeline resident and serve "
                        "POST /generate over HTTP (thin client: paint.py "
                        "--server URL). Ctrl-C to stop.",
        )
        sp.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
        sp.add_argument("--port", type=int, default=DEFAULT_SERVE_PORT,
                        help=f"Bind port (default: {DEFAULT_SERVE_PORT})")
        sp.add_argument("--idle-exit", type=float, default=0, metavar="SEC",
                        help="Exit after SEC seconds without requests "
                             "(default: 0 = run until Ctrl-C)")
        sp.add_argument("--dual-gpu", action="store_true",
                        help="Experimental component split across two GPUs "
                             "(see --dual-gpu in the main CLI). Default: off.")
        sargs = sp.parse_args(sys.argv[2:])
        serve_krea2(sargs.host, sargs.port, idle_exit=sargs.idle_exit,
                    dual_gpu=sargs.dual_gpu)
        return

    # First positional arg can be a model alias.
    if len(sys.argv) > 1 and sys.argv[1] in REMOVED_ALIASES:
        print(
            f"Error: alias '{sys.argv[1]}' was removed — "
            f"{REMOVED_ALIASES[sys.argv[1]]}",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] in MODEL_ALIASES:
        sys.argv[1:2] = ["--model", sys.argv[1]]

    parser = argparse.ArgumentParser(
        description="Generate images with the local Krea 2 diffusers "
                    "pipeline (default: krea2 Turbo FP8). "
                    "Prompt expansion lives in expand.py."
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Text prompt (or '-' to read from stdin; also read from stdin "
             "when omitted)",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Model to run: an alias ({', '.join(MODEL_ALIASES)}) or a "
             f"Krea 2 repo id directly (default: krea2)",
    )
    parser.add_argument(
        "-o", "--output", default="output.png",
        help="Output image file (default: output.png)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Image width (krea2 default: 2048)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Image height (krea2 default: 2048)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Inference steps (krea2 default: 8)",
    )
    parser.add_argument(
        "--cfg", type=float, default=None,
        help="Guidance scale (krea2 default: 0.0 — distilled, no CFG)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--server", default=os.environ.get("PAINT_SERVER"), metavar="URL",
        help=f"Use a running `paint.py serve` instance (e.g. "
             f"http://127.0.0.1:{DEFAULT_SERVE_PORT}) instead of loading the "
             f"model locally; also read from $PAINT_SERVER. Default: local "
             f"one-shot mode (unchanged).",
    )
    parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images (default: 1)")
    parser.add_argument(
        "--dual-gpu", action="store_true",
        help="Experimental (krea2 only): split components across two GPUs "
             "(transformer on the freest GPU, text encoder + VAE on the "
             "other). Adds VRAM headroom on shared GPUs; no denoise "
             "speedup and does not enable 2048² (see "
             "runs/2026-08-21-17-12-57-24ge report). Default: off.",
    )

    args = parser.parse_args()

    # Prompt: positional arg, '-' sentinel, or stdin when omitted
    prompt = args.prompt
    if prompt is None or prompt == "-":
        prompt = sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    # Bare invocation → krea2 alias (Turbo FP8).
    if not args.model:
        args.model = MODEL_ALIASES["krea2"]

    # Thin client mode: forward to a running `paint.py serve`.
    if args.server:
        if args.num_images != 1:
            print("Error: --server mode supports -n 1 per request",
                  file=sys.stderr)
            sys.exit(1)
        if args.model in MODEL_ALIASES:
            args.model = MODEL_ALIASES[args.model]
        client_generate(
            args.server, prompt, args.output,
            width=args.width, height=args.height, steps=args.steps,
            cfg=args.cfg, seed=args.seed,
            model=args.model if args.model != MODEL_ALIASES["krea2"] else None,
        )
        return


    if args.model in REMOVED_ALIASES:
        print(
            f"Error: alias '{args.model}' was removed — "
            f"{REMOVED_ALIASES[args.model]}",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.model in MODEL_ALIASES:
        args.model = MODEL_ALIASES[args.model]

    if "krea" in args.model.lower():
        images = generate_krea2(
            args.model, prompt,
            width=args.width, height=args.height,
            steps=args.steps, cfg=args.cfg,
            seed=args.seed, n=args.num_images,
            dual_gpu=args.dual_gpu,
        )
    else:
        print(
            f"Error: unsupported model: {args.model} "
            f"(supported aliases: {', '.join(MODEL_ALIASES)}; or pass a "
            f"Krea 2 repo id)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not images:
        print("Error: no images returned", file=sys.stderr)
        sys.exit(1)

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
