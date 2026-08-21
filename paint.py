#!/home/yuanqi.xhf/miniconda3/bin/python
"""
paint.py — Text-to-image generation.

Two backends:

1. ideogram4 MaaS (default): plain-text prompts are automatically expanded
   into Ideogram 4's structured JSON caption format using qwen3-a ("magic
   prompt", driven by Ideogram's official open-source magic-prompt system
   prompt). Prompts that are already valid JSON are passed through unchanged.

2. Local diffusers pipelines (Krea 2): select with a model alias as the first
   positional arg (or --model). Weights are downloaded from ModelScope
   (preferred) or HuggingFace on first use.

Usage:
    ./paint.py "a cat sitting on a cloud" -o cat.png
    echo "cyberpunk city at night" | ./paint.py -o city.png
    ./paint.py - --width 1536 --height 864 < prompt.txt
    ./paint.py --no-magic '{"high_level_description": "...", ...}'
    ./paint.py krea2 "a fox walking in the snow" -o fox.png
    ./paint.py krea2_raw "a fox" --steps 52 --cfg 3.5 -o fox.png

Model aliases (local pipelines):
    krea2       Krea 2 Turbo FP8 (sakamakismile/Krea-2-Turbo-FP8) — W8A8
                quantized transformer (~12.8 GB vs ~25 GB bf16) + FP8
                Qwen3-VL text encoder (quantized in-process with modelopt
                on first use, state cached for later runs), requires
                `pip install nvidia-modelopt`
    krea2_raw   Krea 2 Raw bf16   (krea/Krea-2-Raw)   — base model, full sampler

Environment:
    IDEOGRAM_API   ideogram4 service URL   (default: http://localhost:9114)
    QWEN3_API      qwen3-a service URL     (default: http://localhost:9113)
    QWEN3_MODEL    qwen3-a model name      (default: qwen3.8-a)
"""

import argparse
import base64
import glob
import inspect
import io
import json
import math
import os
import sys
import time
import urllib.request

IDEOGRAM_API = os.environ.get("IDEOGRAM_API", "http://localhost:9114")
QWEN3_API = os.environ.get("QWEN3_API", "http://localhost:9113")
QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "qwen3.8-a")
MAGIC_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ideogram4_magic_prompt_v1.txt"
)

# Model aliases — first positional arg can be an alias to select a local
# diffusers model instead of the ideogram4 MaaS backend.
MODEL_ALIASES = {
    "krea2": "sakamakismile/Krea-2-Turbo-FP8",
    "krea2_raw": "krea/Krea-2-Raw",
}

# Aliases that were removed — fail with a helpful message instead of
# silently treating the word as a prompt (which would hit ideogram4).
REMOVED_ALIASES = {
    "krea2_bf16": (
        "the bf16 Turbo alias was removed; use `krea2` (FP8, near-bf16 "
        "quality at ~half the VRAM) or `krea2_raw` (bf16 base model)"
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
#   Turbo — distilled, 8 steps, CFG disabled, mu=1.15, 1k~2k resolution
#   Raw   — base model, full sampler with CFG, up to 1k resolution
KREA2_PRESETS = {
    "turbo": {"steps": 8, "cfg": 0.0, "width": 2048, "height": 2048, "mu": 1.15},
    "raw": {"steps": 52, "cfg": 3.5, "width": 1024, "height": 1024},
}

# Appended to the user message: turn the LLM from a formatter into an expander.
EXPANSION_POLICY = """
EXPANSION POLICY: The user idea may be a brief sketch — expand it into a richly
detailed, complete scene. Fill in medium, style, aesthetics, lighting, background
detail, plausible secondary elements and in-scene text where appropriate, while
staying faithful to everything the user explicitly named (never drop or alter
named subjects, text, colors, or constraints). If the user includes meta
instructions about how to expand (e.g. desired style, mood, richness, additions,
"电影感", "细节丰富一点"), treat them as authoritative and follow them first.
"""


def _http_post_json(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _load_magic_prompt() -> tuple[str, str]:
    """Parse the official Ideogram 4 magic-prompt file into (system, user_template)."""
    with open(MAGIC_PROMPT_FILE, "r") as f:
        raw = f.read()
    # File layout: [META] ... [SYSTEM] <system prompt> [USER] <user template>
    sys_start = raw.index("[SYSTEM]") + len("[SYSTEM]")
    usr_start = raw.index("[USER]", sys_start)
    system = raw[sys_start:usr_start].strip()
    user_template = raw[usr_start + len("[USER]"):].strip()
    return system, user_template


def _aspect_ratio(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def _is_json_prompt(prompt: str) -> bool:
    stripped = prompt.strip()
    if not stripped.startswith("{"):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except json.JSONDecodeError:
        return False


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json / ``` fences
        lines = text.splitlines()
        lines = lines[1:]  # opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in LLM output: {text[:120]!r}")
    return json.loads(_sanitize_json_text(text[start:end + 1]))


def _sanitize_json_text(text: str) -> str:
    """Escape raw control characters (newlines/tabs) inside JSON string literals."""
    out = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
            elif ch == "\\":
                esc = True
                out.append(ch)
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\r":
                out.append("\\r")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _fallback_wrap(prompt: str) -> str:
    """Minimal structured caption when the magic-prompt LLM is unavailable."""
    return json.dumps(
        {
            "high_level_description": prompt,
            "compositional_deconstruction": {
                "background": "As described by the high level description.",
                "elements": [{"type": "obj", "desc": prompt}],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def to_json_prompt(prompt: str, width: int, height: int) -> str:
    """Expand a plain-text prompt into an Ideogram 4 JSON caption via qwen3-a.

    Passthrough if the prompt is already valid JSON. Falls back to a minimal
    wrapper if the LLM call fails, so generation is never blocked.
    """
    if _is_json_prompt(prompt):
        return prompt.strip()

    try:
        system, user_template = _load_magic_prompt()
    except Exception as e:
        print(f"magic-prompt file unavailable ({e}), using fallback wrapper", file=sys.stderr)
        return _fallback_wrap(prompt)

    user = user_template.replace("{{aspect_ratio}}", _aspect_ratio(width, height))
    user = user.replace("{{original_prompt}}", prompt)
    user = user + "\n" + EXPANSION_POLICY

    print(f"Expanding prompt to JSON via {QWEN3_MODEL}...", file=sys.stderr)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    content = ""
    for attempt in range(2):
        try:
            result = _http_post_json(
                f"{QWEN3_API}/v1/chat/completions",
                {
                    "model": QWEN3_MODEL,
                    "messages": messages,
                    "temperature": 0.7 if attempt == 0 else 0.2,
                    "max_tokens": 16384,
                    # Disable thinking: reasoning can exhaust max_tokens before the
                    # caption is emitted (content=None, finish_reason=length).
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180,
            )
            content = result["choices"][0]["message"]["content"]
            if not content:
                raise ValueError(f"empty content (finish_reason={result['choices'][0].get('finish_reason')})")
            caption = _extract_json(content)
            expanded = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
            print(f"JSON caption ready ({len(expanded)} chars):", file=sys.stderr)
            print(json.dumps(caption, ensure_ascii=False, indent=2), file=sys.stderr)
            return expanded
        except Exception as e:
            if attempt == 0:
                print(f"caption JSON invalid ({e}); asking {QWEN3_MODEL} to repair...", file=sys.stderr)
                messages = [
                    {"role": "system", "content": "You output exactly one valid minified JSON object. No commentary, no markdown fences."},
                    {"role": "user", "content": (
                        f"The following JSON is invalid ({e}). Fix it and output ONLY the "
                        f"corrected minified JSON, keeping all content unchanged:\n\n{content}"
                    )},
                ]
            else:
                print(f"magic-prompt expansion failed ({e}), using fallback wrapper", file=sys.stderr)
                return _fallback_wrap(prompt)


def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    guidance_scale: float = None,
    seed: int = None,
    n: int = 1,
) -> list:
    """Generate images via the ideogram4 HTTP API. Returns list of PIL Images."""
    from PIL import Image

    payload = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "n": n,
    }
    # Only send guidance_scale when explicitly set; otherwise the server
    # applies the model's recommended per-step guidance schedule.
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if seed is not None:
        payload["seed"] = seed

    try:
        result = _http_post_json(f"{IDEOGRAM_API}/v1/images/generations", payload, timeout=600)
    except Exception as e:
        print(f"Error calling ideogram4 API: {e}", file=sys.stderr)
        sys.exit(1)

    if "error" in result:
        print(f"ideogram4 API error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    images = []
    for item in result.get("data", []):
        b64 = item.get("b64_json", "")
        if b64:
            images.append(Image.open(io.BytesIO(base64.b64decode(b64))))
    return images


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
            "    pip install nvidia-modelopt\n"
            "Or use the bf16 alias instead: krea2_raw",
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
            "    pip install nvidia-modelopt\n"
            "Or use the bf16 alias instead: krea2_raw",
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
    README recommendations (Turbo: 8 steps, cfg 0.0, 2048x2048;
    Raw: 52 steps, cfg 3.5, 1024x1024).
    """
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
    import torch

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

    preset = KREA2_PRESETS["turbo"] if "turbo" in model_lower else KREA2_PRESETS["raw"]
    width = width or preset["width"]
    height = height or preset["height"]
    steps = steps or preset["steps"]
    if cfg is None:
        cfg = preset["cfg"]

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
    extra = {}
    if "mu" in preset:
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        if "mu" in params:
            extra["mu"] = preset["mu"]

    images = []
    i = 0
    while i < n:
        generator = None
        if seed is not None:
            generator = torch.Generator(device).manual_seed(seed + i)
        print(f"Generating {width}x{height}, {steps} steps, cfg {cfg}...", file=sys.stderr)
        t0 = time.time()
        call_args, call_kwargs = (prompt,), {}
        if dual:
            # encode on the text encoder's GPU, denoise on the transformer's
            emb_dev = torch.device(dev1)
            pe, pem = pipe.encode_prompt(prompt=prompt, device=emb_dev,
                                         num_images_per_prompt=1)
            call_kwargs["prompt_embeds"] = pe.to(dev0)
            call_kwargs["prompt_embeds_mask"] = pem.to(dev0)
            if cfg > 1:
                npe, npem = pipe.encode_prompt(prompt="", device=emb_dev,
                                               num_images_per_prompt=1)
                call_kwargs["negative_prompt_embeds"] = npe.to(dev0)
                call_kwargs["negative_prompt_embeds_mask"] = npem.to(dev0)
            call_args = ()
        try:
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
        print(f"Done in {time.time() - t0:.1f}s", file=sys.stderr)
        i += 1
    return images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # First positional arg can be a model alias (local pipeline mode)
    _used_alias = None
    if len(sys.argv) > 1 and sys.argv[1] in REMOVED_ALIASES:
        print(
            f"Error: alias '{sys.argv[1]}' was removed — "
            f"{REMOVED_ALIASES[sys.argv[1]]}",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] in MODEL_ALIASES:
        _used_alias = sys.argv[1]
        sys.argv[1:2] = ["--model", MODEL_ALIASES[_used_alias]]

    parser = argparse.ArgumentParser(
        description="Generate images with ideogram4 (default) or a local "
                    "diffusers model (krea2 FP8, krea2_raw). "
                    "Plain-text prompts are expanded to Ideogram 4 JSON "
                    "captions via qwen3-a (ideogram4 mode only)."
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Text prompt (or '-' to read from stdin; also read from stdin "
             "when omitted)",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Local model to run instead of ideogram4 "
             f"(aliases: {', '.join(MODEL_ALIASES)})",
    )
    parser.add_argument(
        "-o", "--output", default="output.png",
        help="Output image file (default: output.png)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Image width (ideogram4 default: 1024; krea2: 2048 turbo / 1024 raw)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Image height (ideogram4 default: 1024; krea2: 2048 turbo / 1024 raw)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Inference steps (ideogram4 default: 10; krea2: 8 turbo / 52 raw)",
    )
    parser.add_argument(
        "--cfg", type=float, default=None,
        help="Guidance scale; ideogram4: omit for server-side recommended "
             "schedule (1.0 = fast mode, no CFG); krea2: default 0.0 turbo "
             "(distilled, no CFG) / 3.5 raw",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images (default: 1)")
    parser.add_argument(
        "--no-magic", action="store_true",
        help="Skip JSON expansion via qwen3-a (send the prompt as-is; "
             "implicit for local models)",
    )
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

    # Local diffusers pipeline mode (Krea 2)
    if args.model:
        model_lower = args.model.lower()
        if args.model in REMOVED_ALIASES:
            print(
                f"Error: alias '{args.model}' was removed — "
                f"{REMOVED_ALIASES[args.model]}",
                file=sys.stderr,
            )
            sys.exit(2)
        if "krea" in model_lower:
            images = generate_krea2(
                args.model, prompt,
                width=args.width, height=args.height,
                steps=args.steps, cfg=args.cfg,
                seed=args.seed, n=args.num_images,
                dual_gpu=args.dual_gpu,
            )
        else:
            print(
                f"Error: unsupported local model: {args.model} "
                f"(supported aliases: {', '.join(MODEL_ALIASES)})",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # ideogram4 MaaS mode
        width = args.width or 1024
        height = args.height or 1024
        steps = args.steps or 10
        if not args.no_magic:
            prompt = to_json_prompt(prompt, width, height)
        print(f"Generating {width}x{height}, {steps} steps...", file=sys.stderr)
        images = generate(
            prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=args.cfg,
            seed=args.seed,
            n=args.num_images,
        )

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
