#!/home/yuanqi.xhf/miniconda3/bin/python
"""
paint.py — Text-to-image generation via the ideogram4 MaaS service.

Plain-text prompts are automatically expanded into Ideogram 4's structured
JSON caption format using qwen3-a ("magic prompt", driven by Ideogram's
official open-source magic-prompt system prompt). Prompts that are already
valid JSON are passed through unchanged.

Usage:
    ./paint.py "a cat sitting on a cloud" -o cat.png
    echo "cyberpunk city at night" | ./paint.py -o city.png
    ./paint.py - --width 1536 --height 864 < prompt.txt
    ./paint.py --no-magic '{"high_level_description": "...", ...}'

Environment:
    IDEOGRAM_API   ideogram4 service URL   (default: http://localhost:9114)
    QWEN3_API      qwen3-a service URL     (default: http://localhost:9113)
    QWEN3_MODEL    qwen3-a model name      (default: qwen3.8-a)
"""

import argparse
import base64
import io
import json
import math
import os
import sys
import urllib.request

IDEOGRAM_API = os.environ.get("IDEOGRAM_API", "http://localhost:9114")
QWEN3_API = os.environ.get("QWEN3_API", "http://localhost:9113")
QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "qwen3.8-a")
MAGIC_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ideogram4_magic_prompt_v1.txt"
)

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
    return json.loads(text[start:end + 1])


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
    try:
        result = _http_post_json(
            f"{QWEN3_API}/v1/chat/completions",
            {
                "model": QWEN3_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7,
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with ideogram4. Plain-text prompts are "
                    "expanded to Ideogram 4 JSON captions via qwen3-a."
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Text prompt (or '-' to read from stdin; also read from stdin "
             "when omitted)",
    )
    parser.add_argument(
        "-o", "--output", default="output.png",
        help="Output image file (default: output.png)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps (default: 10)")
    parser.add_argument(
        "--cfg", type=float, default=None,
        help="Guidance scale; omit for server-side recommended schedule "
             "(1.0 = fast mode, no CFG)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("-n", "--num-images", type=int, default=1, help="Number of images (default: 1)")
    parser.add_argument(
        "--no-magic", action="store_true",
        help="Skip JSON expansion via qwen3-a (send the prompt as-is)",
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

    # Expand plain text into Ideogram 4 JSON caption
    if not args.no_magic:
        prompt = to_json_prompt(prompt, args.width, args.height)

    print(f"Generating {args.width}x{args.height}, {args.steps} steps...", file=sys.stderr)
    images = generate(
        prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
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
