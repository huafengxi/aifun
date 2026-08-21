#!/home/yuanqi.xhf/miniconda3/bin/python
"""
expand.py — Plain-text prompt expansion for paint.py via qwen3-a.

Turns a brief prompt sketch into a richly detailed natural-language scene
description (no structured JSON — Krea 2 takes plain text). Designed as a
pipeline filter: reads the prompt from argv or stdin, writes the expanded
text to stdout, all diagnostics to stderr.

Usage:
    ./expand.py "a cat"
    echo "cyberpunk city at night" | ./expand.py
    echo "a cat" | ./expand.py | ./paint.py -o cat.png

If the qwen3-a service is unreachable or fails, the original prompt is
passed through unchanged (warning on stderr) so image generation is never
blocked.

Environment:
    QWEN3_API      qwen3-a service URL   (default: http://localhost:9113)
    QWEN3_MODEL    qwen3-a model name    (default: qwen3.8-a)
"""

import json
import os
import sys
import urllib.request

QWEN3_API = os.environ.get("QWEN3_API", "http://localhost:9113")
QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "qwen3.8-a")

SYSTEM_PROMPT = """
You are a text-to-image prompt expander. The user gives a brief image idea;
you output ONE richly detailed, complete natural-language scene description
(plain text, no JSON, no markdown, no commentary) suitable for a modern
text-to-image model.

Rules:
- Expand the sketch into a full scene: medium, style, aesthetics, lighting,
  composition, background detail and plausible secondary elements where
  appropriate.
- Stay strictly faithful to everything the user explicitly named: never drop
  or alter named subjects, in-scene text, colors, quantities, or stated
  constraints.
- If the user includes meta instructions about how to expand (e.g. desired
  style, mood, richness, additions, "电影感", "细节丰富一点"), treat them as
  authoritative and follow them first.
- Match the user's language when it is unambiguous; otherwise English.
- Output the expanded prompt only.
"""


def _passthrough(prompt: str, reason: str) -> None:
    print(f"expand: qwen3-a unavailable ({reason}); passing prompt through "
          f"unchanged", file=sys.stderr)
    print(prompt)


def main() -> None:
    # Prompt: join argv args, or stdin when none given.
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("expand: empty prompt", file=sys.stderr)
        sys.exit(1)

    payload = {
        "model": QWEN3_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        # Disable thinking: reasoning can exhaust max_tokens before the
        # expanded prompt is emitted (content=None, finish_reason=length).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{QWEN3_API}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        print(f"expand: expanding prompt via {QWEN3_MODEL} at {QWEN3_API}...",
              file=sys.stderr)
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        content = result["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError(
                f"empty content (finish_reason="
                f"{result['choices'][0].get('finish_reason')})"
            )
    except Exception as e:
        _passthrough(prompt, str(e))
        return

    expanded = content.strip()
    print(f"expand: expanded prompt ({len(expanded)} chars)", file=sys.stderr)
    print(expanded)


if __name__ == "__main__":
    main()
