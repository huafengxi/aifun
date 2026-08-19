# paint.py — Text-to-image via ideogram4 MaaS

Generates images through the **ideogram4** MaaS service (`make ideogram4.start`,
port 9114). Plain-text prompts are automatically expanded into Ideogram 4's
structured **JSON caption** format via **qwen3-a** (`make qwen3-a.start`,
port 9113), using Ideogram's official open-source magic-prompt system prompt
(`ideogram4_magic_prompt_v1.txt`).

## Usage

```bash
# Plain-text prompt (auto-expanded to JSON by qwen3-a)
./paint.py "a cat sitting on a cloud" -o cat.png

# Read prompt from stdin
echo "cyberpunk city at night" | ./paint.py -o city.png
./paint.py - --width 1536 --height 864 < prompt.txt

# Prompt is already an Ideogram 4 JSON caption → passed through as-is
./paint.py '{"high_level_description": "...", "compositional_deconstruction": {...}}'

# Skip qwen3-a expansion entirely
./paint.py --no-magic "raw prompt"
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `prompt` | stdin | Text prompt; `-` or omitted reads from stdin |
| `-o`, `--output` | `output.png` | Output image file |
| `--width` | `1024` | Image width (divisible by 16) |
| `--height` | `1024` | Image height (divisible by 16) |
| `--steps` | `10` | Inference steps |
| `--cfg` | server schedule | Guidance scale; `1.0` = fast mode (no CFG) |
| `--seed` | random | Random seed |
| `-n`, `--num-images` | `1` | Number of images |
| `--no-magic` | off | Skip JSON expansion, send prompt as-is |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `IDEOGRAM_API` | `http://localhost:9114` | ideogram4 service URL |
| `QWEN3_API` | `http://localhost:9113` | qwen3-a service URL |
| `QWEN3_MODEL` | `qwen3.8-a` | qwen3-a model name |

## Notes

- If qwen3-a is down, expansion falls back to a minimal JSON wrapper so
  generation still works (lower quality).
- JSON captions follow the Ideogram 4 schema: `high_level_description` +
  `compositional_deconstruction` (`background` + `elements[]`), optionally
  `style_description` with a `color_palette` of uppercase hex colors.
