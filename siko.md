# siko.py — 西口 (siko) instruction detector

Detects "siko" (西口) instructions in Japanese ASMR audio using
**Qwen3-ASR-1.7B** for speech recognition with forced alignment timestamps.

## Usage

```bash
# Basic detection
./siko.py a.mp3

# Output: <ts> <duration> for each siko instruction
# 12.345   1.500
# 30.200   0.800
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3-ASR-1.7B` | Qwen3-ASR model name or path |
| `--aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner for word-level timestamps |
| `--no-aligner` | — | Disable forced aligner (segment-level only) |
| `--device` | `auto` | Device: cuda:0 / cpu |
| `--dtype` | `bfloat16` | Compute dtype: bfloat16 / float16 / float32 |
| `--language` | `ja` | Language code |
| `--output` | `tsv` | Output format: tsv / srt / json |
| `--max-new-tokens` | `256` | Max new tokens for generation |
| `--batch-size` | `1` | Max inference batch size |

## Output formats

- `tsv` — `<start_ts> <duration>` (default, for piping)
- `srt` — SubRip subtitle format
- `json` — JSON array with start, duration, end, text

## Siko patterns

Detects the following patterns in transcribed text:

| Pattern | Type |
|---------|------|
| しこ | Hiragana |
| シコ | Katakana |
| 西口 | Kanji |
| siko | Romaji |
| shico | Romaji |
| shikoko | Romaji |
| しこしこ | Repeated hiragana |
| シコシコ | Repeated katakana |