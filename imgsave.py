#!/usr/bin/env python3
"""imgsave.py — save generated images as JPEG (aifun 产物格式约定).

生成产物一律存 JPEG（quality 默认 90）：w/ 画廊浏览友好，体积约为 PNG 的
1/10。

单文件模式:
    imgsave.py IN OUT [quality]
    IN 可以是:
      - .png（或任何 PIL 能打开的图片）文件
      - base64 文本文件（SGLang /v1/images/generations 返回的 b64_json，
        直接粘贴或 `jq -r .data[0].b64_json` 落盘均可；自动容忍前后空白
        与 data: URL 前缀）
    OUT 按扩展名保存（.jpg/.jpeg → JPEG，quality 默认 90）

批量模式:
    imgsave.py --dir DIR [--quality 90] [--delete-src]
    DIR 下每个 *.png 转成同名 .jpg；--delete-src 时在转换并校验
    （重新打开 jpg 且尺寸一致）后才删除原 png，并把 DIR/index.md 里
    出现的 <name>.png 文件引用重写为 <name>.jpg（index.md 存在时）。

仅依赖标准库 + PIL。

用法示例:
    ./imgsave.py out.png out.jpg
    jq -r .data[0].b64_json resp.json | tee b64.txt >/dev/null && ./imgsave.py b64.txt out.jpg 95
    ./imgsave.py --dir ~/m/run/temp/some-gen --delete-src
"""
import argparse
import base64
import io
import os
import re
import sys

from PIL import Image

JPEG_EXTS = {".jpg", ".jpeg"}


def load_input(path: str) -> Image.Image:
    """Open IN as an image file, or decode it as a base64 text file."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"\x89PNG") or head[:3] == b"\xff\xd8\xff" \
            or head[:2] in (b"BM", b"II", b"MM"):
        img = Image.open(path)
        img.load()
        return img
    # treat as base64 text (b64_json)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path}: empty file, not an image or base64")
    text = re.sub(r"^data:[^;]*;base64,", "", text)
    try:
        raw = base64.b64decode(text, validate=False)
    except Exception as e:
        raise ValueError(
            f"{path}: not a recognized image and not decodable base64 ({e})")
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def save_jpeg(img: Image.Image, out: str, quality: int) -> int:
    """Save img to OUT (format from extension); return bytes written."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")  # RGBA/P 等无 alpha 通道支持的格式
    fmt = Image.registered_extensions().get(
        os.path.splitext(out)[1].lower(), "JPEG")
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(out, format=fmt, quality=quality)
    return os.path.getsize(out)


def single(args) -> None:
    img = load_input(args.infile)
    n = save_jpeg(img, args.out, args.quality)
    print(f"{args.infile} -> {args.out} "
          f"({img.size[0]}x{img.size[1]}, {n} bytes, q{args.quality})",
          file=sys.stderr)


def batch(args) -> None:
    pngs = sorted(f for f in os.listdir(args.dir)
                  if f.lower().endswith(".png")
                  and os.path.isfile(os.path.join(args.dir, f)))
    if not pngs:
        print(f"{args.dir}: no *.png files", file=sys.stderr)
        return
    index_path = os.path.join(args.dir, "index.md")
    index_text = None
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_text = f.read()
    total_png = total_jpg = 0
    renamed = []
    for f in pngs:
        src = os.path.join(args.dir, f)
        dst = os.path.join(args.dir, os.path.splitext(f)[0] + ".jpg")
        img = Image.open(src)
        img.load()
        n = save_jpeg(img, dst, args.quality)
        total_png += os.path.getsize(src)
        total_jpg += n
        if args.delete_src:
            # 校验后才删：重新打开 jpg，尺寸必须一致
            with Image.open(dst) as chk:
                chk.load()
                assert chk.size == img.size, f"{dst}: size mismatch"
            os.remove(src)
            if index_text is not None:
                index_text = index_text.replace(f, os.path.basename(dst))
            renamed.append(f"{f} -> {os.path.basename(dst)}")
        print(f"{f} -> {os.path.basename(dst)} ({n} bytes)",
              file=sys.stderr)
    if index_text is not None and renamed:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_text)
        print(f"index.md: rewrote {len(renamed)} filename(s)", file=sys.stderr)
    ratio = total_jpg / total_png * 100 if total_png else 0
    print(f"converted {len(pngs)} file(s): {total_png} -> {total_jpg} bytes "
          f"({ratio:.1f}%)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description="Save/convert generated images as JPEG (aifun 产物格式"
                    "约定, 默认 q90). 单文件支持 png 或 base64 文本输入 "
                    "(SGLang b64_json); --dir 批量转 *.png, --delete-src "
                    "校验后删源并同步重写 index.md。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法示例:", 1)[-1] if __doc__ else None)
    p.add_argument("infile", nargs="?", help="输入图片或 base64 文本文件")
    p.add_argument("out", nargs="?", help="输出文件（按扩展名定格式）")
    p.add_argument("quality_pos", nargs="?", type=int, metavar="quality",
                   help="JPEG 质量（默认 90）")
    p.add_argument("--dir", help="批量模式：目录下 *.png 转同名 .jpg")
    p.add_argument("--quality", type=int, default=None, help="JPEG 质量（默认 90）")
    p.add_argument("--delete-src", action="store_true",
                   help="批量模式：转换并校验后删除原 png，且重写 index.md")
    args = p.parse_args()

    quality = args.quality_pos or args.quality or 90
    if args.dir:
        if args.infile or args.out:
            p.error("--dir 不能与 IN/OUT 同时使用")
        if not os.path.isdir(args.dir):
            p.error(f"--dir: not a directory: {args.dir}")
        args.quality = quality
        batch(args)
    else:
        if not args.infile or not args.out:
            p.error("需要 IN OUT（或用 --dir 批量模式）")
        args.quality = quality
        try:
            single(args)
        except (ValueError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
