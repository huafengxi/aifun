#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
wximg.py — 微信公众号文章图片抓取器（WeChat article image scraper）

用法:
    ./wximg.py URL [URL...] [-o OUT_DIR] [--list FILE] [--workers N]

示例:
    ./wximg.py https://mp.weixin.qq.com/s/XXXX -o ./wximg-out/
    ./wximg.py --list urls.txt -o ./wximg-out/

功能:
    1. urllib GET 文章原始 HTML（浏览器 UA，文章间 0.5-1s 间隔）；
       失败（网络错误/环境验证页）重试 2 次后报错跳过，不中断批量。
    2. 解析两类图片来源（纯标准库正则实现）:
       - 图片翻页类文章（item_show_type=8）: JS 变量 picture_page_info_list
         中的 cdn_url（跳过 is_qr_code=1 的二维码）;
       - 普通图文文章: 正文 <img data-src="...">，只收 mmbiz.qpic.cn 且
         wx_fmt 为 jpeg/jpg/png/webp/gif 的，去重保序。
       URL 中的 \xNN hex 转义与 &amp;/&lt; 等 HTML 实体会自动还原。
    3. 线程池（≤8）并发下载，带 Referer: https://mp.weixin.qq.com/ 头，
       按序命名 01.<ext>（ext 按 wx_fmt），并校验文件魔数
       (png/jpg/webp/gif) 防假图，损坏的记入 meta.json 的 failures，
       不混入 local_files。
    4. 产物布局: OUT_DIR/<标题slug>-<urlhash6>/
       图片文件 + meta.json（title/author/publish_time/tags/source_url/
       image_urls/local_files/failures/fetched_at）。
       og:image 封面若不在图片列表中，单独存为 cover.<ext>。

已知限制:
    - 账号级文章列表外部不可得（微信无公开接口），需用户自行投喂文章链接。
    - 视频类文章（item_show_type 非图片/图文）无图片可抓。
    - 抓取频率过高会触发微信「当前环境异常」验证页，请降低频率/减少并发；
      本工具已在文章间加 0.5-1s 间隔，仍被拦截时建议加大 --min-delay。

仅使用 Python 标准库（urllib/re/json/hashlib/threading），无第三方依赖。
"""

import argparse
import hashlib
import html as html_mod
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
REFERER = "https://mp.weixin.qq.com/"
ALLOWED_FMT = {"jpeg", "jpg", "png", "webp", "gif"}
FMT_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}
MAGIC = {
    "png":  (b"\x89PNG",),
    "jpg":  (b"\xff\xd8\xff",),
    "webp": (b"RIFF",),          # RIFF....WEBP, second check below
    "gif":  (b"GIF87a", b"GIF89a"),
}
VERIFY_MARKERS = ("当前环境异常", "环境异常，完成验证", "weixin110.qq.com/cgi-bin/mmspam")

CST = timezone(timedelta(hours=8))


# ---------------------------------------------------------------- utilities

def unescape_js(s: str) -> str:
    """还原 \\xNN hex 转义 + HTML 实体（可能多层嵌套，迭代到稳定）。"""
    prev = None
    cur = s
    for _ in range(3):
        cur = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), cur)
        cur = html_mod.unescape(cur)
        if cur == prev:
            break
        prev = cur
    return cur


def http_get(url: str, timeout: int = 30, referer: bool = False) -> bytes:
    req = Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    if referer:
        req.add_header("Referer", REFERER)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_article(url: str, retries: int = 2) -> str:
    """抓取文章 HTML；失败/验证页重试 retries 次后抛异常。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            data = http_get(url)
            text = data.decode("utf-8", errors="replace")
            hits = [m for m in VERIFY_MARKERS if m in text]
            if hits:
                raise RuntimeError("环境验证页(%s)" % hits[0])
            return text
        except (URLError, HTTPError, RuntimeError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("抓取失败: %s" % last_err)


# ---------------------------------------------------------------- parsing

def _scan_js_array(text: str, open_idx: int) -> int:
    """从 '[' 位置扫描到匹配的 ']'，返回其下标（字符串感知）。"""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c in "'\"":
            q = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    break
                i += 1
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_js_objects(array_body: str):
    """把 JS 数组体切成顶层对象字符串列表（括号/引号感知）。"""
    objs = []
    i, n = 0, len(array_body)
    while i < n:
        if array_body[i] == "{":
            depth = 0
            start = i
            while i < n:
                c = array_body[i]
                if c in "'\"":
                    q = c
                    i += 1
                    while i < n:
                        if array_body[i] == "\\":
                            i += 2
                            continue
                        if array_body[i] == q:
                            break
                        i += 1
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(array_body[start:i + 1])
                        break
                i += 1
        i += 1
    return objs


def parse_picture_page(html_text: str):
    """解析 picture_page_info_list（图片翻页类文章）。返回 [(url, w, h), ...]。"""
    m = re.search(r"picture_page_info_list\s*:\s*\[", html_text)
    if not m:
        return []
    open_idx = html_text.index("[", m.start())
    close_idx = _scan_js_array(html_text, open_idx)
    if close_idx < 0:
        return []
    body = html_text[open_idx + 1:close_idx]
    out = []
    for obj in _split_js_objects(body):
        cdn = re.search(r"cdn_url\s*:\s*'([^']*)'", obj)
        if not cdn:
            continue
        qr = re.search(r"is_qr_code\s*:\s*'(\d+)'", obj)
        if qr and qr.group(1) == "1":
            continue  # 二维码，跳过
        w = re.search(r"width\s*:\s*'(\d+)'", obj)
        h = re.search(r"height\s*:\s*'(\d+)'", obj)
        out.append((unescape_js(cdn.group(1)),
                    int(w.group(1)) if w else None,
                    int(h.group(1)) if h else None))
    return out


def parse_data_src(html_text: str):
    """解析普通图文正文 <img data-src="...">，只收 mmbiz.qpic.cn 合法格式，去重保序。"""
    out, seen = [], set()
    for m in re.finditer(r"""data-src\s*=\s*["']([^"']+)["']""", html_text):
        url = unescape_js(m.group(1))
        host = urlparse(url).netloc
        if not host.endswith("mmbiz.qpic.cn"):
            continue
        fmt = (parse_qs(urlparse(url).query).get("wx_fmt") or [""])[0].lower()
        if fmt not in ALLOWED_FMT:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, None, None))
    return out


def _meta(html_text: str, prop: str):
    m = re.search(r'<meta\s+(?:property|name)="%s"\s+content="([^"]*)"' % re.escape(prop),
                  html_text)
    return unescape_js(m.group(1)) if m else None


def parse_meta(html_text: str, url: str) -> dict:
    meta = {
        "title": _meta(html_text, "og:title"),
        "author": _meta(html_text, "author"),
        "cover": _meta(html_text, "og:image"),
        "publish_time": None,
        "tags": [],
        "source_url": url,
    }
    if not meta["author"]:
        m = re.search(r"window\.name\s*=\s*\"([^\"]*)\"", html_text)
        if m:
            meta["author"] = unescape_js(m.group(1))
    if not meta["author"]:
        m = re.search(r"nick_name\s*:\s*'([^']*)'", html_text)
        if m:
            meta["author"] = unescape_js(m.group(1))
    # create_time: 优先可读字符串，其次 unix 时间戳
    m = re.search(r"create_time\.DATA'\)\s*:\s*'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})'", html_text) \
        or re.search(r"create_time\s*:\s*'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})'", html_text)
    if m:
        meta["publish_time"] = m.group(1)
    else:
        m = re.search(r"ori_create_time\.DATA'\)\s*:\s*'(\d{10})'", html_text) \
            or re.search(r"ori_create_time\s*:\s*'(\d{10})'", html_text)
        if m:
            ts = int(m.group(1))
            meta["publish_time"] = datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M")
    # description 里的 #话题# 标签
    desc = None
    m = re.search(r"window\.desc\s*=\s*\"((?:[^\"\\]|\\.)*)\"", html_text)
    if m:
        desc = unescape_js(m.group(1))
    else:
        desc = _meta(html_text, "og:description") or _meta(html_text, "description")
    if desc:
        plain = re.sub(r"<[^>]+>", " ", desc)
        tags, seen = [], set()

        def _add(t):
            t = t.strip("#").strip()
            if t and t not in seen:
                seen.add(t)
                tags.append(t)

        # 优先成对 #话题# 形式，再收单 # 前缀的空格分隔形式
        for t in re.findall(r"#([^#\n]{1,40}?)#", plain):
            _add(t)
        for t in re.findall(r"#([^\s#]+)", plain):
            _add(t)
        meta["tags"] = tags
    return meta


# ---------------------------------------------------------------- downloading

def magic_ok(data: bytes, ext: str) -> bool:
    if ext == "webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return any(data.startswith(sig) for sig in MAGIC.get(ext, ()))


def fmt_of_url(url: str) -> str:
    fmt = (parse_qs(urlparse(url).query).get("wx_fmt") or ["jpg"])[0].lower()
    return FMT_EXT.get(fmt, "jpg")


def download_one(url: str, path: str) -> tuple:
    """下载单个 URL 到 path；返回 (ok, size, error)。"""
    try:
        data = http_get(url, referer=True)
        ext = os.path.splitext(path)[1].lstrip(".")
        if not magic_ok(data, ext):
            return False, len(data), "magic mismatch (got %d bytes, expect %s)" % (len(data), ext)
        with open(path, "wb") as f:
            f.write(data)
        return True, len(data), None
    except Exception as e:  # noqa: BLE001
        return False, 0, str(e)


def slugify(title: str, maxlen: int = 40) -> str:
    if not title:
        return "untitled"
    t = unicodedata.normalize("NFKC", title)
    t = re.sub(r'[\\/:*?"<>|\r\n\t]', "", t)
    t = re.sub(r"\s+", " ", t).strip(" ._")
    if len(t) > maxlen:
        t = t[:maxlen].rstrip(" ._")
    return t or "untitled"


def url_hash6(url: str) -> str:
    norm = re.sub(r"^https?://", "", url.strip()).rstrip("/")
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:6]


# ---------------------------------------------------------------- pipeline

def process_article(url: str, out_root: str, workers: int) -> dict:
    result = {"source_url": url, "ok": False}
    html_text = fetch_article(url)
    meta = parse_meta(html_text, url)

    images = parse_picture_page(html_text)
    source_kind = "picture_page_info_list"
    reason = None
    if not images:
        images = parse_data_src(html_text)
        source_kind = "data-src"
    if not images:
        reason = "无图: picture_page_info_list 与正文 data-src 均为空（可能是视频类/纯文字文章）"

    title = meta["title"] or "untitled"
    dirname = "%s-%s" % (slugify(title), url_hash6(url))
    article_dir = os.path.join(out_root, dirname)
    os.makedirs(article_dir, exist_ok=True)

    jobs = []  # (idx, url, path, kind)
    local_files, failures = [], []
    for i, (img_url, w, h) in enumerate(images, 1):
        ext = fmt_of_url(img_url)
        path = os.path.join(article_dir, "%02d.%s" % (i, ext))
        jobs.append((i, img_url, path, "image", w, h))

    # 封面: 不在图列则单独下载
    cover = meta.get("cover")
    cover_in_list = False
    if cover:
        if cover in [u for _, u, _, _, _, _ in jobs]:
            cover_in_list = True
        else:
            cext = fmt_of_url(cover)
            jobs.append((len(jobs) + 1, cover, os.path.join(article_dir, "cover." + cext),
                         "cover", None, None))

    results_by_idx = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(8, workers)) as pool:
            futs = {pool.submit(download_one, j[1], j[2]): j for j in jobs}
            for fut in as_completed(futs):
                idx, img_url, path, kind, w, h = futs[fut]
                ok, size, err = fut.result()
                results_by_idx[idx] = (ok, size, err, img_url, path, kind, w, h)

    for idx in sorted(results_by_idx):
        ok, size, err, img_url, path, kind, w, h = results_by_idx[idx]
        rec = {"url": img_url, "file": os.path.basename(path), "size": size}
        if w is not None:
            rec["width"] = w
        if h is not None:
            rec["height"] = h
        if ok:
            if kind == "cover":
                meta["cover_local"] = os.path.basename(path)
            else:
                local_files.append(os.path.basename(path))
                meta.setdefault("images_detail", []).append(rec)
        else:
            rec["error"] = err
            rec["kind"] = kind
            failures.append(rec)

    meta["local_files"] = local_files
    meta["image_urls"] = [u for _, u, _, k, _, _ in jobs if k == "image"]
    meta["image_source"] = source_kind if images else None
    if reason:
        meta["note"] = reason
    meta["cover_in_image_list"] = cover_in_list
    meta["fetched_at"] = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %z")

    with open(os.path.join(article_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    result.update(ok=not reason, dir=article_dir, title=title,
                  n_images=len(local_files), n_fail=len(failures), reason=reason)
    return result


# ---------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(
        prog="wximg.py",
        description="微信公众号文章图片抓取器（标准库实现，详见模块 docstring）",
        epilog="已知限制: (1) 账号级文章列表外部不可得，需用户投喂链接; "
               "(2) 视频类文章无图; "
               "(3) 频率过高会触发微信环境验证页，请降低频率。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urls", nargs="*", metavar="URL", help="文章链接（可多个）")
    p.add_argument("--list", dest="list_file", metavar="FILE", help="每行一个 URL 的批量文件")
    p.add_argument("-o", "--out", default="./wximg-out/", help="输出目录（默认 ./wximg-out/）")
    p.add_argument("--workers", type=int, default=8, help="下载并发数（默认 8，上限 8）")
    p.add_argument("--min-delay", type=float, default=0.5, help="文章间最小间隔秒（默认 0.5）")
    p.add_argument("--max-delay", type=float, default=1.0, help="文章间最大间隔秒（默认 1.0）")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    urls = list(args.urls)
    if args.list_file:
        with open(args.list_file, encoding="utf-8") as f:
            urls.extend(line.strip() for line in f
                        if line.strip() and not line.strip().startswith("#"))
    urls = [u for u in urls if u.startswith(("http://", "https://"))]
    if not urls:
        print("错误: 未提供任何文章 URL（位置参数或 --list）", file=sys.stderr)
        return 2

    out_root = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_root, exist_ok=True)

    n_ok = n_fail = 0
    for i, url in enumerate(urls):
        if i:
            time.sleep(random.uniform(args.min_delay, args.max_delay))
        print("[%d/%d] %s" % (i + 1, len(urls), url))
        try:
            r = process_article(url, out_root, args.workers)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print("  !! 失败（重试后仍失败，跳过）: %s" % e, file=sys.stderr)
            continue
        if r["ok"]:
            n_ok += 1
            print("  ok  title=%r  images=%d  fail=%d  -> %s"
                  % (r["title"], r["n_images"], r["n_fail"], r["dir"]))
        else:
            n_ok += 1  # 抓取成功但无图，不算抓取失败
            print("  -- %s -> %s" % (r["reason"], r["dir"]))

    print("完成: 成功 %d / 失败 %d / 共 %d，输出: %s" % (n_ok, n_fail, len(urls), out_root))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
