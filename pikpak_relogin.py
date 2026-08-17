#!/home/yuanqi.xhf/miniconda3/bin/python
"""
pikpak_relogin.py — Re-login the alist PikPak storage when it is logged out.

Background
----------
alist's PikPak driver keeps a `refresh_token` + a short-lived `captcha_token`
in its storage config.  When the captcha token expires (and the refresh token
has rotated somewhere else), the driver cannot log back in by itself because
PikPak's `/v1/auth/signin` now requires a one-time slider captcha (Tencent
TxCaptcha).  The driver reports:

    captcha_invalid / captcha_token expired  (alist log)

The captcha cannot be solved headlessly; a human has to solve it once.
This script makes that single manual step as small as possible.

Usage
-----
    # 1) See what is wrong and get the verification URL(s).
    ./pikpak_relogin.py status
    ./pikpak_relogin.py url

    # 2) Open the URL printed by `url` in a browser and solve the slider.
    #    Capture the resulting token (the captcha page redirects to the
    #    xlaccsdk01:// deep link; the token is the `code`/captcha_token in
    #    that URL, or whatever the page shows).

    # 3a) If you got a verified *captcha_token*, do the signin to obtain a
    #     fresh refresh_token:
    ./pikpak_relogin.py signin --captcha-token <CAPTCHA_TOKEN>

    # 3b) Apply the token(s) to the alist storage and re-init the driver:
    ./pikpak_relogin.py apply --refresh-token <REFRESH_TOKEN> \
        [--captcha-token <CAPTCHA_TOKEN>]

    # 4) Verify upload is healthy again (after this, re-run:
    #     make dav.sync DAV_SKIP_UPLOAD=0)

Proxy note
----------
alist talks to PikPak directly (CN egress).  Pass --proxy if you need the
mihomo proxy for the PikPak API calls, e.g.:

    ./pikpak_relogin.py --proxy http://127.0.0.1:7897 url
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse

import requests

# PikPak web-platform constants (match alist vendors/pikpak/driver.go).
WEB_CLIENT_ID = "YUMx5nI8ZU8Ap8pm"
WEB_CLIENT_SECRET = "dbw2OtmVEeuUvIptb1Coyg"
WEB_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36")
API_DOMAIN = "mypikpak.net"
REDIRECT_URI = "xlaccsdk01://xbase.cloud/callback?state=harbor"


def _load_env_file(path):
    env = {}
    if not os.path.isfile(path):
        return env
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$", line)
            if not m:
                continue
            val = m.group(2).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            env[m.group(1)] = val
    return env


def _load_config():
    """Load PikPak + alist config from the standard env files."""
    base = os.path.dirname(os.path.abspath(__file__))
    m_root = os.path.dirname(base)
    cfg = {}
    for f in (os.path.join(m_root, "env", "pikpak.web"),
              os.path.join(m_root, "env", "webdav.env"),
              os.path.join(m_root, "dav.env")):
        cfg.update(_load_env_file(f))

    return {
        "username": cfg.get("PIKPAK_USERNAME") or cfg.get("ALIST_PIKPAK_USERNAME", ""),
        "password": cfg.get("PIKPAK_PASSWORD") or cfg.get("ALIST_PIKPAK_PASSWORD", ""),
        "alist_base": (cfg.get("ALIST_BASE_URL") or "http://localhost:5244").rstrip("/"),
        "alist_user": cfg.get("ALIST_USERNAME", "admin"),
        "alist_password": cfg.get("ALIST_PASSWORD", "admin"),
    }


def _session(proxy):
    s = requests.Session()
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def _shield_captcha_init(cfg, proxy):
    """POST /v1/shield/captcha/init exactly like alist does at login."""
    s = _session(proxy)
    url = f"https://user.{API_DOMAIN}/v1/shield/captcha/init"
    device_id = hashlib.md5((cfg["username"] + cfg["password"]).encode()).hexdigest()
    body = {
        "action": "POST:/v1/auth/signin",
        "captcha_token": "",
        "client_id": WEB_CLIENT_ID,
        "device_id": device_id,
        "meta": {"email": cfg["username"]},
        "redirect_uri": REDIRECT_URI,
    }
    r = s.post(url, params={"client_id": WEB_CLIENT_ID},
               headers={
                   "User-Agent": WEB_USER_AGENT,
                   "X-Device-ID": device_id,
                   "X-Captcha-Token": "",
                   "Content-Type": "application/json",
               }, json=body, timeout=60)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    data = r.json()
    return {
        "url": data.get("url", ""),
        "captcha_token": data.get("captcha_token", ""),
        "expires_in": data.get("expires_in"),
        "device_id": device_id,
    }


def _signin(cfg, captcha_token, proxy):
    """POST /v1/auth/signin with a verified captcha_token -> tokens."""
    s = _session(proxy)
    url = f"https://user.{API_DOMAIN}/v1/auth/signin"
    device_id = hashlib.md5((cfg["username"] + cfg["password"]).encode()).hexdigest()
    body = {
        "captcha_token": captcha_token,
        "client_id": WEB_CLIENT_ID,
        "client_secret": WEB_CLIENT_SECRET,
        "username": cfg["username"],
        "password": cfg["password"],
    }
    r = s.post(url, params={"client_id": WEB_CLIENT_ID},
               headers={
                   "User-Agent": WEB_USER_AGENT,
                   "X-Device-ID": device_id,
                   "X-Captcha-Token": captcha_token,
                   "Content-Type": "application/json",
               }, json=body, timeout=60)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:500]}"}
    data = r.json()
    if data.get("error"):
        return {"error": json.dumps(data)[:500]}
    return {
        "refresh_token": data.get("refresh_token", ""),
        "access_token": data.get("access_token", ""),
        "sub": data.get("sub", ""),
    }


def _alist_admin(cfg):
    """Login to alist admin and return a token."""
    s = requests.Session()
    r = s.post(f"{cfg['alist_base']}/api/auth/login",
               json={"username": cfg["alist_user"],
                     "password": cfg["alist_password"]}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 200:
        raise RuntimeError(f"alist admin login failed: {data}")
    return data["data"]["token"]


def _alist_storage(cfg, token):
    s = requests.Session()
    r = s.get(f"{cfg['alist_base']}/api/admin/storage/list",
              headers={"Authorization": token}, timeout=30)
    r.raise_for_status()
    data = r.json()
    for st in data["data"]["content"]:
        if st.get("driver") == "PikPak":
            return st
    return None


def _alist_update(cfg, token, storage, refresh_token, captcha_token):
    addition = json.loads(storage["addition"])
    if refresh_token:
        addition["refresh_token"] = refresh_token
    if captcha_token:
        addition["captcha_token"] = captcha_token
    # Preserve device_id derived from username+password like alist does.
    addition.setdefault(
        "device_id",
        hashlib.md5((addition.get("username", "") + addition.get("password", "")).encode()).hexdigest())
    storage["addition"] = json.dumps(addition)

    s = requests.Session()
    r = s.post(f"{cfg['alist_base']}/api/admin/storage/update",
               headers={"Authorization": token, "Content-Type": "application/json"},
               json=storage, timeout=60)
    r.raise_for_status()
    return r.json()


def cmd_url(cfg, proxy):
    res = _shield_captcha_init(cfg, proxy)
    if res.get("error"):
        print("ERROR:", res["error"], file=sys.stderr)
        return 1
    if not res.get("url"):
        print("PikPak returned a captcha_token directly (no verify needed):", file=sys.stderr)
        print(res["captcha_token"], file=sys.stderr)
        print("\nRe-run signin with it:", file=sys.stderr)
        print(f"  ./pikpak_relogin.py signin --captcha-token {res['captcha_token']}", file=sys.stderr)
        return 0
    print("device_id:", res["device_id"], file=sys.stderr)
    print("credit token (ck0, not yet verified):", res["captcha_token"], file=sys.stderr)
    print("expires_in:", res["expires_in"], file=sys.stderr)
    print("\nOpen this URL in a browser and solve the slider:\n", file=sys.stderr)
    print(res["url"])
    print(file=sys.stderr)
    print("After solving, the page redirects to the xlaccsdk01:// deep link;", file=sys.stderr)
    print("capture the resulting code/captcha_token from that URL.", file=sys.stderr)
    return 0


def cmd_signin(cfg, captcha_token, proxy):
    if not captcha_token:
        print("ERROR: --captcha-token is required", file=sys.stderr)
        return 2
    res = _signin(cfg, captcha_token, proxy)
    if res.get("error"):
        print("signin failed:", res["error"], file=sys.stderr)
        return 1
    print("refresh_token:", res["refresh_token"])
    print("access_token: ", res["access_token"])
    print("sub:          ", res["sub"])
    print(file=sys.stderr)
    print("Apply it with:", file=sys.stderr)
    print(f"  ./pikpak_relogin.py apply --refresh-token {res['refresh_token']} "
          f"--captcha-token {captcha_token}", file=sys.stderr)
    return 0


def cmd_apply(cfg, refresh_token, captcha_token):
    token = _alist_admin(cfg)
    storage = _alist_storage(cfg, token)
    if storage is None:
        print("ERROR: no PikPak storage found in alist", file=sys.stderr)
        return 1
    res = _alist_update(cfg, token, storage, refresh_token, captcha_token)
    if res.get("code") != 200:
        print("alist update failed:", res, file=sys.stderr)
        return 1
    print("alist storage updated and driver re-initialized.", file=sys.stderr)
    print("Check `make dav.status` (or logs/dav_sync.log) to confirm listing works again.",
          file=sys.stderr)
    return 0


def cmd_status(cfg, proxy):
    try:
        token = _alist_admin(cfg)
    except Exception as e:
        print(f"alist admin unavailable: {e}", file=sys.stderr)
        return 1
    storage = _alist_storage(cfg, token)
    if storage is None:
        print("no PikPak storage in alist", file=sys.stderr)
        return 1

    addition = json.loads(storage["addition"])
    print("mount:", storage["mount_path"])
    print("status:", storage.get("status"))
    print("username:", addition.get("username"))
    print("platform:", addition.get("platform"))
    print("device_id:", addition.get("device_id"))
    print("refresh_token:", (addition.get("refresh_token") or "")[:24] + "…"
          if addition.get("refresh_token") else "(empty)")
    print("captcha_token:", (addition.get("captcha_token") or "")[:24] + "…"
          if addition.get("captcha_token") else "(empty)")

    # Test the exact operation dav_sync performs: list the mount's children
    # (Depth: 1 forces the driver to call PikPak's list API and exposes the
    # real auth state; Depth: 0 only returns the collection itself).
    try:
        r = requests.request(
            "PROPFIND",
            f"{cfg['alist_base']}/dav{storage['mount_path']}/",
            auth=(cfg["alist_user"], cfg["alist_password"]),
            headers={"Depth": "1"}, timeout=60)
        if r.status_code in (200, 207):
            print("\nlist /pikpak/: OK")
        else:
            print(f"\nlist /pikpak/: HTTP {r.status_code}")
            print("    -> likely captcha_token expired; run `./pikpak_relogin.py url`")
    except Exception as e:
        print(f"\nlist /pikpak/: {e}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Re-login alist PikPak storage")
    p.add_argument("--proxy", default="",
                   help="HTTP proxy for PikPak API calls (e.g. http://127.0.0.1:7897)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show PikPak storage status")
    sub.add_parser("url", help="Print the captcha verification URL")

    sp = sub.add_parser("signin", help="Sign in with a verified captcha token")
    sp.add_argument("--captcha-token", required=True)

    ap = sub.add_parser("apply", help="Apply refresh/captcha tokens to alist")
    ap.add_argument("--refresh-token")
    ap.add_argument("--captcha-token")

    args = p.parse_args()
    cfg = _load_config()

    if args.cmd == "status":
        return cmd_status(cfg, args.proxy)
    if args.cmd == "url":
        return cmd_url(cfg, args.proxy)
    if args.cmd == "signin":
        return cmd_signin(cfg, args.captcha_token, args.proxy)
    if args.cmd == "apply":
        return cmd_apply(cfg, args.refresh_token, args.captcha_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())