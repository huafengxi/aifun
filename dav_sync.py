#!/home/yuanqi.xhf/miniconda3/bin/python
"""
dav_sync.py — Sync files between local mirror and remote storage.

Backends (DAV_BACKEND env, default webdav):
    webdav  — direct WebDAV (WEBDAV_ENDPOINT_URL; the former alist-mount route was removed)
    pikpak  — direct PikPak API via ../cloud-storage/pikpak.py

Usage:
    ./dav_sync.py download <remote_dir> <local_mirror>
    ./dav_sync.py upload <local_mirror> <remote_dir>
    ./dav_sync.py demosaic_clean_remote <remote_dir> <remote_done_dir>
    ./dav_sync.py clean_local <local_dir> <remote_dir> <remote_done_dir>
"""

import argparse
import io
import os
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime

# ---------------------------------------------------------------------------
# Name fuzz (same rule as fsu.py nfuzz) — symmetric, fuzz==defuzz
# ---------------------------------------------------------------------------

_CHAR_MAP = 'iFXhbcNYDuUgsjrIMJwTpPAqnyvOfSxeEzWBkdtQmlZCoRVKLGHa'

def _fuzz_str(s):
    """Fuzz a string using the nfuzz rule. Symmetric: _fuzz_str(_fuzz_str(x)) == x."""
    def _translate(c):
        i = _CHAR_MAP.find(c)
        return _CHAR_MAP[i ^ 1] if i >= 0 else c
    return ''.join(map(_translate, s))


# ---------------------------------------------------------------------------
# WebDAV helpers
# ---------------------------------------------------------------------------

def _dec_secret(cipher):
    """Decrypt an enc1: cipher value via bin/envdec.py (in-memory only)."""
    envdec = os.path.join(os.path.dirname(__file__), "..", "bin", "envdec.py")
    return subprocess.run(
        [sys.executable, envdec, "--value", cipher],
        capture_output=True, text=True, check=True,
    ).stdout


def _load_webdav_env():
    """Load WebDAV credentials from env files."""
    env = {}
    env_files = [
        os.path.join(os.path.dirname(__file__), "..", "env", "webdav.env"),
        os.path.join(os.path.dirname(__file__), "..", "dav.env"),
        os.path.expanduser("~/m/env/webdav.env"),
        os.path.expanduser("~/m/dav.env"),
    ]
    for f in env_files:
        if os.path.isfile(f):
            with open(f) as fh:
                for line in fh:
                    m = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)\s*$", line)
                    if not m:
                        continue
                    val = m.group(2).strip()
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    if val.startswith("enc1:"):
                        val = _dec_secret(val[5:])
                    env[m.group(1)] = val
            break

    # Direct WebDAV only (the alist-mount route was removed with alist).
    url = env.get("WEBDAV_ENDPOINT_URL") or env.get("WEBDAV_URL")
    if url:
        p = urllib.parse.urlsplit(url)
        return {
            "hostname": f"{p.scheme}://{p.hostname}" + (f":{p.port}" if p.port else ""),
            "username": p.username or env.get("WEBDAV_USERNAME", ""),
            "password": p.password or env.get("WEBDAV_PASSWORD", ""),
            "root": env.get("WEBDAV_ROOT", "/"),
        }
    return {
        "hostname": env.get("WEBDAV_HOSTNAME", ""),
        "username": env.get("WEBDAV_USERNAME", ""),
        "password": env.get("WEBDAV_PASSWORD", ""),
        "root": env.get("WEBDAV_ROOT", "/"),
    }


def _get_webdav_client():
    """Create a webdav4 Client from env config."""
    from webdav4.client import Client
    creds = _load_webdav_env()
    hostname = creds["hostname"].rstrip("/")
    root = "/" + creds["root"].strip("/")
    base_url = f"{hostname}{root}"
    return Client(
        base_url,
        auth=(creds["username"], creds["password"]),
        verify=False,
        follow_redirects=True,
        timeout=120.0,
        trust_env=False,
    )


def _list_webdav_dir(dav, path):
    """List a WebDAV directory, returning list of file names.

    Returns ``None`` when the directory cannot be listed (e.g. remote is
    unreachable).  Callers must NOT interpret ``None`` as an empty directory,
    otherwise destructive steps would run against an unknown remote state.
    """
    import os as _os
    try:
        items = dav.ls(path, detail=True)
    except Exception as e:
        print(f"Error listing {path}: {e}", file=sys.stderr)
        return None

    names = []
    for item in items:
        name = item.get("name", "")
        name = _os.path.split(name)[1]
        is_dir = (
            item.get("isdir") or
            item.get("href", "").endswith("/") or
            item.get("content_type") in ("httpd/unix-directory", "directory")
        )
        if not is_dir and name:
            names.append(name)
    return names


def _download_from_webdav(dav, remote_path, local_path):
    """Download a file from WebDAV to local path via a .part temp file.

    The final file only appears after the download completes, so watchers
    (demosaic.py) never see a partial file.
    """
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    part_path = local_path + ".part"
    try:
        with open(part_path, "wb") as f:
            dav.download_fileobj(remote_path, f)
        os.replace(part_path, local_path)
        return True
    except Exception as e:
        print(f"Download error {remote_path}: {e}", file=sys.stderr)
        try:
            os.remove(part_path)
        except OSError:
            pass
        return False


def _upload_to_webdav(dav, local_path, remote_path):
    """Upload a local file to WebDAV."""
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        buffer = io.BytesIO(data)
        dav.upload_fileobj(buffer, remote_path, overwrite=True)
        return True
    except Exception as e:
        print(f"Upload error {remote_path}: {e}", file=sys.stderr)
        return False


def _move_webdav(dav, src_path, dst_path):
    """Move/rename a file on WebDAV."""
    try:
        dav.move(src_path, dst_path, overwrite=True)
        return True
    except Exception as e:
        print(f"Move error {src_path} -> {dst_path}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Backends — uniform interface: list / download / upload / move.
# list() returns None when the remote is unreachable/unknown (callers must
# NOT treat that as an empty dir).
# ---------------------------------------------------------------------------

class WebDAVBackend:
    name = "webdav"

    def __init__(self):
        self.dav = _get_webdav_client()

    def list(self, path):
        return _list_webdav_dir(self.dav, path)

    def download(self, remote_path, local_path):
        return _download_from_webdav(self.dav, remote_path, local_path)

    def upload(self, local_path, remote_path):
        return _upload_to_webdav(self.dav, local_path, remote_path)

    def move(self, src_path, dst_path):
        return _move_webdav(self.dav, src_path, dst_path)


class PikPakBackend:
    """Direct PikPak API backend (cloud-storage/pikpak.py)."""
    name = "pikpak"

    def __init__(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "cloud-storage"))
        import pikpak
        self.pk = pikpak

    @staticmethod
    def _p(path):
        return "/" + path.strip("/") if path.strip("/") else "/"

    def list(self, path):
        p = self._p(path)
        try:
            parent = "" if p == "/" else self.pk._get_folder_id(p)
            names = []
            page_token = None
            while True:
                result = self.pk.list_files(parent_id=parent,
                                            page_token=page_token)
                names.extend(f["name"] for f in result["files"]
                             if f["kind"] != "drive#folder")
                page_token = result.get("next_page_token")
                if not page_token:
                    break
            return names
        except Exception as e:
            print(f"Error listing {p}: {e}", file=sys.stderr)
            return None

    def download(self, remote_path, local_path):
        """Download via .part temp file (watchers never see partials)."""
        part_path = local_path + ".part"
        try:
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            self.pk.download_file(self._p(remote_path), part_path)
            os.replace(part_path, local_path)
            return True
        except Exception as e:
            print(f"Download error {remote_path}: {e}", file=sys.stderr)
            try:
                os.remove(part_path)
            except OSError:
                pass
            return False

    def upload(self, local_path, remote_path):
        try:
            self.pk.upload_file(local_path, self._p(remote_path))
            return True
        except Exception as e:
            print(f"Upload error {remote_path}: {e}", file=sys.stderr)
            return False

    def move(self, src_path, dst_path):
        try:
            src = self._p(src_path)
            dst = self._p(dst_path)
            src_parent = self.pk._get_folder_id(os.path.dirname(src))
            f = self.pk._find_child(src_parent, os.path.basename(src))
            if not f:
                print(f"Move error: {src_path} not found", file=sys.stderr)
                return False
            dst_parent = self.pk._get_folder_id(os.path.dirname(dst),
                                                create=True)
            self.pk._api_request(
                "POST", "/drive/v1/files:batchMove",
                json={"ids": [f["id"]], "to": {"parent_id": dst_parent}})
            return True
        except Exception as e:
            print(f"Move error {src_path} -> {dst_path}: {e}",
                  file=sys.stderr)
            return False


def _get_backend():
    name = os.environ.get("DAV_BACKEND", "webdav").lower()
    if name == "pikpak":
        return PikPakBackend()
    return WebDAVBackend()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_download(args):
    """Download all files from remote_dir to local_mirror."""
    backend = _get_backend()
    remote_dir = args.remote_dir.strip("/")
    local_mirror = args.local_mirror

    os.makedirs(local_mirror, exist_ok=True)

    remote_files = backend.list(remote_dir)
    if remote_files is None:
        print(f"SKIP download: remote /{remote_dir} unreachable", file=sys.stderr)
        return
    print(f"Remote: {len(remote_files)} files in /{remote_dir}", file=sys.stderr)
    print(f"Local: {local_mirror}", file=sys.stderr)

    # 清理孤儿 .part：目标文件已存在，或远端已不再有该文件
    expected_finals = set()
    for fname in remote_files:
        base, ext = os.path.splitext(fname)
        expected_finals.add(_fuzz_str(base) + ext)
    for entry in os.listdir(local_mirror):
        if not entry.endswith(".part"):
            continue
        target = entry[:-len(".part")]
        in_remote = target in expected_finals
        final_exists = os.path.isfile(os.path.join(local_mirror, target))
        if in_remote and not final_exists:
            continue  # 正在下载/待下载的临时文件，保留
        try:
            os.remove(os.path.join(local_mirror, entry))
            print(f"  Clean .part: {entry}", file=sys.stderr)
        except OSError:
            pass

    for fname in remote_files:
        base, ext = os.path.splitext(fname)
        local_name = _fuzz_str(base) + ext
        local_path = os.path.join(local_mirror, local_name)
        if os.path.isfile(local_path):
            print(f"  Skip (exists): {fname} -> {local_name}", file=sys.stderr)
            continue

        remote_path = f"{remote_dir}/{fname}"
        print(f"  Download: {fname} -> {local_name}", file=sys.stderr)
        if backend.download(remote_path, local_path):
            print(f"    OK", file=sys.stderr)
        else:
            print(f"    FAILED", file=sys.stderr)

    print("Download complete.", file=sys.stderr)


def cmd_upload(args):
    """Upload .restored.mp4 files from local_mirror to remote_dir."""
    backend = _get_backend()
    local_mirror = args.local_mirror
    remote_dir = args.remote_dir.strip("/")

    remote_files = backend.list(remote_dir)
    if remote_files is None:
        print(f"SKIP upload: remote /{remote_dir} unreachable", file=sys.stderr)
        return
    remote_set = set(remote_files)
    print(f"Remote: {len(remote_files)} files in /{remote_dir}", file=sys.stderr)
    print(f"Local: {local_mirror}", file=sys.stderr)

    for entry in sorted(os.listdir(local_mirror)):
        if not entry.endswith(".restored.mp4"):
            continue
        # Defuzz local name back to original remote name
        base = entry[:-len(".restored.mp4")]
        remote_name = _fuzz_str(base) + ".restored.mp4"
        if remote_name in remote_set:
            print(f"  Skip (exists): {entry} -> {remote_name}", file=sys.stderr)
            continue

        local_path = os.path.join(local_mirror, entry)
        remote_path = f"{remote_dir}/{remote_name}"
        print(f"  Upload: {entry} -> {remote_name}", file=sys.stderr)
        if backend.upload(local_path, remote_path):
            print(f"    OK", file=sys.stderr)
        else:
            print(f"    FAILED", file=sys.stderr)

    print("Upload complete.", file=sys.stderr)


def cmd_demosaic_clean_remote(args):
    """Move <foo>.mp4 to <remote_done_dir> if <foo>.restored.mp4 exists."""
    backend = _get_backend()
    remote_dir = args.remote_dir.strip("/")
    remote_done_dir = args.remote_done_dir.strip("/")

    remote_files = backend.list(remote_dir)
    if remote_files is None:
        print(f"SKIP clean remote: remote /{remote_dir} unreachable", file=sys.stderr)
        return
    print(f"Remote: {len(remote_files)} files in /{remote_dir}", file=sys.stderr)
    print(f"Done dir: /{remote_done_dir}", file=sys.stderr)

    restored_files = {f for f in remote_files if f.endswith(".restored.mp4")}
    mp4_files = {f for f in remote_files if f.endswith(".mp4") and not f.endswith(".restored.mp4")}

    for mp4 in sorted(mp4_files):
        base = mp4[:-4]  # remove .mp4
        restored_name = f"{base}.restored.mp4"
        if restored_name in restored_files:
            src = f"{remote_dir}/{mp4}"
            dst = f"{remote_done_dir}/{mp4}"
            print(f"  Move: {mp4} -> done/", file=sys.stderr)
            if backend.move(src, dst):
                print(f"    OK", file=sys.stderr)
            else:
                print(f"    FAILED", file=sys.stderr)

    print("Clean remote complete.", file=sys.stderr)


def cmd_clean_local(args):
    """Clean local files based on remote state.

    - Delete <foo>.mp4 if <remote_dir> has no such file (original was moved).
    - Delete <foo>.restored.mp4 if <remote_dir> has such file (restored was uploaded).
    """
    backend = _get_backend()
    local_dir = args.local_dir
    remote_dir = args.remote_dir.strip("/")

    remote_files = backend.list(remote_dir)
    if remote_files is None:
        print(f"SKIP clean local: remote /{remote_dir} unreachable", file=sys.stderr)
        return
    remote_set = set(remote_files)
    print(f"Remote: {len(remote_files)} files in /{remote_dir}", file=sys.stderr)
    print(f"Local: {local_dir}", file=sys.stderr)

    for entry in sorted(os.listdir(local_dir)):
        local_path = os.path.join(local_dir, entry)
        if not os.path.isfile(local_path):
            continue

        if entry.endswith(".restored.mp4"):
            # Defuzz to match remote name; delete if uploaded to remote
            base = entry[:-len(".restored.mp4")]
            remote_name = _fuzz_str(base) + ".restored.mp4"
            if remote_name in remote_set:
                print(f"  Delete: {entry} (uploaded as {remote_name})", file=sys.stderr)
                os.remove(local_path)
            continue

        if entry.endswith(".mp4"):
            # Defuzz to match remote name; delete if original no longer in remote
            base, ext = os.path.splitext(entry)
            remote_name = _fuzz_str(base) + ext
            if remote_name not in remote_set:
                print(f"  Delete: {entry} (moved from remote: {remote_name})", file=sys.stderr)
                os.remove(local_path)

    print("Clean local complete.", file=sys.stderr)


def cmd_sync(args):
    """Run download, upload, demosaic_clean_remote, clean_local in a loop."""
    from argparse import Namespace
    remote_dir = args.remote_dir
    local_mirror = args.local_mirror
    remote_done_dir = args.remote_done_dir
    interval = args.interval
    skip_upload = args.skip_upload or os.environ.get("DAV_SKIP_UPLOAD") == "1"

    print(f"dav.sync loop started", file=sys.stderr)
    print(f"  remote: {remote_dir}", file=sys.stderr)
    print(f"  local:  {local_mirror}", file=sys.stderr)
    print(f"  done:   {remote_done_dir}", file=sys.stderr)
    print(f"  interval: {interval}s", file=sys.stderr)
    print(f"  upload:  {'SKIP' if skip_upload else 'enabled'}", file=sys.stderr)

    os.makedirs(local_mirror, exist_ok=True)

    while True:
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] --- download ---", file=sys.stderr)
            cmd_download(Namespace(remote_dir=remote_dir, local_mirror=local_mirror))

            if skip_upload:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] --- upload --- SKIP", file=sys.stderr)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] --- upload ---", file=sys.stderr)
                cmd_upload(Namespace(local_mirror=local_mirror, remote_dir=remote_dir))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] --- clean remote ---", file=sys.stderr)
            cmd_demosaic_clean_remote(Namespace(
                remote_dir=remote_dir, remote_done_dir=remote_done_dir))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] --- clean local ---", file=sys.stderr)
            cmd_clean_local(Namespace(
                local_dir=local_mirror, remote_dir=remote_dir,
                remote_done_dir=remote_done_dir))

        except KeyboardInterrupt:
            print("\nStopping dav.sync...", file=sys.stderr)
            break
        except Exception as e:
            print(f"Sync loop error: {e}", file=sys.stderr)

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync files between local mirror and WebDAV remote."
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # download
    p_dl = sub.add_parser("download", help="Download files from remote to local mirror")
    p_dl.add_argument("remote_dir", help="Remote WebDAV directory")
    p_dl.add_argument("local_mirror", help="Local mirror directory")

    # upload
    p_ul = sub.add_parser("upload", help="Upload .restored.mp4 files to remote")
    p_ul.add_argument("local_mirror", help="Local mirror directory")
    p_ul.add_argument("remote_dir", help="Remote WebDAV directory")

    # demosaic_clean_remote
    p_cr = sub.add_parser("demosaic_clean_remote",
                          help="Move <foo>.mp4 to done dir if <foo>.restored.mp4 exists")
    p_cr.add_argument("remote_dir", help="Remote WebDAV directory")
    p_cr.add_argument("remote_done_dir", help="Remote done directory")

    # clean_local
    p_cl = sub.add_parser("clean_local",
                          help="Clean local files based on remote state")
    p_cl.add_argument("local_dir", help="Local mirror directory")
    p_cl.add_argument("remote_dir", help="Remote WebDAV directory")
    p_cl.add_argument("remote_done_dir", help="Remote done directory")

    # sync (loop)
    p_sync = sub.add_parser("sync",
                            help="Run download/upload/clean in a loop")
    p_sync.add_argument("remote_dir", help="Remote WebDAV directory")
    p_sync.add_argument("local_mirror", help="Local mirror directory")
    p_sync.add_argument("remote_done_dir", help="Remote done directory")
    p_sync.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    p_sync.add_argument("--skip-upload", action="store_true",
                        help="Skip the upload step (also honours DAV_SKIP_UPLOAD=1)")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "demosaic_clean_remote":
        cmd_demosaic_clean_remote(args)
    elif args.command == "clean_local":
        cmd_clean_local(args)
    elif args.command == "sync":
        cmd_sync(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()