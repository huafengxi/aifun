#!/home/yuanqi.xhf/miniconda3/bin/python
"""
pikpak_client.py — PikPak API client with SOCKS5 proxy support.

Replaces alist dependency for essential PikPak operations:
- login (token management)
- list files
- upload files
- download files
- delete files

Usage:
    source ../env/pikpak.web
    ./pikpak_client.py list [/path]
    ./pikpak_client.py upload <local_file> <remote_path>
    ./pikpak_client.py download <remote_path> [local_path]
    ./pikpak_client.py delete <remote_path>

Proxy:
    export SOCKS_PROXY=socks5://127.0.0.1:7897
    export HTTPS_PROXY=http://127.0.0.1:7897
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIENT_ID = "YUMx5nI8ZU8Ap8pm"
CLIENT_SECRET = ""
API_HOST = "https://api-drive.mypikpak.net"
AUTH_HOST = "https://user.mypikpak.net"
USER_AGENT = "PikPakClient/1.0"

TOKEN_FILE = os.path.expanduser("~/.pikpak_token.json")


# ---------------------------------------------------------------------------
# Proxy support
# ---------------------------------------------------------------------------

def _get_proxies():
    """Get proxy settings from environment."""
    proxies = {}
    for env_var in ["SOCKS_PROXY", "socks_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
        val = os.environ.get(env_var)
        if val:
            if "socks" in env_var.lower():
                proxies["http"] = val
                proxies["https"] = val
            elif "https" in env_var.lower():
                proxies["https"] = val
            elif "http" in env_var.lower():
                proxies["http"] = val
    return proxies if proxies else None


def _get_session():
    """Create a requests session with proxy support."""
    session = requests.Session()
    proxies = _get_proxies()
    if proxies:
        session.proxies.update(proxies)
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    })
    return session


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _load_token():
    """Load cached token from file."""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_token(token_data):
    """Save token to cache file."""
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f)


def _get_device_id():
    """Generate a deterministic device ID."""
    hostname = os.uname().nodename
    return hashlib.md5(f"pikpak-{hostname}".encode()).hexdigest()


def login(username=None, password=None, refresh_token=None):
    """Login to PikPak and return access token.
    
    Tries refresh_token first, then username/password.
    """
    session = _get_session()
    device_id = _get_device_id()

    # Try refresh token
    token = refresh_token or _load_token().get("refresh_token")
    if token:
        try:
            resp = session.post(
                f"{AUTH_HOST}/v1/auth/token",
                params={"client_id": CLIENT_ID},
                json={
                    "refresh_token": token,
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                data["device_id"] = device_id
                _save_token(data)
                return data.get("access_token")
        except Exception:
            pass

    # Login with username/password
    if not username or not password:
        username = os.environ.get("PIKPAK_USERNAME", "")
        password = os.environ.get("PIKPAK_PASSWORD", "")

    if not username or not password:
        raise RuntimeError(
            "No valid token and no PIKPAK_USERNAME/PIKPAK_PASSWORD set. "
            "Source ../env/pikpak.web or set env vars."
        )

    resp = session.post(
        f"{AUTH_HOST}/v1/auth/token",
        params={"client_id": CLIENT_ID},
        json={
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "device_id": device_id,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Login failed: {resp.status_code} {resp.text}")

    data = resp.json()
    data["device_id"] = device_id
    _save_token(data)
    return data.get("access_token")


def get_access_token():
    """Get a valid access token, refreshing if needed."""
    token_data = _load_token()
    access_token = token_data.get("access_token")
    if access_token:
        return access_token
    return login()


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def _api_request(method, path, **kwargs):
    """Make an authenticated API request."""
    token = get_access_token()
    session = _get_session()
    session.headers["Authorization"] = f"Bearer {token}"

    url = f"{API_HOST}{path}"
    resp = session.request(method, url, timeout=120, **kwargs)

    if resp.status_code == 401:
        # Token expired, re-login and retry
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
        token = login()
        session.headers["Authorization"] = f"Bearer {token}"
        resp = session.request(method, url, timeout=120, **kwargs)

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    return resp.json()


def list_files(path="/", parent_id=None, limit=100, page_token=None):
    """List files in a directory.
    
    Args:
        path: Directory path (e.g., "/" or "/videos")
        parent_id: Parent folder ID (alternative to path)
        limit: Max files per page
        page_token: Pagination token
    
    Returns:
        dict with 'files' list and optional 'next_page_token'
    """
    params = {
        "parent_id": parent_id or "",
        "page_token": page_token or "",
        "limit": limit,
        "thumbnail_size": "SIZE_LARGE",
        "with_audit": "true",
    }
    
    # Build filters
    filters = {
        "phase": {"eq": "PHASE_TYPE_COMPLETE"},
        "trashed": {"eq": False},
    }
    
    params["filters"] = json.dumps(filters)
    
    result = _api_request("GET", "/drive/v1/files", params=params)
    
    files = []
    for f in result.get("files", []):
        files.append({
            "id": f.get("id"),
            "name": f.get("name"),
            "kind": f.get("kind"),  # "drive#file" or "drive#folder"
            "size": int(f.get("size", 0)),
            "mime_type": f.get("mime_type", ""),
            "created_time": f.get("created_time"),
            "modified_time": f.get("modified_time"),
            "parent_id": f.get("parent_id"),
            "md5_checksum": f.get("md5_checksum"),
        })
    
    return {
        "files": files,
        "next_page_token": result.get("next_page_token"),
    }


def _get_folder_id(path):
    """Resolve a path to a folder ID by walking the directory tree."""
    if path == "/" or path == "":
        return ""
    
    parts = [p for p in path.strip("/").split("/") if p]
    current_parent = ""
    
    for part in parts:
        found = False
        page_token = None
        while True:
            result = list_files(parent_id=current_parent, page_token=page_token, limit=100)
            for f in result["files"]:
                if f["kind"] == "drive#folder" and f["name"] == part:
                    current_parent = f["id"]
                    found = True
                    break
            if found:
                break
            page_token = result.get("next_page_token")
            if not page_token:
                break
        
        if not found:
            raise RuntimeError(f"Folder not found: '{part}' in path '{path}'")
    
    return current_parent


def upload_file(local_path, remote_path):
    """Upload a local file to PikPak.
    
    Args:
        local_path: Path to local file
        remote_path: Remote path (e.g., "/videos/myfile.mp4")
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise RuntimeError(f"File not found: {local_path}")
    
    file_name = local_path.name
    file_size = local_path.stat().st_size
    
    # Determine parent folder
    remote_parent = str(Path(remote_path).parent)
    parent_id = _get_folder_id(remote_parent)
    
    # Calculate SHA1 hash of first 1MB (PikPak requirement)
    sha1 = hashlib.sha1()
    with open(local_path, "rb") as f:
        sha1.update(f.read(1024 * 1024))
    file_hash = sha1.hexdigest()
    
    # Step 1: Request upload URL
    token = get_access_token()
    session = _get_session()
    session.headers["Authorization"] = f"Bearer {token}"
    
    upload_req = {
        "kind": "drive#file",
        "name": file_name,
        "size": file_size,
        "hash": file_hash,
        "upload_type": "UPLOAD_TYPE_RESUMABLE",
        "parent_id": parent_id,
    }
    
    resp = session.post(
        f"{API_HOST}/drive/v1/files",
        params={"upload_type": "resumable"},
        json=upload_req,
        timeout=30,
    )
    
    if resp.status_code != 200:
        raise RuntimeError(f"Upload request failed: {resp.status_code} {resp.text}")
    
    upload_info = resp.json()
    upload_url = upload_info.get("upload_url") or upload_info.get("resumable", {}).get("upload_url")
    
    if not upload_url:
        raise RuntimeError(f"No upload URL in response: {upload_info}")
    
    # Step 2: Upload file content
    print(f"Uploading {file_name} ({file_size} bytes) ...", file=sys.stderr)
    
    with open(local_path, "rb") as f:
        upload_resp = session.put(
            upload_url,
            data=f,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(file_size),
            },
            timeout=600,
        )
    
    if upload_resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Upload failed: {upload_resp.status_code} {upload_resp.text}")
    
    print(f"Upload complete: {file_name}", file=sys.stderr)
    return upload_info.get("id") or upload_info.get("file", {}).get("id")


def download_file(remote_path, local_path=None):
    """Download a file from PikPak.
    
    Args:
        remote_path: Remote path (e.g., "/videos/myfile.mp4")
        local_path: Local save path (default: current dir with original name)
    """
    # Find the file
    parent_path = str(Path(remote_path).parent)
    file_name = Path(remote_path).name
    parent_id = _get_folder_id(parent_path)
    
    file_id = None
    file_size = 0
    page_token = None
    while True:
        result = list_files(parent_id=parent_id, page_token=page_token)
        for f in result["files"]:
            if f["kind"] == "drive#file" and f["name"] == file_name:
                file_id = f["id"]
                file_size = f["size"]
                break
        if file_id:
            break
        page_token = result.get("next_page_token")
        if not page_token:
            break
    
    if not file_id:
        raise RuntimeError(f"File not found: {remote_path}")
    
    # Get download URL
    token = get_access_token()
    session = _get_session()
    session.headers["Authorization"] = f"Bearer {token}"
    
    resp = session.get(
        f"{API_HOST}/drive/v1/files/{file_id}",
        params={"usage": "download"},
        timeout=30,
    )
    
    if resp.status_code != 200:
        raise RuntimeError(f"Get download URL failed: {resp.status_code} {resp.text}")
    
    file_info = resp.json()
    download_url = file_info.get("web_content_link") or file_info.get("download_url")
    
    if not download_url:
        raise RuntimeError(f"No download URL in response: {file_info}")
    
    # Download
    local_path = local_path or file_name
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {file_name} ({file_size} bytes) -> {local_path}", file=sys.stderr)
    
    dl_resp = session.get(download_url, timeout=600, stream=True)
    if dl_resp.status_code != 200:
        raise RuntimeError(f"Download failed: {dl_resp.status_code}")
    
    downloaded = 0
    with open(local_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if file_size > 0:
                pct = downloaded * 100 // file_size
                print(f"\r  {downloaded}/{file_size} ({pct}%)", end="", file=sys.stderr)
    
    print(file=sys.stderr)
    print(f"Download complete: {local_path}", file=sys.stderr)
    return str(local_path)


def delete_file(remote_path):
    """Delete a file from PikPak."""
    parent_path = str(Path(remote_path).parent)
    file_name = Path(remote_path).name
    parent_id = _get_folder_id(parent_path)
    
    file_id = None
    page_token = None
    while True:
        result = list_files(parent_id=parent_id, page_token=page_token)
        for f in result["files"]:
            if f["name"] == file_name:
                file_id = f["id"]
                break
        if file_id:
            break
        page_token = result.get("next_page_token")
        if not page_token:
            break
    
    if not file_id:
        raise RuntimeError(f"File not found: {remote_path}")
    
    _api_request("DELETE", f"/drive/v1/files/{file_id}")
    print(f"Deleted: {remote_path}", file=sys.stderr)


def create_folder(path):
    """Create a folder on PikPak."""
    parent_path = str(Path(path).parent)
    folder_name = Path(path).name
    parent_id = _get_folder_id(parent_path)
    
    _api_request("POST", "/drive/v1/files", json={
        "kind": "drive#folder",
        "name": folder_name,
        "parent_id": parent_id,
    })
    print(f"Created folder: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_size(size):
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _format_time(ts):
    """Format ISO time string."""
    if not ts:
        return ""
    try:
        # Parse ISO 8601
        dt = ts[:19].replace("T", " ")
        return dt
    except Exception:
        return ts


def cmd_list(args):
    """List files command."""
    path = args.path or "/"
    print(f"Listing: {path}", file=sys.stderr)
    print(file=sys.stderr)
    
    parent_id = _get_folder_id(path) if path != "/" else ""
    page_token = None
    total = 0
    
    print(f"{'Type':<6} {'Size':>10} {'Modified':<20} {'Name'}")
    print("-" * 80)
    
    while True:
        result = list_files(parent_id=parent_id, page_token=page_token)
        for f in result["files"]:
            kind = "DIR" if f["kind"] == "drive#folder" else "FILE"
            size = _format_size(f["size"]) if f["kind"] == "drive#file" else "-"
            mtime = _format_time(f.get("modified_time", ""))
            print(f"{kind:<6} {size:>10} {mtime:<20} {f['name']}")
            total += 1
        
        page_token = result.get("next_page_token")
        if not page_token:
            break
    
    print(f"\n{total} items", file=sys.stderr)


def cmd_upload(args):
    """Upload file command."""
    upload_file(args.local_file, args.remote_path)


def cmd_download(args):
    """Download file command."""
    download_file(args.remote_path, args.local_path)


def cmd_delete(args):
    """Delete file command."""
    delete_file(args.remote_path)


def cmd_mkdir(args):
    """Create folder command."""
    create_folder(args.path)


def main():
    parser = argparse.ArgumentParser(
        description="PikPak API client with SOCKS proxy support"
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # list
    p_list = sub.add_parser("list", help="List files in a directory")
    p_list.add_argument("path", nargs="?", default="/", help="Directory path (default: /)")

    # upload
    p_upload = sub.add_parser("upload", help="Upload a file")
    p_upload.add_argument("local_file", help="Local file path")
    p_upload.add_argument("remote_path", help="Remote path (e.g., /videos/myfile.mp4)")

    # download
    p_download = sub.add_parser("download", help="Download a file")
    p_download.add_argument("remote_path", help="Remote file path")
    p_download.add_argument("local_path", nargs="?", help="Local save path (default: current dir)")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a file")
    p_delete.add_argument("remote_path", help="Remote file path")

    # mkdir
    p_mkdir = sub.add_parser("mkdir", help="Create a folder")
    p_mkdir.add_argument("path", help="Folder path")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "mkdir":
        cmd_mkdir(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()