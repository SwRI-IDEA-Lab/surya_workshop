#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests


def extract_file_id(url_or_id: str) -> str:
    """
    Accepts either a raw file ID or a Google Drive URL and returns the file ID.
    Supports common URL formats:
      - https://drive.google.com/file/d/<ID>/view?usp=sharing
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
    """
    if re.fullmatch(r"[-\w]{20,}", url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)

    # /file/d/<ID>/...
    m = re.search(r"/file/d/([-\w]+)", parsed.path)
    if m:
        return m.group(1)

    # ?id=<ID>
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    raise ValueError("Could not extract Google Drive file ID from input.")


def get_confirm_token(resp: requests.Response) -> str | None:
    """
    Google sometimes requires a confirmation token for large files.
    Try cookies first, then HTML fallback.
    """
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v

    m = re.search(r'confirm=([0-9A-Za-z_]+)', resp.text)
    if m:
        return m.group(1)

    return None


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def print_progress(downloaded: int, total: int | None) -> None:
    if total and total > 0:
        pct = downloaded / total
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"\r[{bar}] {pct * 100:6.2f}%  "
            f"{format_bytes(downloaded)} / {format_bytes(total)}"
        )
    else:
        msg = f"\rDownloaded {format_bytes(downloaded)}"
    sys.stdout.write(msg)
    sys.stdout.flush()


def save_response_content(resp: requests.Response, destination: Path, chunk_size: int = 32768):
    resp.raise_for_status()

    total = None
    content_length = resp.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        total = int(content_length)

    downloaded = 0
    with destination.open("wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print_progress(downloaded, total)

    sys.stdout.write("\n")
    sys.stdout.flush()


def download_public_drive_file(url_or_id: str, output_path: str):
    file_id = extract_file_id(url_or_id)
    session = requests.Session()

    base_url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}

    # First request
    resp = session.get(base_url, params=params, stream=True)
    token = get_confirm_token(resp)

    # If confirmation token exists, request again with confirm
    if token:
        params["confirm"] = token
        resp = session.get(base_url, params=params, stream=True)

    save_response_content(resp, Path(output_path))
    print(f"Downloaded to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_public_gdrive.py <google_drive_url_or_file_id> <output_path>")
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2]
    download_public_drive_file(source, output)