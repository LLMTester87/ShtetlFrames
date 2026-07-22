"""Smoke-test British Pathé resolve + short HLS download (no YouTube proxy).

Usage (from repo root):
  .venv\\Scripts\\python.exe scripts/probe_pathe_hls.py
  .venv\\Scripts\\python.exe scripts/probe_pathe_hls.py --asset 197462
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="197462", help="British Pathé asset id")
    args = ap.parse_args()

    from britishpathe import asset_page_url, prepare_pathe_job, resolve_asset

    asset_url = asset_page_url(args.asset)
    print(f"resolve {asset_url}")
    resolved = resolve_asset(asset_url)
    print(
        f"ok asset={resolved['asset_id']} cached={resolved.get('cached')} "
        f"title={resolved.get('title', '')[:80]!r}"
    )
    print(f"m3u8={resolved['m3u8_url'][:100]}…")

    job = prepare_pathe_job(asset_url, resolved.get("title") or "")
    assert job and job["m3u8_url"]
    assert job.get("source") == "britishpathe"

    import subprocess
    import urllib.request

    # Prove CDN accepts Referer (no residential proxy).
    req = urllib.request.Request(
        job["download_url"],
        headers={
            "User-Agent": "ShtetlFrames/1.0",
            "Referer": job["referer"],
            "Origin": "https://www.britishpathe.com",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", "replace")
    if "#EXTM3U" not in body or "m3u8" not in body.lower():
        print("FAIL master playlist unexpected")
        print(body[:300])
        return 1
    print(f"master playlist ok ({len(body)} bytes)")

    # List formats via yt-dlp (no ffmpeg required).
    cmd_f = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-F",
        "--referer",
        job["referer"],
        "--add-header",
        "Origin:https://www.britishpathe.com",
        "-4",
        job["download_url"],
    ]
    print("yt-dlp -F (no proxy)…")
    proc_f = subprocess.run(cmd_f, capture_output=True, text=True, timeout=90)
    out_f = (proc_f.stdout or "") + (proc_f.stderr or "")
    print(out_f[-600:])
    if proc_f.returncode != 0 or ("240p" not in out_f and "480p" not in out_f and "mp4" not in out_f.lower()):
        print("FAIL yt-dlp could not list Pathé formats")
        return 1

    outdir = Path(tempfile.gettempdir()) / "pathe_hls_probe"
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob("probe.*"):
        try:
            old.unlink()
        except OSError:
            pass
    out = outdir / "probe.%(ext)s"

    # Full 240p download — no proxy (same path the GPU worker uses).
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bv*[height<=240]+ba/b[height<=240]/b",
        "--merge-output-format",
        "mp4",
        "--referer",
        job["referer"],
        "--add-header",
        "Origin:https://www.britishpathe.com",
        "-o",
        str(out),
        "--no-playlist",
        "-4",
        job["download_url"],
    ]
    print("yt-dlp HLS 240p (no proxy)…")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    files = [
        p
        for p in outdir.glob("probe.*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm"} and p.stat().st_size > 50_000
    ]
    print(f"yt-dlp rc={proc.returncode} files={[(p.name, p.stat().st_size) for p in files]}")
    if files:
        print("PASS — Pathé HLS download without YouTube proxy")
        return 0
    print((proc.stderr or proc.stdout or "")[-900:])
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
