"""Download public seed / discovery videos with provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

from config import DATA_DIR, VIDEOS_DIR, YT_COOKIES_BROWSER, YT_PLAYER_CLIENTS
from shtetl_core.textutil import slugify

USER_AGENT = "ShtetlFrames/1.0 (research; respectful archival use)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_http(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1024 * 256):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
    return dest


def _ytdlp_base_cmd(
    pattern: str,
    url: str,
    *,
    player_client: str | None,
    use_cookies: bool,
    proxy_url: str | None = None,
    proxy_insecure: bool = False,
    referer: str | None = None,
    format_selector: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        format_selector or "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format",
        "mp4",
        "-o",
        pattern,
        "--no-playlist",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--sleep-requests",
        "1",
        "--newline",
        "-4",
    ]
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
        if proxy_insecure:
            cmd.append("--no-check-certificates")
    if referer:
        cmd.extend(["--referer", referer])
        try:
            origin = f"{urlparse(referer).scheme}://{urlparse(referer).netloc}"
            if origin.startswith("http"):
                cmd.extend(["--add-header", f"Origin:{origin}"])
        except Exception:
            pass
    if player_client:
        cmd.extend(["--extractor-args", f"youtube:player_client={player_client}"])
    if use_cookies:
        # Prefer a pre-exported Netscape jar (more reliable than live Chrome DB on Windows).
        # Cookies + residential proxy together helps YouTube bot-checks.
        try:
            from yt_cookies import cookies_path, read_cookies_text

            jar = cookies_path()
            if jar.is_file() and read_cookies_text():
                cmd.extend(["--cookies", str(jar)])
            elif YT_COOKIES_BROWSER and YT_COOKIES_BROWSER not in ("none", "off", "0"):
                cmd.extend(["--cookies-from-browser", YT_COOKIES_BROWSER])
        except Exception:
            if YT_COOKIES_BROWSER and YT_COOKIES_BROWSER not in ("none", "off", "0"):
                cmd.extend(["--cookies-from-browser", YT_COOKIES_BROWSER])
    cmd.append(url)
    return cmd


def _ytdlp_try_attempts(
    url: str,
    dest_dir: Path,
    out_name: str,
    *,
    attempts: list[tuple[str | None, bool]],
    proxy_url: str | None = None,
    proxy_insecure: bool = False,
    referer: str | None = None,
    format_selector: str | None = None,
) -> tuple[Path | None, str]:
    pattern = str(dest_dir / f"{out_name}.%(ext)s")
    last_err = ""
    seen: set[tuple[str | None, bool]] = set()
    for client, use_cookies in attempts:
        key = (client, use_cookies)
        if key in seen:
            continue
        seen.add(key)
        cmd = _ytdlp_base_cmd(
            pattern,
            url,
            player_client=client,
            use_cookies=use_cookies,
            proxy_url=proxy_url,
            proxy_insecure=proxy_insecure,
            referer=referer,
            format_selector=format_selector,
        )
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            last_err = "yt-dlp_timeout"
            continue
        matches = [
            p
            for p in dest_dir.glob(f"{out_name}.*")
            if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"} and p.stat().st_size > 0
        ]
        if proc.returncode == 0 and matches:
            return matches[0], ""
        err = (proc.stderr or proc.stdout or "")[-400:]
        last_err = err.strip() or f"exit_{proc.returncode}"
        if use_cookies and any(
            x in last_err.lower()
            for x in ("could not copy", "could not find", "failed to load cookies", "unable to find")
        ):
            continue
        if matches:
            return matches[0], ""
    return None, last_err


def _youtube_attempts(*, with_cookies: bool) -> list[tuple[str | None, bool]]:
    attempts: list[tuple[str | None, bool]] = []
    if with_cookies:
        for client in ("tv", "web", "mweb", "tv_embedded"):
            attempts.append((client, True))
        attempts.append((None, True))
    for client in YT_PLAYER_CLIENTS:
        attempts.append((client, False))
    attempts.append((None, False))
    return attempts


def _proxy_alive(proxy_url: str, timeout: float = 12.0) -> bool:
    """Quick probe — fail fast if the residential proxy is dead."""
    try:
        r = requests.get(
            "https://ipv4.icanhazip.com",
            proxies={"http": proxy_url, "https": proxy_url},
            timeout=timeout,
        )
        return r.status_code == 200 and bool((r.text or "").strip())
    except Exception:
        return False


def download_ytdlp_via_proxy(url: str, dest_dir: Path, out_name: str) -> Path | None:
    """Force residential proxy (Scrapfly / ScrapingDog) after a Google/YouTube block."""
    from yt_proxy import (
        proxy_configured,
        proxy_needs_insecure_ssl,
        proxy_provider_name,
        residential_proxy_url,
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not proxy_configured():
        return None
    proxy = residential_proxy_url()
    label = proxy_provider_name()
    if not proxy:
        return None
    if not _proxy_alive(proxy):
        print(f"yt-dlp: {label} proxy not responding — skipping", flush=True)
        (dest_dir / f"{out_name}.ytdlp_error.txt").write_text(
            f"{label}: proxy_timeout_or_dead", encoding="utf-8"
        )
        return None
    print(f"yt-dlp: downloading via {label} residential…", flush=True)
    # Cookies + residential IP is the intended combo for YouTube.
    sf_attempts: list[tuple[str | None, bool]] = [
        ("tv", True),
        ("web", True),
        ("mweb", True),
        ("android", False),
        (None, True),
    ]
    path, err = _ytdlp_try_attempts(
        url,
        dest_dir,
        out_name,
        attempts=sf_attempts,
        proxy_url=proxy,
        proxy_insecure=proxy_needs_insecure_ssl(),
    )
    if path:
        return path
    (dest_dir / f"{out_name}.ytdlp_error.txt").write_text(
        f"{label}: {err}"[-2000:], encoding="utf-8"
    )
    return None


# Back-compat alias


def download_ytdlp(
    url: str,
    dest_dir: Path,
    out_name: str,
    *,
    allow_proxy: bool = True,
) -> Path | None:
    """Download via yt-dlp; residential proxy only if Google/YouTube bot-blocks."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.glob(f"{out_name}.*"))
    existing = [p for p in existing if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}]
    if existing:
        return existing[0]

    is_youtube = "youtube.com" in url.lower() or "youtu.be" in url.lower()
    attempts = _youtube_attempts(with_cookies=True) if is_youtube else [(None, False)]

    path, last_err = _ytdlp_try_attempts(url, dest_dir, out_name, attempts=attempts)
    if path:
        return path

    # Only use residential proxy when Google/YouTube actually blocked us.
    from yt_proxy import is_google_block_error, proxy_configured

    if (
        allow_proxy
        and is_youtube
        and proxy_configured()
        and is_google_block_error(last_err)
    ):
        path = download_ytdlp_via_proxy(url, dest_dir, out_name)
        if path:
            return path
        side = dest_dir / f"{out_name}.ytdlp_error.txt"
        px_err = side.read_text(encoding="utf-8", errors="ignore") if side.exists() else ""
        last_err = f"{last_err} | {px_err}"[-2000:]

    print(f"yt-dlp failed for {url}: {last_err[-500:]}")
    (dest_dir / f"{out_name}.ytdlp_error.txt").write_text(last_err[-2000:], encoding="utf-8")
    return None


def download_archive_org(identifier: str, dest_dir: Path) -> Path | None:
    """Download best video file from an Internet Archive item."""
    meta_url = f"https://archive.org/metadata/{identifier}"
    r = requests.get(meta_url, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    meta = r.json()
    files = meta.get("files") or []
    video_exts = {".mp4", ".webm", ".ogv", ".avi", ".mkv", ".mpg", ".mpeg"}
    candidates = []
    for f in files:
        name = f.get("name", "")
        ext = Path(name).suffix.lower()
        if ext in video_exts and not name.endswith(".thumbs"):
            size = int(f.get("size") or 0)
            candidates.append((size, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    name = candidates[0][1]
    url = f"https://archive.org/download/{identifier}/{name}"
    dest = dest_dir / f"ia_{slugify(identifier)}_{Path(name).name}"
    return download_http(url, dest)


def is_direct_video_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".webm", ".mp4", ".ogv", ".mkv", ".avi", ".mov"))


def download_britishpathe(
    url: str,
    dest_dir: Path,
    out_name: str,
    *,
    title: str = "",
) -> Path | None:
    """Resolve Pathé asset (or YT Pathé title) → HLS via yt-dlp + Referer (no residential proxy)."""
    from britishpathe import prepare_pathe_job

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        p
        for p in dest_dir.glob(f"{out_name}.*")
        if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    ]
    if existing:
        return existing[0]

    job = prepare_pathe_job(url, title or out_name)
    if not job:
        return None
    path, err = _ytdlp_try_attempts(
        job["download_url"],
        dest_dir,
        out_name,
        attempts=[(None, False)],
        referer=job.get("referer"),
        format_selector="bv*[height<=480]+ba/b[height<=480]/b",
    )
    if path:
        return path
    (dest_dir / f"{out_name}.ytdlp_error.txt").write_text(
        f"britishpathe: {err}"[-2000:], encoding="utf-8"
    )
    return None


def download_entry(url: str, title: str, video_id: str | None = None) -> dict:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    vid = video_id or slugify(title)
    meta_path = VIDEOS_DIR / f"{vid}.meta.json"
    result = {"video_id": vid, "url": url, "title": title, "path": None, "sha256": None, "error": None}

    try:
        if is_direct_video_url(url):
            ext = Path(urlparse(url).path).suffix or ".mp4"
            dest = VIDEOS_DIR / f"{vid}{ext}"
            path = download_http(url, dest)
        elif "archive.org/details/" in url:
            identifier = url.rstrip("/").split("/")[-1].split("?")[0]
            path = download_archive_org(identifier, VIDEOS_DIR)
            if path is None:
                raise RuntimeError(f"No video files for IA item {identifier}")
        elif "archive.org/download/" in url:
            # Direct file from crawl multifile expansion
            path_name = Path(urlparse(url).path).name or "ia_file.mp4"
            dest = VIDEOS_DIR / f"{vid}_{path_name}"
            path = download_http(url, dest)
        elif "youtube.com/watch" in url or "youtu.be/" in url:
            path = download_ytdlp(url, VIDEOS_DIR, vid)
            if path is None:
                err_side = VIDEOS_DIR / f"{vid}.ytdlp_error.txt"
                detail = ""
                if err_side.exists():
                    detail = ": " + err_side.read_text(encoding="utf-8", errors="ignore")[:500]
                raise RuntimeError(f"yt-dlp returned no file{detail}")
        else:
            result["error"] = "not_a_direct_download_url"
            meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        result["path"] = str(path)
        result["sha256"] = sha256_file(path)
    except Exception as e:
        result["error"] = str(e)

    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import csv

    seed = DATA_DIR / "seed_videos.csv"
    rows = list(csv.DictReader(seed.open(encoding="utf-8")))
    downloaded = []
    for i, row in enumerate(rows):
        url = (row.get("url") or "").strip()
        title = (row.get("title") or f"item_{i}").strip()
        # Skip search/results pages and catalog-only pointers without direct media
        if any(
            x in url
            for x in (
                "youtube.com/results",
                "archive.org/search",
                "commons.wikimedia.org/wiki/Category",
                "commons.wikimedia.org/w/index.php",
                "resources.ushmm.org",
                "jfc.org.il",
                "digital.tcl.sc.edu",
                "catalog.archives.gov",
                "dp.la/",
                "encyclopedia.ushmm.org",
            )
        ):
            print(f"SKIP catalog/search: {title[:60]}")
            continue
        print(f"GET {title[:70]}...")
        vid = slugify(title)
        # Stable IDs for calibration
        if "upload.wikimedia.org" in url and "Chofetz" in url:
            vid = "agudah_1923_commons"
        elif "rp1OeIf0D0w" in url:
            vid = "munkacs_1933_yt"
        elif "VOD5ztsIqao" in url or "87XlDRjmPME" in url:
            vid = slugify(title)  # mirrors; scanned separately if downloaded
        info = download_entry(url, title, video_id=vid)
        downloaded.append(info)
        print(f"  -> {info.get('path') or info.get('error')}")

    out = DATA_DIR / "download_manifest.json"
    out.write_text(json.dumps(downloaded, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
