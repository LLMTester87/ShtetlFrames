from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from .common import _year_from


def _normalize_youtube_hub(url: str) -> str:
    """Prefer /videos tab for @handle / channel /c/ hubs so we list uploads, not tabs."""
    u = url.strip()
    low = u.lower()
    if "youtube.com/watch" in low or "youtu.be/" in low or "/playlist" in low or "list=" in low:
        return u
    if "youtube.com/results" in low:
        return u
    # Already on a tab
    if re.search(r"/(videos|shorts|streams|playlists)(/|$|\?)", low):
        return u.split("?")[0]  # drop tracking query
    # @handle, /channel/UC…, /c/name, /user/name → append /videos
    if re.search(r"youtube\.com/(@[^/?]+|channel/[^/?]+|c/[^/?]+|user/[^/?]+)/?$", low):
        return u.rstrip("/") + "/videos"
    if "/@" in low or "/channel/" in low or "/c/" in low or "/user/" in low:
        # strip trailing slash only
        base = u.split("?")[0].rstrip("/")
        if not re.search(r"/(videos|shorts|streams|playlists)$", base.lower()):
            return base + "/videos"
    return u


def _is_youtube_video_id(vid: str) -> bool:
    return bool(re.fullmatch(r"[\w-]{11}", vid or ""))


def _crawl_ytdlp(url: str, max_items: int, kind: str = "youtube", on_status=None) -> dict:
    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    # Prefer streaming --print so status can update live for large hubs.
    # Fall back to dump-single-json for tab detection / empty stream cases.
    note(f"Starting yt-dlp flat listing (max {max_items:,})…")
    stream = _crawl_ytdlp_stream(url, max_items, kind=kind, on_status=on_status)
    if stream.get("ok") and stream.get("entries"):
        return stream
    if stream.get("ok") is False and stream.get("fatal"):
        return stream

    note("Retrying with JSON listing…")
    return _crawl_ytdlp_json(url, max_items, kind=kind, on_status=on_status)


def _crawl_ytdlp_stream(url: str, max_items: int, kind: str = "youtube", on_status=None) -> dict:
    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--no-download",
        "--ignore-errors",
        "--no-warnings",
        "--playlist-end",
        str(max_items),
        "--print",
        "%(id)s\t%(title)s\t%(url)s\t%(webpage_url)s\t%(_type)s\t%(upload_date|)s",
        url,
    ]
    timeout = max(180, min(3600, 60 + max_items // 20))
    t0 = time.time()
    entries: list[dict] = []
    playlist_tabs = 0
    last_note = 0.0
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as e:
        return {"ok": False, "fatal": True, "error": f"yt_dlp_spawn: {e}", "entries": [], "source_url": url, "kind": kind}

    assert proc.stdout is not None
    try:
        while True:
            if time.time() - t0 > timeout:
                proc.kill()
                note(f"Timed out after {int(time.time() - t0)}s with {len(entries):,} videos listed")
                return {
                    "ok": False,
                    "fatal": True,
                    "error": "yt_dlp_timeout — try a smaller Max",
                    "entries": entries,
                    "source_url": url,
                    "kind": kind,
                }
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                # Heartbeat while waiting for next line
                now = time.time()
                if now - last_note >= 2.0:
                    elapsed = int(now - t0)
                    note(
                        f"Listing… {len(entries):,}/{max_items:,} videos · {elapsed}s elapsed"
                    )
                    last_note = now
                time.sleep(0.05)
                continue

            parts = line.rstrip("\n").split("\t")
            while len(parts) < 6:
                parts.append("")
            vid, title, watch, webpage, etype, upload_date = parts[:6]
            etype = (etype or "").strip().lower()
            if etype in ("playlist", "multi_video"):
                playlist_tabs += 1
                continue
            watch = (watch or webpage or "").strip()
            title = (title or vid or "Untitled")[:200]
            if title in ("[Private video]", "[Deleted video]"):
                continue
            if "youtube" in kind or "youtube.com" in url.lower():
                if not _is_youtube_video_id(vid):
                    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", watch or "")
                    if m:
                        vid = m.group(1)
                    else:
                        continue
                watch = f"https://www.youtube.com/watch?v={vid}"
            else:
                if not watch.startswith("http"):
                    continue
            entries.append(
                {
                    "url": watch.split("&list=")[0],
                    "title": title,
                    "year": _year_from(upload_date),
                    "identifier": vid,
                    "source": "YouTube (crawl)" if "youtube" in kind else f"{kind} (crawl)",
                    "downloadable": "yes",
                    "notes": f"Crawled from {urlparse(url).netloc}{urlparse(url).path[:60]}",
                }
            )
            now = time.time()
            if len(entries) == 1 or len(entries) % 50 == 0 or now - last_note >= 2.0:
                note(f"Listed {len(entries):,}/{max_items:,} · {int(now - t0)}s")
                last_note = now
            if len(entries) >= max_items:
                try:
                    proc.kill()
                except Exception:
                    pass
                break

        # Drain briefly
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    stderr = ""
    try:
        if proc.stderr:
            stderr = proc.stderr.read()[-800:]
    except Exception:
        pass

    if not entries:
        # Channel homepage may only emit playlist tabs — let JSON path recurse
        if playlist_tabs:
            note(f"Saw {playlist_tabs} playlist tab(s), no video rows yet — trying /videos…")
            videos_url = _normalize_youtube_hub(url)
            if videos_url != url and "/videos" in videos_url:
                return _crawl_ytdlp_stream(videos_url, max_items, kind=kind, on_status=on_status)
        if proc.returncode not in (0, None, -9, 1) and stderr:
            return {
                "ok": False,
                "fatal": True,
                "error": f"yt_dlp: {stderr}",
                "entries": [],
                "source_url": url,
                "kind": kind,
            }
        return {"ok": False, "entries": [], "source_url": url, "kind": kind}

    note(f"Listing complete — {len(entries):,} videos in {int(time.time() - t0)}s")
    return {
        "ok": True,
        "kind": kind,
        "source_url": url,
        "entries": entries[:max_items],
        "truncated": len(entries) >= max_items,
        "n_found": len(entries),
    }


def _crawl_ytdlp_json(url: str, max_items: int, kind: str = "youtube", on_status=None) -> dict:
    def note(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--no-download",
        "--ignore-errors",
        "--playlist-end",
        str(max_items),
        url,
    ]
    timeout = max(180, min(3600, 60 + max_items // 20))
    t0 = time.time()
    note(f"yt-dlp JSON dump (max {max_items:,}, timeout {timeout}s)…")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        bucket: list[str | None] = [None, None]

        def _consume() -> None:
            assert proc.stdout is not None and proc.stderr is not None
            bucket[0] = proc.stdout.read()
            bucket[1] = proc.stderr.read()
            proc.wait()

        reader = threading.Thread(target=_consume, daemon=True)
        reader.start()
        while reader.is_alive():
            elapsed = int(time.time() - t0)
            if elapsed > timeout:
                proc.kill()
                note(f"JSON listing timed out after {elapsed}s")
                return {
                    "ok": False,
                    "error": "yt_dlp_timeout — try a smaller Max",
                    "entries": [],
                    "source_url": url,
                    "kind": kind,
                }
            note(f"Waiting on yt-dlp… {elapsed}s / {timeout}s (still crawling)")
            reader.join(timeout=2.5)
        raw = bucket[0] or ""
        err_out = bucket[1] or ""
    except Exception as e:
        return {"ok": False, "error": f"yt_dlp: {e}", "entries": [], "source_url": url, "kind": kind}

    if proc.returncode != 0:
        err = (err_out or raw or "yt-dlp failed")[-500:]
        note(f"yt-dlp failed (code {proc.returncode})")
        return {"ok": False, "error": f"yt_dlp: {err}", "entries": [], "source_url": url, "kind": kind}

    raw = (raw or "").strip()
    if not raw:
        return {"ok": False, "error": "yt_dlp_empty", "entries": [], "source_url": url, "kind": kind}

    note("Parsing yt-dlp JSON…")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {"ok": False, "error": "yt_dlp_bad_json", "entries": [], "source_url": url, "kind": kind}
        data = json.loads(raw[start : end + 1])

    entries_raw = data.get("entries") or []
    if entries_raw and all((e or {}).get("_type") == "playlist" for e in entries_raw if e):
        videos_url = _normalize_youtube_hub(url)
        if videos_url != url and "/videos" in videos_url:
            note(f"Channel tabs detected — switching to {videos_url}")
            return _crawl_ytdlp(videos_url, max_items, kind=kind, on_status=on_status)
        for e in entries_raw:
            if not e:
                continue
            title = (e.get("title") or "").lower()
            if "video" in title:
                tab = e.get("url") or e.get("webpage_url")
                if tab:
                    note(f"Opening Videos tab…")
                    return _crawl_ytdlp(tab, max_items, kind=kind, on_status=on_status)

    if not entries_raw and data.get("id") and data.get("_type") != "playlist":
        entries_raw = [data]

    entries = []
    for e in entries_raw:
        if not e:
            continue
        if e.get("_type") in ("playlist", "multi_video"):
            continue
        vid = str(e.get("id") or "")
        watch = e.get("url") or e.get("webpage_url") or ""
        if "youtube" in kind or "youtube.com" in url.lower():
            if not _is_youtube_video_id(vid):
                m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", watch or "")
                if m:
                    vid = m.group(1)
                else:
                    continue
            watch = f"https://www.youtube.com/watch?v={vid}"
        else:
            if watch and not watch.startswith("http"):
                continue
            if not watch.startswith("http"):
                continue
        title = (e.get("title") or vid or "Untitled")[:200]
        if title in ("[Private video]", "[Deleted video]"):
            continue
        entries.append(
            {
                "url": watch.split("&list=")[0],
                "title": title,
                "year": _year_from(e.get("upload_date") or e.get("release_year") or ""),
                "identifier": vid,
                "source": "YouTube (crawl)" if "youtube" in kind else f"{kind} (crawl)",
                "downloadable": "yes",
                "notes": f"Crawled from {urlparse(url).netloc}{urlparse(url).path[:60]}",
            }
        )
        if len(entries) >= max_items:
            break
        if len(entries) % 500 == 0:
            note(f"Parsed {len(entries):,} / {max_items:,}")

    if not entries:
        return {
            "ok": False,
            "error": "no_videos_found — for channels use …/@name/videos",
            "entries": [],
            "source_url": url,
            "kind": kind,
        }

    truncated = bool(data.get("playlist_count") and int(data.get("playlist_count") or 0) > len(entries))
    if not truncated and len(entries_raw) >= max_items:
        truncated = True

    note(f"Parsed {len(entries):,} videos from JSON")
    return {
        "ok": True,
        "kind": kind,
        "source_url": url,
        "entries": entries,
        "truncated": truncated,
        "n_found": len(entries),
    }
