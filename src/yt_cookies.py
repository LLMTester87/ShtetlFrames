"""Export YouTube cookies from a local browser for yt-dlp bot-check bypass.

On Windows, Edge locks its Cookies SQLite DB while running (yt-dlp #7271).
yt-dlp's error text still says "Chrome" even when the browser is Edge.
Export needs Edge fully quit, or use Scrapfly/ScrapingDog residential for Google bot-checks instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from config import DATA_DIR, ROOT

COOKIES_PATH = DATA_DIR / "youtube_cookies.txt"
_export_lock = threading.Lock()
_last_export_ok = 0.0
_last_export_err = ""

_CHROMIUM_ROOTS = (
    ("edge", "Microsoft/Edge/User Data"),
    ("chrome", "Google/Chrome/User Data"),
    ("brave", "BraveSoftware/Brave-Browser/User Data"),
    ("opera", "Opera Software/Opera Stable"),
)


def cookies_browser() -> str:
    try:
        from config import load_env

        load_env()
    except Exception:
        pass
    browser = (os.environ.get("YT_COOKIES_BROWSER") or "edge").strip().lower()
    if browser in ("", "0", "none", "off", "false"):
        return ""
    return browser


def cookies_path() -> Path:
    custom = (os.environ.get("YT_COOKIES_FILE") or "").strip()
    if custom:
        return Path(custom)
    return COOKIES_PATH


def cookies_look_valid(text: str) -> bool:
    t = text or ""
    if len(t) < 40:
        return False
    low = t.lower()
    return "youtube.com" in low or ".google.com" in low


def read_cookies_text() -> str | None:
    path = cookies_path()
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return text if cookies_look_valid(text) else None


def _host_is_youtube_family(host: str) -> bool:
    h = (host or "").strip().lower().lstrip(".")
    if not h:
        return False
    exact = (
        "youtube.com",
        "youtu.be",
        "youtubekids.com",
        "googlevideo.com",
        "ytimg.com",
        "ggpht.com",
        "googleapis.com",
        "gstatic.com",
        "google.com",
    )
    for n in exact:
        if h == n or h.endswith("." + n):
            return True
    # google.co.uk, google.de, …
    if h.startswith("google.") or ".google." in h:
        return True
    return False


def _normalize_cookie_domain(domain: str, fallback_host: str = "") -> str:
    d = (domain or "").strip().lower()
    if not d:
        d = (fallback_host or "").strip().lower()
    d = d.lstrip(".")
    if not d:
        return ""
    # Prefer leading-dot form for host-wide cookies (yt-dlp / Netscape).
    if d.count(".") >= 1:
        return "." + d
    return d


def _cookie_expiry(raw) -> int:
    if raw is None or raw == "":
        return 0
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return 0
    # HAR sometimes uses ms
    if n > 1e12:
        n = n / 1000.0
    if n <= 0:
        return 0
    return int(n)


def _parse_set_cookie_header(header: str, fallback_host: str) -> dict | None:
    chunks = [c.strip() for c in (header or "").split(";") if c.strip()]
    if not chunks or "=" not in chunks[0]:
        return None
    name, value = chunks[0].split("=", 1)
    name = name.strip()
    if not name:
        return None
    domain = ""
    path = "/"
    secure = False
    expires = 0
    for attr in chunks[1:]:
        low = attr.lower()
        if low.startswith("domain="):
            domain = attr.split("=", 1)[1].strip()
        elif low.startswith("path="):
            path = attr.split("=", 1)[1].strip() or "/"
        elif low.startswith("expires="):
            # Ignore HTTP-date; prefer Max-Age when present. Session if unknown.
            pass
        elif low.startswith("max-age="):
            try:
                age = int(float(attr.split("=", 1)[1].strip()))
                if age > 0:
                    expires = int(time.time()) + age
            except ValueError:
                pass
        elif low == "secure":
            secure = True
    domain = _normalize_cookie_domain(domain, fallback_host)
    if not domain or not _host_is_youtube_family(domain):
        return None
    return {
        "domain": domain,
        "path": path or "/",
        "secure": secure,
        "expires": expires,
        "name": name,
        "value": value,
    }


def _cookie_from_har_obj(obj: dict, fallback_host: str) -> dict | None:
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or "").strip()
    if not name:
        return None
    value = obj.get("value")
    if value is None:
        return None
    domain = _normalize_cookie_domain(str(obj.get("domain") or ""), fallback_host)
    if not domain or not _host_is_youtube_family(domain):
        return None
    path = str(obj.get("path") or "/") or "/"
    secure = bool(obj.get("secure"))
    expires = _cookie_expiry(obj.get("expires") if obj.get("expires") is not None else obj.get("expirationDate"))
    return {
        "domain": domain,
        "path": path,
        "secure": secure,
        "expires": expires,
        "name": name,
        "value": str(value),
    }


def _cookies_from_cookie_header(header: str, fallback_host: str) -> list[dict]:
    out: list[dict] = []
    domain = _normalize_cookie_domain("", fallback_host)
    if not domain or not _host_is_youtube_family(domain):
        return out
    for part in (header or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if not name:
            continue
        out.append(
            {
                "domain": domain,
                "path": "/",
                "secure": domain.endswith("google.com") or "youtube" in domain,
                "expires": 0,
                "name": name,
                "value": value,
            }
        )
    return out


def har_to_netscape(har: dict | str | bytes) -> tuple[str, int]:
    """Extract YouTube/Google cookies from a HAR export → Netscape cookie file text."""
    if isinstance(har, (bytes, bytearray)):
        har = har.decode("utf-8", errors="ignore")
    if isinstance(har, str):
        har = json.loads(har)
    if not isinstance(har, dict):
        raise ValueError("HAR must be a JSON object")

    log = har.get("log") if isinstance(har.get("log"), dict) else har
    entries = log.get("entries") if isinstance(log, dict) else None
    if not isinstance(entries, list):
        raise ValueError("HAR has no log.entries")

    jar: dict[tuple[str, str, str], dict] = {}

    def absorb(c: dict | None) -> None:
        if not c:
            return
        key = (c["domain"], c["path"], c["name"])
        jar[key] = c

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        req = entry.get("request") if isinstance(entry.get("request"), dict) else {}
        res = entry.get("response") if isinstance(entry.get("response"), dict) else {}
        url = str(req.get("url") or entry.get("url") or "")
        host = (urlparse(url).hostname or "").lower()

        if host and _host_is_youtube_family(host):
            for c in req.get("cookies") or []:
                absorb(_cookie_from_har_obj(c if isinstance(c, dict) else {}, host))
            for h in req.get("headers") or []:
                if not isinstance(h, dict):
                    continue
                if str(h.get("name") or "").lower() == "cookie":
                    for c in _cookies_from_cookie_header(str(h.get("value") or ""), host):
                        absorb(c)

        for c in res.get("cookies") or []:
            absorb(_cookie_from_har_obj(c if isinstance(c, dict) else {}, host))
        for h in res.get("headers") or []:
            if not isinstance(h, dict):
                continue
            if str(h.get("name") or "").lower() == "set-cookie":
                absorb(_parse_set_cookie_header(str(h.get("value") or ""), host))

    if not jar:
        return "", 0

    lines = [
        "# Netscape HTTP Cookie File",
        "# Sourced from browser HAR export (ShtetlFrames)",
    ]
    for c in sorted(jar.values(), key=lambda x: (x["domain"], x["path"], x["name"])):
        domain = c["domain"]
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if c["secure"] else "FALSE"
        # Netscape: domain, subdomainFlag, path, secure, expiry, name, value
        lines.append(
            f"{domain}\t{include_sub}\t{c['path']}\t{secure}\t{int(c['expires'])}\t{c['name']}\t{c['value']}"
        )
    text = "\n".join(lines) + "\n"
    return text, len(jar)


def import_cookies_from_har(har: dict | str | bytes) -> dict:
    """Write YouTube cookies from a HAR file into data/youtube_cookies.txt."""
    global _last_export_ok, _last_export_err
    dest = cookies_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        text, n = har_to_netscape(har)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "path": str(dest),
            "message": "Not valid JSON — save the file as HAR (HTTP Archive) from Edge/Chrome DevTools.",
            "bytes": 0,
            "browser": "har",
            "count": 0,
        }
    except ValueError as e:
        return {
            "ok": False,
            "path": str(dest),
            "message": str(e),
            "bytes": 0,
            "browser": "har",
            "count": 0,
        }

    if not text or not cookies_look_valid(text):
        _last_export_err = "har_no_youtube_cookies"
        return {
            "ok": False,
            "path": str(dest),
            "message": (
                "No YouTube/Google cookies found in that HAR. "
                "In Edge: open youtube.com while signed in → F12 → Network → reload → "
                "right-click the list → Save all as HAR with content → upload that file."
            ),
            "bytes": 0,
            "browser": "har",
            "count": 0,
        }

    with _export_lock:
        dest.write_text(text, encoding="utf-8")
        _last_export_ok = time.time()
        _last_export_err = ""
    return {
        "ok": True,
        "path": str(dest),
        "message": f"Imported {n} cookies from HAR ({dest.stat().st_size} bytes)",
        "bytes": dest.stat().st_size,
        "browser": "har",
        "count": n,
    }


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))


def _roaming_appdata() -> Path:
    return Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))


def _cookie_db_paths(profile_dir: Path) -> list[Path]:
    return [
        profile_dir / "Network" / "Cookies",
        profile_dir / "Cookies",
    ]


def _chromium_profiles(user_data: Path) -> list[str]:
    if not user_data.is_dir():
        return []
    names: list[str] = []
    candidates = ["Default", "Profile 1", "Profile 2", "Profile 3"]
    try:
        for p in sorted(user_data.iterdir()):
            if p.is_dir() and (p.name == "Default" or p.name.startswith("Profile ")):
                if p.name not in candidates:
                    candidates.append(p.name)
    except OSError:
        pass
    for name in candidates:
        base = user_data / name
        if any(p.is_file() for p in _cookie_db_paths(base)):
            names.append(name)
    return names


def _user_data_for(browser: str) -> Path | None:
    local = _local_appdata()
    for name, rel in _CHROMIUM_ROOTS:
        if name == browser:
            p = local / Path(rel)
            return p if p.is_dir() else None
    return None


def detect_browser_candidates(preferred: str = "") -> list[tuple[str, str, Path | None]]:
    """
    Returns list of (browser, profile_name_or_empty, user_data_root|None).
    """
    preferred = (preferred or "edge").strip().lower()
    if preferred in ("", "auto"):
        preferred = "edge"

    order = [preferred] + [b for b, _ in _CHROMIUM_ROOTS if b != preferred]
    out: list[tuple[str, str, Path | None]] = []
    seen: set[tuple[str, str]] = set()

    for browser in order:
        if browser in ("none", "off", "0", "firefox"):
            continue
        root = _user_data_for(browser)
        if not root:
            if browser == preferred:
                key = (browser, "")
                if key not in seen:
                    seen.add(key)
                    out.append((browser, "", None))
            continue
        profiles = _chromium_profiles(root)
        if not profiles:
            key = (browser, "Default")
            if key not in seen:
                seen.add(key)
                out.append((browser, "Default", root))
            continue
        for prof in profiles:
            key = (browser, prof)
            if key in seen:
                continue
            seen.add(key)
            out.append((browser, prof, root))

    # Firefox last (no locked Chromium DB issue)
    ff = _roaming_appdata() / "Mozilla" / "Firefox" / "Profiles"
    if ff.is_dir():
        out.append(("firefox", "", None))
    return out


def _copy_locked_file_win(src: Path, dst: Path) -> None:
    """Copy a file even when Edge/Chrome holds a share lock (yt-dlp #7271)."""
    import ctypes
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x1
    FILE_SHARE_WRITE = 0x2
    FILE_SHARE_DELETE = 0x4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE = wintypes.HANDLE(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CreateFileW = kernel32.CreateFileW
    CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    CreateFileW.restype = wintypes.HANDLE
    ReadFile = kernel32.ReadFile
    CloseHandle = kernel32.CloseHandle
    GetFileSizeEx = kernel32.GetFileSizeEx

    handle = CreateFileW(
        str(src),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE or handle is None:
        raise OSError(f"CreateFileW failed for {src} (err={ctypes.get_last_error()})")

    try:
        size = ctypes.c_longlong(0)
        if not GetFileSizeEx(handle, ctypes.byref(size)):
            raise OSError("GetFileSizeEx failed")
        remaining = int(size.value)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out:
            buf = ctypes.create_string_buffer(1024 * 1024)
            read = wintypes.DWORD(0)
            while remaining > 0:
                to_read = min(remaining, len(buf))
                ok = ReadFile(handle, buf, to_read, ctypes.byref(read), None)
                if not ok or read.value == 0:
                    break
                out.write(buf.raw[: read.value])
                remaining -= read.value
    finally:
        CloseHandle(handle)


def _copy_file_best_effort(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
        return
    except OSError:
        pass
    if os.name == "nt":
        _copy_locked_file_win(src, dst)
        return
    raise OSError(f"cannot copy {src}")


def _snapshot_chromium_profile(user_data: Path, profile: str) -> Path | None:
    """
    Build a temp User Data tree yt-dlp can read while Edge is open.
    Returns path to the temp *profile* directory (…/Default).
    """
    profile_dir = user_data / (profile or "Default")
    cookie_src = next((p for p in _cookie_db_paths(profile_dir) if p.is_file()), None)
    if cookie_src is None:
        return None
    local_state = user_data / "Local State"
    if not local_state.is_file():
        return None

    tmp = Path(tempfile.mkdtemp(prefix="shtetl_cookies_"))
    # Mimic Chromium layout: User Data / Local State + Profile / Network / Cookies
    fake_user = tmp / "User Data"
    fake_profile = fake_user / (profile or "Default")
    if cookie_src.name == "Cookies" and cookie_src.parent.name == "Network":
        cookie_dst = fake_profile / "Network" / "Cookies"
    else:
        cookie_dst = fake_profile / "Cookies"
    try:
        _copy_file_best_effort(local_state, fake_user / "Local State")
        _copy_file_best_effort(cookie_src, cookie_dst)
        # Sidecars (best effort)
        for suf in ("-journal", "-wal", "-shm"):
            side = Path(str(cookie_src) + suf)
            if side.is_file():
                try:
                    _copy_file_best_effort(side, Path(str(cookie_dst) + suf))
                except OSError:
                    pass
        return fake_profile
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def _run_ytdlp_cookie_export(browser_spec: str, dest: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies-from-browser",
        browser_spec,
        "--cookies",
        str(dest),
        "--skip-download",
        "--no-playlist",
        "https://www.youtube.com/",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "cookie_export_timeout"
    except Exception as e:
        return False, str(e)[:240]

    text = dest.read_text(encoding="utf-8", errors="ignore") if dest.is_file() else ""
    if proc.returncode == 0 and cookies_look_valid(text):
        return True, f"Exported YouTube cookies from {browser_spec}"
    err = (proc.stderr or proc.stdout or "cookie_export_failed").strip()
    err = " ".join(line.strip() for line in err.splitlines() if line.strip())[-500:]
    return False, err or "cookie_export_failed"


def _export_one(browser: str, profile: str, user_data: Path | None, dest: Path) -> tuple[bool, str]:
    # 1) Direct yt-dlp (works if browser is closed)
    if browser == "firefox":
        return _run_ytdlp_cookie_export("firefox", dest)
    spec = f"{browser}:{profile}" if profile else browser
    ok, msg = _run_ytdlp_cookie_export(spec, dest)
    if ok:
        return True, msg

    # 2) Windows locked-DB snapshot → yt-dlp against temp profile path
    if user_data and profile and os.name == "nt":
        snap = _snapshot_chromium_profile(user_data, profile)
        if snap is not None:
            try:
                # Absolute profile path form supported by yt-dlp
                snap_spec = f"{browser}:{snap}"
                ok2, msg2 = _run_ytdlp_cookie_export(snap_spec, dest)
                if ok2:
                    return True, f"{msg2} (live Edge/Chrome snapshot)"
                return False, f"{msg} | snapshot: {msg2}"
            finally:
                shutil.rmtree(snap.parent.parent, ignore_errors=True)
    return False, msg


def _friendly_lock_message(raw: str) -> str:
    """Collapse yt-dlp #7271 spam into one Edge-specific line."""
    low = (raw or "").lower()
    locked = "could not copy" in low or "cookie database" in low or "sharing" in low
    try:
        from yt_proxy import proxy_configured

        px = proxy_configured()
    except Exception:
        px = False
    if locked and px:
        return (
            "Edge is open and Windows locked its cookie database "
            "(yt-dlp still says “Chrome” — that’s normal). "
            "A residential proxy is configured and will bypass Google blocks — you can keep browsing. "
            "To export cookies anyway: upload a HAR from Edge DevTools, or fully quit Edge → Refresh YouTube cookies."
        )
    if locked:
        return (
            "Edge is open and Windows locked its cookie database "
            "(yt-dlp still says “Chrome” — that’s normal). "
            "Upload a HAR from Edge DevTools (youtube.com → F12 → Network → Save all as HAR), "
            "or fully quit Edge and Refresh YouTube cookies — or add Scrapfly/ScrapingDog in Settings."
        )
    return (raw or "cookie_export_failed")[-700:]


def export_youtube_cookies(*, force: bool = False, min_age_sec: float = 300.0) -> dict:
    """Pull cookies into data/youtube_cookies.txt (Edge must be fully quit on Windows).

    Never deletes a valid jar until a new browser export succeeds — HAR imports
    and prior exports stay usable when Edge locks its cookie DB.
    """
    global _last_export_ok, _last_export_err
    preferred = cookies_browser()
    dest = cookies_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _existing_ok() -> dict | None:
        if not dest.is_file() or dest.stat().st_size <= 200:
            return None
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if not cookies_look_valid(text):
            return None
        return {
            "ok": True,
            "path": str(dest),
            "message": f"Using cached cookies ({dest.stat().st_size} bytes)",
            "bytes": dest.stat().st_size,
            "browser": "cache",
        }

    if not preferred:
        existing = _existing_ok()
        if existing:
            return existing
        return {
            "ok": False,
            "path": "",
            "message": "YT_COOKIES_BROWSER is off — residential proxy will handle Google blocks",
            "bytes": 0,
            "browser": "",
        }

    with _export_lock:
        now = time.time()
        existing = _existing_ok()
        if existing and not force:
            # Prefer file mtime so HAR imports survive process restarts.
            try:
                age = now - float(dest.stat().st_mtime)
            except OSError:
                age = float("inf")
            if age < float(min_age_sec):
                if _last_export_ok <= 0:
                    _last_export_ok = now - age
                return existing

        errors: list[str] = []
        for browser, profile, user_data in detect_browser_candidates(preferred):
            # Write to a temp path — only replace the jar after a successful export.
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            ok, msg = _export_one(browser, profile, user_data, tmp)
            if ok and tmp.is_file() and cookies_look_valid(
                tmp.read_text(encoding="utf-8", errors="ignore")
            ):
                try:
                    tmp.replace(dest)
                except OSError:
                    dest.write_text(tmp.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                _last_export_ok = time.time()
                _last_export_err = ""
                return {
                    "ok": True,
                    "path": str(dest),
                    "message": msg,
                    "bytes": dest.stat().st_size,
                    "browser": f"{browser}:{profile}" if profile else browser,
                }
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            errors.append(f"{browser}:{profile or '-'}: {msg}")

        _last_export_err = " | ".join(errors)[-700:]
        # Keep HAR / previous jar if browser export failed (Edge lock, etc.).
        if existing:
            existing = dict(existing)
            existing["message"] = (
                f"{existing['message']} — browser refresh failed, kept existing jar"
            )
            return existing
        return {
            "ok": False,
            "path": str(dest),
            "message": _friendly_lock_message(_last_export_err),
            "bytes": 0,
            "browser": "",
        }


def ensure_cookies_for_scrape() -> dict:
    """Best-effort refresh before a RunPod scrape batch."""
    return export_youtube_cookies(force=False, min_age_sec=180.0)


def is_bot_check_error(msg: str) -> bool:
    low = (msg or "").lower()
    return "not a bot" in low or "sign in to confirm" in low
