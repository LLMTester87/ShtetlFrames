"""Local web server: discover → scrape → review (SQLite)."""

from __future__ import annotations

import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import api_health
import api_jobs
import api_queue
import api_review
import api_runpod
import api_summary
from api_http import bytes_response, cors, file_response, json_response, parse_json_body
from config import (
    CONTACT_DIR,
    OUTPUT_DIR,
    ROOT,
    VIDEOS_DIR,
    effective_scan_backend,
    load_env,
    runpod_configured,
)
import config as app_config
from db import init_db, reset_stale_jobs

WEB_DIR = ROOT / "web"
PORT = 8787


def find_video_file(video_id: str) -> Path | None:
    if not VIDEOS_DIR.exists():
        return None
    for p in VIDEOS_DIR.iterdir():
        if p.stem == video_id and p.suffix.lower() in {
            ".mp4",
            ".webm",
            ".mkv",
            ".avi",
            ".mov",
            ".ogv",
        }:
            return p
    for p in VIDEOS_DIR.iterdir():
        if video_id in p.stem and p.suffix.lower() in {
            ".mp4",
            ".webm",
            ".mkv",
            ".avi",
            ".mov",
            ".ogv",
        }:
            return p
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        try:
            from console_dash import is_enabled

            if is_enabled():
                return
        except Exception:
            pass
        print(f"[web] {self.address_string()} {fmt % args}")

    def _cors(self) -> None:
        cors(self)

    def _json(self, code: int, payload: object) -> None:
        json_response(self, code, payload)

    def _bytes(self, code: int, data: bytes, content_type: str, *, no_cache: bool = False) -> None:
        bytes_response(self, code, data, content_type, no_cache=no_cache)

    def _file(self, path: Path, content_type: str | None = None) -> None:
        file_response(self, path, content_type)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/summary":
            api_summary.handle_get_summary(self, parsed)
            return
        if path == "/api/health":
            api_health.handle_get_health(self, parsed)
            return
        if path == "/api/settings":
            from settings_store import settings_public_view

            load_env()
            self._json(200, settings_public_view())
            return
        if path == "/api/runpod/build":
            api_runpod.handle_get_build(self, parsed)
            return
        if path == "/api/runpod/go":
            api_runpod.handle_get_go(self, parsed)
            return
        if path == "/api/runpod/pod":
            api_runpod.handle_get_pod(self, parsed)
            return
        if path == "/api/queue" or path == "/api/queue/items":
            api_queue.handle_get_queue(self, parsed)
            return
        if path == "/api/pathe/summary":
            import api_pathe

            api_pathe.handle_get_summary(self, parsed)
            return
        if path == "/api/pathe/queue":
            import api_pathe

            api_pathe.handle_get_queue(self, parsed)
            return
        if path == "/api/errors":
            api_jobs.handle_get_errors(self, parsed)
            return
        if path == "/api/jobs":
            api_jobs.handle_get_jobs(self, parsed)
            return
        if path.startswith("/api/jobs/"):
            jid = path.split("/api/jobs/", 1)[1].strip("/")
            api_jobs.handle_get_job(self, jid)
            return
        if path == "/api/candidates":
            api_review.handle_get_candidates(self, parsed)
            return
        if path == "/api/review/label_stats":
            api_review.handle_get_label_stats(self)
            return
        if path == "/api/crops":
            import api_crops

            api_crops.handle_get_crops(self, parsed)
            return

        if path.startswith("/media/sheet/"):
            name = Path(path.split("/media/sheet/", 1)[1]).name
            self._file(CONTACT_DIR / name, "image/jpeg")
            return

        if path.startswith("/media/video/"):
            vid = path.split("/media/video/", 1)[1]
            vid = re.sub(r"[^\w.\-\[\] (),]", "", vid)
            f = find_video_file(vid)
            if not f:
                self._json(404, {"error": "video not found", "video_id": vid})
                return
            ctype = mimetypes.guess_type(str(f))[0] or "video/mp4"
            data = f.read_bytes()
            self._bytes(200, data, ctype)
            return

        if path in ("/", "/index.html"):
            self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/review" or path == "/review.html":
            self._file(WEB_DIR / "review.html", "text/html; charset=utf-8")
            return
        if path == "/crops" or path == "/crops.html":
            self._file(WEB_DIR / "crops.html", "text/html; charset=utf-8")
            return
        if path == "/pathe" or path == "/pathe.html":
            self._file(WEB_DIR / "pathe.html", "text/html; charset=utf-8")
            return
        if path == "/health" or path == "/health.html":
            self._file(WEB_DIR / "health.html", "text/html; charset=utf-8")
            return
        if path.startswith("/assets/"):
            name = Path(path.split("/assets/", 1)[1]).name
            self._file(WEB_DIR / "assets" / name)
            return

        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = parse_json_body(self)
        if body is None:
            return

        if path in ("/api/youtube/cookies", "/api/youtube/cookies/har"):
            from yt_cookies import export_youtube_cookies, import_cookies_from_har

            payload = body if isinstance(body, dict) else {}
            har = payload.get("har") or payload.get("har_text") or payload.get("har_json")
            # Allow posting the HAR document itself as the JSON body.
            if har is None and path.endswith("/har") and isinstance(payload.get("log"), dict):
                har = payload
            if har is not None or path.endswith("/har"):
                if har is None:
                    self._json(
                        400,
                        {"ok": False, "error": "missing har field (upload via the UI or POST {\"har\": ...})"},
                    )
                    return
                result = import_cookies_from_har(har)
                self._json(200 if result.get("ok") else 400, {"ok": bool(result.get("ok")), **result})
                return
            force = bool(payload.get("force", True))
            result = export_youtube_cookies(force=force)
            self._json(200 if result.get("ok") else 400, {"ok": bool(result.get("ok")), **result})
            return

        if path == "/api/settings":
            from settings_store import set_settings, settings_public_view

            try:
                values = set_settings(body if isinstance(body, dict) else {})
            except ValueError as e:
                self._json(400, {"ok": False, "error": str(e)})
                return
            load_env()
            self._json(
                200,
                {
                    "ok": True,
                    **settings_public_view(values),
                    "scan": {
                        "backend": effective_scan_backend(),
                        "requested": app_config.SCAN_BACKEND,
                        "runpod_configured": runpod_configured(),
                        "image_set": bool(app_config.RUNPOD_DOCKER_IMAGE),
                        "pod_id": (app_config.RUNPOD_POD_ID or "")[:12],
                        "gpu_type": app_config.RUNPOD_GPU_TYPE,
                        "max_inflight": app_config.RUNPOD_MAX_INFLIGHT,
                        "stop_when_done": app_config.RUNPOD_STOP_WHEN_DONE,
                    },
                },
            )
            return

        if path == "/api/runpod/build":
            api_runpod.handle_post_build(self, body)
            return

        if path == "/api/runpod/go":
            api_runpod.handle_post_go(self, body)
            return

        if path == "/api/runpod/pod/start":
            api_runpod.handle_post_pod_start(self, body)
            return

        if path == "/api/runpod/pod/stop":
            api_runpod.handle_post_pod_stop(self, body)
            return

        if path == "/api/runpod/pod/reload":
            api_runpod.handle_post_pod_reload(self, body)
            return

        if path == "/api/discover":
            api_queue.handle_post_discover(self, body)
            return

        if path == "/api/pathe/discover":
            import api_pathe

            api_pathe.handle_post_discover(self, body)
            return

        if path == "/api/pathe/scrape":
            import api_pathe

            api_pathe.handle_post_scrape(self, body)
            return

        if path == "/api/console/refresh":
            try:
                from console_dash import draw, is_enabled, refresh_from_jobs

                if not is_enabled():
                    json_response(self, 200, {"ok": False, "error": "console_disabled"})
                    return
                synced = refresh_from_jobs()
                draw(force=True)
                json_response(self, 200, {"ok": True, "synced": synced})
            except Exception as e:
                json_response(self, 500, {"ok": False, "error": str(e)[:200]})
            return

        if path == "/api/pathe/queue/clear":
            import api_pathe

            api_pathe.handle_post_queue_clear(self, body)
            return

        if path == "/api/queue/clear":
            api_queue.handle_post_queue_clear(self, body)
            return

        if path == "/api/scrape":
            api_queue.handle_post_scrape(self, body)
            return

        if path == "/api/queue/delete":
            api_queue.handle_post_queue_delete(self, body)
            return

        if path == "/api/queue/priority":
            api_queue.handle_post_queue_priority(self, body)
            return

        if path == "/api/crops":
            import api_crops

            api_crops.handle_post_crop(self, body if isinstance(body, dict) else {})
            return
        if path == "/api/review":
            api_review.handle_post_review(self, body)
            return

        self._json(404, {"error": "not found"})


def main() -> None:
    load_env()
    init_db()
    try:
        from settings_store import ensure_settings_table, get_all_settings, set_settings

        ensure_settings_table()
        # Seed SQLite from current environ/.env once so UI shows values
        current = get_all_settings()
        set_settings(current)
    except Exception:
        pass
    reset_stale_jobs()
    try:
        from logutil import _ensure

        _ensure()
    except Exception:
        pass
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "assets").mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, PORT), Handler)
    try:
        from console_dash import enable, set_idle

        enable()
        set_idle(note="Browser should open on its own.")
    except Exception:
        print(f"ShtetlFrames -> http://{host}:{PORT}")
        print(f"Review workspace -> http://{host}:{PORT}/review")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        try:
            from console_dash import is_enabled, set_idle

            if is_enabled():
                set_idle(note="Stopped. You can close this window.")
            else:
                print("Stopped.")
        except Exception:
            print("Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
