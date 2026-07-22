"""GET/POST /api/runpod/* routes."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import ParseResult

from api_http import json_response
from config import load_env


def handle_get_build(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    from docker_build import build_status

    json_response(handler, 200, {"ok": True, "job": build_status()})


def handle_get_go(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    from runpod_go import go_status

    json_response(handler, 200, {"ok": True, "job": go_status()})


def handle_get_pod(handler: BaseHTTPRequestHandler, parsed: ParseResult) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from api_health import probe_pod
    from runpod_client import get_pod_base_url
    from runpod_provision import find_shtetl_pods

    load_env()
    pods = find_shtetl_pods()
    pool: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(12, len(pods) or 1))) as ex:
        futs = [ex.submit(probe_pod, p) for p in pods if p.get("id")]
        for fut in as_completed(futs):
            try:
                pool.append(fut.result())
            except Exception as e:
                pool.append({"healthy": False, "error": str(e)[:200], "busy": False})
    pool.sort(key=lambda r: (r.get("name") or ""))
    healthy_n = sum(1 for p in pool if p.get("healthy"))
    primary = pool[0] if pool else None
    try:
        pool_base = get_pod_base_url()
    except Exception:
        pool_base = primary["base_url"] if primary else None
    json_response(
        handler,
        200,
        {
            "ok": True,
            "pod": primary,
            "base_url": pool_base,
            "healthy": healthy_n > 0,
            "healthy_count": healthy_n,
            "pod_count": len(pool),
            "pods": pool,
        },
    )


def handle_post_build(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from docker_build import start_build_and_push

    load_env()
    result = start_build_and_push(
        image=(body.get("image") or "").strip(),
        push=bool(body.get("push", True)),
    )
    code = 200 if result.get("ok") else 409
    json_response(handler, code, result)


def handle_post_go(handler: BaseHTTPRequestHandler, body: dict) -> None:
    try:
        from runpod_go import start_runpod_scrape

        load_env()
        settings = body.get("settings")
        if not isinstance(settings, dict):
            settings = {
                k: v
                for k, v in body.items()
                if k not in ("max_videos", "workers", "force_build", "settings")  # force_build ignored
            }
        result = start_runpod_scrape(
            settings=settings,
            max_videos=body.get("max_videos", "all"),
            workers=body.get("workers", 2),
        )
        code = 200 if result.get("ok") else 409
        json_response(handler, code, result)
    except Exception as e:
        json_response(handler, 500, {"ok": False, "error": f"Start scrape failed: {e}"[:800]})


def handle_post_pod_start(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from runpod_client import set_pod_base_url
    from runpod_provision import ensure_pod

    load_env()
    try:
        msgs: list[str] = []

        def on_status(m: str) -> None:
            msgs.append(m)

        base = ensure_pod(on_status=on_status)
        set_pod_base_url(base)
        json_response(handler, 200, {"ok": True, "base_url": base, "log": msgs[-8:]})
    except Exception as e:
        json_response(handler, 500, {"ok": False, "error": str(e)[:800]})


def handle_post_pod_stop(handler: BaseHTTPRequestHandler, body: dict) -> None:
    from runpod_provision import stop_pod

    load_env()
    try:
        stop_pod()
        json_response(handler, 200, {"ok": True})
    except Exception as e:
        json_response(handler, 500, {"ok": False, "error": str(e)[:800]})


def handle_post_pod_reload(handler: BaseHTTPRequestHandler, body: dict) -> None:
    """Ask all pods to pull latest GitHub worker code (hot-reload, no recreate)."""
    from runpod_client import reload_all_pod_workers

    load_env()
    try:
        result = reload_all_pod_workers()
        code = 200 if result.get("ok") else 502
        json_response(handler, code, result)
    except Exception as e:
        json_response(handler, 500, {"ok": False, "error": str(e)[:800]})
