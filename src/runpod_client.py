"""Submit scan jobs to auto-provisioned RunPod GPU Pods (HTTP /scan)."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import requests

import config as app_config
from config import load_env

OnStatus = Callable[[str], None]

_POD_BASE_URL: str | None = None
_POD_POOL: list[str] = []
_pool_lock = threading.Lock()
_rr = 0
# Soft-claim idle pods so two workers don't both pile onto the same free GPU.
_reserved_until: dict[str, float] = {}
# Prefer least-used URLs among equal-rank candidates (avoids starving one GPU).
_pick_counts: dict[str, int] = {}
_RESERVE_SEC = 6.0
_PHASE_RANK = {
    "idle": 0,
    "done": 0,
    "": 0,
    "queued": 1,
    "download": 2,
    "upload": 3,
    "scan": 4,
    "unknown": 5,
    "warming": 6,
    "broken": 98,
    "dead": 99,
}

_DEAD_PROXY_CODES = frozenset({404, 502, 503, 520, 521, 522, 523, 524})
# Terminate + recreate only on hard proxy death (not transient 503 overload).
_TERMINATE_PROXY_CODES = frozenset({404, 502, 520, 521, 522, 523, 524})
_BROKEN_WARM_MARKERS = (
    "importerror",
    "attributeerror",
    "cannot import",
    "min_person_aspect",
    "modulenotfounderror",
    "syntaxerror",
    "warmup failed",
    "warm_failed",
    # YOLO dropped / never loaded — replace the pod, don't hard-fail the queue row.
    "has no attribute 'predict'",
    "nonetype' object has no attribute 'predict",
)

# Pods that already received this checkout's handler via /sync_push (Catbox-less stills).
_handler_pushed: set[str] = set()
_handler_push_lock = threading.Lock()
# Self-heal: strike counts + terminate cooldown (avoid replace storms).
_pod_strikes: dict[str, int] = {}
_warming_since: dict[str, float] = {}
_terminate_at: dict[str, float] = {}
_heal_lock = threading.Lock()
_TERMINATE_COOLDOWN_SEC = 90.0
_WARMING_REPLACE_SEC = 600.0
_BROKEN_STRIKES = 2


def _ensure_local_handler(base: str, *, on_status: OnStatus | None = None) -> bool:
    """Push local worker files so Review stills use still_b64 + GET /still (not Catbox).

    GitHub main may lag this checkout; without a push, pods keep ``upload_failed``
    and Review gets no contact sheet.
    """
    root = (base or "").rstrip("/")
    if not root:
        return False
    with _handler_push_lock:
        if root in _handler_pushed:
            return True
    files = _local_worker_files_for_push()
    if not files:
        return False
    try:
        if on_status:
            on_status("pushing local still-handler to pod…")
        r = requests.post(
            f"{root}/sync_push",
            json={"files": files},
            timeout=120,
        )
        ok = r.status_code == 200
        body: dict[str, Any] = {}
        try:
            body = r.json() if r.content else {}
        except Exception:
            body = {}
        if ok and isinstance(body, dict) and body.get("ok"):
            with _handler_push_lock:
                _handler_pushed.add(root)
            if on_status:
                on_status("pod still-handler pinned (local sync)")
            return True
        # Older pods without /sync_push — fall through; PC hydrate/ensure will try.
        if r.status_code == 404:
            return False
    except requests.RequestException:
        return False
    return False

# Serialize / debounce ensure_pods so N workers don't stampede RunPod.
_refresh_lock = threading.Lock()
_refresh_mono = 0.0
_REFRESH_DEBOUNCE_SEC = 8.0

# Dedicated pod for Pathé catalog discover (Scrapfly listing pages).
# Kept out of the scrape round-robin pool so discover doesn't steal GPU scan slots.
_DISCOVER_POD_URL: str | None = None
_discover_lock = threading.Lock()

# Pathé client-side stacking: ceiling from Settings PATHE_STACK_MAX (1–6).
_PATHE_STACK_HARD_CAP = 6
_pathe_stack_limit = 3
_pathe_stack_cfg_max = 3  # last seen UI/env ceiling
_pathe_stack_ok_streak = 0
_pathe_stack_fail_streak = 0
_pathe_stack_lock = threading.Lock()
_PATHE_SCALE_UP_AFTER_OK = 1  # recover stacking immediately after a clean job
_PATHE_FAIL_STRIKES = 2  # non-overload failures before −1
_PATHE_OVERLOAD_MARKERS = (
    "pod_saturated",
    "http_503",
    "http_524",
    "http_502",
    "gateway time-out",
    "gateway timeout",
    "gpu pod not ready",
)

_INFRA_MARKERS = (
    "gpu pod not ready",
    "ensure_pod",
    "ensure_pods",
    "all gpu proxies dead",
    "need ensure_pods",
    "http_404",
    "http_502",
    "http_503",
    "http_524",
    "pod_worker_died",
    "pod_scan_http",
    "pod_scan_timeout",
    "pod_bad_json",
    "pod_saturated",
    "models_not_ready",
    "gateway time-out",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "max retries exceeded",
    "failed to establish a new connection",
    "name or service not known",
    "temporary failure in name resolution",
    "has no attribute 'predict'",
)


def _is_dead_proxy_status(code: int) -> bool:
    return int(code or 0) in _DEAD_PROXY_CODES


def pod_id_from_proxy_url(url: str | None) -> str | None:
    """Extract RunPod id from ``https://{id}-8000.proxy.runpod.net``."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return None
    m = re.match(r"https?://([a-z0-9]+)-\d+\.proxy\.runpod\.net/?$", u, re.I)
    return m.group(1) if m else None


def pool_size() -> int:
    with _pool_lock:
        return len(_POD_POOL)


def _may_terminate(pod_id: str) -> bool:
    now = time.time()
    with _heal_lock:
        last = float(_terminate_at.get(pod_id) or 0.0)
        if now - last < _TERMINATE_COOLDOWN_SEC:
            return False
        _terminate_at[pod_id] = now
        return True


def drop_pod_url(
    url: str | None,
    *,
    terminate: bool = False,
    reason: str = "",
) -> None:
    """Remove a dead proxy from the pool; optionally terminate the RunPod GPU."""
    global _POD_BASE_URL, _rr
    u = (url or "").rstrip("/")
    if not u:
        return
    with _pool_lock:
        _POD_POOL[:] = [x for x in _POD_POOL if x.rstrip("/") != u]
        _reserved_until.pop(u, None)
        if _POD_BASE_URL and _POD_BASE_URL.rstrip("/") == u:
            _POD_BASE_URL = _POD_POOL[0] if _POD_POOL else None
        _rr = 0
    with _handler_push_lock:
        _handler_pushed.discard(u)
    with _heal_lock:
        _pod_strikes.pop(u, None)
        _warming_since.pop(u, None)
    if not terminate:
        return
    pid = pod_id_from_proxy_url(u)
    if not pid or not _may_terminate(pid):
        return
    try:
        from runpod_provision import terminate_pod

        terminate_pod(pid)
        print(
            f"[shtetl] self-heal terminate {pid[:12]}… ({reason or 'dead'})",
            flush=True,
        )
    except Exception as e:
        print(f"[shtetl] self-heal terminate failed {pid[:12]}: {e}"[:160], flush=True)


def set_pod_base_url(url: str | None) -> None:
    global _POD_BASE_URL, _POD_POOL, _rr
    u = (url or "").rstrip("/") or None
    _POD_BASE_URL = u
    with _pool_lock:
        _POD_POOL = [u] if u else []
        _rr = 0
        _reserved_until.clear()


def set_pod_pool(urls: list[str]) -> None:
    """Set round-robin pool of healthy pod proxy bases."""
    global _POD_BASE_URL, _POD_POOL, _rr
    cleaned = [u.rstrip("/") for u in urls if (u or "").strip()]
    # Stable order, de-dupe (preserve first occurrence).
    seen: set[str] = set()
    uniq: list[str] = []
    for u in cleaned:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    cleaned = uniq
    with _discover_lock:
        disc = (_DISCOVER_POD_URL or "").rstrip("/")
    if disc:
        cleaned = [u for u in cleaned if u != disc]
    with _pool_lock:
        _POD_POOL = cleaned
        _rr = 0
        _reserved_until.clear()
        # Drop pick counts for URLs no longer in the pool.
        for u in list(_pick_counts):
            if u not in cleaned:
                _pick_counts.pop(u, None)
    _POD_BASE_URL = cleaned[0] if cleaned else None


def get_pod_pool() -> list[str]:
    """Copy of current scrape pod proxy bases (excludes reserved discover pod)."""
    with _pool_lock:
        return list(_POD_POOL)


def _attach_vision_verify_payload(payload: dict[str, Any]) -> None:
    """Attach OpenAI or open-VLM verify credentials for on-pod still checks.

    Localhost open-VLM URLs are skipped on the pod (unreachable); the PC filter
    runs after segments return.
    """
    try:
        from openai_verify import (
            POD_OLLAMA_URL,
            _api_key,
            openai_model,
            openai_verify_enabled,
            open_vlm_api_key,
            open_vlm_base_url,
            open_vlm_model,
            open_vlm_runs_on_pod,
            verify_backend,
        )

        if not openai_verify_enabled():
            return
        backend = verify_backend()
        payload["verify_backend"] = backend
        if backend in ("open_vlm", "ollama_then_openai"):
            # Default: Ollama on the RunPod GPU (pod loopback). Remote URL = OpenRouter etc.
            if open_vlm_runs_on_pod():
                payload["open_vlm_base_url"] = POD_OLLAMA_URL
            else:
                payload["open_vlm_base_url"] = open_vlm_base_url()
            payload["open_vlm_model"] = open_vlm_model()
            vlm_key = open_vlm_api_key()
            if vlm_key:
                payload["open_vlm_api_key"] = vlm_key
            if backend == "ollama_then_openai":
                oai = _api_key()
                if oai:
                    payload["openai_api_key"] = oai
                    payload["openai_model"] = openai_model()
            return
        key = _api_key()
        if key:
            # Verify on the GPU while the JPEG is local (Catbox is unreliable).
            payload["openai_api_key"] = key
            payload["openai_model"] = openai_model()
    except Exception:
        pass


def _local_worker_files_for_push() -> dict[str, str]:
    """Critical worker files from this checkout (UTF-8) for POST /sync_push."""
    from config import ROOT

    rels = [
        "runpod_worker/entry.py",
        "runpod_worker/handler.py",
        "runpod_worker/worker_sync.py",
        "runpod_worker/ollama_pod.py",
        "src/openai_verify.py",
        "src/label_feedback.py",
        "src/shtetl_core/__init__.py",
        "src/shtetl_core/cues.py",
        "src/shtetl_core/scoring.py",
        "src/shtetl_core/scan.py",
        "src/shtetl_core/segments.py",
        "src/shtetl_core/textutil.py",
        "src/shtetl_core/upload.py",
    ]
    out: dict[str, str] = {}
    for rel in rels:
        path = ROOT / rel
        if not path.is_file():
            continue
        # Pod paths are relative to worker root (no runpod_worker/ / src/ prefix).
        if rel.startswith("runpod_worker/"):
            dest = rel.split("/", 1)[1]
        elif rel.startswith("src/"):
            dest = rel.split("/", 1)[1]
        else:
            dest = Path(rel).name
        out[dest] = path.read_text(encoding="utf-8")
    return out


def reload_all_pod_workers(
    *,
    on_status: OnStatus | None = None,
    push_local: bool = True,
) -> dict[str, Any]:
    """POST /reload on every shtetl pod — pull GitHub + hot-reload (no pod recreate)."""
    load_env()
    from runpod_provision import find_shtetl_pods, pod_proxy_url

    pods = find_shtetl_pods()
    push_files = _local_worker_files_for_push() if push_local else {}
    results: list[dict[str, Any]] = []
    ok_n = 0
    for pod in pods:
        pid = pod.get("id") or ""
        name = pod.get("name") or pid[:12]
        if not pid:
            continue
        base = pod_proxy_url(pid)
        row: dict[str, Any] = {"name": name, "id": pid[:16], "base": base}
        try:
            r = requests.post(f"{base.rstrip('/')}/reload", timeout=90)
            row["http"] = r.status_code
            try:
                row["body"] = r.json() if r.content else {}
            except Exception:
                row["body"] = {"raw": (r.text or "")[:200]}
            # Bypass stale CDN: push this checkout's cues/handler when endpoint exists.
            if push_files:
                try:
                    pr = requests.post(
                        f"{base.rstrip('/')}/sync_push",
                        json={"files": push_files},
                        timeout=120,
                    )
                    row["push_http"] = pr.status_code
                    try:
                        row["push_body"] = pr.json() if pr.content else {}
                    except Exception:
                        row["push_body"] = {"raw": (pr.text or "")[:160]}
                    if pr.status_code == 200 and isinstance(row.get("push_body"), dict):
                        if row["push_body"].get("ok"):
                            ok_n += 1
                            if on_status:
                                on_status(f"push {name}: ok")
                            results.append(row)
                            continue
                except Exception as pe:
                    row["push_error"] = str(pe)[:120]
            if r.status_code == 200 and isinstance(row.get("body"), dict) and row["body"].get("ok"):
                ok_n += 1
            elif r.status_code == 404:
                row["error"] = "reload_not_on_pod — recreate once to install github sync"
            if on_status:
                on_status(f"reload {name}: HTTP {r.status_code}")
        except Exception as e:
            row["error"] = str(e)[:160]
            if on_status:
                on_status(f"reload {name}: {e}"[:120])
        results.append(row)
    return {
        "ok": ok_n > 0,
        "reloaded": ok_n,
        "pod_count": len(results),
        "pods": results,
    }


def get_pathe_discover_pod() -> str | None:
    with _discover_lock:
        return _DISCOVER_POD_URL


def reserve_pathe_discover_pod(
    *,
    on_status: OnStatus | None = None,
    scrape_pods: int | None = None,
) -> str:
    """Reserve at most one healthy pod for Pathé listing (never a scrape fleet).

    Reuses an existing ready pod. Does not call ``ensure_pods`` with count>1
    and does not background-fill — scrape start owns multi-GPU creates.
    """
    del scrape_pods  # discover must never size a scrape pool
    global _DISCOVER_POD_URL
    load_env()
    with _discover_lock:
        cur = _DISCOVER_POD_URL
    if cur and _probe_pod_phase(cur) != "dead":
        # Keep it out of the scrape pool.
        with _pool_lock:
            pool = [u for u in list(_POD_POOL) if u.rstrip("/") != cur.rstrip("/")]
        set_pod_pool(pool)
        return cur.rstrip("/")

    from runpod_provision import find_shtetl_pods, pod_proxy_url

    # Fast path: use any already-healthy pod — do not block on cold boots.
    for p in find_shtetl_pods():
        pid = p.get("id")
        if not pid:
            continue
        base = pod_proxy_url(pid).rstrip("/")
        phase = _probe_pod_phase(base)
        if phase == "dead":
            continue
        with _discover_lock:
            _DISCOVER_POD_URL = base
        with _pool_lock:
            pool = [u for u in list(_POD_POOL) if u.rstrip("/") != base]
        set_pod_pool(pool)
        if on_status:
            on_status(
                f"Pathé discover using ready pod · "
                f"{base.split('//')[-1][:28]}"
            )
        return base

    # No healthy pod — do NOT spin a fleet. Caller falls back to local Scrapfly.
    # (Creating even 1 GPU here is reserved for Start scrape / explicit ensure.)
    raise RuntimeError("no_ready_discover_pod")


def release_pathe_discover_pod() -> None:
    """Return the discover pod to the scrape pool (or drop if gone)."""
    global _DISCOVER_POD_URL
    with _discover_lock:
        disc = _DISCOVER_POD_URL
        _DISCOVER_POD_URL = None
    if not disc:
        return
    disc = disc.rstrip("/")
    with _pool_lock:
        pool = list(_POD_POOL)
    if disc not in pool and _probe_pod_phase(disc) != "dead":
        set_pod_pool(pool + [disc])


def fetch_pathe_list_html_remote(
    page_url: str,
    *,
    base: str | None = None,
    rendering_wait: int = 4000,
    auto_scroll: bool = False,
) -> str:
    """Fetch Pathé listing HTML via the dedicated discover pod's /pathe_list."""
    load_env()
    pod = (base or get_pathe_discover_pod() or "").rstrip("/")
    if not pod:
        raise RuntimeError("pathe_discover_pod_missing")
    key = (
        getattr(app_config, "SCRAPFLY_API_KEY", None)
        or __import__("os").environ.get("SCRAPFLY_API_KEY")
        or __import__("os").environ.get("SCRAPFLY_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError("SCRAPFLY_API_KEY required for Pathé discover")
    r = requests.post(
        f"{pod}/pathe_list",
        json={
            "url": page_url,
            "scrapfly_api_key": key,
            "rendering_wait": int(rendering_wait),
            "auto_scroll": bool(auto_scroll),
            "country": (
                __import__("os").environ.get("SCRAPFLY_COUNTRY") or "us"
            ).strip()
            or "us",
        },
        timeout=200,
    )
    if r.status_code == 404:
        raise RuntimeError(
            "pathe_list_not_on_pod — push worker code + recreate pods "
            "(missing POST /pathe_list)"
        )
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"pathe_list_bad_json http_{r.status_code}: {e}") from e
    if r.status_code >= 400 or not data.get("ok"):
        raise RuntimeError(
            str(data.get("error") or f"pathe_list_http_{r.status_code}")[:300]
        )
    html = data.get("html") or ""
    if not html:
        raise RuntimeError("pathe_list_empty_html")
    return html


def get_pod_base_url() -> str:
    load_env()
    with _pool_lock:
        if _POD_POOL:
            return _POD_POOL[0]
    if _POD_BASE_URL:
        return _POD_BASE_URL
    return refresh_pod_pool(count=1)[0]


def refresh_pod_pool(
    *,
    count: int = 1,
    on_status: OnStatus | None = None,
    force: bool = False,
) -> list[str]:
    """Ensure at least one healthy pod. Debounced — concurrent callers share one refresh."""
    global _refresh_mono
    from runpod_provision import MAX_PARALLEL_PODS, ensure_pods

    n = max(1, min(int(count or 1), MAX_PARALLEL_PODS))
    with _refresh_lock:
        now = time.monotonic()
        with _pool_lock:
            have = list(_POD_POOL)
        if not force and have and (now - _refresh_mono) < _REFRESH_DEBOUNCE_SEC:
            return have
        if on_status:
            on_status(f"GPU pool refresh — ensuring {n} pod(s)…")
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                bases = ensure_pods(
                    count=n,
                    on_status=on_status,
                    recreate=False,
                    min_ready=1,
                    extra_fill_sec=0,
                )
                if bases:
                    set_pod_pool(bases)
                    _refresh_mono = time.monotonic()
                    # New/reused pods may be on stale GitHub code — pin local handler.
                    _push_handlers_best_effort(bases)
                    return bases
                last_err = RuntimeError("ensure_pods returned no proxies")
            except Exception as e:
                last_err = e
                if on_status:
                    on_status(f"pod refresh try {attempt}/3 failed: {e}"[:160])
                time.sleep(min(12.0, 2.0 * attempt))
        raise RuntimeError(
            f"GPU pod not ready — ensure_pods failed: {last_err}"
        ) from last_err


def _classify_pod(base: str) -> str:
    """Return idle|done|queued|download|upload|scan|warming|broken|unknown|dead."""
    root = (base or "").rstrip("/")
    if not root:
        return "dead"
    try:
        r = requests.get(f"{root}/health", timeout=2.5)
    except requests.RequestException:
        return "dead"
    if int(r.status_code or 0) in _TERMINATE_PROXY_CODES:
        return "dead"
    if r.status_code == 503:
        # Overloaded or models_not_ready — not always a kill, but not pickable idle.
        try:
            data = r.json() if r.content else {}
        except Exception:
            data = {}
        err = str((data or {}).get("error") or (data or {}).get("warm_error") or "").lower()
        if any(m in err for m in _BROKEN_WARM_MARKERS):
            return "broken"
        if "models_not_ready" in err or (isinstance(data, dict) and data.get("models_ready") is False):
            return "warming"
        return "unknown"
    if r.status_code != 200:
        return "unknown"
    try:
        data = r.json() if r.content else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        return "unknown"
    warm_err = str(data.get("warm_error") or "").lower()
    if any(m in warm_err for m in _BROKEN_WARM_MARKERS):
        return "broken"
    if data.get("models_ready") is False or data.get("ok") is False:
        return "warming"
    try:
        r2 = requests.get(f"{root}/progress", timeout=2.5)
    except requests.RequestException:
        return "dead"
    if int(r2.status_code or 0) in _TERMINATE_PROXY_CODES:
        return "dead"
    if r2.status_code != 200:
        return "unknown"
    try:
        prog = r2.json() if r2.content else {}
    except Exception:
        return "unknown"
    if not isinstance(prog, dict):
        return "unknown"
    phase = str(prog.get("phase") or "").strip().lower()
    if phase in _PHASE_RANK:
        return phase
    return phase or "idle"


def _probe_pod_phase(base: str) -> str:
    """Return current pod phase (idle/download/scan/…), 'dead' if proxy is gone."""
    kind = _classify_pod(base)
    if kind == "broken":
        return "dead"  # pick/rank treats as unusable
    if kind == "warming":
        return "unknown"
    return kind


def maintain_pod_pool(
    *,
    target: int = 1,
    on_status: OnStatus | None = None,
) -> list[str]:
    """Health-check pool; terminate broken/dead GPUs; refill + push local handler."""
    from runpod_provision import MAX_PARALLEL_PODS, find_shtetl_pods, pod_proxy_url

    want = max(1, min(int(target or 1), MAX_PARALLEL_PODS))
    with _pool_lock:
        pool = list(_POD_POOL)
    if not pool:
        bases = refresh_pod_pool(count=want, on_status=on_status, force=True)
        _push_handlers_best_effort(bases)
        return bases

    alive: list[str] = []
    replaced = 0
    now = time.time()
    kinds: dict[str, str] = {}
    for raw in pool:
        u = raw.rstrip("/")
        kind = _classify_pod(u)
        kinds[u.split("//")[-1][:18]] = kind
        if kind == "dead":
            drop_pod_url(u, terminate=True, reason="proxy_dead")
            replaced += 1
            continue
        if kind == "broken":
            with _heal_lock:
                strikes = int(_pod_strikes.get(u) or 0) + 1
                _pod_strikes[u] = strikes
            if strikes >= _BROKEN_STRIKES:
                drop_pod_url(u, terminate=True, reason="warm_broken")
                replaced += 1
            continue
        if kind == "warming":
            with _heal_lock:
                started = float(_warming_since.get(u) or 0.0)
                if started <= 0:
                    _warming_since[u] = now
                    started = now
            # Cold boot OK for a while; replace only if stuck forever.
            if now - started >= _WARMING_REPLACE_SEC:
                drop_pod_url(u, terminate=True, reason="warm_stuck")
                replaced += 1
            else:
                # Keep in pool so we don't over-create while it's booting.
                alive.append(u)
            continue
        with _heal_lock:
            _pod_strikes.pop(u, None)
            _warming_since.pop(u, None)
        alive.append(u)
    # #region agent log
    if replaced or any(k in ("dead", "broken", "warming", "unknown") for k in kinds.values()):
        _agent_log(
            "C",
            "runpod_client.py:maintain_pod_pool",
            "heal_pass",
            {
                "want": want,
                "pool": len(pool),
                "alive": len(alive),
                "replaced": replaced,
                "kinds": kinds,
            },
        )
    # #endregion

    # GraphQL orphans: RUNNING but broken/dead — kill so ensure_pods can refill.
    known = set(alive)
    try:
        for p in find_shtetl_pods():
            pid = p.get("id")
            if not pid:
                continue
            base = pod_proxy_url(pid).rstrip("/")
            if not base or base in known:
                continue
            kind = _classify_pod(base)
            if kind in ("dead", "broken"):
                drop_pod_url(base, terminate=True, reason=f"orphan_{kind}")
                replaced += 1
                continue
            if kind == "warming":
                continue
            if len(alive) < MAX_PARALLEL_PODS:
                alive.append(base)
                known.add(base)
    except Exception:
        pass

    if {u.rstrip("/") for u in alive} != {u.rstrip("/") for u in pool}:
        set_pod_pool(alive)

    # Critical: do NOT block the scrape coordinator on a full ensure_pods probe
    # every few seconds just because alive < want (e.g. 5/8). That froze the
    # claim/result loop for ~15–40s per pass and made the UI look crashed at 0%.
    need_block_refresh = (not alive) or replaced > 0
    if need_block_refresh:
        try:
            if on_status and replaced:
                on_status(f"self-heal: replaced {replaced} dead GPU(s)…")
            # #region agent log
            _agent_log(
                "F",
                "runpod_client.py:maintain_pod_pool",
                "block_refresh",
                {"want": want, "alive": len(alive), "replaced": replaced},
            )
            # #endregion
            alive = refresh_pod_pool(
                count=want, on_status=on_status, force=True
            )
        except Exception:
            pass
    elif len(alive) < want:
        # #region agent log
        _agent_log(
            "F",
            "runpod_client.py:maintain_pod_pool",
            "async_fill_skip_block",
            {"want": want, "alive": len(alive), "replaced": replaced},
        )
        # #endregion

        def _bg_refill() -> None:
            try:
                more = refresh_pod_pool(count=want, on_status=None, force=False)
                if more:
                    set_pod_pool(more)
                    _push_handlers_best_effort(more)
            except Exception:
                pass

        threading.Thread(
            target=_bg_refill, daemon=True, name="pod-pool-async-fill"
        ).start()
    _push_handlers_best_effort(alive)
    return alive


def _push_handlers_best_effort(urls: list[str]) -> None:
    """Push local still-handler to pods that have not received it yet."""
    for u in urls or []:
        try:
            _ensure_local_handler(u.rstrip("/"), on_status=None)
        except Exception:
            pass


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "30525a",
            "runId": "idle-gpu-pick",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        path = Path(__file__).resolve().parents[1] / "debug-30525a.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        try:
            requests.post(
                "http://127.0.0.1:7406/ingest/637a1fe8-1535-4387-b632-3fb6093e59a2",
                headers={"Content-Type": "application/json", "X-Debug-Session-Id": "30525a"},
                json=payload,
                timeout=1.5,
            )
        except Exception:
            pass
    except Exception:
        pass
    # #endregion


def _pick_pod(
    *,
    idle_only: bool = False,
    reserve_sec: float | None = None,
) -> str:
    """Prefer an idle GPU pod so free pods get work instead of queuing on busy ones.

    ``idle_only=True`` (Pathé) refuses busy pods so we never stack HLS jobs.
    """
    last_err: Exception | None = None
    hold_for = float(reserve_sec) if reserve_sec is not None else _RESERVE_SEC
    # Pathé: wait longer for an idle GPU rather than piling onto a busy one.
    max_attempts = 12 if idle_only else 4
    for attempt in range(1, max_attempts + 1):
        try:
            with _pool_lock:
                if not _POD_POOL:
                    pool = [_POD_BASE_URL] if _POD_BASE_URL else []
                else:
                    pool = list(_POD_POOL)
                now = time.time()
                reserved = {u: t for u, t in _reserved_until.items() if t > now}
                _reserved_until.clear()
                _reserved_until.update(reserved)

            if not pool:
                pool = list(refresh_pod_pool(count=1, force=attempt > 1))

            if len(pool) == 1:
                # Still verify single pod isn't a zombie proxy.
                ph = _probe_pod_phase(pool[0])
                if ph == "dead":
                    drop_pod_url(pool[0])
                    refresh_pod_pool(count=1, force=True)
                    continue
                if idle_only and ph not in ("idle", "done", ""):
                    time.sleep(min(8.0, 1.2 * attempt))
                    continue
                with _pool_lock:
                    _reserved_until[pool[0]] = time.time() + hold_for
                return pool[0]

            phases: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=min(8, len(pool))) as ex:
                futs = {ex.submit(_probe_pod_phase, u): u for u in pool}
                for fut in as_completed(futs):
                    u = futs[fut]
                    try:
                        phases[u] = fut.result()
                    except Exception:
                        phases[u] = "unknown"

            def rank(u: str) -> tuple[int, int]:
                if reserved.get(u, 0) > now:
                    return (9, 0)
                ph = phases.get(u, "unknown")
                return (_PHASE_RANK.get(ph, 5), 0)

            for u, ph in list(phases.items()):
                if ph == "dead":
                    drop_pod_url(u)
            alive = [u for u in pool if phases.get(u) != "dead"]
            if not alive:
                refresh_pod_pool(count=1, force=True)
                continue
            idle = [
                u
                for u in alive
                if phases.get(u) in ("idle", "done", "")
                and reserved.get(u, 0) <= now
            ]
            if idle_only:
                if not idle:
                    time.sleep(min(8.0, 1.2 * attempt))
                    continue
                pool = idle
            elif idle:
                # Fill empty GPUs before stacking onto busy ones (max throughput).
                pool = idle
            else:
                pool = alive

            scored = sorted(pool, key=rank)
            best_rank = rank(scored[0])[0]
            candidates = [u for u in scored if rank(u)[0] == best_rank]
            with _pool_lock:
                global _rr
                # Least-used among equal rank, then RR as tie-break.
                candidates = sorted(
                    candidates,
                    key=lambda u: (_pick_counts.get(u, 0), _rr),
                )
                choice = candidates[0]
                _rr += 1
                _pick_counts[choice] = _pick_counts.get(choice, 0) + 1
                _reserved_until[choice] = time.time() + hold_for

            # #region agent log
            _agent_log(
                "H1",
                "runpod_client.py:_pick_pod",
                "pod_pick",
                {
                    "choice_tail": choice.split("//")[-1][:28],
                    "best_rank": best_rank,
                    "idle_only": idle_only,
                    "phases": {u.split("//")[-1][:18]: phases.get(u, "?") for u in pool},
                    "reserved_tails": [
                        u.split("//")[-1][:18] for u, t in reserved.items() if t > now
                    ],
                    "n_candidates": len(candidates),
                    "pick_count": _pick_counts.get(choice, 0),
                },
            )
            # #endregion
            return choice
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 1.5 * attempt))
    raise RuntimeError(
        f"GPU pod not ready — _pick_pod failed: {last_err}"
    ) from last_err


def _format_progress(data: dict[str, Any], queue_id: int | None) -> str | None:
    phase = (data.get("phase") or "").strip()
    if not phase or phase == "idle":
        return None
    q = data.get("queue_id")
    if queue_id is not None and q is not None and int(q) != int(queue_id):
        title = (data.get("title") or "")[:40]
        return f"waiting · GPU busy on #{q}" + (f" ({title})" if title else "")

    # Keep progress updates short; the CMD dashboard shows friendly phase names.
    msg = (data.get("message") or phase).strip()
    detail = (data.get("detail") or "").strip()
    pct = data.get("pct")
    parts = [phase]
    if msg and msg != phase:
        parts.append(msg)
    if pct is not None:
        parts.append(f"{pct:g}%")
    line = " · ".join(parts)
    if detail:
        line = f"{line} · {detail}"
    return line[:220]


_PERMANENT_YT_MARKERS = (
    "members-only",
    "members only",
    "this video is available to this channel's members",
    "join this channel to get access",
    "private video",
    "video unavailable",
    "this video is private",
    "this video has been removed",
    "has been removed",
    "copyright",
    "who has blocked you",
    "account associated with this video has been terminated",
    "login required",
    "confirm your age",
    "age-restricted",
    "url_required",
)


def is_permanent_youtube_skip(msg: str) -> bool:
    """True for YouTube failures that will never succeed by retrying pods/proxy."""
    low = (msg or "").lower()
    return any(m in low for m in _PERMANENT_YT_MARKERS)


def pathe_stack_max() -> int:
    """Configured Pathé jobs-per-GPU ceiling (Settings → PATHE_STACK_MAX, 1–6)."""
    global _pathe_stack_limit, _pathe_stack_cfg_max
    try:
        load_env()
        mx = int(getattr(app_config, "PATHE_STACK_MAX", None) or 3)
    except Exception:
        mx = 3
    mx = max(1, min(_PATHE_STACK_HARD_CAP, mx))
    with _pathe_stack_lock:
        if mx > _pathe_stack_cfg_max:
            # User raised the UI ceiling — apply immediately.
            _pathe_stack_limit = mx
        elif mx < _pathe_stack_limit:
            _pathe_stack_limit = mx
        _pathe_stack_cfg_max = mx
    return mx


def pathe_stack_limit() -> int:
    """Current Pathé jobs-per-pod preference. Scales down on proxy storms."""
    mx = pathe_stack_max()
    with _pathe_stack_lock:
        return max(1, min(_pathe_stack_limit, mx))


def note_pathe_stack_outcome(*, ok: bool, err: str = "") -> None:
    """Client AIMD for Pathé stacking: −1 on overload, 2-strike −1 otherwise; fast up."""
    global _pathe_stack_limit, _pathe_stack_ok_streak, _pathe_stack_fail_streak
    mx = pathe_stack_max()
    err_l = (err or "").lower()
    overload = (not ok) and any(m in err_l for m in _PATHE_OVERLOAD_MARKERS)
    with _pathe_stack_lock:
        if ok:
            _pathe_stack_ok_streak += 1
            _pathe_stack_fail_streak = 0
            if (
                _pathe_stack_ok_streak >= _PATHE_SCALE_UP_AFTER_OK
                and _pathe_stack_limit < mx
            ):
                _pathe_stack_limit += 1
                _pathe_stack_ok_streak = 0
        else:
            _pathe_stack_ok_streak = 0
            if overload:
                if _pathe_stack_limit > 1:
                    _pathe_stack_limit = max(1, _pathe_stack_limit - 1)
                _pathe_stack_fail_streak = 0
            else:
                _pathe_stack_fail_streak += 1
                if (
                    _pathe_stack_fail_streak >= _PATHE_FAIL_STRIKES
                    and _pathe_stack_limit > 1
                ):
                    _pathe_stack_limit = max(1, _pathe_stack_limit - 1)
                    _pathe_stack_fail_streak = 0


def is_infra_error(msg: str) -> bool:
    """True for pod/network failures that must not permanently fail a video."""
    low = (msg or "").lower()
    if is_permanent_youtube_skip(low):
        return False
    if any(m in low for m in _INFRA_MARKERS):
        return True
    if "404" in low and ("pod" in low or "proxy" in low or "runpod" in low):
        return True
    if "timed out" in low or "timeout" in low:
        # Pod HTTP timeouts are infra; YouTube "timeout" in yt-dlp often is too — retry.
        return True
    return False


def _is_retryable_remote_error(msg: str) -> bool:
    low = (msg or "").lower()
    if is_permanent_youtube_skip(low):
        return False
    if is_infra_error(low):
        return True
    retryable = (
        "numpy is not available",
        "numpy._core",
        "no module named 'numpy",
        "429",
        "temporary failure",
        "cuda out of memory",
        "unable to download",
        "fragment",
        "not a bot",
        "sign in to confirm",
    )
    return any(m in low for m in retryable)


def _cookies_payload() -> str | None:
    try:
        from yt_cookies import ensure_cookies_for_scrape, read_cookies_text

        ensure_cookies_for_scrape()
        return read_cookies_text()
    except Exception:
        return None


def process_video_remote(
    *,
    url: str,
    title: str = "",
    queue_id: int | None = None,
    sample_fps: float = 1.5,
    score_threshold: float = 0.10,
    source_url: str = "",
    download_sections: list[str] | None = None,
    on_status: OnStatus | None = None,
    max_attempts: int = 3,
    cookies_text: str | None = None,
    local_fallback: bool = False,
) -> dict[str, Any]:
    """POST /scan on a pod (download+scan on GPU). Never downloads video on this PC.

    local_fallback is ignored (kept for call-site compat) — PC download/upload is disabled.
    """
    load_env()
    from yt_proxy import (
        acquire_scrapfly_slot,
        fallback_proxy_provider,
        provider_proxy_url,
        proxy_configured,
        proxy_cooldown_remaining,
        proxy_needs_insecure_ssl,
        proxy_provider_name,
        release_scrapfly_slot,
        residential_proxy_url,
        try_acquire_scrapfly_slot,
    )

    cookies = cookies_text if cookies_text is not None else _cookies_payload()
    proxy = residential_proxy_url() if proxy_configured() else None
    provider = proxy_provider_name()
    # Without cookies, residential proxy is the only reliable path — skip doomed guest tries.
    force_proxy = bool(proxy and not cookies)
    payload: dict[str, Any] = {
        "url": url,
        "title": title or url,
        "queue_id": queue_id,
        "sample_fps": sample_fps,
        "score_threshold": score_threshold,
        "source_url": source_url or url,
        "force_proxy": force_proxy,
        "proxy_provider": provider,
        "proxy_insecure": bool(proxy and proxy_needs_insecure_ssl()),
    }
    if download_sections:
        payload["download_sections"] = list(download_sections)
    if cookies:
        payload["cookies_text"] = cookies
    # Only attach Scrapfly when we must — cookies-first jobs omit proxy_url so the pod
    # never burns Scrapfly quota falling through after a cookie miss.
    if proxy and force_proxy:
        payload["proxy_url"] = proxy
    _attach_vision_verify_payload(payload)
    # #region agent log
    _agent_log(
        "H5",
        "runpod_client.py:process_video_remote",
        "server_only_download",
        {
            "queue_id": queue_id,
            "has_cookies": bool(cookies),
            "proxy_provider": provider,
            "has_proxy": bool(proxy),
            "force_proxy": force_proxy,
            "proxy_attached": bool(payload.get("proxy_url")),
            "local_fallback": False,
        },
    )
    # #endregion
    timeout = float(app_config.RUNPOD_JOB_TIMEOUT_SEC or 1800)
    last_err: Exception | None = None
    # Extra attempts: infra blips (dead proxy / cold pod) should not burn the video.
    total_attempts = max(max_attempts, 6 if proxy else 5)

    def _on_scrapfly_wait(sec: float) -> None:
        if on_status and sec > 0:
            on_status(f"Scrapfly Retry-After — waiting {sec:.0f}s before next request…")

    # Scrapfly slot only when this attempt will actually use Scrapfly.
    # If Scrapfly is cooling down / busy and ScrapingDog exists — switch immediately.
    scrapfly_held = {"v": False}
    if provider == "scrapfly" and proxy and force_proxy:
        if try_acquire_scrapfly_slot():
            scrapfly_held["v"] = True
        elif fallback_proxy_provider("scrapfly") == "scrapingdog":
            dog_url = provider_proxy_url("scrapingdog")
            if dog_url:
                provider = "scrapingdog"
                proxy = dog_url
                payload["proxy_url"] = dog_url
                payload["proxy_provider"] = "scrapingdog"
                payload["proxy_insecure"] = False
                if on_status:
                    why = (
                        "rate-limited"
                        if proxy_cooldown_remaining() > 0
                        else "slot busy"
                    )
                    on_status(f"Scrapfly {why} — falling back to ScrapingDog…")
        else:
            scrapfly_held["v"] = acquire_scrapfly_slot(on_wait=_on_scrapfly_wait)

    try:
        return _process_video_remote_attempts(
            payload=payload,
            provider=provider,
            proxy=proxy,
            cookies=cookies,
            queue_id=queue_id,
            on_status=on_status,
            timeout=timeout,
            total_attempts=total_attempts,
            download_sections=download_sections,
            scrapfly_held=scrapfly_held,
            on_scrapfly_wait=_on_scrapfly_wait,
        )
    finally:
        if scrapfly_held["v"]:
            release_scrapfly_slot()


def _process_video_remote_attempts(
    *,
    payload: dict[str, Any],
    provider: str,
    proxy: str | None,
    cookies: str | None,
    queue_id: int | None,
    on_status: OnStatus | None,
    timeout: float,
    total_attempts: int,
    download_sections: list[str] | None,
    scrapfly_held: dict[str, bool] | None = None,
    on_scrapfly_wait: OnStatus | None = None,
    idle_only: bool = False,
    reserve_sec: float | None = None,
) -> dict[str, Any]:
    from yt_proxy import (
        acquire_scrapfly_slot,
        fallback_proxy_provider,
        is_google_block_error,
        is_proxy_throttle_error,
        is_scrapfly_hard_fail,
        note_proxy_throttle,
        provider_proxy_url,
        proxy_cooldown_remaining,
        proxy_needs_insecure_ssl,
        release_scrapfly_slot,
        try_acquire_scrapfly_slot,
        wait_proxy_ready,
    )

    last_err: Exception | None = None
    held_box = scrapfly_held if scrapfly_held is not None else {"v": False}
    active_provider = (provider or "none").strip().lower() or "none"
    active_proxy = proxy
    scrapfly_fail_count = 0

    def _failover_to_scrapingdog(reason: str) -> bool:
        nonlocal active_provider, active_proxy
        nxt = fallback_proxy_provider(active_provider)
        if nxt != "scrapingdog":
            return False
        dog_url = provider_proxy_url("scrapingdog")
        if not dog_url:
            return False
        active_provider = "scrapingdog"
        active_proxy = dog_url
        payload["force_proxy"] = True
        payload["proxy_url"] = dog_url
        payload["proxy_provider"] = "scrapingdog"
        payload["proxy_insecure"] = False
        if held_box["v"]:
            release_scrapfly_slot()
            held_box["v"] = False
        if on_status:
            on_status(f"Scrapfly failed ({reason[:80]}) — falling back to ScrapingDog…")
        return True

    is_pathe_job = str(payload.get("source") or "").lower() == "britishpathe"

    for attempt in range(1, total_attempts + 1):
        # Never sit on Scrapfly Retry-After when ScrapingDog is available.
        if active_provider == "scrapfly" and payload.get("proxy_url"):
            if proxy_cooldown_remaining() > 0:
                if _failover_to_scrapingdog("rate_limit_cooldown"):
                    pass
                else:
                    wait_proxy_ready(
                        provider=active_provider,
                        on_wait=lambda sec: on_status(
                            f"Scrapfly Retry-After — waiting {sec:.0f}s…"
                        )
                        if on_status and sec > 0
                        else None,
                    )
        # Pathé: idle-only when scaled back to 1; allow light stacking when limit ≥ 2.
        use_idle_only = idle_only
        if is_pathe_job:
            use_idle_only = pathe_stack_limit() <= 1
        base = _pick_pod(idle_only=use_idle_only, reserve_sec=reserve_sec)
        # Critical: GitHub main may still ship Catbox upload → upload_failed + no still.
        _ensure_local_handler(base, on_status=on_status)
        if on_status:
            bits = []
            if cookies:
                bits.append("cookies")
            if active_proxy or payload.get("proxy_url"):
                bits.append(
                    active_provider
                    if payload.get("force_proxy")
                    else f"{active_provider}-ready"
                )
            if download_sections:
                bits.append(f"{len(download_sections)} sections")
            if use_idle_only:
                bits.append("idle-only")
            elif is_pathe_job:
                bits.append(f"stack≤{pathe_stack_limit()}")
            mode = (" · " + "+".join(bits)) if bits else ""
            suffix = f" · try {attempt}/{total_attempts}" if attempt > 1 else ""
            on_status(f"pod: submitting… ({base.split('//')[-1][:18]}…){mode}{suffix}")
        # #region agent log
        _agent_log(
            "H1",
            "runpod_client.py:process_video_remote",
            "submit_scan",
            {
                "queue_id": queue_id,
                "attempt": attempt,
                "pod_tail": base.split("//")[-1][:28],
                "force_proxy": bool(payload.get("force_proxy")),
                "proxy_provider": active_provider,
                "has_proxy": bool(active_proxy or payload.get("proxy_url")),
                "idle_only": use_idle_only,
                "pathe_stack": pathe_stack_limit() if is_pathe_job else None,
            },
        )
        # #endregion

        stop = threading.Event()
        last_line = {"v": ""}

        proxy_dead = {"v": False}

        def _poll() -> None:
            while not stop.wait(2.0):
                try:
                    q = f"?queue_id={queue_id}" if queue_id is not None else ""
                    r = requests.get(f"{base}/progress{q}", timeout=8)
                    if _is_dead_proxy_status(r.status_code):
                        proxy_dead["v"] = True
                        return
                    if r.status_code != 200:
                        continue
                    data = r.json() if r.content else {}
                    line = _format_progress(data if isinstance(data, dict) else {}, queue_id)
                    if line and line != last_line["v"] and on_status:
                        last_line["v"] = line
                        on_status(line)
                except requests.RequestException:
                    proxy_dead["v"] = True
                    return
                except Exception:
                    pass

        poller = threading.Thread(target=_poll, name=f"pod-progress-{queue_id}", daemon=True)
        poller.start()
        t0 = time.time()
        try:
            # Short POST: worker accepts async (202) so RunPod's ~100s proxy does not 524.
            try:
                r = requests.post(
                    f"{base}/scan",
                    json=payload,
                    timeout=90,
                )
            except requests.Timeout as e:
                raise TimeoutError(f"pod_scan_timeout after {int(time.time() - t0)}s") from e
            except requests.RequestException as e:
                drop_pod_url(base, terminate=True, reason="scan_http")
                raise RuntimeError(f"pod_scan_http: {e}") from e

            if _is_dead_proxy_status(r.status_code):
                kill = int(r.status_code or 0) in _TERMINATE_PROXY_CODES
                drop_pod_url(base, terminate=kill, reason=f"http_{r.status_code}")
                raise RuntimeError(f"http_{r.status_code}")

            try:
                out = r.json() if r.content else {}
            except Exception as e:
                raise RuntimeError(f"pod_bad_json: {r.status_code} {r.text[:400]}") from e
            if not isinstance(out, dict):
                raise RuntimeError(f"pod_bad_json: {r.status_code}")

            if r.status_code == 524 or "524" in (r.text or "")[:80]:
                drop_pod_url(base, terminate=True, reason="http_524")
                raise RuntimeError("http_524 gateway timeout on /scan accept")

            async_mode = bool(out.get("accepted") and out.get("async")) or r.status_code == 202
            if async_mode:
                result_qid = out.get("queue_id") if out.get("queue_id") is not None else queue_id
                if result_qid is None:
                    raise RuntimeError("pod_scan_accepted_without_queue_id")
                if on_status:
                    on_status("pod accepted · polling result…")
                deadline = t0 + timeout
                out = None
                consecutive_dead = 0
                while time.time() < deadline:
                    if proxy_dead["v"]:
                        drop_pod_url(base, terminate=True, reason="proxy_dead_poll")
                        raise RuntimeError("http_404")
                    try:
                        pr = requests.get(
                            f"{base}/result",
                            params={"queue_id": result_qid},
                            timeout=45,
                        )
                        if _is_dead_proxy_status(pr.status_code):
                            consecutive_dead += 1
                            if consecutive_dead >= 2:
                                kill = int(pr.status_code or 0) in _TERMINATE_PROXY_CODES
                                drop_pod_url(
                                    base,
                                    terminate=kill,
                                    reason=f"http_{pr.status_code}",
                                )
                                raise RuntimeError(f"http_{pr.status_code}")
                        elif pr.status_code == 200 and pr.content:
                            consecutive_dead = 0
                            data = pr.json()
                            if isinstance(data, dict) and not data.get("pending", True):
                                if data.get("error") == "worker_died":
                                    raise RuntimeError("pod_worker_died")
                                out = data
                                break
                    except RuntimeError:
                        raise
                    except (requests.RequestException, ValueError):
                        consecutive_dead += 1
                        if consecutive_dead >= 3:
                            drop_pod_url(base, terminate=True, reason="result_unreachable")
                            raise RuntimeError("http_404")
                    time.sleep(2.0)
                if out is None:
                    raise TimeoutError(f"pod_scan_timeout after {int(time.time() - t0)}s")
            elif r.status_code >= 400 or not out.get("ok", True):
                err = out.get("error") or f"http_{r.status_code}"
                if _is_dead_proxy_status(r.status_code):
                    kill = int(r.status_code or 0) in _TERMINATE_PROXY_CODES
                    # models_not_ready / import break often arrives as 503 with warm_error.
                    err_l = str(err).lower()
                    if any(m in err_l for m in _BROKEN_WARM_MARKERS) or "models_not_ready" in err_l:
                        kill = True
                    drop_pod_url(base, terminate=kill, reason=str(err)[:80])
                raise RuntimeError(str(err))

            if not out.get("ok", True):
                err = out.get("error") or "scan_failed"
                raise RuntimeError(str(err))

            out = dict(out)
            out["job_id"] = f"pod-{queue_id or 'x'}"
            _hydrate_segment_stills(base, out)
            _materialize_segment_stills(out)
            if is_pathe_job:
                note_pathe_stack_outcome(ok=True)
            if on_status:
                on_status(f"pod done · {int(time.time() - t0)}s · {out.get('n_hits', 0)} hits")
            return out
        except Exception as e:
            last_err = e
            err_s = str(e)
            if is_pathe_job:
                note_pathe_stack_outcome(ok=False, err=err_s)

            # Scrapfly failing (throttle, SSL, bot-check) → ScrapingDog.
            scrapfly_in_use = active_provider == "scrapfly" and bool(payload.get("proxy_url"))
            scrapfly_blew = scrapfly_in_use and (
                is_proxy_throttle_error(err_s)
                or is_scrapfly_hard_fail(err_s)
                or "yt-dlp failed" in err_s.lower()
                or "download_failed" in err_s.lower()
                or is_google_block_error(err_s)
            )
            if scrapfly_blew:
                scrapfly_fail_count += 1
                if is_proxy_throttle_error(err_s):
                    note_proxy_throttle(err_s)
                if _failover_to_scrapingdog(err_s[:120]):
                    time.sleep(1.0)
                    continue
                if is_proxy_throttle_error(err_s):
                    wait_s = note_proxy_throttle(err_s)
                    if on_status:
                        on_status(
                            f"Scrapfly throttled — Retry-After {wait_s:.0f}s "
                            "(no ScrapingDog fallback configured)…"
                        )
                    wait_proxy_ready(
                        provider=active_provider,
                        on_wait=lambda sec: on_status(
                            f"Scrapfly Retry-After — waiting {sec:.0f}s…"
                        )
                        if on_status and sec > 0
                        else None,
                    )
                    continue

            # Google block / cookie miss → attach primary residential proxy (usually Scrapfly).
            need_proxy = (
                active_proxy
                and not payload.get("force_proxy")
                and (
                    is_google_block_error(err_s)
                    or "yt-dlp failed" in err_s.lower()
                    or "download_failed" in err_s.lower()
                    or "sign in to confirm" in err_s.lower()
                )
            )
            if need_proxy:
                # Prefer ScrapingDog over parking every worker on Scrapfly Retry-After.
                if active_provider == "scrapfly" and not held_box["v"]:
                    if try_acquire_scrapfly_slot():
                        held_box["v"] = True
                    elif _failover_to_scrapingdog("slot_busy_or_cooldown"):
                        time.sleep(1.0)
                        continue
                    else:
                        held_box["v"] = acquire_scrapfly_slot(
                            on_wait=on_scrapfly_wait
                        )
                payload["force_proxy"] = True
                payload["proxy_url"] = active_proxy
                payload["proxy_provider"] = active_provider
                payload["proxy_insecure"] = proxy_needs_insecure_ssl(active_provider)
                # #region agent log
                _agent_log(
                    "H5",
                    "runpod_client.py:process_video_remote",
                    "escalate_proxy_on_pod",
                    {
                        "queue_id": queue_id,
                        "attempt": attempt,
                        "proxy_provider": active_provider,
                        "err": err_s[:160],
                    },
                )
                # #endregion
                if on_status:
                    on_status(
                        f"Cookies alone failed — retrying with {active_provider} on GPU…"
                    )
                time.sleep(2.0)
                continue

            if "not a bot" in err_s.lower() or "sign in to confirm" in err_s.lower():
                try:
                    from yt_cookies import export_youtube_cookies, read_cookies_text

                    export_youtube_cookies(force=True)
                    refreshed = read_cookies_text()
                    if refreshed:
                        payload["cookies_text"] = refreshed
                except Exception:
                    pass
            # Dead worker / broken warm — terminate that GPU then refill.
            err_l = err_s.lower()
            if (
                "pod_worker_died" in err_l
                or "worker_died" in err_l
                or "models_not_ready" in err_l
                or any(m in err_l for m in _BROKEN_WARM_MARKERS)
            ):
                drop_pod_url(base, terminate=True, reason=err_s[:80])

            # Dead / empty / overloaded pod pool — self-heal + refresh, then retry.
            if is_infra_error(err_s):
                try:
                    if on_status:
                        on_status("GPU busy — self-healing pod pool…")
                    want = max(1, int(getattr(app_config, "RUNPOD_MAX_INFLIGHT", 1) or 1))
                    maintain_pod_pool(target=want, on_status=None)
                except Exception as refresh_err:
                    if on_status:
                        on_status(f"pod heal failed: {refresh_err}"[:160])
                    try:
                        refresh_pod_pool(count=want, on_status=None, force=True)
                    except Exception:
                        pass
                # Always burn another attempt on infra — never give up early in-loop.
                if attempt < total_attempts:
                    if on_status:
                        on_status(f"pod retry {attempt}/{total_attempts}: {err_s[:120]}")
                    time.sleep(min(25.0, 4.0 * attempt))
                    continue
                break
            if attempt < total_attempts and _is_retryable_remote_error(err_s):
                if on_status:
                    on_status(f"pod retry {attempt}/{total_attempts}: {err_s[:120]}")
                time.sleep(min(20.0, 3.0 * attempt))
                continue
            break
        finally:
            stop.set()
            poller.join(timeout=3)

    assert last_err is not None
    raise last_err


def _hydrate_segment_stills(base: str, out: dict[str, Any]) -> None:
    """Fill missing still_b64 from pod GET /still (local-only; no Catbox)."""
    import base64
    import time as _time

    segs = out.get("segments")
    if not isinstance(segs, list) or not segs:
        return
    qid = out.get("queue_id")
    root = (base or "").rstrip("/")
    for i, seg in enumerate(segs, 1):
        if not isinstance(seg, dict):
            continue
        if seg.get("still_b64") or seg.get("image_b64"):
            continue
        got = False
        if root and qid is not None:
            idx = seg.get("still_index")
            try:
                idx_i = int(idx) if idx is not None else i
            except (TypeError, ValueError):
                idx_i = i
            # Try string + int queue_id — some pods key stills by whichever form arrived.
            qids = [qid]
            if str(qid).isdigit():
                qids.append(int(qid))
            qids.append(str(qid))
            seen: set[str] = set()
            for attempt in range(3):
                if got:
                    break
                if attempt:
                    _time.sleep(0.4 * attempt)
                for qtry in qids:
                    key = str(qtry)
                    if key in seen and attempt == 0:
                        continue
                    seen.add(key)
                    try:
                        r = requests.get(
                            f"{root}/still",
                            params={"queue_id": qtry, "index": idx_i},
                            timeout=60,
                        )
                    except requests.RequestException:
                        continue
                    if r.status_code != 200 or len(r.content) < 200:
                        continue
                    ctype = (r.headers.get("content-type") or "").lower()
                    body = r.content
                    looks_img = (
                        body[:3] == b"\xff\xd8\xff"
                        or body[:8] == b"\x89PNG\r\n\x1a\n"
                        or "image/jpeg" in ctype
                        or "image/jpg" in ctype
                        or "image/png" in ctype
                    )
                    if not looks_img:
                        continue
                    seg["still_b64"] = base64.standard_b64encode(body).decode("ascii")
                    # Drop stale cloud URLs — Review uses local contact_sheets only.
                    seg["image_url"] = None
                    got = True
                    break
                seen.clear()
        if not got:
            note = str(seg.get("notes") or "")
            if "still_hydrate_miss" not in note:
                # Prefix so long OpenAI reasons cannot truncate the flag away.
                seg["notes"] = (f"still_hydrate_miss {note}".strip())[:500]
            # Never keep catbox URLs as the Review image source.
            url = str(seg.get("image_url") or "").lower()
            if "catbox" in url:
                seg["image_url"] = None


def _materialize_segment_stills(out: dict[str, Any]) -> None:
    """Decode still_b64 to temp JPEGs so insert never loses bytes after JSON round-trips."""
    import base64
    import tempfile

    segs = out.get("segments")
    if not isinstance(segs, list):
        return
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        if seg.get("_local_still"):
            continue
        b64 = seg.get("still_b64") or seg.get("image_b64")
        if not b64:
            continue
        try:
            raw = base64.standard_b64decode(str(b64).encode("ascii"), validate=False)
        except Exception:
            continue
        if not raw or len(raw) < 200 or raw[:3] != b"\xff\xd8\xff":
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(raw)
                seg["_local_still"] = tmp.name
        except OSError:
            pass


def segments_to_candidate_rows(out: dict[str, Any], source_url: str = "") -> list[dict]:
    rows = []
    for s in out.get("segments") or []:
        img = s.get("image_url")
        if img and "catbox" in str(img).lower():
            img = None
        row = {
            "video_id": s.get("video_id") or out.get("video_id") or "unknown",
            "start_sec": s.get("start_sec"),
            "end_sec": s.get("end_sec"),
            "peak_score": s.get("peak_score"),
            "mean_score": s.get("mean_score"),
            "rank_score": s.get("rank_score"),
            "hit_count": s.get("hit_count"),
            "best_cue": s.get("best_cue"),
            "source_url": s.get("source_url") or source_url,
            "image_url": img,
            "notes": s.get("notes") or "",
        }
        # Inline still — saved to contact_sheets/ on insert.
        b64 = s.get("still_b64") or s.get("image_b64")
        if b64:
            row["still_b64"] = b64
        local = s.get("_local_still") or s.get("local_still")
        if local:
            row["_local_still"] = str(local)
        rows.append(row)
    return rows


def process_pathe_remote(
    url: str,
    title: str = "",
    *,
    queue_id: int | None = None,
    sample_fps: float = 1.5,
    score_threshold: float = 0.10,
    source_url: str = "",
    on_status: OnStatus | None = None,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """GPU scan for British Pathé asset URLs — HLS only, no YouTube proxy/cookies."""
    load_env()
    from britishpathe import is_britishpathe_asset_url, prepare_pathe_job

    if not is_britishpathe_asset_url(url):
        raise RuntimeError("process_pathe_remote expects a britishpathe.com/asset/… URL")

    try:
        job = prepare_pathe_job(url, title or "", on_status=on_status)
    except Exception as e:
        raise RuntimeError(f"britishpathe_resolve_failed: {e}") from e
    if not job:
        raise RuntimeError(
            "britishpathe_resolve_failed — not a Pathé asset URL or resolve returned empty"
        )

    payload: dict[str, Any] = {
        "url": job["download_url"],
        "title": job.get("title") or title or url,
        "queue_id": queue_id,
        "sample_fps": sample_fps,
        "score_threshold": score_threshold,
        "source_url": source_url or job.get("asset_url") or url,
        "source": "britishpathe",
        "m3u8_url": job["m3u8_url"],
        "referer": job["referer"],
        "force_proxy": False,
        "proxy_provider": "none",
        "proxy_insecure": False,
    }
    _attach_vision_verify_payload(payload)
    if on_status:
        on_status(
            "British Pathé HLS — GPU download (no YouTube proxy)…"
            + (" · cached" if job.get("cached") else "")
        )
    timeout = float(app_config.RUNPOD_JOB_TIMEOUT_SEC or 1800)
    # Pathé: prefer idle GPUs but allow stacking when limit≥2 (short reserve).
    stack = pathe_stack_limit()
    stack_mx = pathe_stack_max()
    payload["pathe_max_inflight"] = stack_mx
    if on_status and stack > 1:
        on_status(
            f"Pathé adaptive stack try={stack}/{stack_mx} per GPU "
            f"(scales down on 503/524)…"
        )
    return _process_video_remote_attempts(
        payload=payload,
        provider="none",
        proxy=None,
        cookies=None,
        queue_id=queue_id,
        on_status=on_status,
        timeout=timeout,
        total_attempts=max(max_attempts, 4),
        download_sections=None,
        scrapfly_held={"v": False},
        on_scrapfly_wait=None,
        idle_only=stack <= 1,
        reserve_sec=8.0 if stack <= 1 else 2.0,
    )
