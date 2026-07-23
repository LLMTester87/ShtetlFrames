"""Auto-provision a RunPod GPU Pod for ShtetlFrames scans (no endpoint ID needed)."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests

import config as app_config
from config import load_env

# #region agent log
_DBG_LOG = Path(__file__).resolve().parents[1] / "debug-30525a.log"
_DBG_LOCK = threading.Lock()


def _dbg(hypothesis_id: str, location: str, message: str, **data: object) -> None:
    try:
        payload = {
            "sessionId": "30525a",
            "runId": "scrape-stuck",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "tid": threading.get_ident(),
        }
        with _DBG_LOCK:
            with _DBG_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# #endregion

POD_NAME = "shtetlframes-scan"
POD_NAME_PREFIX = "shtetlframes-scan"
HTTP_PORT = 8000
GRAPHQL = "https://api.runpod.io/graphql"
# Scrape GPUs (RUNPOD_MAX_INFLIGHT). Hard account ceiling adds +1 discover.
MAX_PARALLEL_PODS = 8
MAX_DISCOVER_EXTRA = 1
MAX_SHTETL_PODS = MAX_PARALLEL_PODS + MAX_DISCOVER_EXTRA  # 9 absolute

# Serialize creates / trims so concurrent ensure_pods + background fills cannot stampede.
_ensure_lock = threading.RLock()
_bg_fill_lock = threading.Lock()
_bg_fill_active = False
# Names reserved by in-flight creates (GraphQL list lags → duplicate names otherwise).
_claimed_names: set[str] = set()
_claimed_names_lock = threading.Lock()
# When True, ensure_pods / create_pod will not spin up new machines (discover-only).
_pod_creates_blocked = False
_pod_creates_lock = threading.Lock()
# Soft cap on live shtetl pods while discover runs (scrape clears this).
# None = normal account cap (MAX_INFLIGHT + discover spare).
_pod_create_ceiling: int | None = None


def set_pod_creates_blocked(blocked: bool = True) -> None:
    """Block (or allow) creating new ShtetlFrames RunPod GPUs (persisted)."""
    import os

    global _pod_creates_blocked
    flag = bool(blocked)
    with _pod_creates_lock:
        _pod_creates_blocked = flag
    os.environ["POD_CREATES_BLOCKED"] = "1" if flag else "0"
    try:
        from settings_store import set_settings

        set_settings({"POD_CREATES_BLOCKED": "1" if flag else "0"})
    except Exception:
        pass


def pod_creates_blocked() -> bool:
    import os

    with _pod_creates_lock:
        if _pod_creates_blocked:
            return True
    raw = (os.environ.get("POD_CREATES_BLOCKED") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        from settings_store import get_setting

        raw = (get_setting("POD_CREATES_BLOCKED") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")
    except Exception:
        return False


def set_pod_create_ceiling(n: int | None) -> None:
    """Limit how many shtetl pods may exist (creates + trim keep).

    Discover sets this to ``1`` so listing cannot grow a scrape fleet.
    Pathé / YouTube scrape clears it (``None``) before ``ensure_pods``.
    """
    global _pod_create_ceiling
    with _pod_creates_lock:
        if n is None:
            _pod_create_ceiling = None
        else:
            _pod_create_ceiling = max(1, int(n))


def pod_create_ceiling() -> int | None:
    with _pod_creates_lock:
        return _pod_create_ceiling

# Prefer cheaper capable GPUs first; fall back when a pool is full.
GPU_FALLBACKS = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA L4",
    "Tesla T4",
]

OnStatus = Callable[[str], None]


def _api_key() -> str:
    load_env()
    key = (app_config.RUNPOD_API_KEY or "").strip()
    if not key:
        raise RuntimeError("RUNPOD_API_KEY missing — set it in Settings")
    return key


def _gql(query: str, variables: dict | None = None) -> dict[str, Any]:
    r = requests.post(
        f"{GRAPHQL}?api_key={_api_key()}",
        headers={"Content-Type": "application/json"},
        json={"query": query, "variables": variables or {}},
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"runpod_graphql_http_{r.status_code}: {r.text[:500]}")
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"runpod_graphql: {data['errors']}")
    return data.get("data") or {}


def _err_is_capacity(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "does not have the resources",
        "no longer any instances",
        "not enough",
        "no instances available",
        "enough disk space",
        "try a different machine",
        "unavailable",
    )
    return any(n in msg for n in needles)


def _err_is_bad_image(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "was not found on the registry" in msg or "image" in msg and "not found" in msg


def list_pods() -> list[dict]:
    data = _gql(
        """
        query {
          myself {
            pods {
              id
              name
              desiredStatus
              imageName
              runtime {
                uptimeInSeconds
                ports {
                  privatePort
                  publicPort
                  type
                  isIpPublic
                }
              }
            }
          }
        }
        """
    )
    return list((data.get("myself") or {}).get("pods") or [])


def _is_shtetl_name(name: str) -> bool:
    n = (name or "").strip()
    return n == POD_NAME or n.startswith(POD_NAME_PREFIX + "-")


def pod_slot_name(index: int) -> str:
    """index 0 → shtetlframes-scan, else shtetlframes-scan-2, -3, …"""
    if index <= 0:
        return POD_NAME
    return f"{POD_NAME_PREFIX}-{index + 1}"


def _name_slot(name: str) -> int:
    n = (name or "").strip()
    if n == POD_NAME:
        return 0
    prefix = POD_NAME_PREFIX + "-"
    if n.startswith(prefix):
        try:
            return max(0, int(n[len(prefix) :]) - 1)
        except ValueError:
            return 999
    return 999


def shtetl_account_cap() -> int:
    """Hard ceiling on shtetlframes-scan* pods for this account (scrape + discover)."""
    load_env()
    scrape = max(
        1,
        min(int(getattr(app_config, "RUNPOD_MAX_INFLIGHT", None) or 8), MAX_PARALLEL_PODS),
    )
    cap = min(MAX_SHTETL_PODS, scrape + MAX_DISCOVER_EXTRA)
    ceiling = pod_create_ceiling()
    if ceiling is not None:
        return max(1, min(cap, int(ceiling)))
    return cap


def find_shtetl_pod() -> dict | None:
    pods = find_shtetl_pods()
    return pods[0] if pods else None


def find_shtetl_pods() -> list[dict]:
    out = [p for p in list_pods() if _is_shtetl_name(p.get("name") or "")]
    out.sort(key=lambda p: (_name_slot(p.get("name") or ""), p.get("name") or "", p.get("id") or ""))
    return out


def _pod_has_http_ports(pod: dict) -> bool:
    runtime = pod.get("runtime")
    ports = (runtime or {}).get("ports") or [] if isinstance(runtime, dict) else []
    return bool(ports)


def trim_shtetl_pods(
    *,
    keep: int | None = None,
    on_status: OnStatus | None = None,
) -> int:
    """Terminate surplus shtetl pods so account count never exceeds ``keep``.

    Keeps lowest slot numbers; among duplicate names, keeps the one with HTTP ports.
    Returns how many were terminated.
    """
    cap = int(keep) if keep is not None else shtetl_account_cap()
    cap = max(1, min(cap, MAX_SHTETL_PODS))
    with _ensure_lock:
        pods = find_shtetl_pods()
        if len(pods) <= cap:
            return 0

        # Rank: low slot, has ports, RUNNING, stable id. Drop the rest.
        ranked = sorted(
            pods,
            key=lambda p: (
                _name_slot(p.get("name") or ""),
                # Prefer unique names: when equal slot/name, prefer ports + RUNNING.
                0 if _pod_has_http_ports(p) else 1,
                0 if (p.get("desiredStatus") or "").upper() == "RUNNING" else 1,
                p.get("name") or "",
                p.get("id") or "",
            ),
        )
        # Among same name, keep only the best-ranked one.
        seen_names: set[str] = set()
        keep_list: list[dict] = []
        kill_list: list[dict] = []
        for p in ranked:
            name = (p.get("name") or "").strip()
            if name and name in seen_names:
                kill_list.append(p)
                continue
            if name:
                seen_names.add(name)
            if len(keep_list) < cap:
                keep_list.append(p)
            else:
                kill_list.append(p)

        for p in kill_list:
            pid = p.get("id")
            if not pid:
                continue
            name = p.get("name") or pid[:12]
            if on_status:
                on_status(f"terminating surplus pod {name} (cap={cap})…")
            try:
                terminate_pod(pid)
            except Exception as e:
                if on_status:
                    on_status(f"terminate surplus warning: {e}"[:160])
        return len(kill_list)


def pod_proxy_url(pod_id: str, port: int = HTTP_PORT) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _gpu_try_order(preferred: str) -> list[str]:
    pref = (preferred or "").strip()
    out: list[str] = []
    if pref:
        out.append(pref)
    for g in GPU_FALLBACKS:
        if g not in out:
            out.append(g)
    return out


def create_pod_once(
    *,
    image: str,
    gpu_type: str,
    docker_args: str = "",
    cloud_type: str = "ALL",
    container_disk_gb: int = 20,
    min_vcpu: int = 2,
    min_memory_gb: int = 8,
    name: str = POD_NAME,
) -> dict:
    """Single deploy attempt. Raises RuntimeError on GraphQL / capacity errors."""
    q = """
    mutation ($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        imageName
        desiredStatus
        machine { podHostId }
      }
    }
    """
    input_body = {
        "cloudType": cloud_type,
        "gpuCount": 1,
        "volumeInGb": 0,
        "containerDiskInGb": container_disk_gb,
        "minVcpuCount": min_vcpu,
        "minMemoryInGb": min_memory_gb,
        "gpuTypeId": gpu_type,
        "name": name or POD_NAME,
        "imageName": image,
        "dockerArgs": docker_args or "",
        "ports": f"{HTTP_PORT}/http",
        "volumeMountPath": "/workspace",
        "env": [
            {"key": "SHTETL_HTTP", "value": "1"},
            {"key": "PYTHONUNBUFFERED", "value": "1"},
        ],
    }
    # Do NOT bake residential proxy into pod env — that makes cookies-only jobs still
    # hit Scrapfly via YT_PROXY_URL (rate-limit burn). Proxy URL is sent per-job.
    data = _gql(q, {"input": input_body})
    pod = data.get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        raise RuntimeError(f"pod_create_failed: {data}")
    return pod


def create_pod(
    *,
    image: str,
    gpu_type: str,
    docker_args: str = "",
    on_status: OnStatus | None = None,
    skip_images: set[str] | None = None,
    name: str = POD_NAME,
) -> dict:
    """
    Deploy on-demand GPU pod. Retries across images / GPU types / lighter resource asks
    when RunPod reports capacity misses or a bad image tag.
    """
    if pod_creates_blocked():
        raise RuntimeError("pod_creates_blocked — not creating new GPUs")
    from runpod_bootstrap import IMAGE_CANDIDATES

    skip = {s.strip() for s in (skip_images or set()) if s and s.strip()}
    images: list[str] = []
    for i in [image, *IMAGE_CANDIDATES]:
        i = (i or "").strip()
        if i and i not in images and i not in skip:
            images.append(i)
    if not images:
        raise RuntimeError("no RunPod images left to try (all skipped after failed starts)")

    last_err: Exception | None = None
    for img in images:
        if on_status:
            on_status(f"image {img}")
        for gpu in _gpu_try_order(gpu_type):
            attempts = [
                {"cloud_type": "ALL", "container_disk_gb": 20, "min_vcpu": 2, "min_memory_gb": 8},
                {"cloud_type": "COMMUNITY", "container_disk_gb": 20, "min_vcpu": 2, "min_memory_gb": 8},
                {"cloud_type": "SECURE", "container_disk_gb": 20, "min_vcpu": 2, "min_memory_gb": 8},
                {"cloud_type": "ALL", "container_disk_gb": 30, "min_vcpu": 4, "min_memory_gb": 15},
            ]
            for attempt in attempts:
                label = f"{gpu} · {attempt['cloud_type']} · disk={attempt['container_disk_gb']}GB"
                if on_status:
                    on_status(f"trying {label}…")
                try:
                    pod = create_pod_once(
                        image=img,
                        gpu_type=gpu,
                        docker_args=docker_args,
                        cloud_type=attempt["cloud_type"],
                        container_disk_gb=attempt["container_disk_gb"],
                        min_vcpu=attempt["min_vcpu"],
                        min_memory_gb=attempt["min_memory_gb"],
                        name=name or POD_NAME,
                    )
                    if on_status:
                        on_status(f"pod created · {name or POD_NAME} · {gpu} · {img}")
                    return pod
                except Exception as e:
                    last_err = e
                    if _err_is_bad_image(e):
                        if on_status:
                            on_status(f"image not on registry — try next image")
                        break  # next image
                    if _err_is_capacity(e):
                        if on_status:
                            on_status(f"no capacity on {label} — next…")
                        time.sleep(1.2)
                        continue
                    if on_status:
                        on_status(f"failed {label}: {str(e)[:160]}")
                    time.sleep(0.4)
                    continue
            else:
                continue
            break  # bad image → outer image loop
        else:
            continue
        # bad image broke inner loops
        continue

    raise RuntimeError(
        "Could not create a RunPod GPU pod (images / capacity). "
        "Wait a minute and Start scrape again. "
        f"Last error: {last_err}"
    )


def terminate_pod(pod_id: str) -> None:
    _gql(
        f"""
        mutation {{
          podTerminate(input: {{ podId: "{pod_id}" }})
        }}
        """
    )


def resume_pod(pod_id: str) -> None:
    _gql(
        f"""
        mutation {{
          podResume(input: {{ podId: "{pod_id}", gpuCount: 1 }}) {{
            id desiredStatus
          }}
        }}
        """
    )


def stop_pod(pod_id: str | None = None) -> None:
    load_env()
    if pod_id:
        targets = [pod_id]
    else:
        targets = [p["id"] for p in find_shtetl_pods() if p.get("id")]
        if not targets:
            pid = (app_config.RUNPOD_POD_ID or "").strip()
            if pid:
                targets = [pid]
    for pid in targets:
        try:
            _gql(
                f"""
                mutation {{
                  podStop(input: {{ podId: "{pid}" }}) {{
                    id desiredStatus
                  }}
                }}
                """
            )
        except Exception:
            pass


def terminate_shtetl_pods(*, on_status: OnStatus | None = None) -> None:
    global _claimed_names
    for p in find_shtetl_pods():
        pid = p.get("id")
        if not pid:
            continue
        if on_status:
            on_status(f"terminating {p.get('name') or pid[:12]}…")
        try:
            terminate_pod(pid)
        except Exception as e:
            if on_status:
                on_status(f"terminate warning: {e}")
    with _claimed_names_lock:
        _claimed_names.clear()
    # Wait until GraphQL stops listing them — otherwise live-count gates
    # think the account is still full and under-create the replacement pool.
    deadline = time.time() + 120.0
    while time.time() < deadline:
        left = find_shtetl_pods()
        if not left:
            break
        if on_status:
            on_status(f"waiting for {len(left)} pod(s) to finish terminate…")
        time.sleep(3.0)
    time.sleep(1.0)


def _image_driver_risky(image: str) -> bool:
    from runpod_bootstrap import DRIVER_RISKY_IMAGE_MARKERS

    img = (image or "").lower()
    return any(m in img for m in DRIVER_RISKY_IMAGE_MARKERS)


def wait_healthy(
    base_url: str,
    *,
    pod_id: str | None = None,
    timeout_sec: float = 1200,
    on_status: OnStatus | None = None,
) -> None:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_sec:
        elapsed = int(time.time() - t0)
        note = f"waiting for GPU pod · {elapsed}s · first boot installs deps (~5–15 min) · {base_url}"
        if on_status and note != last:
            on_status(note)
            last = note
        if pod_id and elapsed >= 30:
            try:
                pod = next((p for p in list_pods() if p.get("id") == pod_id), None)
                if pod:
                    status = (pod.get("desiredStatus") or "").upper()
                    runtime = pod.get("runtime")
                    ports = (runtime or {}).get("ports") or []
                    # runtime=null or no HTTP ports = container not up yet.
                    # GPU attach can take a few minutes — do not treat as dead early.
                    dead_runtime = runtime is None or not ports
                    if status in ("EXITED", "FAILED"):
                        raise RuntimeError(
                            f"pod_container_dead status={status or '?'} "
                            f"runtime={'null' if runtime is None else 'ok'} "
                            f"ports={len(ports)}"
                        )
                    if (
                        elapsed >= 480
                        and status == "RUNNING"
                        and dead_runtime
                    ):
                        raise RuntimeError(
                            f"pod_container_dead status={status or '?'} "
                            f"runtime={'null' if runtime is None else 'ok'} "
                            f"ports={len(ports)} "
                            f"(often CUDA driver < image requirement — try older CUDA image)"
                        )
            except RuntimeError:
                raise
            except Exception:
                pass
        try:
            r = requests.get(f"{base_url.rstrip('/')}/health", timeout=15)
            # 502/503/404 while bootstrap installs deps + fetches worker is normal
            # for ~5–15 min (proxy port mapped, uvicorn not up yet). Do NOT treat
            # early 404 as fatal — that was killing every supplemental pod at ~4 min.
            # Only give up on persistent 404 near the end of the health window.
            if (
                r.status_code == 404
                and elapsed >= min(900, max(600, timeout_sec * 0.75))
                and pod_id
            ):
                raise RuntimeError(
                    f"pod_proxy_dead http_404 after {elapsed}s — replacing"
                )
            if r.status_code == 200 and r.content:
                data = r.json() if r.content else {}
                if isinstance(data, dict) and data.get("ok") and data.get("models_ready", True):
                    if on_status:
                        on_status(f"GPU pod ready · {elapsed}s")
                    return
                if isinstance(data, dict) and data.get("warm_error") and elapsed >= 180:
                    raise RuntimeError(f"pod_warm_failed: {data.get('warm_error')}"[:240])
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"pod_health_timeout after {int(timeout_sec)}s: {base_url}")


def _persist_pod_id(pod_id: str) -> None:
    try:
        from settings_store import ensure_settings_table
        from db import db
        import time as _t

        ensure_settings_table()
        with db() as conn:
            conn.execute(
                """INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                ("RUNPOD_POD_ID", pod_id, _t.time()),
            )
        import os

        os.environ["RUNPOD_POD_ID"] = pod_id
        app_config.RUNPOD_POD_ID = pod_id
    except Exception:
        pass


def ensure_pod(*, on_status: OnStatus | None = None, recreate: bool = False) -> str:
    """Find or create one ShtetlFrames GPU pod; return base HTTP URL."""
    urls = ensure_pods(count=1, on_status=on_status, recreate=recreate)
    return urls[0]


def ensure_pods(
    *,
    count: int = 1,
    on_status: OnStatus | None = None,
    recreate: bool = False,
    min_ready: int = 1,
    extra_fill_sec: float = 90.0,
) -> list[str]:
    """Ensure N parallel GPU pods; return proxy base URLs (round-robin pool).

    Reuses healthy pods first. Once ``min_ready`` are up, only spends
    ``extra_fill_sec`` trying to fill the rest. Pass ``extra_fill_sec=0`` to
    return immediately after the first healthy pod (background fill can expand later).

    Hard account cap is ``shtetl_account_cap()`` (scrape max + 1 discover) — never
    creates past that, even when multiple callers / background fills race.
    """
    from runpod_bootstrap import DEFAULT_BASE_IMAGE, IMAGE_CANDIDATES, docker_start_args

    load_env()
    account_cap = shtetl_account_cap()
    n = max(1, min(int(count or 1), account_cap, MAX_SHTETL_PODS))
    min_ready = max(1, min(int(min_ready or 1), n))
    preferred = (app_config.RUNPOD_DOCKER_IMAGE or "").strip() or DEFAULT_BASE_IMAGE
    gpu = (app_config.RUNPOD_GPU_TYPE or "NVIDIA GeForce RTX 3090").strip()
    args = docker_start_args()
    creates_blocked = pod_creates_blocked()
    if creates_blocked and on_status:
        on_status("Pod creates blocked — reusing healthy pods only")

    # Prefer the image already proven healthy on this account (matches live pods).
    proven_images: list[str] = []
    for p in find_shtetl_pods():
        img = (p.get("imageName") or "").strip()
        if img and img not in proven_images:
            proven_images.append(img)

    images: list[str] = []
    for i in [*proven_images, preferred, *IMAGE_CANDIDATES]:
        i = (i or "").strip()
        if i and i not in images:
            images.append(i)

    if recreate:
        if on_status:
            on_status(f"recreating up to {n} GPU pod(s)…")
        terminate_shtetl_pods(on_status=on_status)
    else:
        trimmed = trim_shtetl_pods(keep=account_cap, on_status=on_status)
        if trimmed and on_status:
            on_status(f"trimmed {trimmed} surplus pod(s) → cap {account_cap}")

    ready: list[tuple[str, str]] = []  # (pod_id, base_url)
    # Pods still cold-starting (null runtime / deps installing). Must NOT be
    # terminated as "zombies" — that caused replace storms + over-create.
    booting: list[tuple[str, str, str]] = []  # name, pid, base
    # Include every healthy paid GPU in the return pool (up to scrape cap),
    # even when create-target ``n`` is lower — otherwise orphan pods sit idle.
    scrape_pool_cap = min(MAX_PARALLEL_PODS, account_cap)
    existing = find_shtetl_pods()
    for p in existing:
        pid = p.get("id")
        if not pid:
            continue
        name = p.get("name") or pid[:12]
        old_image = (p.get("imageName") or "").strip()
        status = (p.get("desiredStatus") or "").upper()
        wrong_image = bool(old_image) and (
            not any(x in old_image for x in ("runpod/pytorch", "pytorch/pytorch"))
            or _image_driver_risky(old_image)
        )
        if wrong_image:
            if on_status:
                on_status(f"replacing bad image pod {name}…")
            try:
                terminate_pod(pid)
            except Exception:
                pass
            continue
        if status in ("EXITED", "STOPPED", ""):
            if on_status:
                on_status(f"resuming {name}…")
            try:
                resume_pod(pid)
            except Exception as e:
                if on_status:
                    on_status(f"resume failed ({e}) — will create new")
                try:
                    terminate_pod(pid)
                except Exception:
                    pass
                continue
        runtime = p.get("runtime")
        ports = (runtime or {}).get("ports") or []
        base = pod_proxy_url(pid)
        # Null runtime / no ports = normal for several minutes after create.
        # Leave the pod alone and count it toward the fill target.
        if status == "RUNNING" and (runtime is None or not ports):
            if on_status:
                on_status(f"pod still booting · {name} (runtime not up yet)")
            booting.append((name, pid, base))
            continue
        if len(ready) >= scrape_pool_cap:
            continue
        try:
            # Quick probe: warm pods answer immediately. Cold boots 404/ports=0 —
            # keep this short so Pathé discover is not stuck 45s×N on zombies.
            wait_healthy(base, pod_id=pid, on_status=on_status, timeout_sec=8)
            ready.append((pid, base))
            if on_status:
                on_status(f"pod ready · {name} · {len(ready)}/{max(n, len(ready))}")
        except (TimeoutError, RuntimeError) as e:
            err = str(e)
            # Definite container death → replace. Soft bootstrap misses → keep.
            fatal = (
                "pod_container_dead" in err
                or "pod_warm_failed" in err
                or status in ("EXITED", "FAILED")
            )
            if fatal:
                if on_status:
                    on_status(f"pod dead ({err}) — replacing")
                try:
                    terminate_pod(pid)
                except Exception:
                    pass
            else:
                if on_status:
                    on_status(f"pod still booting · {name} ({err[:80]})")
                booting.append((name, pid, base))

    # Do not truncate ready to ``n`` — keep all healthy pods in the scrape pool.

    # Soft fill: once min_ready are up, optionally return early and create the rest
    # in a background thread (extra_fill_sec=0). Otherwise spend extra_fill_sec creating.
    fill_deadline: float | None = None
    return_early = False
    live_n = len(find_shtetl_pods())
    have_or_booting = max(len(ready) + len(booting), live_n)
    if len(ready) >= min_ready and have_or_booting < n:
        if float(extra_fill_sec) <= 0:
            return_early = True
            if on_status:
                on_status(
                    f"{len(ready)}/{n} pods ready"
                    + (f", {len(booting)} booting" if booting else "")
                    + f" — starting now; filling to {n} in background…"
                )
        else:
            fill_deadline = time.time() + float(extra_fill_sec)
            if on_status:
                on_status(
                    f"{len(ready)}/{n} pods ready — scanning can start; "
                    f"trying for more (~{int(extra_fill_sec)}s)…"
                )
    elif len(ready) >= min_ready and have_or_booting >= n and len(ready) < n:
        # Enough pods exist (ready+booting / live); just wait — don't create more.
        if float(extra_fill_sec) <= 0:
            return_early = True
            if on_status:
                on_status(
                    f"{len(ready)}/{n} ready, live={live_n}/{account_cap} — "
                    f"waiting for bootstrap in background…"
                )
        else:
            fill_deadline = time.time() + float(extra_fill_sec)
    elif len(ready) < min_ready and float(extra_fill_sec) > 0:
        # Still short of min_ready — honor extra_fill_sec instead of deadline=None
        # (which used to block forever in _create_until_full).
        fill_deadline = time.time() + float(extra_fill_sec)

    # #region agent log
    _dbg(
        "A",
        "runpod_provision.py:ensure_pods",
        "after_probe_branch",
        n=n,
        min_ready=min_ready,
        extra_fill_sec=float(extra_fill_sec),
        ready=len(ready),
        booting=len(booting),
        live_n=live_n,
        have_or_booting=have_or_booting,
        return_early=return_early,
        fill_deadline=fill_deadline,
        creates_blocked=creates_blocked,
        caller_tid=threading.get_ident(),
    )
    # #endregion

    def _live_count() -> int:
        try:
            return len(find_shtetl_pods())
        except Exception:
            return account_cap  # fail closed: do not create

    def _create_until_full(
        ready_now: list[tuple[str, str]],
        *,
        already_booting: list[tuple[str, str, str]] | None = None,
        deadline: float | None,
        status_cb: OnStatus | None,
        target: int,
    ) -> list[tuple[str, str]]:
        """Create missing pods, re-checking the live account count before each create."""
        target = max(1, min(int(target), account_cap, MAX_SHTETL_PODS))
        slot = 0
        last_err: Exception | None = None
        out = list(ready_now)
        pending: list[tuple[str, str, str]] = list(already_booting or [])

        def _wait_one(item: tuple[str, str, str]) -> tuple[str, str] | None:
            name, pid, base = item
            try:
                wait_healthy(
                    base,
                    pod_id=pid,
                    on_status=status_cb,
                    timeout_sec=1200.0,
                )
                if status_cb:
                    status_cb(f"pod ready · {name}")
                return (pid, base)
            except (TimeoutError, RuntimeError) as e:
                if status_cb:
                    status_cb(f"{name} unhealthy ({e}) — dropping")
                try:
                    terminate_pod(pid)
                except Exception:
                    pass
                return None

        # At most 2 replacement waves — old code ran 3 and over-created under races.
        for wave in range(1, 3):
            if len(out) >= target:
                break
            if deadline is not None and time.time() >= deadline:
                if status_cb:
                    status_cb(
                        f"starting with {len(out)}/{target} pods "
                        f"(not waiting longer for more)"
                    )
                break

            while True:
                if deadline is not None and time.time() >= deadline:
                    break
                live = _live_count()
                # Hard stop: never exceed account cap OR this call's target.
                if live >= account_cap or live >= target:
                    break
                if len(out) + len(pending) >= target:
                    break

                with _ensure_lock:
                    # Re-check under lock right before the GraphQL create.
                    live = _live_count()
                    if live >= account_cap or live >= target:
                        break
                    used_names = {(p.get("name") or "") for p in find_shtetl_pods()}
                    used_names |= {nm for nm, _, _ in pending}
                    with _claimed_names_lock:
                        used_names |= set(_claimed_names)
                    name = None
                    # Prefer slots 0..cap-1; if GraphQL still lists ghosts, keep going
                    # a few past cap with unique higher indexes (trimmed later).
                    while slot < account_cap + 4:
                        cand = pod_slot_name(slot)
                        slot += 1
                        if cand not in used_names:
                            name = cand
                            break
                    if not name:
                        if status_cb:
                            status_cb(
                                f"no free pod slot under cap {account_cap} "
                                f"(live={live}) — stop creating"
                            )
                        break
                    with _claimed_names_lock:
                        _claimed_names.add(name)

                    tried_dead: set[str] = set()
                    created = False
                    try:
                        for img in images:
                            if img in tried_dead:
                                continue
                            if deadline is not None and time.time() >= deadline:
                                break
                            # Final gate inside lock.
                            live = _live_count()
                            if live >= account_cap or live >= target:
                                break
                            if status_cb:
                                status_cb(
                                    f"creating {name} · {img} "
                                    f"(live {live}/{account_cap}, want {target}"
                                    f"{'' if wave == 1 else f', wave {wave}'})…"
                                )
                            try:
                                pod = create_pod(
                                    image=img,
                                    gpu_type=gpu,
                                    docker_args=args,
                                    on_status=status_cb,
                                    skip_images=tried_dead,
                                    name=name,
                                )
                                pid = pod["id"]
                                base = pod_proxy_url(pid)
                                pending.append((name, pid, base))
                                created = True
                                if status_cb:
                                    status_cb(
                                        f"pod created · {name} — booting "
                                        f"(live≈{live + 1}/{account_cap})"
                                    )
                                break
                            except (TimeoutError, RuntimeError) as e:
                                last_err = e
                                used = img
                                try:
                                    dead = next(
                                        (
                                            p
                                            for p in find_shtetl_pods()
                                            if (p.get("name") or "") == name
                                        ),
                                        None,
                                    )
                                    if dead and dead.get("id"):
                                        used = (dead.get("imageName") or img).strip()
                                        terminate_pod(dead["id"])
                                except Exception:
                                    pass
                                tried_dead.add(used)
                                if status_cb:
                                    status_cb(f"{name} failed ({e}) — try next image…")
                                time.sleep(1.5)
                                if "could not create a runpod" in str(e).lower():
                                    break
                    finally:
                        if not created:
                            with _claimed_names_lock:
                                _claimed_names.discard(name)
                    if not created:
                        break
                # end lock — allow other readers between creates

            if pending:
                if status_cb:
                    status_cb(
                        f"waiting for {len(pending)} pod(s) to finish bootstrap "
                        f"(~5–15 min)…"
                    )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, len(pending))
                ) as pool:
                    for result in pool.map(_wait_one, pending):
                        if result:
                            out.append(result)
                pending = []

            # After a wave, trim again in case something raced past the cap.
            trim_shtetl_pods(keep=account_cap, on_status=status_cb)

            if len(out) < target and wave < 2 and status_cb:
                # Only replace if we are under target AND under account cap.
                if _live_count() < min(target, account_cap):
                    # #region agent log
                    _dbg(
                        "A",
                        "runpod_provision.py:_create_until_full",
                        "wave_retry",
                        wave=wave,
                        out=len(out),
                        target=target,
                        live=_live_count(),
                        deadline=deadline,
                        deadline_left=(
                            None
                            if deadline is None
                            else round(deadline - time.time(), 1)
                        ),
                    )
                    # #endregion
                    status_cb(
                        f"{len(out)}/{target} healthy after wave {wave} — "
                        f"retrying replacements…"
                    )
                else:
                    break

        if not out:
            raise RuntimeError(
                "Could not start a healthy RunPod GPU pod (CUDA image / capacity). "
                f"Last error: {last_err}"
            )
        if len(out) < target and status_cb:
            status_cb(f"only {len(out)}/{target} pods available — continuing with those")
        # #region agent log
        _dbg(
            "A",
            "runpod_provision.py:_create_until_full",
            "exit",
            out=len(out),
            target=target,
            deadline=deadline,
        )
        # #endregion
        return out

    def _start_bg_fill(
        snapshot: list[tuple[str, str]],
        boot_snap: list[tuple[str, str, str]],
    ) -> None:
        global _bg_fill_active
        with _bg_fill_lock:
            if _bg_fill_active:
                if on_status:
                    on_status("background pod fill already running — skip duplicate")
                return
            _bg_fill_active = True

        def _bg_fill() -> None:
            global _bg_fill_active

            def _bg_status(msg: str) -> None:
                # Do not call on_status — Pathé scrape uses it to set_job(message=…)
                # and bg "waiting for GPU" lines were stomping live scrape progress.
                print(f"[shtetl] bg-fill: {(msg or '')[:140]}", flush=True)

            try:
                _create_until_full(
                    snapshot,
                    already_booting=boot_snap,
                    deadline=None,
                    status_cb=_bg_status,
                    target=n,
                )
            except Exception as e:
                _bg_status(f"background pod fill failed: {e}"[:160])
            finally:
                with _bg_fill_lock:
                    _bg_fill_active = False

        threading.Thread(
            target=_bg_fill, daemon=True, name="runpod-ensure-fill"
        ).start()

    if creates_blocked:
        # Discover-only / manual freeze — never create or bg-fill.
        if not ready:
            raise RuntimeError("pod_creates_blocked and no healthy pods")
        _persist_pod_id(ready[0][0])
        return [base for _, base in ready]

    if return_early and ready:
        # #region agent log
        _dbg(
            "A",
            "runpod_provision.py:ensure_pods",
            "return_early",
            ready=len(ready),
            booting=len(booting),
            n=n,
        )
        # #endregion
        _start_bg_fill(list(ready), list(booting))
        _persist_pod_id(ready[0][0])
        return [base for _, base in ready]

    # Already have enough healthy pods — do not block on cold boots.
    if len(ready) >= max(min_ready, 1) and len(ready) >= n:
        if booting:
            _start_bg_fill(list(ready), list(booting))
        _persist_pod_id(ready[0][0])
        # #region agent log
        _dbg(
            "A",
            "runpod_provision.py:ensure_pods",
            "return_full",
            ready=len(ready),
            n=n,
        )
        # #endregion
        return [base for _, base in ready]

    if len(ready) < n:
        # extra_fill_sec=0 means: block only until min_ready, then scrape while
        # a background thread fills to ``n``. Previously deadline=None + target=n
        # blocked the scrape coordinator until ALL pods were healthy (e.g. stuck
        # at "6/8 healthy after wave 1 — retrying replacements…").
        block_target = min_ready if float(extra_fill_sec) <= 0 else n
        # #region agent log
        _dbg(
            "A",
            "runpod_provision.py:ensure_pods",
            "enter_create_until_full",
            ready=len(ready),
            booting=len(booting),
            n=n,
            min_ready=min_ready,
            block_target=block_target,
            fill_deadline=fill_deadline,
            soft_return=float(extra_fill_sec) <= 0,
        )
        # #endregion
        if len(ready) < block_target:
            ready = _create_until_full(
                ready,
                already_booting=list(booting),
                deadline=fill_deadline,
                status_cb=on_status,
                target=block_target,
            )
        if float(extra_fill_sec) <= 0 and ready:
            # #region agent log
            _dbg(
                "A",
                "runpod_provision.py:ensure_pods",
                "soft_return_min_ready",
                ready=len(ready),
                n=n,
                min_ready=min_ready,
            )
            # #endregion
            if on_status:
                on_status(
                    f"{len(ready)}/{n} pods ready — starting scrape; "
                    f"filling remaining in background…"
                )
            _start_bg_fill(list(ready), list(booting))
            _persist_pod_id(ready[0][0])
            return [base for _, base in ready]

    if not ready:
        raise RuntimeError("Could not start GPU pods.")

    _persist_pod_id(ready[0][0])
    # #region agent log
    _dbg(
        "A",
        "runpod_provision.py:ensure_pods",
        "return_after_create",
        ready=len(ready),
        n=n,
    )
    # #endregion
    return [base for _, base in ready]
