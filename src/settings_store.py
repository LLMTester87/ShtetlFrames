"""UI-persisted app settings (overrides .env). Stored in SQLite + synced to .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ROOT
from db import db, init_db

SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL
);
"""

# Keys editable from the UI — RunPod API key only (no Docker / Hub)
SETTING_DEFS: dict[str, dict[str, Any]] = {
    "RUNPOD_API_KEY": {
        "label": "RunPod API key",
        "type": "password",
        "default": "",
        "help": "From RunPod → Settings → API Keys. Cloud GPU starts automatically — no Docker.",
        "secret": True,
    },
    "RUNPOD_GPU_TYPE": {
        "label": "GPU type (preferred)",
        "type": "text",
        "default": "NVIDIA GeForce RTX 3090",
        "help": "Tried first; if busy, auto-falls back to 4090 / A5000 / A4000 / T4 etc.",
    },
    "RUNPOD_MAX_INFLIGHT": {
        "label": "Parallel GPU pods",
        "type": "number",
        "default": "8",
        "min": 1,
        "max": 8,
        "help": "Scrape GPU pods (1–8). Pathé may add +1 dedicated discover pod (hard cap 9).",
    },
    "PATHE_STACK_MAX": {
        "label": "Pathé jobs per GPU",
        "type": "number",
        "default": "3",
        "min": 1,
        "max": 6,
        "help": "How many Pathé scans each GPU may run at once (1–6). Applies live while scrape is running. Higher = faster, more 503/OOM risk — watch /health.",
    },
    "RUNPOD_STOP_WHEN_DONE": {
        "label": "Stop pod when done",
        "type": "select",
        "options": ["1", "0"],
        "default": "1",
        "help": "1 = stop GPU pod after scrape (saves money)",
    },
    "SCORE_THRESHOLD": {
        "label": "Hit score threshold",
        "type": "number",
        "default": "0.04",
        "min": -0.5,
        "max": 0.25,
        "step": 0.01,
        "help": "CLIP gate before vision verify (default 0.04). Raise toward 0.08–0.10 to cut Pathé noise.",
    },
    "YT_COOKIES_BROWSER": {
        "label": "YouTube cookies browser",
        "type": "select",
        "options": ["edge", "chrome", "firefox", "brave", "opera", "none"],
        "default": "edge",
        "help": "Browser signed into YouTube. Export tries Edge/Chrome/Brave profiles automatically. If Chrome DB is missing, pick Edge — or set a residential proxy below.",
    },
    "PROXY_PROVIDER": {
        "label": "Provider",
        "type": "select",
        "options": ["auto", "scrapfly", "scrapingdog", "none"],
        "option_labels": {
            "auto": "auto — Scrapfly, then ScrapingDog",
            "scrapfly": "Scrapfly",
            "scrapingdog": "ScrapingDog",
            "none": "none",
        },
        "default": "auto",
        "section": "YouTube proxy",
        "help": "Used on the GPU when YouTube blocks the cloud IP. Scrapfly preferred; ScrapingDog is the fallback.",
    },
    "SCRAPFLY_API_KEY": {
        "label": "Scrapfly API key",
        "type": "password",
        "default": "",
        "section": "YouTube proxy",
        "visible_for": ["auto", "scrapfly"],
        "help": "https://scrapfly.io/dashboard — YouTube proxy + British Pathé HTML scrape.",
        "secret": True,
    },
    "SCRAPINGDOG_API_KEY": {
        "label": "ScrapingDog API key",
        "type": "password",
        "default": "",
        "section": "YouTube proxy",
        "visible_for": ["auto", "scrapingdog"],
        "help": "https://www.scrapingdog.com/dashboard — fallback when Scrapfly is busy or unset.",
        "secret": True,
    },
    "OPENAI_VERIFY": {
        "label": "Vision second pass",
        "type": "select",
        "options": ["1", "0"],
        "default": "1",
        "help": "1 = re-check each CLIP hit with a vision model before Review. 0 = CLIP only.",
        "section": "Vision verify",
    },
    "VERIFY_BACKEND": {
        "label": "Verify backend",
        "type": "select",
        "options": ["ollama_then_openai", "openai", "open_vlm"],
        "option_labels": {
            "ollama_then_openai": "Ollama first → OpenAI on keeps only",
            "openai": "OpenAI only (GPT vision)",
            "open_vlm": "Ollama / open VLM only",
        },
        "default": "openai",
        "section": "Vision verify",
        "help": "openai = GPT only (current). ollama_then_openai = Ollama first, OpenAI on keeps. open_vlm = Ollama only.",
    },
    "OPENAI_API_KEY": {
        "label": "OpenAI API key",
        "type": "password",
        "default": "",
        "section": "Vision verify",
        "visible_for_key": "VERIFY_BACKEND",
        "visible_for": ["openai", "ollama_then_openai"],
        "help": "https://platform.openai.com/api-keys — cascade: only called when Ollama keeps.",
        "secret": True,
    },
    "OPEN_VLM_BASE_URL": {
        "label": "Ollama / open VLM base URL",
        "type": "text",
        "default": "pod",
        "section": "Vision verify",
        "visible_for_key": "VERIFY_BACKEND",
        "visible_for": ["open_vlm", "ollama_then_openai"],
        "help": "pod = Ollama on the RunPod GPU (recommended). Or a remote OpenAI-compatible URL e.g. https://openrouter.ai/api/v1.",
    },
    "OPEN_VLM_MODEL": {
        "label": "Ollama / open VLM model",
        "type": "text",
        "default": "qwen2.5vl:3b",
        "section": "Vision verify",
        "visible_for_key": "VERIFY_BACKEND",
        "visible_for": ["open_vlm", "ollama_then_openai"],
        "help": "Pulled on each GPU pod. Default qwen2.5vl:3b (faster). OpenRouter: qwen/qwen2.5-vl-72b-instruct.",
    },
    "OPEN_VLM_API_KEY": {
        "label": "Open VLM API key",
        "type": "password",
        "default": "",
        "section": "Vision verify",
        "visible_for_key": "VERIFY_BACKEND",
        "visible_for": ["open_vlm", "ollama_then_openai"],
        "help": "Leave empty for pod Ollama; required for OpenRouter.",
        "secret": True,
    },
}

# Still applied from .env / advanced use, but not shown in UI
_HIDDEN_KEYS = (
    "SCAN_BACKEND",
    "RUNPOD_DOCKER_IMAGE",
    "DOCKER_HUB_USER",
    "DOCKER_HUB_TOKEN",
    "RUNPOD_JOB_TIMEOUT_SEC",
    "RUNPOD_POD_ID",
    "SCRAPFLY_PROXY_URL",
    "SCRAPFLY_COUNTRY",
    "SCRAPFLY_PROXY_OPTS",
    "SCRAPINGDOG_PROXY_URL",
    "SPARSE_SECTION_SEC",
    "SPARSE_STRIDE_SEC",
    "DENSE_PAD_SEC",
    "MAX_DENSE_SEC",
)


def ensure_settings_table() -> None:
    init_db()
    with db() as conn:
        conn.executescript(SETTINGS_SCHEMA)


def get_setting(key: str, default: str = "") -> str:
    ensure_settings_table()
    with db() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    return str(row["value"])


def get_all_settings() -> dict[str, str]:
    """Defaults ← .env/os.environ ← SQLite UI values."""
    import os

    ensure_settings_table()
    out = {k: str(d["default"]) for k, d in SETTING_DEFS.items()}
    for k in SETTING_DEFS:
        if os.environ.get(k):
            out[k] = os.environ[k].strip()
    with db() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    for r in rows:
        if r["key"] in out:
            out[r["key"]] = str(r["value"])
    # Always force cloud backend when using this UI path
    out["SCAN_BACKEND"] = "runpod"
    return out


def set_settings(updates: dict[str, Any]) -> dict[str, str]:
    """Validate + save settings; sync to .env; return current values."""
    import time

    ensure_settings_table()
    existing = get_all_settings()
    cleaned: dict[str, str] = {}
    for key, raw in updates.items():
        if key not in SETTING_DEFS and key != "SCAN_BACKEND":
            continue
        if key == "SCAN_BACKEND":
            cleaned[key] = "runpod"
            continue
        meta = SETTING_DEFS[key]
        val = "" if raw is None else str(raw).strip()
        if meta.get("secret"):
            clear_list = updates.get("_clear_secrets") or []
            if key in clear_list:
                val = ""
            elif not val or set(val.replace(" ", "")) <= {"*"} or (val.startswith("*") and key in existing):
                val = existing.get(key, "")
        if meta["type"] == "select":
            opts = meta.get("options") or []
            # Legacy IPRoyal choice → auto (Scrapfly then ScrapingDog).
            if key == "PROXY_PROVIDER" and val == "iproyal":
                val = "auto"
            if val not in opts:
                raise ValueError(f"{key} must be one of {opts}")
        if meta["type"] == "number":
            try:
                num = float(val)
            except ValueError as e:
                raise ValueError(f"{key} must be a number") from e
            mn, mx = meta.get("min"), meta.get("max")
            if mn is not None and num < mn:
                raise ValueError(f"{key} min is {mn}")
            if mx is not None and num > mx:
                raise ValueError(f"{key} max is {mx}")
            if meta.get("step") == 1 or meta.get("step") is None:
                val = str(int(num))
            else:
                val = str(num)
        cleaned[key] = val

    cleaned["SCAN_BACKEND"] = "runpod"
    now = time.time()
    with db() as conn:
        for k, v in cleaned.items():
            conn.execute(
                """INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (k, v, now),
            )
    sync_env_file()
    apply_settings_to_environ()
    # Live Pathé scrape picks this up without restart.
    if "PATHE_STACK_MAX" in cleaned:
        try:
            from config import load_env

            load_env()
            from runpod_client import pathe_stack_max

            pathe_stack_max()
        except Exception:
            pass
    return get_all_settings()


def apply_settings_to_environ() -> None:
    import os

    ensure_settings_table()
    with db() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    for r in rows:
        os.environ[str(r["key"])] = str(r["value"])
    os.environ["SCAN_BACKEND"] = "runpod"


def sync_env_file() -> None:
    """Write known keys into .env (preserve unknown lines)."""
    import os

    ensure_settings_table()
    path = ROOT / ".env"
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    with db() as conn:
        rows = {str(r["key"]): str(r["value"]) for r in conn.execute("SELECT key, value FROM app_settings")}

    keys_to_write = set(SETTING_DEFS) | {"SCAN_BACKEND"} | set(_HIDDEN_KEYS)
    values = {k: rows.get(k, os.environ.get(k, "")) for k in keys_to_write}
    values["SCAN_BACKEND"] = "runpod"

    kept: list[str] = []
    seen: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            kept.append(line)
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in values:
            kept.append(f"{k}={values[k]}")
            seen.add(k)
        else:
            kept.append(line)
    for k, v in values.items():
        if k not in seen and v != "":
            kept.append(f"{k}={v}")
    if "SCAN_BACKEND" not in seen:
        # ensure line exists
        if not any(l.strip().startswith("SCAN_BACKEND=") for l in kept):
            kept.append("SCAN_BACKEND=runpod")

    path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")


def settings_public_view(values: dict[str, str] | None = None) -> dict[str, Any]:
    vals = values or get_all_settings()
    # Coerce legacy IPRoyal choice so the select can render a valid option.
    if vals.get("PROXY_PROVIDER") == "iproyal":
        vals = dict(vals)
        vals["PROXY_PROVIDER"] = "auto"
    fields = []
    for key, meta in SETTING_DEFS.items():
        item = {
            "key": key,
            "label": meta["label"],
            "type": meta["type"],
            "help": meta.get("help") or "",
            "value": vals.get(key, meta.get("default", "")),
            "section": meta.get("section") or "",
        }
        if meta.get("visible_for"):
            item["visible_for"] = list(meta["visible_for"])
        if meta.get("visible_for_key"):
            item["visible_for_key"] = str(meta["visible_for_key"])
        if meta.get("secret"):
            raw = vals.get(key, "")
            item["has_value"] = bool(raw)
            item["value"] = ""
        if meta["type"] == "select":
            item["options"] = meta.get("options") or []
            labels = meta.get("option_labels") or {}
            item["option_labels"] = {o: labels.get(o, o) for o in item["options"]}
        if meta["type"] == "number":
            item["min"] = meta.get("min")
            item["max"] = meta.get("max")
            item["step"] = meta.get("step", 1)
        fields.append(item)
    return {"ok": True, "fields": fields, "values": _safe_values(vals)}


def _safe_values(vals: dict[str, str]) -> dict[str, str]:
    safe = dict(vals)
    for key, meta in SETTING_DEFS.items():
        if not meta.get("secret"):
            continue
        k = safe.get(key) or ""
        if k:
            safe[key] = ("*" * max(0, len(k) - 4)) + k[-4:]
    return safe
