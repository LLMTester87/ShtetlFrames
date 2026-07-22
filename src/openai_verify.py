"""Second-pass vision check on CLIP-flagged stills (OpenAI or open VLM).

Confirms the frame shows an Orthodox Jewish man with a visible Jewish head covering
(yarmulke / black hat / shtreimel / spodik). Soft CLIP may flood candidates; this
pass must stay strict. Visual filter only — not identity.
Uses human Keep/Pass few-shots when available (see label_feedback).

Backends (Settings → VERIFY_BACKEND):
- openai — GPT vision via OPENAI_API_KEY
- open_vlm — OpenAI-compatible endpoint (Ollama / vLLM / OpenRouter) for Qwen2.5-VL etc.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

OnStatus = Callable[[str], None]

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OPEN_VLM_MODEL = "qwen2.5vl:3b"
POD_OLLAMA_URL = "http://127.0.0.1:11434/v1"
_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
# Cap parallel vision calls so scrape workers don't pile up on OpenAI/SQLite.
_VERIFY_SEM = threading.Semaphore(2)
_DISABLE_LOCK = threading.Lock()
_disabled_reason: str | None = None
_DEFAULT_TIMEOUT = 25.0
_OPEN_VLM_TIMEOUT = 90.0

_SYSTEM = (
    "You review archival film stills for a research tool. "
    "Decide two visual questions separately, then KEEP only if BOTH are true: "
    "(1) looks_jewish: does at least one person overall look Jewish in an Orthodox/"
    "Hasidic/Litvish sense (beard+payot, traditional dress, community look, etc.)? "
    "This is visual appearance only — not identity or ethnicity certainty. "
    "(2) head_covered: is that person's head covered by a specifically Jewish/"
    "Orthodox head covering? COUNT AS YES: yarmulke/kippah/skullcap; black Orthodox "
    "fedora/homburg/Borsalino with Orthodox dress; AND especially a shtreimel or spodik "
    "(large round/tall dark fur Hasidic hat) — fur Hasidic hats ARE Jewish head coverings, "
    "even if the fur texture is grainy on old film. Do NOT reject a clear fur shtreimel/"
    "spodik as 'secular' or 'unclear Jewish covering'. "
    "Do NOT reject a dark brimmed hat on a clearly Orthodox/Hasidic man (beard, payot, "
    "kapote) as merely 'secular fedora' — mark head_covered=true. "
    "COUNT AS NO: bare heads; military/naval caps; turbans; keffiyeh; biretta; mitre; "
    "bowler/top hat on secular dress; hair/payot alone. A long coat alone is NEVER enough. "
    "KEEP only when looks_jewish=true AND head_covered=true. "
    "HARD REJECT Pathé-style false keeps: sports crowds, royal parades, military, other "
    "faiths' clergy, generic bearded Western men without Orthodox cues, costume caricature, "
    "empty/unclear frames. "
    "If the only doubt is whether a fur hat is a shtreimel but the man otherwise looks "
    "Hasidic, PREFER keep with head_covered=true. "
    "Do not claim anyone is a rabbi. When human KEEP/PASS examples are provided, match "
    "that bar. Reply with JSON only."
)

_USER = (
    "For this still: (1) does someone overall look Jewish/Orthodox (looks_jewish)? "
    "(2) do they wear a clear Jewish head covering — yarmulke, Orthodox black hat, "
    "shtreimel, or spodik/fur Hasidic hat (head_covered)? "
    "Fur shtreimel/spodik counts as head_covered=true. Keep only if BOTH are true. "
    'Respond JSON: {"keep": true|false, "looks_jewish": true|false, '
    '"head_covered": true|false, "confidence": 0.0-1.0, "reason": "short"}'
)


def _load_env() -> None:
    try:
        from config import load_env

        load_env()
    except Exception:
        pass


def verify_backend() -> str:
    """Return ``ollama_then_openai``, ``openai``, or ``open_vlm``."""
    _load_env()
    raw = (os.environ.get("VERIFY_BACKEND") or "openai").strip().lower()
    if raw in (
        "ollama_then_openai",
        "cascade",
        "ollama+openai",
        "vlm_then_openai",
        "open_vlm_then_openai",
    ):
        return "ollama_then_openai"
    if raw in ("open_vlm", "vlm", "ollama", "qwen", "open-vlm"):
        return "open_vlm"
    return "openai"


def openai_configured() -> bool:
    _load_env()
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


def running_on_pod() -> bool:
    """True inside a ShtetlFrames RunPod worker process."""
    if (os.environ.get("SHTETL_POD") or "").strip() in ("1", "true", "yes"):
        return True
    try:
        return Path("/workspace/shtetl/entry.py").is_file()
    except Exception:
        return False


def open_vlm_runs_on_pod() -> bool:
    """True when OPEN_VLM_BASE_URL means Ollama on the RunPod GPU (not the PC)."""
    _load_env()
    raw = (os.environ.get("OPEN_VLM_BASE_URL") or "pod").strip().lower().rstrip("/")
    if raw in ("", "pod", "gpu", "runpod", "local-pod"):
        return True
    # Explicit loopback in Settings also means pod Ollama when cascade is on.
    if raw in ("http://127.0.0.1:11434/v1", "http://localhost:11434/v1"):
        return True
    return False


def open_vlm_base_url() -> str:
    _load_env()
    raw = (os.environ.get("OPEN_VLM_BASE_URL") or "pod").strip().rstrip("/")
    if raw.lower() in ("", "pod", "gpu", "runpod", "local-pod"):
        return POD_OLLAMA_URL
    return raw


def open_vlm_configured() -> bool:
    # ``pod`` / empty resolves to loopback — configured for on-GPU Ollama.
    return bool(open_vlm_base_url())


def open_vlm_model() -> str:
    _load_env()
    return (
        (os.environ.get("OPEN_VLM_MODEL") or DEFAULT_OPEN_VLM_MODEL).strip()
        or DEFAULT_OPEN_VLM_MODEL
    )


def open_vlm_api_key() -> str:
    _load_env()
    return (os.environ.get("OPEN_VLM_API_KEY") or "").strip()


def verify_note_prefix() -> str:
    """Notes tag prefix used in Review filters (``openai:`` or ``vlm:``)."""
    backend = verify_backend()
    if backend == "open_vlm":
        return "vlm"
    # Cascade finals are usually openai:; Ollama-only drops stay vlm:.
    return "openai"


def openai_verify_enabled() -> bool:
    """True when the vision second pass is on and the selected backend is configured."""
    if _disabled_reason:
        return False
    _load_env()
    flag = (os.environ.get("OPENAI_VERIFY") or "1").strip().lower()
    if flag in ("0", "false", "off", "no", "none"):
        return False
    backend = verify_backend()
    if backend == "open_vlm":
        return open_vlm_configured()
    if backend == "ollama_then_openai":
        # Cascade needs Ollama/VLM; OpenAI is optional (only runs on VLM keeps).
        return open_vlm_configured()
    return openai_configured()


def openai_disabled_reason() -> str | None:
    return _disabled_reason


def disable_openai_verify(reason: str) -> None:
    """Turn off verify for this process after hard API failures (model/auth)."""
    global _disabled_reason
    with _DISABLE_LOCK:
        if _disabled_reason:
            return
        _disabled_reason = (reason or "disabled")[:240]
    try:
        from logutil import status

        status(f"Vision verify disabled for this run: {_disabled_reason}", job="scrape", persist=True)
    except Exception:
        pass


def openai_model() -> str:
    return (os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _should_disable_http(status_code: int, body: str) -> bool:
    if status_code in (401, 403):
        return True
    low = (body or "").lower()
    return status_code == 400 and (
        "does not have access to model" in low
        or "model_not_found" in low
        or "invalid_api_key" in low
    )


def _api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _chat_completions_url(base: str) -> str:
    root = (base or "").strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def open_vlm_url_is_local(url: str | None = None) -> bool:
    """True when pods cannot reach this host (localhost / private loopback)."""
    raw = (url or open_vlm_base_url() or "").strip()
    if not raw:
        return False
    try:
        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        return False
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _notes_tag_match(notes: str | None, tag: str) -> bool:
    """True if any line starts with openai:{tag} or vlm:{tag}."""
    want = f":{tag}"
    for line in (notes or "").splitlines():
        low = line.strip().lower()
        if low.startswith("openai" + want) or low.startswith("vlm" + want):
            return True
    low0 = (notes or "").strip().lower()
    return low0.startswith("openai" + want) or low0.startswith("vlm" + want)


def _parse_verdict(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {
            "keep": False,
            "looks_jewish": False,
            "head_covered": False,
            "confidence": 0.0,
            "reason": "empty_model_reply",
            "skipped": False,
        }
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        low = text.lower()
        keep = False
        if '"keep": true' in low or '"keep":true' in low:
            keep = True
        if '"keep": false' in low or '"keep":false' in low:
            keep = False
        head = '"head_covered": true' in low or '"head_covered":true' in low
        jewish = '"looks_jewish": true' in low or '"looks_jewish":true' in low
        if any(
            w in low
            for w in ("bare head", "bareheaded", "no hat", "no yarmulke", "uncovered")
        ):
            keep = False
            head = False
        if keep and (not head or not jewish):
            keep = False
        return {
            "keep": bool(keep and head and jewish),
            "looks_jewish": bool(jewish),
            "head_covered": bool(head),
            "confidence": 0.4,
            "reason": text[:240],
            "skipped": False,
        }

    keep = bool(data.get("keep"))
    if "head_covered" in data:
        head = bool(data.get("head_covered"))
    else:
        # Legacy replies without the field: do not trust keep alone.
        head = False
        keep = False
    if "looks_jewish" in data:
        jewish = bool(data.get("looks_jewish"))
    else:
        # Without an explicit overall-Jewish call, fail closed.
        jewish = False
        keep = False
    try:
        conf = float(data.get("confidence") if data.get("confidence") is not None else 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason") or "")[:240]
    # Hard rule: need Jewish head covering AND overall Jewish/Orthodox look.
    if not head or not jewish:
        keep = False
    return {
        "keep": keep,
        "looks_jewish": jewish,
        "head_covered": head,
        "confidence": conf,
        "reason": reason,
        "skipped": False,
    }


def notes_openai_approved(notes: str | None) -> bool:
    """True only when a real vision-verify keep was recorded (Review gate)."""
    return _notes_tag_match(notes, "keep")


def notes_openai_dropped(notes: str | None) -> bool:
    """True when vision verify rejected the still (openai:/vlm: drop)."""
    return _notes_tag_match(notes, "drop")


def notes_openai_uncertain(notes: str | None) -> bool:
    """True when verify marked low-confidence / uncertain (not auto-kept)."""
    return _notes_tag_match(notes, "uncertain")


def notes_already_verified(notes: str | None) -> bool:
    """True when a prior openai:/vlm: keep|drop|uncertain note is present."""
    return (
        notes_openai_approved(notes)
        or notes_openai_dropped(notes)
        or notes_openai_uncertain(notes)
    )


def verdict_is_keep(verdict: dict[str, Any]) -> bool:
    """Require keep + Jewish head covering + overall Jewish look."""
    if verdict.get("skipped") or verdict.get("uncertain"):
        return False
    if not bool(verdict.get("keep")):
        return False
    if "head_covered" in verdict and not bool(verdict.get("head_covered")):
        return False
    if "looks_jewish" in verdict and not bool(verdict.get("looks_jewish")):
        return False
    return True


def format_verdict_notes(verdict: dict[str, Any]) -> str:
    if verdict.get("uncertain"):
        tag = "uncertain"
    elif verdict_is_keep(verdict):
        tag = "keep"
    else:
        tag = "drop"
    head = verdict.get("head_covered")
    jewish = verdict.get("looks_jewish")
    head_s = "" if head is None else f" head={'yes' if head else 'no'}"
    jew_s = "" if jewish is None else f" jewish={'yes' if jewish else 'no'}"
    prefix = str(verdict.get("provider") or verify_note_prefix()).strip() or "openai"
    if prefix not in ("openai", "vlm"):
        prefix = "openai"
    return (
        f"{prefix}:{tag}"
        f" conf={float(verdict.get('confidence') or 0):.2f}"
        f"{jew_s}{head_s}"
        f" {verdict.get('reason') or ''}"
    )[:500]


def _sniff_image_mime(raw: bytes, hint: str = "") -> str | None:
    """Return image/* mime or None if bytes do not look like a still."""
    if len(raw) < 32:
        return None
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    hint = (hint or "").split(";")[0].strip().lower()
    if hint in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        return "image/jpg" if hint == "image/jpg" else hint
    return None


def _fetch_image_bytes(url: str, *, timeout: float = 45.0) -> tuple[bytes, str] | None:
    """Download a public still ourselves — OpenAI's crawler often cannot reach Catbox."""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ShtetlFrames/1.0; openai-verify)"
                },
                allow_redirects=True,
            )
            if r.status_code != 200:
                last_err = RuntimeError(f"http_{r.status_code}")
                time.sleep(0.4 * attempt)
                continue
            mime = _sniff_image_mime(r.content, r.headers.get("Content-Type") or "")
            if not mime:
                last_err = RuntimeError("not_image")
                time.sleep(0.4 * attempt)
                continue
            return r.content, mime
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.5 * attempt)
    _ = last_err
    return None


def _image_part_from_bytes(raw: bytes, mime: str) -> dict[str, Any] | dict[str, str]:
    if len(raw) > 20_000_000:
        return {"error": "image_too_large"}
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def verify_still(
    *,
    image_path: Path | str | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """
    Returns {keep, confidence, reason, skipped, error?, provider?}.
    If verify is off/unconfigured, keep=True and skipped=True (caller treats as passthrough).
    When enabled: fail closed — errors/skips set keep=False so Review stays empty of unverified hits.

    ``ollama_then_openai``: Ollama/VLM first; OpenAI only when that pass keeps.
    """
    backend = verify_backend()
    if backend == "ollama_then_openai":
        return _verify_cascade(
            image_path=image_path,
            image_url=image_url,
            image_b64=image_b64,
            timeout=timeout,
        )
    return _verify_still_one(
        backend,
        image_path=image_path,
        image_url=image_url,
        image_b64=image_b64,
        timeout=timeout,
    )


def verify_stills_any(
    image_paths: list[Path | str],
    *,
    timeout: float | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Try top stills in order; keep if any keep.

    CLIP's peak frame is sometimes an OpenAI false drop while a nearby hit
    in the same segment clearly shows a kippah (e.g. asset 71170 t=22 vs t=20).
    """
    seen: set[str] = set()
    paths: list[Path] = []
    for raw in image_paths:
        if not raw:
            continue
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        paths.append(p)
        if len(paths) >= max(1, int(max_attempts)):
            break
    if not paths:
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": "no_stills",
            "skipped": True,
            "error": "no_stills",
        }
    last: dict[str, Any] | None = None
    for p in paths:
        v = verify_still(image_path=p, timeout=timeout)
        last = v
        if verdict_is_keep(v):
            out = dict(v)
            out["verified_path"] = str(p)
            out["verify_attempts"] = paths.index(p) + 1
            return out
    assert last is not None
    out = dict(last)
    out["verified_path"] = str(paths[-1])
    out["verify_attempts"] = len(paths)
    return out


def _verify_cascade(
    *,
    image_path: Path | str | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ollama/VLM gate, then OpenAI only on positives."""
    if not openai_verify_enabled():
        return {
            "keep": True,
            "confidence": 0.0,
            "reason": _disabled_reason or "vision_verify_off",
            "skipped": True,
            "provider": "vlm",
        }
    vlm = _verify_still_one(
        "open_vlm",
        image_path=image_path,
        image_url=image_url,
        image_b64=image_b64,
        timeout=timeout,
        require_enabled=False,
    )
    if not verdict_is_keep(vlm):
        # Drop / uncertain / error — never spend OpenAI tokens.
        vlm["provider"] = "vlm"
        prior = str(vlm.get("reason") or "")
        if not prior.startswith("vlm_only"):
            vlm["reason"] = f"vlm_only {prior}".strip()[:240]
        return vlm
    if not openai_configured():
        # No OpenAI key — accept Ollama keep (still cheaper than GPT-on-everything).
        vlm["provider"] = "vlm"
        prior = str(vlm.get("reason") or "")
        vlm["reason"] = f"vlm_keep_no_openai {prior}".strip()[:240]
        return vlm
    oai = _verify_still_one(
        "openai",
        image_path=image_path,
        image_url=image_url,
        image_b64=image_b64,
        timeout=timeout,
        require_enabled=False,
    )
    oai["provider"] = "openai"
    vlm_r = str(vlm.get("reason") or "")[:80]
    oai_r = str(oai.get("reason") or "")
    oai["reason"] = f"after_vlm({vlm_r}) {oai_r}".strip()[:240]
    oai["vlm_keep"] = True
    oai["vlm_confidence"] = vlm.get("confidence")
    return oai


def _verify_still_one(
    backend: str,
    *,
    image_path: Path | str | None = None,
    image_url: str | None = None,
    image_b64: str | None = None,
    timeout: float | None = None,
    require_enabled: bool = True,
) -> dict[str, Any]:
    """Single-provider vision call (``openai`` or ``open_vlm``)."""
    provider = "vlm" if backend == "open_vlm" else "openai"
    if require_enabled and not openai_verify_enabled():
        return {
            "keep": True,
            "confidence": 0.0,
            "reason": _disabled_reason or "vision_verify_off",
            "skipped": True,
            "provider": provider,
        }

    if backend == "open_vlm":
        if not open_vlm_configured():
            return {
                "keep": False,
                "confidence": 0.0,
                "reason": "open_vlm_unconfigured",
                "skipped": True,
                "error": "unconfigured",
                "provider": provider,
            }
        # Ollama runs on the RunPod GPU — do not fall back to the PC.
        if open_vlm_runs_on_pod() and not running_on_pod():
            return {
                "keep": False,
                "confidence": 0.0,
                "reason": "ollama_gpu_pod_only",
                "skipped": True,
                "error": "pod_only",
                "provider": provider,
            }
        api_url = _chat_completions_url(open_vlm_base_url())
        key = open_vlm_api_key()
        model = open_vlm_model()
        use_json_format = False
        req_timeout = float(timeout if timeout is not None else _OPEN_VLM_TIMEOUT)
    else:
        if not openai_configured():
            return {
                "keep": False,
                "confidence": 0.0,
                "reason": "openai_unconfigured",
                "skipped": True,
                "error": "unconfigured",
                "provider": provider,
            }
        api_url = _OPENAI_API_URL
        key = _api_key()
        model = openai_model()
        use_json_format = True
        req_timeout = float(timeout if timeout is not None else _DEFAULT_TIMEOUT)

    path = Path(image_path) if image_path else None
    raw: bytes | None = None
    mime = "image/jpeg"
    if path and path.is_file():
        raw = path.read_bytes()
        mime = _sniff_image_mime(raw) or "image/jpeg"
    elif image_b64:
        try:
            raw = base64.standard_b64decode(str(image_b64).encode("ascii"), validate=False)
        except Exception:
            raw = None
        if raw:
            mime = _sniff_image_mime(raw) or "image/jpeg"
    elif image_url and str(image_url).startswith(("http://", "https://")):
        fetched = _fetch_image_bytes(str(image_url))
        if fetched:
            raw, mime = fetched

    if not raw or not _sniff_image_mime(raw, mime):
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": "no_image" if not raw else "image_fetch_failed",
            "skipped": True,
            "error": "no_image" if not raw else "image_fetch_failed",
            "provider": provider,
        }
    part = _image_part_from_bytes(raw, mime if mime.startswith("image/") else "image/jpeg")
    if part.get("error"):
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": str(part["error"]),
            "skipped": True,
            "error": str(part["error"]),
            "provider": provider,
        }

    fewshot_parts: list[dict[str, Any]] = []
    try:
        from label_feedback import build_fewshot_content_parts

        fewshot_parts, _meta = build_fewshot_content_parts()
    except Exception:
        fewshot_parts = []

    user_content: list[dict[str, Any]] = [
        *fewshot_parts,
        {"type": "text", "text": _USER},
        part,
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }
    if use_json_format:
        payload["response_format"] = {"type": "json_object"}
    # gpt-5.6-sol (and similar) only accept default temperature — omit 0.
    model_l = model.lower()
    if not any(tok in model_l for tok in ("gpt-5.6", "-sol", "o1-", "o3-", "o4-")):
        payload["temperature"] = 0
    resp = None
    last_net: Exception | None = None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with _VERIFY_SEM:
        if require_enabled and not openai_verify_enabled():
            return {
                "keep": True,
                "confidence": 0.0,
                "reason": _disabled_reason or "vision_verify_off",
                "skipped": True,
                "provider": provider,
            }
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=req_timeout,
                )
            except requests.RequestException as e:
                last_net = e
                time.sleep(0.6 * attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(8.0, 1.2 * attempt))
                continue
            break
        else:
            if last_net is not None:
                return {
                    "keep": False,
                    "confidence": 0.0,
                    "reason": f"{provider}_network:{last_net}"[:240],
                    "skipped": True,
                    "error": "network",
                    "provider": provider,
                }

    if resp is None:
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": f"{provider}_no_response",
            "skipped": True,
            "error": "network",
            "provider": provider,
        }

    if resp.status_code >= 400:
        body = (resp.text or "")[:300]
        if _should_disable_http(resp.status_code, body):
            disable_openai_verify(f"http_{resp.status_code}:{body[:160]}")
            return {
                "keep": True,
                "confidence": 0.0,
                "reason": f"{provider}_disabled:{body[:120]}",
                "skipped": True,
                "error": f"http_{resp.status_code}",
                "provider": provider,
            }
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": f"{provider}_http_{resp.status_code}:{body[:160]}",
            "skipped": True,
            "error": f"http_{resp.status_code}",
            "provider": provider,
        }

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            # Some open VLMs return content parts instead of a plain string.
            chunks: list[str] = []
            for part_c in content:
                if isinstance(part_c, dict) and part_c.get("type") == "text":
                    chunks.append(str(part_c.get("text") or ""))
                elif isinstance(part_c, str):
                    chunks.append(part_c)
            content = "\n".join(chunks)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {
            "keep": False,
            "confidence": 0.0,
            "reason": f"{provider}_bad_response:{e}",
            "skipped": True,
            "error": "bad_response",
            "provider": provider,
        }

    try:
        from label_feedback import apply_confidence_gate

        verdict = apply_confidence_gate(_parse_verdict(str(content)))
    except Exception:
        verdict = _parse_verdict(str(content))
    verdict["provider"] = provider
    return verdict


def _passthrough_rows(rows: list[dict], tag: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        note = (r.get("notes") or "").strip()
        label = tag[:200]
        r["notes"] = f"{note} {label}".strip() if note else label
        out.append(r)
    return out


def filter_candidates_openai(
    rows: list[dict],
    *,
    on_status: OnStatus | None = None,
) -> list[dict]:
    """Tag each CLIP hit with vision keep/drop notes; persist both for Review filters.

    Default Review UI still shows keeps only; ``openai=drop`` surfaces rejections.
    """
    if not rows:
        return rows
    backend = verify_backend()
    prefix = verify_note_prefix()
    if backend == "ollama_then_openai":
        label = "Ollama→OpenAI"
    elif backend == "open_vlm":
        label = "Open VLM"
    else:
        label = "OpenAI"
    if not openai_verify_enabled():
        return _passthrough_rows(rows, f"{prefix}:skip {_disabled_reason or 'verify_off'}")

    out: list[dict] = []
    dropped = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        if not openai_verify_enabled():
            out.extend(
                _passthrough_rows(rows[i - 1 :], f"{prefix}:skip {_disabled_reason}")
            )
            if on_status:
                on_status(f"{label} disabled — keeping {len(rows) - i + 1} CLIP hits")
            break
        if on_status and (i == 1 or i % 2 == 0 or i == total):
            on_status(f"{label} verify {i}/{total}")
        row = dict(row)
        prior = (row.get("notes") or "").strip()
        # Pod may have already verified the local JPEG — don't re-fetch Catbox.
        if notes_already_verified(prior):
            if not notes_openai_approved(prior):
                dropped += 1
            out.append(row)
            continue
        path = row.get("_local_still") or row.get("local_still")
        url = row.get("image_url") or ""
        b64 = row.get("still_b64") or row.get("image_b64") or ""
        verdict = verify_still(
            image_path=path if path else None,
            image_b64=str(b64) if b64 else None,
            image_url=url if url else None,
        )
        # Hard disable can trip mid-call; passthrough this + remaining CLIP hits.
        if verdict.get("skipped") and not openai_verify_enabled():
            out.extend(
                _passthrough_rows(rows[i - 1 :], f"{prefix}:skip {_disabled_reason}")
            )
            if on_status:
                on_status(f"{label} disabled — keeping {len(rows) - i + 1} CLIP hits")
            break
        note = format_verdict_notes(verdict)
        row["notes"] = note
        if not verdict_is_keep(verdict):
            dropped += 1
        out.append(row)

    if on_status and dropped:
        on_status(
            f"{label} dropped {dropped}/{total} — saved; filter “AI failed” in Review"
        )
    return out
