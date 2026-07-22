"""Persist Review stills on disk so Catbox expiry does not blank the UI."""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from config import CONTACT_DIR, USER_AGENT_CLOUD


def candidate_still_name(cand_id: int) -> str:
    return f"cand_{int(cand_id)}.jpg"


def candidate_still_path(cand_id: int) -> Path:
    return CONTACT_DIR / candidate_still_name(cand_id)


def candidate_strip_name(cand_id: int) -> str:
    return f"cand_{int(cand_id)}_strip.jpg"


def candidate_strip_path(cand_id: int) -> Path:
    return CONTACT_DIR / candidate_strip_name(cand_id)


def candidate_crop_name(cand_id: int) -> str:
    return f"cand_{int(cand_id)}_crop.jpg"


def candidate_crop_path(cand_id: int) -> Path:
    return CONTACT_DIR / candidate_crop_name(cand_id)


def local_still_url(cand_id: int) -> str | None:
    path = candidate_still_path(cand_id)
    if path.is_file() and path.stat().st_size > 200:
        return f"/media/sheet/{path.name}"
    return None


def local_strip_url(cand_id: int) -> str | None:
    path = candidate_strip_path(cand_id)
    if path.is_file() and path.stat().st_size > 500:
        return f"/media/sheet/{path.name}"
    return None


def local_crop_url(cand_id: int) -> str | None:
    path = candidate_crop_path(cand_id)
    if path.is_file() and path.stat().st_size > 500:
        return f"/media/sheet/{path.name}"
    return None


def _looks_like_image(raw: bytes) -> bool:
    if len(raw) < 200:
        return False
    if raw[:3] == b"\xff\xd8\xff":
        return True
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return True
    return False


def save_candidate_still(
    cand_id: int,
    *,
    raw: bytes | None = None,
    path: Path | str | None = None,
    b64: str | None = None,
    image_url: str | None = None,
) -> Path | None:
    """Write a durable JPEG under contact_sheets/cand_{id}.jpg."""
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    data = raw
    if data is None and path:
        p = Path(path)
        if p.is_file():
            data = p.read_bytes()
    if data is None and b64:
        try:
            data = base64.standard_b64decode(str(b64).encode("ascii"), validate=False)
        except Exception:
            data = None
    if data is None and image_url and str(image_url).startswith(("http://", "https://")):
        try:
            resp = requests.get(
                str(image_url),
                timeout=45,
                headers={"User-Agent": USER_AGENT_CLOUD},
                allow_redirects=True,
            )
            if resp.status_code == 200:
                data = resp.content
        except Exception:
            data = None
    if not data or not _looks_like_image(data):
        return None
    dest = candidate_still_path(cand_id)
    dest.write_bytes(data)
    return dest
