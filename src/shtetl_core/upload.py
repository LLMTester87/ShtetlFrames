"""Upload hit stills to Catbox (no API key)."""

from __future__ import annotations

from pathlib import Path

import requests

USER_AGENT = "ShtetlFrames/1.0 (research; image upload)"
CATBOX_URL = "https://catbox.moe/user/api.php"
_MIN_IMAGE_BYTES = 500


def _looks_like_image(raw: bytes) -> bool:
    if len(raw) < _MIN_IMAGE_BYTES:
        return False
    if raw[:3] == b"\xff\xd8\xff":
        return True
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return True
    return False


def _public_url_ok(url: str, *, user_agent: str) -> bool:
    """Catbox sometimes returns HTTP 200 with an empty body — reject those."""
    try:
        r = requests.get(
            url,
            timeout=45,
            headers={"User-Agent": user_agent},
            allow_redirects=True,
        )
        return r.status_code == 200 and _looks_like_image(r.content)
    except Exception:
        return False


def _catbox_upload(path: Path, *, user_agent: str) -> str | None:
    with path.open("rb") as f:
        response = requests.post(
            CATBOX_URL,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (path.name, f, "image/jpeg")},
            headers={"User-Agent": user_agent},
            timeout=120,
        )
    if response.status_code == 200 and response.text.strip().startswith("http"):
        return response.text.strip()
    return None


def upload_image(path: Path, *, user_agent: str = USER_AGENT) -> str | None:
    """Upload a local image to Catbox; return a public URL that serves bytes, or None."""
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    local = path.read_bytes()
    if not _looks_like_image(local):
        return None

    for _ in range(3):
        try:
            url = _catbox_upload(path, user_agent=user_agent)
        except Exception:
            url = None
        if url and _public_url_ok(url, user_agent=user_agent):
            return url
    return None
