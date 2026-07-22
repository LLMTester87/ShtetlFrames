"""Upload hit stills to free cloud hosts (no local retention)."""

from __future__ import annotations

from pathlib import Path

from config import USER_AGENT_CLOUD
from shtetl_core.upload import upload_image as _upload_image


def upload_image(path: Path) -> str | None:
    return _upload_image(path, user_agent=USER_AGENT_CLOUD)
