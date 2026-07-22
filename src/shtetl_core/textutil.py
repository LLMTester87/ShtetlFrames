"""Small string helpers shared by local and worker paths."""

from __future__ import annotations

import re


def slugify(text: str, max_len: int = 80) -> str:
    """Filesystem-safe slug from a title or URL fragment."""
    cleaned = re.sub(r"[^\w\s-]", "", (text or "video").lower())
    cleaned = re.sub(r"[-\s]+", "_", cleaned).strip("_")
    return cleaned[:max_len] or "video"
