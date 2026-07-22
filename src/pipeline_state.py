"""Shared pipeline job locks and active flags."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_active = {
    "discover": False,
    "scrape": False,
    "pathe_discover": False,
    "pathe_scrape": False,
}
