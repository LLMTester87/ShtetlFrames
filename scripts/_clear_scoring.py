"""Clear review + discovery queues/caches. Keep settings, videos, cookies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CONTACT_DIR, DATA_DIR, OUTPUT_DIR, load_env  # noqa: E402
from db import clear_candidates, clear_queue, db, init_db  # noqa: E402

# Discovery / catalog artifacts (not videos or credentials).
_DISCOVERY_FILES = [
    DATA_DIR / "pathe_catalog.jsonl",
    DATA_DIR / "pathe_resolve_cache.json",
    DATA_DIR / "download_manifest.json",
    OUTPUT_DIR / "candidates.jsonl",
    OUTPUT_DIR / "review_queue.csv",
    OUTPUT_DIR / "ia_batch_discoveries.csv",
    OUTPUT_DIR / "ia_batch_discoveries.json",
    OUTPUT_DIR / "bulk_queue.csv",
    OUTPUT_DIR / "bulk_queue.json",
    OUTPUT_DIR / "bulk_queue_summary.json",
    OUTPUT_DIR / "jobs.json",
    OUTPUT_DIR / "scan_summary.json",
]


def main() -> int:
    load_env()
    init_db()

    before: dict[str, int] = {}
    with db() as conn:
        for t in ("candidates", "queue_items", "jobs", "app_settings"):
            try:
                before[t] = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            except Exception:
                before[t] = -1

    stills = list(CONTACT_DIR.glob("*.jpg")) if CONTACT_DIR.is_dir() else []
    print(f"before: {before} stills={len(stills)}", flush=True)

    clear_candidates()
    clear_queue()
    with db(write=True) as conn:
        conn.execute("DELETE FROM jobs")

    removed = 0
    if CONTACT_DIR.is_dir():
        for p in CONTACT_DIR.glob("*.jpg"):
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                print(f"still delete fail {p.name}: {e}", flush=True)

    disc_removed = 0
    for p in _DISCOVERY_FILES:
        if p.is_file():
            try:
                sz = p.stat().st_size
                p.unlink()
                disc_removed += 1
                print(f"removed discovery file {p.name} ({sz} bytes)", flush=True)
            except OSError as e:
                print(f"discovery delete fail {p}: {e}", flush=True)

    after: dict[str, int] = {}
    with db() as conn:
        for t in ("candidates", "queue_items", "jobs", "app_settings"):
            try:
                after[t] = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            except Exception:
                after[t] = -1

    left = len(list(CONTACT_DIR.glob("*.jpg"))) if CONTACT_DIR.is_dir() else 0
    print(f"after:  {after} stills={left} (removed {removed})", flush=True)
    print(f"discovery files removed: {disc_removed}", flush=True)
    print("settings / videos / cookies preserved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
