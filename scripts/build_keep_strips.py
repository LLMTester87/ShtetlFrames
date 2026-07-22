"""Build labeled ±10s @ 0.5s timeline strips for kept Review candidates.

Downloads each source once, writes output/contact_sheets/cand_{id}_strip.jpg
(each frame stamped with source + timestamp), then deletes the local video.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env  # noqa: E402
from db import db, init_db  # noqa: E402
from frame_strip import generate_strip_for_candidate  # noqa: E402
from still_store import candidate_strip_path, local_strip_url  # noqa: E402


def kept_missing_strips() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, video_id, start_sec, end_sec, source_url, decision
            FROM candidates
            WHERE decision='accept'
            ORDER BY video_id, start_sec
            """
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if local_strip_url(d["id"]):
            continue
        if not (d.get("source_url") or "").strip():
            continue
        out.append(d)
    return out


def main() -> int:
    load_env()
    init_db()
    rows = kept_missing_strips()
    print(f"Kept candidates needing strips: {len(rows)}", flush=True)
    if not rows:
        return 0

    by_src: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in rows:
        key = ((d.get("video_id") or "").strip(), (d.get("source_url") or "").strip())
        by_src[key].append(d)

    ok = 0
    fail = 0
    for (video_id, url), group in by_src.items():
        print(f"\n=== {video_id or url[:60]} ({len(group)} hit(s)) ===", flush=True)
        # Download once for the group.
        from frame_strip import _download_source
        from run_archive import delete_local_video

        video = _download_source(url, video_id or f"cand_{group[0]['id']}", video_id or "clip")
        if not video:
            print("  download failed", flush=True)
            fail += len(group)
            continue
        try:
            for d in group:
                cid = int(d["id"])
                print(f"  #{cid} strip…", flush=True)
                path = generate_strip_for_candidate(
                    d,
                    video_path=video,
                    download_if_needed=False,
                    delete_video_after=False,
                )
                if path:
                    ok += 1
                    print(f"    OK {path.name} ({path.stat().st_size} bytes)", flush=True)
                else:
                    fail += 1
                    print("    FAIL", flush=True)
        finally:
            try:
                delete_local_video(video)
            except Exception:
                pass

    print(f"\nDone. ok={ok} fail={fail}", flush=True)
    print(
        f"Example strip path: {candidate_strip_path(rows[0]['id']) if rows else ''}",
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
