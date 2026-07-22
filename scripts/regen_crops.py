"""Regenerate existing cand_*_crop.jpg with caption-below + higher-res logic."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CONTACT_DIR, load_env  # noqa: E402

load_env()

from db import db, init_db  # noqa: E402
from frame_strip import (  # noqa: E402
    CROP_MAX_BYTES,
    CROP_THUMB_H,
    _download_source,
    generate_crop_for_candidate,
)
from serve import find_video_file  # noqa: E402
from still_store import candidate_crop_path  # noqa: E402


def main() -> int:
    print("thumb_h", CROP_THUMB_H, "max_bytes", CROP_MAX_BYTES)
    init_db()
    ids: list[int] = []
    for p in sorted(CONTACT_DIR.glob("cand_*_crop.jpg")):
        m = re.match(r"cand_(\d+)_crop\.jpg$", p.name)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        print("no crops to regen")
        return 0
    print("regen ids", ids)
    with db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, video_id, start_sec, end_sec, source_url "
                f"FROM candidates WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
        ]
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"] or f"cand_{r['id']}"].append(r)

    ok = fail = 0
    for vid, group in by_vid.items():
        url = (group[0].get("source_url") or "").strip()
        video = find_video_file(vid)
        owned = False
        if not video or not Path(video).is_file():
            video = _download_source(url, vid, vid)
            owned = True
        print("video", vid, "path", video, "n", len(group))
        if not video or not Path(video).is_file():
            print("  DOWNLOAD FAIL")
            fail += len(group)
            continue
        try:
            for r in group:
                before = candidate_crop_path(r["id"])
                old = before.stat().st_size if before.is_file() else 0
                path = generate_crop_for_candidate(
                    r,
                    video_path=Path(video),
                    download_if_needed=False,
                    delete_video_after=False,
                    force=True,
                )
                if path and path.is_file():
                    sz = path.stat().st_size
                    print(f"  #{r['id']} {old} -> {sz} bytes (cap {CROP_MAX_BYTES})")
                    if sz <= CROP_MAX_BYTES:
                        ok += 1
                    else:
                        print("  OVERSIZE")
                        fail += 1
                else:
                    print(f"  #{r['id']} FAIL")
                    fail += 1
        finally:
            if owned and video and Path(video).is_file():
                try:
                    from run_archive import delete_local_video

                    delete_local_video(Path(video))
                except Exception:
                    Path(video).unlink(missing_ok=True)
    print("done ok", ok, "fail", fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
