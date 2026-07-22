"""SQLite persistence for ShtetlFrames queue, jobs, and review candidates."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import DB_PATH, OUTPUT_DIR

# Serialize writers — many scrape threads opening connections was freezing on Windows.
_DB_WRITE_LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS queue_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT NOT NULL UNIQUE,
  title TEXT,
  year TEXT DEFAULT '',
  source TEXT DEFAULT '',
  downloadable TEXT DEFAULT 'yes',
  status TEXT DEFAULT 'pending',
  hub_url TEXT DEFAULT '',
  error TEXT DEFAULT '',
  detail TEXT DEFAULT '',
  created_at REAL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT DEFAULT 'idle',
  phase TEXT DEFAULT 'idle',
  message TEXT DEFAULT '',
  progress REAL DEFAULT 0,
  discovered INTEGER DEFAULT 0,
  total INTEGER DEFAULT 0,
  completed INTEGER DEFAULT 0,
  hits INTEGER DEFAULT 0,
  max_videos TEXT DEFAULT 'all',
  workers INTEGER DEFAULT 2,
  hub_url TEXT DEFAULT '',
  error TEXT DEFAULT '',
  updated_at REAL
);

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id TEXT NOT NULL,
  start_sec REAL NOT NULL,
  end_sec REAL NOT NULL,
  peak_score REAL,
  mean_score REAL,
  rank_score REAL,
  hit_count INTEGER,
  best_cue TEXT,
  source_url TEXT,
  image_url TEXT,
  decision TEXT DEFAULT '',
  notes TEXT DEFAULT '',
  label TEXT DEFAULT 'orthodox_dress_candidate_not_identity',
  created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_cand_rank ON candidates(rank_score DESC);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(status);
"""


def _connect() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


@contextmanager
def db(*, write: bool = False) -> Iterator[sqlite3.Connection]:
    """Open SQLite. Pass write=True for mutations so writers serialize without blocking reads."""
    acquired = False
    if write:
        _DB_WRITE_LOCK.acquire()
        acquired = True
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        if acquired:
            _DB_WRITE_LOCK.release()


def init_db() -> None:
    with db(write=True) as conn:
        conn.executescript(SCHEMA)
        # Migrations for older DBs
        cols = {r[1] for r in conn.execute("PRAGMA table_info(queue_items)").fetchall()}
        if "error" not in cols:
            conn.execute("ALTER TABLE queue_items ADD COLUMN error TEXT DEFAULT ''")
        if "detail" not in cols:
            conn.execute("ALTER TABLE queue_items ADD COLUMN detail TEXT DEFAULT ''")
        for jid in ("discover", "scrape", "pathe_discover", "pathe_scrape"):
            conn.execute(
                "INSERT OR IGNORE INTO jobs (id, status, phase, updated_at) VALUES (?, 'idle', 'idle', ?)",
                (jid, time.time()),
            )


def set_job(job_id: str, **kwargs: Any) -> dict:
    kwargs["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in kwargs)
    with db(write=True) as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*kwargs.values(), job_id))
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else {}


def get_job(job_id: str) -> dict:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else {"id": job_id, "status": "idle", "phase": "idle"}


def list_jobs() -> dict[str, dict]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM jobs").fetchall()
    return {r["id"]: dict(r) for r in rows}


def clear_queue() -> None:
    with db(write=True) as conn:
        conn.execute("DELETE FROM queue_items")


def insert_queue_items(items: list[dict], hub_url: str = "") -> dict:
    """Batch-insert discovered items; skip duplicates. Returns n_added / n_skipped.

    Pathé asset URLs are normalized to a canonical form so host/slash variants
    do not create duplicate rows (UNIQUE on url + INSERT OR IGNORE).
    """
    added = 0
    skipped = 0
    now = time.time()
    rows = []
    seen_batch: set[str] = set()
    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            skipped += 1
            continue
        if "britishpathe.com" in url.lower():
            try:
                from britishpathe import normalize_asset_url

                url = normalize_asset_url(url) or url
            except Exception:
                pass
        if url in seen_batch:
            skipped += 1
            continue
        seen_batch.add(url)
        rows.append(
            (
                url,
                (it.get("title") or url)[:300],
                it.get("year") or "",
                it.get("source") or "",
                it.get("downloadable") or "yes",
                hub_url,
                now,
            )
        )
    if not rows:
        return {"n_added": 0, "n_skipped": skipped}
    with db(write=True) as conn:
        before = conn.execute("SELECT COUNT(*) AS n FROM queue_items").fetchone()["n"]
        conn.executemany(
            """INSERT OR IGNORE INTO queue_items
               (url, title, year, source, downloadable, status, hub_url, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            rows,
        )
        after = conn.execute("SELECT COUNT(*) AS n FROM queue_items").fetchone()["n"]
        added = after - before
        skipped += len(rows) - added
        # Refresh placeholder titles when we later learn a real name.
        for url, title, _year, _src, _dl, _hub, _ts in rows:
            if not title or title.lower().startswith("british pathé asset"):
                continue
            if title.lower().startswith("asset ") and title[6:].isdigit():
                continue
            conn.execute(
                """UPDATE queue_items SET title=?
                   WHERE url=? AND (
                     title LIKE 'British Pathé asset %'
                     OR title LIKE 'British Pathe asset %'
                     OR title LIKE 'Asset %'
                     OR title='' OR title IS NULL
                   )""",
                (title, url),
            )
    return {"n_added": added, "n_skipped": skipped}


def list_queue_page(
    *,
    offset: int = 0,
    limit: int = 100,
    status: str = "",
    q: str = "",
) -> dict:
    """Paginated queue for large discovers — returns items + total matching."""
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 500))
    clauses = ["1=1"]
    params: list = []
    if status:
        clauses.append("status=?")
        params.append(status)
    if q:
        clauses.append("(title LIKE ? OR url LIKE ? OR error LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where = " AND ".join(clauses)
    with db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM queue_items WHERE {where}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT id, url, title, year, source, downloadable, status, error, detail, hub_url, created_at
                FROM queue_items WHERE {where}
                ORDER BY
                  CASE status
                    WHEN 'downloading' THEN 0
                    WHEN 'scanning' THEN 1
                    WHEN 'uploading' THEN 2
                    WHEN 'error' THEN 3
                    WHEN 'queued' THEN 4
                    WHEN 'pending' THEN 5
                    WHEN 'done' THEN 6
                    ELSE 7
                  END,
                  id DESC
                LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def reset_stale_jobs() -> None:
    """Clear 'running' jobs left over after a server kill/crash. Call once at startup only."""
    now = time.time()
    with db(write=True) as conn:
        for jid in ("discover", "scrape", "pathe_discover", "pathe_scrape"):
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
            if row and row["status"] == "running":
                conn.execute(
                    """UPDATE jobs SET status='idle', phase='idle',
                       message='Previous run interrupted — click Start again.',
                       error='', progress=0, updated_at=? WHERE id=?""",
                    (now, jid),
                )
        conn.execute(
            "UPDATE queue_items SET status='pending', detail='' WHERE status IN "
            "('queued','scanning','downloading','uploading')"
        )


def reclaim_inflight_queue() -> int:
    """Put stuck scanning/downloading/uploading rows back to pending (safe after kill)."""
    with db(write=True) as conn:
        cur = conn.execute(
            "UPDATE queue_items SET status='pending', detail='', error='' "
            "WHERE status IN ('scanning','downloading','uploading','queued')"
        )
        return int(cur.rowcount or 0)


def queue_stats() -> dict:
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM queue_items").fetchone()["n"]
        dl = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items WHERE downloadable='yes'"
        ).fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items "
            "WHERE status IN ('pending','queued','scanning','downloading','uploading') "
            "AND downloadable='yes'"
        ).fetchone()["n"]
        done = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status='done'"
        ).fetchone()["n"]
        active = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items "
            "WHERE status IN ('scanning','downloading','uploading')"
        ).fetchone()["n"]
        errn = conn.execute(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status='error'"
        ).fetchone()["n"]
    return {
        "n_queue": total,
        "n_downloadable": dl,
        "n_pending": pending,
        # Errors are cleared and retried when Start scrape runs again.
        "n_retryable": errn,
        "n_done": done,
        "n_active": active,
        "n_error": errn,
    }


def list_queue(limit: int = 500) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM queue_items ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_queue_url(url: str) -> bool:
    with db(write=True) as conn:
        cur = conn.execute("DELETE FROM queue_items WHERE url=?", (url,))
        return cur.rowcount > 0


def take_pending(limit: int | None) -> list[dict]:
    """Fetch claimable downloadable rows; optionally cap. Marks them 'queued'.

    Reclaims stuck in-flight rows and previous errors so Start scrape retries them.
    British Pathé URLs are excluded — use the dedicated Pathé page scrape.
    """
    claim = (
        "status IN ('pending','queued','scanning','downloading','uploading','error') "
        "AND downloadable='yes' "
        "AND url NOT LIKE '%britishpathe.com%'"
    )
    order = (
        "ORDER BY CASE status "
        "WHEN 'pending' THEN 0 "
        "WHEN 'queued' THEN 1 "
        "WHEN 'error' THEN 2 "
        "ELSE 3 END, id"
    )
    with db(write=True) as conn:
        if limit is None:
            rows = conn.execute(
                f"SELECT * FROM queue_items WHERE {claim} {order}"
            ).fetchall()
        else:
            # Prefer fresh pending first, then retry errors / stuck jobs.
            rows = conn.execute(
                f"SELECT * FROM queue_items WHERE {claim} {order} LIMIT ?",
                (limit,),
            ).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            conn.executemany(
                "UPDATE queue_items SET status='queued', error='', detail='' WHERE id=?",
                [(i,) for i in ids],
            )
    return [dict(r) for r in rows]


_PATHE_URL_SQL = "url LIKE '%britishpathe.com%'"


def take_pending_pathe(limit: int | None, *, only_pending: bool = True) -> list[dict]:
    """Claim British Pathé rows for scrape. Default: only fresh ``pending``.

    Use ``only_pending=False`` to also reclaim stuck in-flight / error rows
    (manual cold start). Continuous discover+scrape must keep ``only_pending=True``
    so in-flight work is never double-claimed.
    """
    if only_pending:
        claim = f"status='pending' AND downloadable='yes' AND {_PATHE_URL_SQL}"
        order = "ORDER BY id"
    else:
        claim = (
            "status IN ('pending','queued','scanning','downloading','uploading','error') "
            f"AND downloadable='yes' AND {_PATHE_URL_SQL}"
        )
        order = (
            "ORDER BY CASE status "
            "WHEN 'pending' THEN 0 "
            "WHEN 'queued' THEN 1 "
            "WHEN 'error' THEN 2 "
            "ELSE 3 END, id"
        )
    with db(write=True) as conn:
        if limit is None:
            rows = conn.execute(
                f"SELECT * FROM queue_items WHERE {claim} {order}"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM queue_items WHERE {claim} {order} LIMIT ?",
                (int(limit),),
            ).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            conn.executemany(
                "UPDATE queue_items SET status='queued', error='', detail='' WHERE id=?",
                [(i,) for i in ids],
            )
    return [dict(r) for r in rows]


def requeue_pathe_errors() -> int:
    """Reset Pathé error rows to pending so a new scrape can retry them."""
    with db(write=True) as conn:
        cur = conn.execute(
            f"UPDATE queue_items SET status='pending', error='', detail='' "
            f"WHERE status='error' AND {_PATHE_URL_SQL}"
        )
        return int(cur.rowcount or 0)


def requeue_pathe_stuck() -> int:
    """Reset Pathé in-flight / error rows to pending (dead pod / crashed scrape)."""
    with db(write=True) as conn:
        cur = conn.execute(
            f"UPDATE queue_items SET status='pending', error='', detail='' "
            f"WHERE {_PATHE_URL_SQL} AND status IN "
            f"('queued','scanning','downloading','uploading','error')"
        )
        return int(cur.rowcount or 0)


def clear_queue_pathe() -> int:
    """Delete only British Pathé rows from the queue. Returns rows deleted."""
    with db(write=True) as conn:
        cur = conn.execute(f"DELETE FROM queue_items WHERE {_PATHE_URL_SQL}")
        return int(cur.rowcount or 0)


def list_youtube_pathe_titles(*, limit: int = 5000) -> list[str]:
    """Titles from the crawled @britishpathe YouTube hub (for Pathé name→URL)."""
    limit = max(1, min(int(limit), 100_000))
    with db() as conn:
        rows = conn.execute(
            """
            SELECT title FROM queue_items
            WHERE url LIKE '%youtube.com%'
              AND hub_url LIKE '%britishpathe%'
              AND title IS NOT NULL
              AND TRIM(title) != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        t = (r["title"] or "").strip()
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def queue_stats_pathe() -> dict:
    """Queue counters scoped to britishpathe.com URLs."""
    with db() as conn:
        row = conn.execute(
            f"""
            SELECT
              COUNT(*) AS n_queue,
              SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS n_pending,
              SUM(CASE WHEN status IN ('queued','scanning','downloading','uploading') THEN 1 ELSE 0 END) AS n_active,
              SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS n_done,
              SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS n_error
            FROM queue_items
            WHERE {_PATHE_URL_SQL}
            """
        ).fetchone()
    return {
        "n_queue": int(row["n_queue"] or 0),
        "n_pending": int(row["n_pending"] or 0),
        "n_active": int(row["n_active"] or 0),
        "n_done": int(row["n_done"] or 0),
        "n_error": int(row["n_error"] or 0),
    }


def list_queue_page_pathe(
    *,
    offset: int = 0,
    limit: int = 100,
    status: str = "",
    q: str = "",
) -> dict:
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 500))
    where = [_PATHE_URL_SQL]
    args: list = []
    if status:
        where.append("status=?")
        args.append(status)
    if q:
        where.append("(title LIKE ? OR url LIKE ?)")
        like = f"%{q}%"
        args.extend([like, like])
    clause = " AND ".join(where)
    with db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM queue_items WHERE {clause}", args
        ).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT id, url, title, year, source, downloadable, status, error, detail, hub_url, created_at
                FROM queue_items WHERE {clause}
                ORDER BY id DESC LIMIT ? OFFSET ?""",
            [*args, limit, offset],
        ).fetchall()
    return {
        "items": [dict(r) for r in rows],
        "offset": offset,
        "limit": limit,
        "total": int(total or 0),
    }


def set_queue_status(item_id: int, status: str, error: str = "", detail: str = "") -> None:
    # #region agent log
    import json
    import threading
    from pathlib import Path

    t0 = time.time()
    got_lock = _DB_WRITE_LOCK.acquire(timeout=0.0)
    if got_lock:
        _DB_WRITE_LOCK.release()
    try:
        logp = Path(__file__).resolve().parents[1] / "debug-30525a.log"
        with logp.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "30525a",
                        "hypothesisId": "D",
                        "location": "db.py:set_queue_status",
                        "message": "enter set_queue_status",
                        "data": {
                            "item_id": item_id,
                            "status": status,
                            "detail": (detail or "")[:80],
                            "lock_free": bool(got_lock),
                            "tid": threading.get_ident(),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            with db(write=True) as conn:
                conn.execute(
                    "UPDATE queue_items SET status=?, error=?, detail=? WHERE id=?",
                    (status, (error or "")[:1000], (detail or "")[:500], item_id),
                )
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(0.15 * attempt)
    if last_err is not None:
        raise RuntimeError(f"set_queue_status_failed:{last_err}") from last_err
    # #region agent log
    try:
        logp = Path(__file__).resolve().parents[1] / "debug-30525a.log"
        with logp.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "30525a",
                        "hypothesisId": "D",
                        "location": "db.py:set_queue_status",
                        "message": "exit set_queue_status",
                        "data": {
                            "item_id": item_id,
                            "status": status,
                            "elapsed_ms": int((time.time() - t0) * 1000),
                            "tid": threading.get_ident(),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion


def insert_candidates(rows: list[dict]) -> int:
    """Insert candidates and persist stills locally for Review.

    Rows may include ``still_b64`` / ``image_b64`` / ``_local_still``; those are
    saved under ``output/contact_sheets/cand_{id}.jpg`` (not kept in SQLite).
    Missing stills are queued for background frame extract from source video.
    """
    from still_ensure import enqueue_ensure_still
    from still_store import save_candidate_still

    now = time.time()
    with db(write=True) as conn:
        for r in rows:
            cur = conn.execute(
                """INSERT INTO candidates
                   (video_id, start_sec, end_sec, peak_score, mean_score, rank_score,
                    hit_count, best_cue, source_url, image_url, decision, notes, label, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)""",
                (
                    r.get("video_id"),
                    r.get("start_sec"),
                    r.get("end_sec"),
                    r.get("peak_score"),
                    r.get("mean_score"),
                    r.get("rank_score"),
                    r.get("hit_count"),
                    r.get("best_cue"),
                    r.get("source_url"),
                    r.get("image_url"),
                    (r.get("notes") or "")[:1000],
                    r.get("label") or "orthodox_dress_candidate_not_identity",
                    now,
                ),
            )
            cid = int(cur.lastrowid)
            try:
                saved = save_candidate_still(
                    cid,
                    path=r.get("_local_still") or r.get("local_still"),
                    b64=r.get("still_b64") or r.get("image_b64"),
                    image_url=r.get("image_url"),
                )
                if saved is None:
                    note = (r.get("notes") or "").strip()
                    if "no_still_bytes" not in note:
                        conn.execute(
                            "UPDATE candidates SET notes=? WHERE id=?",
                            ((f"{note} no_still_bytes".strip())[:1000], cid),
                        )
                    # Permanent recovery: extract from source video in background.
                    enqueue_ensure_still(
                        {
                            "id": cid,
                            "source_url": r.get("source_url"),
                            "video_id": r.get("video_id"),
                            "start_sec": r.get("start_sec"),
                            "end_sec": r.get("end_sec"),
                            "image_url": r.get("image_url"),
                        }
                    )
            except Exception as e:
                try:
                    note = (r.get("notes") or "").strip()
                    conn.execute(
                        "UPDATE candidates SET notes=? WHERE id=?",
                        ((f"{note} still_save_err:{e}"[:200]).strip()[:1000], cid),
                    )
                except Exception:
                    pass
                enqueue_ensure_still(
                    {
                        "id": cid,
                        "source_url": r.get("source_url"),
                        "video_id": r.get("video_id"),
                        "start_sec": r.get("start_sec"),
                        "end_sec": r.get("end_sec"),
                        "image_url": r.get("image_url"),
                    }
                )
    return len(rows)


def list_candidates(limit: int = 2000) -> list[dict]:
    from still_ensure import enqueue_ensure_still
    from still_store import local_crop_url, local_still_url, local_strip_url, save_candidate_still

    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM candidates ORDER BY rank_score DESC LIMIT ?", (limit,)
        ).fetchall()
    crop_status_fn = None
    try:
        from frame_strip import crop_status as crop_status_fn
    except Exception:
        crop_status_fn = None
    out = []
    for i, r in enumerate(rows, 1):
        d = dict(r)
        d["rank"] = i
        d["key"] = f"{d['id']}"
        # Prefer durable local still over cloud hosts that expire (litter.catbox).
        local = local_still_url(d["id"])
        cloud = (d.get("image_url") or "").strip()
        if not local and cloud and "litter.catbox.moe" not in cloud.lower():
            # Hydrate on read — pull cloud bytes into contact_sheets/ once.
            try:
                if save_candidate_still(int(d["id"]), image_url=cloud):
                    local = local_still_url(d["id"])
            except Exception:
                pass
        if not local:
            enqueue_ensure_still(d)
        if local:
            d["contact_url"] = local
        elif cloud and "litter.catbox.moe" in cloud.lower():
            # Temporary Litterbox links expire; don't send a known-dead URL to Review.
            d["contact_url"] = None
        else:
            d["contact_url"] = cloud or None
        d["strip_url"] = local_strip_url(d["id"])
        d["crop_url"] = local_crop_url(d["id"])
        if crop_status_fn is not None:
            st = crop_status_fn(d["id"])
            d["crop_status"] = st.get("status") or "none"
            d["crop_error"] = st.get("error")
            if st.get("crop_url"):
                d["crop_url"] = st["crop_url"]
        else:
            d["crop_status"] = "ready" if d["crop_url"] else "none"
            d["crop_error"] = None
        d["source_path"] = ""
        d["video_url"] = ""
        out.append(d)
    return out


def candidate_stats() -> dict:
    with db() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE decision IS NULL OR decision=''"
        ).fetchone()["n"]
        accepted = conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE decision='accept'"
        ).fetchone()["n"]
        videos = conn.execute(
            "SELECT COUNT(DISTINCT video_id) AS n FROM candidates"
        ).fetchone()["n"]
    return {
        "n_candidates": n,
        "n_pending": pending,
        "n_accepted": accepted,
        "videos_scanned": videos,
    }


def update_review(cand_id: int, decision: str, notes: str) -> None:
    with db(write=True) as conn:
        # Preserve OpenAI keep/drop tag so Review gating survives human note edits.
        prev = conn.execute("SELECT notes FROM candidates WHERE id=?", (cand_id,)).fetchone()
        prev_notes = str(prev["notes"] or "") if prev else ""
        new_notes = notes or ""
        low_new = new_notes.lower()
        for prefix in ("openai:keep", "openai:drop", "openai:uncertain"):
            if prefix in prev_notes.lower() and prefix not in low_new:
                tag = next(
                    (
                        ln
                        for ln in prev_notes.splitlines()
                        if ln.strip().lower().startswith(prefix)
                    ),
                    prefix,
                )
                new_notes = f"{tag}\n{new_notes}".strip()
                low_new = new_notes.lower()
                break
        conn.execute(
            "UPDATE candidates SET decision=?, notes=? WHERE id=?",
            (decision, new_notes, cand_id),
        )


def clear_candidates() -> None:
    with db(write=True) as conn:
        conn.execute("DELETE FROM candidates")
