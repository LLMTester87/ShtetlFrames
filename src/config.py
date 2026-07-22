"""Cue prompts and scan defaults for Orthodox (Hasidic + Litvish) visual candidates."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_DIR = ROOT / "output"
CANDIDATES_PATH = OUTPUT_DIR / "candidates.jsonl"
REVIEW_CSV = OUTPUT_DIR / "review_queue.csv"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"
CROPS_DIR = OUTPUT_DIR / "hit_crops"
DB_PATH = OUTPUT_DIR / "shtetlframes.db"
BATCH_IA_CSV = OUTPUT_DIR / "ia_batch_discoveries.csv"
BULK_QUEUE_CSV = OUTPUT_DIR / "bulk_queue.csv"
ARCHIVE_TARGETS_MD = OUTPUT_DIR / "ARCHIVE_TARGETS.md"

USER_AGENT_CLOUD = "ShtetlFrames/1.0 (research; image upload)"
DEFAULT_WORKERS = 2
MAX_WORKERS = 16
DEFAULT_DISCOVER_MAX = 5000
DISCOVER_HARD_CAP = 1_000_000  # UI max for discover listing
QUEUE_PAGE_SIZE = 100

# YouTube anti-bot (yt-dlp wiki): try cookies from a real logged-in browser
# Set to edge / chrome / firefox / brave / opera — empty / none skips cookies
# Default edge: many Windows PCs have Edge but not Chrome.
YT_COOKIES_BROWSER = (os.environ.get("YT_COOKIES_BROWSER") or "edge").strip().lower()
YT_COOKIES_FILE = (os.environ.get("YT_COOKIES_FILE") or "").strip()
# YouTube proxy: scrapfly | scrapingdog | auto (see yt_proxy.py)
PROXY_PROVIDER = (os.environ.get("PROXY_PROVIDER") or "auto").strip().lower()
SCRAPFLY_API_KEY = (os.environ.get("SCRAPFLY_API_KEY") or "").strip()
SCRAPINGDOG_API_KEY = (os.environ.get("SCRAPINGDOG_API_KEY") or "").strip()
# Sparse section sampling (cut full-video proxy bandwidth on long videos)
try:
    SPARSE_SECTION_SEC = float(os.environ.get("SPARSE_SECTION_SEC") or "20")
except ValueError:
    SPARSE_SECTION_SEC = 20.0
try:
    SPARSE_STRIDE_SEC = float(os.environ.get("SPARSE_STRIDE_SEC") or "60")
except ValueError:
    SPARSE_STRIDE_SEC = 60.0
try:
    DENSE_PAD_SEC = float(os.environ.get("DENSE_PAD_SEC") or "30")
except ValueError:
    DENSE_PAD_SEC = 30.0
try:
    MAX_DENSE_SEC = float(os.environ.get("MAX_DENSE_SEC") or "600")
except ValueError:
    MAX_DENSE_SEC = 600.0
# Player clients to try in order if one fails (see yt-dlp YouTube extractor docs)
# Cookie-capable clients first when a cookie jar is available (web/mweb/tv).
YT_PLAYER_CLIENTS = ("tv", "web", "mweb", "android_vr", "android", "ios")


# Cue / model defaults — single source of truth in shtetl_core.cues
from shtetl_core.cues import (  # noqa: E402
    CLIP_MODEL,
    CLIP_PRETRAINED,
    DEFAULT_FPS,
    DEFAULT_SCORE_THRESHOLD,
    HEADCOVER_PROMPTS,
    MAX_GAP_SEC,
    MIN_HEADCOVER_SCORE,
    MIN_PERSON_AREA,
    MIN_POS_SCORE,
    MIN_SEGMENT_SEC,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    TOP_K_CUES,
    YOLO_CONF,
    YOLO_WEIGHTS,
)

# Scan backend: "local" or "runpod" (auto GPU Pod via API — no endpoint ID)
SCAN_BACKEND = (os.environ.get("SCAN_BACKEND") or "local").strip().lower()
RUNPOD_API_KEY = (os.environ.get("RUNPOD_API_KEY") or "").strip()
RUNPOD_DOCKER_IMAGE = (os.environ.get("RUNPOD_DOCKER_IMAGE") or "").strip()
RUNPOD_GPU_TYPE = (os.environ.get("RUNPOD_GPU_TYPE") or "NVIDIA GeForce RTX 3090").strip()
RUNPOD_POD_ID = (os.environ.get("RUNPOD_POD_ID") or "").strip()
RUNPOD_STOP_WHEN_DONE = (os.environ.get("RUNPOD_STOP_WHEN_DONE") or "1").strip() in (
    "1",
    "true",
    "yes",
    "on",
)
RUNPOD_MAX_INFLIGHT = int(os.environ.get("RUNPOD_MAX_INFLIGHT") or "8")
try:
    PATHE_STACK_MAX = int(os.environ.get("PATHE_STACK_MAX") or "3")
except ValueError:
    PATHE_STACK_MAX = 3
PATHE_STACK_MAX = max(1, min(6, PATHE_STACK_MAX))
RUNPOD_JOB_TIMEOUT_SEC = int(os.environ.get("RUNPOD_JOB_TIMEOUT_SEC") or "1800")
RUNPOD_POLL_SEC = float(os.environ.get("RUNPOD_POLL_SEC") or "2.5")
# Deprecated — kept so old .env keys do not crash; ignored
RUNPOD_ENDPOINT_ID = (os.environ.get("RUNPOD_ENDPOINT_ID") or "").strip()

try:
    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD") or DEFAULT_SCORE_THRESHOLD)
except ValueError:
    SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD


def load_env() -> None:
    """Load .env then UI/DB settings (UI wins). Refresh module globals."""
    global SCAN_BACKEND, RUNPOD_API_KEY, RUNPOD_DOCKER_IMAGE, RUNPOD_GPU_TYPE
    global RUNPOD_POD_ID, RUNPOD_STOP_WHEN_DONE, RUNPOD_ENDPOINT_ID
    global RUNPOD_MAX_INFLIGHT, PATHE_STACK_MAX, RUNPOD_JOB_TIMEOUT_SEC, RUNPOD_POLL_SEC
    global SCORE_THRESHOLD, YT_COOKIES_BROWSER, YT_COOKIES_FILE
    global PROXY_PROVIDER, SCRAPFLY_API_KEY, SCRAPINGDOG_API_KEY
    global SPARSE_SECTION_SEC, SPARSE_STRIDE_SEC, DENSE_PAD_SEC, MAX_DENSE_SEC
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    try:
        from settings_store import apply_settings_to_environ

        apply_settings_to_environ()
    except Exception:
        pass
    SCAN_BACKEND = (os.environ.get("SCAN_BACKEND") or "local").strip().lower()
    RUNPOD_API_KEY = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    RUNPOD_DOCKER_IMAGE = (os.environ.get("RUNPOD_DOCKER_IMAGE") or "").strip()
    RUNPOD_GPU_TYPE = (os.environ.get("RUNPOD_GPU_TYPE") or "NVIDIA GeForce RTX 3090").strip()
    RUNPOD_POD_ID = (os.environ.get("RUNPOD_POD_ID") or "").strip()
    RUNPOD_STOP_WHEN_DONE = (os.environ.get("RUNPOD_STOP_WHEN_DONE") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    RUNPOD_ENDPOINT_ID = (os.environ.get("RUNPOD_ENDPOINT_ID") or "").strip()
    try:
        RUNPOD_MAX_INFLIGHT = int(os.environ.get("RUNPOD_MAX_INFLIGHT") or "8")
    except ValueError:
        RUNPOD_MAX_INFLIGHT = 2
    try:
        PATHE_STACK_MAX = int(os.environ.get("PATHE_STACK_MAX") or "3")
    except ValueError:
        PATHE_STACK_MAX = 3
    PATHE_STACK_MAX = max(1, min(6, PATHE_STACK_MAX))
    try:
        RUNPOD_JOB_TIMEOUT_SEC = int(os.environ.get("RUNPOD_JOB_TIMEOUT_SEC") or "1800")
    except ValueError:
        RUNPOD_JOB_TIMEOUT_SEC = 1800
    try:
        RUNPOD_POLL_SEC = float(os.environ.get("RUNPOD_POLL_SEC") or "2.5")
    except ValueError:
        RUNPOD_POLL_SEC = 2.5
    try:
        SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD") or DEFAULT_SCORE_THRESHOLD)
    except ValueError:
        SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
    SCORE_THRESHOLD = max(-0.5, min(0.35, SCORE_THRESHOLD))
    YT_COOKIES_BROWSER = (os.environ.get("YT_COOKIES_BROWSER") or "edge").strip().lower()
    YT_COOKIES_FILE = (os.environ.get("YT_COOKIES_FILE") or "").strip()
    PROXY_PROVIDER = (os.environ.get("PROXY_PROVIDER") or "auto").strip().lower()
    SCRAPFLY_API_KEY = (os.environ.get("SCRAPFLY_API_KEY") or "").strip()
    SCRAPINGDOG_API_KEY = (os.environ.get("SCRAPINGDOG_API_KEY") or "").strip()
    try:
        SPARSE_SECTION_SEC = float(os.environ.get("SPARSE_SECTION_SEC") or "20")
    except ValueError:
        SPARSE_SECTION_SEC = 20.0
    try:
        SPARSE_STRIDE_SEC = float(os.environ.get("SPARSE_STRIDE_SEC") or "60")
    except ValueError:
        SPARSE_STRIDE_SEC = 60.0
    try:
        DENSE_PAD_SEC = float(os.environ.get("DENSE_PAD_SEC") or "30")
    except ValueError:
        DENSE_PAD_SEC = 30.0
    try:
        MAX_DENSE_SEC = float(os.environ.get("MAX_DENSE_SEC") or "600")
    except ValueError:
        MAX_DENSE_SEC = 600.0
    SPARSE_SECTION_SEC = max(5.0, SPARSE_SECTION_SEC)
    SPARSE_STRIDE_SEC = max(SPARSE_SECTION_SEC, SPARSE_STRIDE_SEC)
    DENSE_PAD_SEC = max(5.0, DENSE_PAD_SEC)
    MAX_DENSE_SEC = max(60.0, MAX_DENSE_SEC)


def runpod_configured() -> bool:
    """Ready for RunPod when API key is set (public image + bootstrap, no Docker on PC)."""
    return bool(RUNPOD_API_KEY)


def effective_scan_backend() -> str:
    if SCAN_BACKEND == "runpod" and runpod_configured():
        return "runpod"
    return "local"
