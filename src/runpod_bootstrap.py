"""Bootstrap RunPod pods from a public CUDA image - no local Docker / Hub push.

Worker files are pulled from the public GitHub repo so dockerArgs stay small
(large inline base64 payloads get truncated by RunPod and break uvicorn).
"""

from __future__ import annotations

import base64

# Valid Hub tags (older naming like 2.2.0-py3.10-... no longer exist on Hub).
# Prefer CUDA 12.4: many RunPod hosts reject cu129 (nvidia-container-cli: cuda>=12.9).
# Do not auto-fallback to cu129 - only use it if RUNPOD_DOCKER_IMAGE is set explicitly.
IMAGE_CANDIDATES = [
    "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204",
    "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204",
    "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
]

DEFAULT_BASE_IMAGE = IMAGE_CANDIDATES[0]

# Images that often fail nvidia-container-cli on older host drivers.
DRIVER_RISKY_IMAGE_MARKERS = ("cu1290", "cu129-", "-cu129")

# Pin after each cues/OpenAI gate change so pods do not keep old sidelocks-only prompts.
# Prefer jsDelivr — raw.githubusercontent.com/main often serves stale cues for minutes.
WORKER_COMMIT = "main"
_RAW = f"https://cdn.jsdelivr.net/gh/AIQAEngineer/ShtetlFrames@{WORKER_COMMIT}"
WORKER_RAW_BASE = f"{_RAW}/runpod_worker"
CORE_RAW_BASE = f"{_RAW}/src/shtetl_core"

# Shared vision package files fetched onto the pod next to handler.py
CORE_MODULES = (
    "__init__.py",
    "cues.py",
    "scoring.py",
    "scan.py",
    "segments.py",
    "textutil.py",
    "upload.py",
)

# Keep numpy/torch ABI happy; pin OpenCV below 5.x (needs numpy>=2 and fights torch).
PIP_NUMPY = "'numpy>=1.26.4,<2'"
PIP_PKGS = (
    "fastapi 'uvicorn[standard]' python-multipart ultralytics open-clip-torch "
    "'opencv-python-headless>=4.8,<4.12' Pillow requests yt-dlp"
)


def bootstrap_shell_script() -> str:
    """Shell that runs on the pod: install deps, fetch worker + shtetl_core, serve /scan."""
    core_curls = "\n".join(
        f'curl -fsSL "{CORE_RAW_BASE}/{name}" -o "shtetl_core/{name}"' for name in CORE_MODULES
    )
    return f"""#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p /workspace/shtetl/shtetl_core
cd /workspace/shtetl

echo "[shtetl] apt..."
apt-get update -qq || true
apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl ca-certificates || true

PY=python
command -v python >/dev/null 2>&1 || PY=python3

echo "[shtetl] pip..."
$PY -m pip install -q --upgrade pip
# numpy first; python-multipart required for FastAPI File/Form (/scan_file).
$PY -m pip install -q --force-reinstall {PIP_NUMPY}
$PY -m pip install -q {PIP_PKGS}
$PY -m pip install -q -U yt-dlp
# Re-assert numpy<2 after deps that may pull numpy 2.x.
$PY -m pip install -q --force-reinstall {PIP_NUMPY}
$PY -c "import numpy; import torch; torch.from_numpy(numpy.zeros(1)); print('numpy_torch_ok', numpy.__version__, torch.__version__)"

echo "[shtetl] install Ollama (GPU VLM verify)…"
export OLLAMA_HOST=127.0.0.1:11434
curl -fsSL https://ollama.com/install.sh | sh || true
if command -v ollama >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for i in $(seq 1 45); do
    curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  # Pull in background so uvicorn can start; entry warm waits / finishes pull.
  nohup ollama pull qwen2.5vl:3b >/tmp/ollama-pull.log 2>&1 &
else
  echo "[shtetl] ollama binary missing — entry warm will retry"
fi

echo "[shtetl] fetch worker + shtetl_core from GitHub..."
curl -fsSL "{WORKER_RAW_BASE}/entry.py" -o entry.py
curl -fsSL "{WORKER_RAW_BASE}/handler.py" -o handler.py
curl -fsSL "{WORKER_RAW_BASE}/worker_sync.py" -o worker_sync.py
curl -fsSL "{WORKER_RAW_BASE}/ollama_pod.py" -o ollama_pod.py
curl -fsSL "{_RAW}/src/openai_verify.py" -o openai_verify.py
curl -fsSL "{_RAW}/src/label_feedback.py" -o label_feedback.py
{core_curls}
ls -la entry.py handler.py worker_sync.py ollama_pod.py openai_verify.py label_feedback.py shtetl_core
$PY -c "import entry; print('entry_import_ok', entry.app)"

echo "[shtetl] starting HTTP :8000"
export PYTHONPATH=/workspace/shtetl
export SHTETL_POD=1
export OPEN_VLM_BASE_URL=http://127.0.0.1:11434/v1
export OPEN_VLM_MODEL=qwen2.5vl:3b
export VERIFY_BACKEND=openai
exec $PY -m uvicorn entry:app --host 0.0.0.0 --port 8000
"""


def docker_start_args() -> str:
    """Short dockerArgs: decode a compact base64 script and run it."""
    script_b64 = base64.b64encode(bootstrap_shell_script().encode("utf-8")).decode("ascii")
    return f"bash -c 'echo {script_b64} | base64 -d | bash'"
