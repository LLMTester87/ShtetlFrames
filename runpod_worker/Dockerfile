# ShtetlFrames RunPod GPU Pod worker (HTTP /scan)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHTETL_HTTP=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY handler.py entry.py ./

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "entry:app", "--host", "0.0.0.0", "--port", "8000"]
