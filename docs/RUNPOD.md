# RunPod guide

## Default path (recommended)

No Docker Desktop, no custom image push, no Serverless endpoint.

1. Create an API key: https://www.runpod.io/console/user/settings  
2. In ShtetlFrames Settings → paste key → backend **runpod**  
3. Click **Start scrape**

The app will:

1. Terminate/replace broken or CUDA-too-new pods named `shtetlframes-scan`  
2. Deploy an on-demand GPU pod from a public `runpod/pytorch:…` image  
3. Run bootstrap (`dockerArgs`) to install deps and pull worker code from GitHub  
4. Wait until `GET /health` returns `models_ready: true`  
5. Send videos to `POST /scan` and poll `GET /progress`  
6. Optionally stop the pod when the scrape finishes  

## Pod HTTP API

Base URL: `https://{podId}-8000.proxy.runpod.net`

### `GET /health`

```json
{
  "ok": true,
  "service": "shtetlframes",
  "device": "cuda",
  "models_ready": true,
  "warm_error": null,
  "progress": { "phase": "idle", "message": "", "...": "..." }
}
```

### `GET /progress`

Live status for the job currently holding the GPU lock.

| Field | Meaning |
|-------|---------|
| `phase` | `idle` / `download` / `scan` / `upload` / `done` / `error` |
| `message` | Short status |
| `pct` | 0–100 when known |
| `detail` | Speed, ETA, scan time, hit counts |
| `queue_id` | Local queue row id when provided |
| `title` / `url` | Current video |

### `POST /scan`

Body:

```json
{
  "url": "https://www.youtube.com/watch?v=…",
  "title": "Optional title",
  "queue_id": 1234,
  "sample_fps": 1.5,
  "score_threshold": 0.05,
  "source_url": "https://…"
}
```

Success (`200`):

```json
{
  "ok": true,
  "video_id": "…",
  "segments": [
    {
      "start_sec": 12.0,
      "end_sec": 18.5,
      "peak_score": 0.21,
      "mean_score": 0.18,
      "rank_score": 0.4,
      "hit_count": 5,
      "best_cue": "…",
      "source_url": "…",
      "image_url": "https://…"
    }
  ],
  "n_hits": 1,
  "device": "cuda"
}
```

Failure (`500`): `{ "ok": false, "error": "…", "detail": "…" }`.

Scans are **serialized** with a threading lock (one GPU job at a time).

## Worker pin

`src/runpod_bootstrap.py` sets:

```python
WORKER_COMMIT = "<git sha>"
WORKER_RAW_BASE = f"https://raw.githubusercontent.com/LLMTester87/ShtetlFrames/{WORKER_COMMIT}/runpod_worker"
```

After changing `runpod_worker/*`, commit + push, then update `WORKER_COMMIT` to that SHA and push again. New pods curl that exact tree (avoids stale CDN copies of `main`).

## Images and CUDA drivers

Tried in order (unless `RUNPOD_DOCKER_IMAGE` overrides):

1. `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204` ← default  
2. `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204`  
3. `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`  

If you see:

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9
```

the host driver is older than the image. Use CUDA 12.4 (default) or set an older image explicitly. Do not use `cu1290` unless the machine advertises a new enough driver.

## GPU capacity fallbacks

If the preferred GPU type is full, provisioning walks a fallback list (4090 → 3090 → A5000 → … → T4) and retries Community/Secure cloud + disk sizes.

## Parallelism

| Setting | Effect |
|---------|--------|
| `RUNPOD_MAX_INFLIGHT` | How many `/scan` HTTP calls your PC keeps open |
| Pod `_scan_lock` | Only one call runs YOLO/CLIP at a time |

High inflight (e.g. 6–11) mostly queues HTTP waiters; it does not multiply GPU throughput. Use 1–2.

## Common errors

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `cuda>=12.9` on container start | Image too new for host driver | Stay on cu124; terminate pod; Start scrape again |
| `ModuleNotFoundError: runpod` | Old worker imported serverless SDK at import time | Ensure pin ≥ commit that lazy-imports `runpod` |
| `pod_bad_json: 524` | RunPod proxy timeout (long download/scan) | Raise timeout; shorter videos; retry |
| `yt-dlp returned no file` / members-only | Video unavailable or members-only | Expected; row marked `error` |
| UI stuck on `pod scanning…` with no % | Old pod without `/progress` | Recreate pod after updating `WORKER_COMMIT` |
| Health 403 from random curl | Proxy / UA quirks | App uses the same proxy; rely on in-app health wait |

## Optional: custom Docker image

Only needed if you want a prebaked image (faster boot, no pip on start):

```bash
cd runpod_worker
docker build -t YOU/shtetlframes-runpod:latest .
docker push YOU/shtetlframes-runpod:latest
```

Set `RUNPOD_DOCKER_IMAGE=YOU/shtetlframes-runpod:latest` in Settings. Your image must still expose port `8000` with the same HTTP API (or keep using bootstrap `dockerArgs` if the entrypoint allows it).

## Cost control

- Enable **Stop pod when done** (`RUNPOD_STOP_WHEN_DONE=1`).  
- Terminate stuck `shtetlframes-scan` pods in the RunPod console if a scrape is aborted.  
- Prefer shorter sample rates only via code defaults if you change `DEFAULT_FPS` (trade recall vs cost).
