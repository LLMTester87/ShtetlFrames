# RunPod GPU worker

Code that runs **inside** the ShtetlFrames GPU pod. The local app usually pulls these files at pod boot from a pinned GitHub commit (see `src/runpod_bootstrap.py`). You normally do **not** build this image yourself.

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + `models_ready` + optional progress snapshot |
| `GET` | `/progress` | Live download / scan / upload status |
| `POST` | `/scan` | Download URL → GPU scan → JSON segments |

See [docs/RUNPOD.md](../docs/RUNPOD.md) for request/response examples and operations.

## Local iteration (optional)

```bash
pip install -r requirements.txt
uvicorn entry:app --host 0.0.0.0 --port 8000
```

## Custom image (optional)

```bash
docker build -t YOU/shtetlframes-runpod:latest .
docker push YOU/shtetlframes-runpod:latest
```

Set `RUNPOD_DOCKER_IMAGE` in the app Settings. Prefer CUDA 12.4 tags for broader RunPod host driver compatibility.

## Notes

- `/scan` is serialized with a lock (one GPU job at a time).  
- Do not add a top-level `import runpod` — HTTP pod mode does not install that package. Serverless entry remains under `if __name__ == "__main__"`.  
- Never commit API keys.
