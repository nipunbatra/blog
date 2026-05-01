"""FastAPI backend for the Sapiens2 try-it app.

Endpoints:
  GET  /                       -> index.html
  GET  /static/{path}          -> static assets
  GET  /samples/{path}         -> sample images
  GET  /videos/{path}          -> sample videos
  GET  /api/samples            -> JSON list of image samples
  GET  /api/videos             -> JSON list of video samples (with thumb)
  POST /api/predict/image      -> JSON {image_b64, vis_b64?, meta}
  POST /api/predict/video      -> JSON {video_url, info, frames}

Run on Bhaskar:
    cd /DATA/nipun.batra/sapiens2/app
    source ../venv/bin/activate
    export SAPIENS_CHECKPOINT_ROOT=/DATA/nipun.batra/sapiens2/checkpoints
    python api.py            # serves on 0.0.0.0:8511
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import src

ROOT = Path("/DATA/nipun.batra/sapiens2")
SAMPLES = ROOT / "samples"
VIDEOS = ROOT / "videos"
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
THUMBS_DIR = Path("/tmp/sapiens2_video_thumbs")
THUMBS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("/tmp/sapiens2_video_cache")
CACHE_DIR.mkdir(exist_ok=True)


IMAGE_SAMPLES = {
    "RGB (in-distribution)": [
        ("desk_worker.jpg", "desk worker"),
        ("person1.jpg",     "frontal portrait"),
        ("person4.jpg",     "mountain climber"),
    ],
    "Thermal — grayscale (ThermEval)": [
        ("thermeval_head.png",                "head/shoulders"),
        ("thermeval_standing.png",            "standing"),
        ("thermeval_seated_arms.png",         "seated, arms crossed"),
        ("thermeval_seated_crosslegged.png",  "seated cross-legged"),
    ],
    "Thermal — FLIR iron palette (harder)": [
        ("therm_floor_crosslegged.jpg",  "floor, cross-legged"),
        ("therm_distant_stairs.jpg",     "distant on stairs"),
        ("therm_holding_object.jpg",     "cap + object"),
        ("therm_sunglasses_seated.jpg",  "sunglasses, seated"),
        ("therm_reclining.jpg",          "reclining"),
        ("therm_at_desk.jpg",            "at desk"),
    ],
    "Thermal — Wikimedia (CC)": [
        ("thermal_fae.jpg",    "Fae (CC-BY-SA 3.0)"),
        ("thermal_mensch.jpg", "Mensch (CC-BY-SA 2.5)"),
    ],
}

VIDEO_SAMPLES = [
    ("man_crunches.mp4",   "man doing crunches (indoor)"),
    ("man_meditating.mp4", "man bowing in yoga pose"),
    ("downward_dog.mp4",   "downward-dog pose (close-up)"),
    ("man_yoga_pose.mp4",  "outdoor yoga (feet view)"),
    ("walking_sunset.mp4", "silhouette walking at sunset"),
]


# ---------- model cache (LRU) + GPU serialisation lock --------------------
# A 1B Sapiens2 model is ~6 GB on the A5000, and a single forward pass adds
# ~8 GB of activations. With FastAPI's default threadpool, multiple concurrent
# requests would each try to load their own copy and race for GPU memory →
# OOM after ~10–15 images even with 24 GB. We solve this two ways:
#   1. _gpu_lock serialises ALL predict calls (one inference at a time).
#   2. _MAX_MODELS=1 keeps only the active task's weights resident.
import gc
import threading
from collections import OrderedDict

_MAX_MODELS = 1
_models: "OrderedDict[str, object]" = OrderedDict()
_gpu_lock = threading.Lock()


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model(task: str):
    """Must be called under _gpu_lock to avoid concurrent loads."""
    if task in _models:
        _models.move_to_end(task)
        return _models[task]
    while len(_models) >= _MAX_MODELS:
        evict_name, evict_model = _models.popitem(last=False)
        del evict_model
        _free()
    loader, _ = src.PREDICTORS[task]
    try:
        m = loader()
    except torch.cuda.OutOfMemoryError:
        _models.clear()
        _free()
        m = loader()
    _models[task] = m
    return m


# ---------- helpers --------------------------------------------------------
def to_b64_jpeg(bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def to_b64_png(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def video_thumb(name: str) -> Path:
    p = VIDEOS / name
    out = THUMBS_DIR / (Path(name).stem + ".jpg")
    if out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
        return out
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 4))
    ok, fr = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out), fr)
    return out


def encode_h264(src_path: str, dst_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", dst_path],
        check=True, capture_output=True,
    )


# ---------- app ------------------------------------------------------------
app = FastAPI(title="Sapiens2 try-it")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/samples", StaticFiles(directory=str(SAMPLES)),     name="samples")
app.mount("/videos",  StaticFiles(directory=str(VIDEOS)),      name="videos")
app.mount("/cached",  StaticFiles(directory=str(CACHE_DIR)),   name="cached")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def api_status():
    free_b, total_b = (0, 0)
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpus":   torch.cuda.device_count(),
        "loaded_models": list(_models.keys()),
        "max_models":    _MAX_MODELS,
        "gpu_free_gb":   round(free_b / 1e9, 2),
        "gpu_total_gb":  round(total_b / 1e9, 2),
        "disabled_tasks": sorted(DISABLED_TASKS),
    }


@app.get("/api/samples")
def api_samples():
    out = []
    for group, lst in IMAGE_SAMPLES.items():
        for fname, desc in lst:
            if (SAMPLES / fname).exists():
                out.append(dict(group=group, fname=fname, desc=desc,
                                url=f"/samples/{fname}"))
    return out


@app.get("/api/videos")
def api_videos():
    out = []
    for fname, desc in VIDEO_SAMPLES:
        p = VIDEOS / fname
        if not p.exists():
            continue
        thumb = video_thumb(fname)
        out.append(dict(fname=fname, desc=desc,
                        url=f"/videos/{fname}",
                        thumb=f"/cached/{thumb.name}" if thumb.parent == CACHE_DIR
                              else f"/static/thumbs/{thumb.name}"))
    # Move thumbnails into static/thumbs for serving
    static_thumbs = STATIC_DIR / "thumbs"
    static_thumbs.mkdir(exist_ok=True)
    for fname, _ in VIDEO_SAMPLES:
        src_thumb = THUMBS_DIR / (Path(fname).stem + ".jpg")
        dst = static_thumbs / src_thumb.name
        if src_thumb.exists() and (not dst.exists() or
                                   dst.stat().st_mtime < src_thumb.stat().st_mtime):
            shutil.copy2(src_thumb, dst)
    # Re-emit with corrected thumb URLs
    for item in out:
        item["thumb"] = f"/static/thumbs/{Path(item['fname']).stem}.jpg"
    return out


def load_image(sample: Optional[str], file: Optional[UploadFile]) -> np.ndarray:
    if sample:
        path = SAMPLES / sample
        if not path.exists():
            raise HTTPException(404, f"sample {sample} not found")
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif file:
        data = file.file.read()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    else:
        raise HTTPException(400, "provide sample or file")
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    return src.to_bgr(bgr)


DISABLED_TASKS = {"normal", "pointmap"}  # heavy / unstable on this hardware


def _error_payload(exc: Exception, task: str) -> dict:
    import traceback
    tb = traceback.format_exc()
    # Always free GPU after a crash — half-loaded models leak.
    _models.pop(task, None)
    _free()
    return {"error": str(exc), "task": task,
            "traceback": tb[-1500:], "type": type(exc).__name__}


@app.post("/api/predict/image")
def predict_image(task: str = Form(...),
                  sample: Optional[str] = Form(None),
                  file: Optional[UploadFile] = File(None)):
    if task not in src.PREDICTORS:
        raise HTTPException(400, f"bad task: {task}")
    if task in DISABLED_TASKS:
        return JSONResponse(
            {"error": f"{task} is disabled in this build (heavy / unstable on the A5000). "
                      f"Re-enable in api.py: DISABLED_TASKS.", "task": task},
            status_code=503)
    bgr = vis_bgr = meta = None
    try:
        bgr = load_image(sample, file)
        H, W = bgr.shape[:2]
        t0 = time.perf_counter()
        # Serialise ALL GPU work — one inference at a time, regardless of
        # how many concurrent requests FastAPI's threadpool dispatches.
        with _gpu_lock:
            model = get_model(task)
            _, predict = src.PREDICTORS[task]
            vis_bgr, meta = predict(model, bgr)
        dt = time.perf_counter() - t0

        result = {
            "task": task, "input_size": [W, H],
            "input": to_b64_jpeg(bgr),
            "forward_s": float(meta.get("forward_s", 0.0)),
            "wall_s": float(dt),
        }
        if task == "pose":
            kp = meta["keypoints"]; sc = meta["scores"]; names = meta["names"]
            thr = meta["kpt_thr"]
            kps = []
            for i, ((x, y), s, n) in enumerate(zip(kp, sc, names)):
                if s < thr or x < 0 or y < 0 or x >= W or y >= H:
                    continue
                kps.append({"id": i, "name": n, "x": float(x), "y": float(y),
                            "score": float(s), "is_body": i < 17})
            result.update({
                "keypoints": kps,
                "skeleton": list(src.SKELETON),
                "kpts_above_thr": int(meta["kpts_above_thr"]),
                "total_kpts": int(meta["total_kpts"]),
            })
        elif task == "seg":
            labels = meta["labels"].astype(np.uint8)
            names = {int(k): v for k, v in meta["class_names"].items()}
            result.update({
                "vis":         to_b64_jpeg(vis_bgr),
                "labels_png":  to_b64_png(labels),
                "class_names": names,
                "fg_pct":      float(meta["fg_pct"]),
                "top_classes": [{"id": c, "name": names.get(c, f"class_{c}"),
                                 "px": n} for c, n in meta["top_classes"]],
            })
            del labels
        else:  # normal / pointmap
            result["vis"] = to_b64_jpeg(vis_bgr)
            result["meta"] = {k: float(v) if isinstance(v, (int, float, np.floating))
                              else v for k, v in meta.items()
                              if isinstance(v, (int, float, np.floating))}
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(_error_payload(e, task), status_code=500)
    finally:
        # Per-request cleanup: release the input image, the visualisation,
        # and any references the predict() function returned (which include
        # CPU numpy arrays and possibly GPU tensors via meta["labels"]).
        # Also release uploaded file bytes that FastAPI buffered.
        try:
            if file is not None:
                file.file.close()
        except Exception:
            pass
        del bgr, vis_bgr, meta
        _free()


@app.post("/api/predict/video")
def predict_video(task: str = Form(...),
                  fps: float = Form(1.0),
                  sample: Optional[str] = Form(None),
                  file: Optional[UploadFile] = File(None)):
    if task not in ("pose", "seg", "normal", "pointmap"):
        raise HTTPException(400, f"bad task: {task}")
    if task in DISABLED_TASKS:
        return JSONResponse(
            {"error": f"{task} is disabled (heavy)", "task": task}, status_code=503)
    try:
        if sample:
            in_path = VIDEOS / sample
            if not in_path.exists():
                raise HTTPException(404, "no such sample video")
            in_path = str(in_path)
        elif file:
            tmp = CACHE_DIR / f"upload_{uuid.uuid4().hex}.mp4"
            with open(tmp, "wb") as f:
                f.write(file.file.read())
            in_path = str(tmp)
        else:
            raise HTTPException(400, "provide sample or file")

        cap = cv2.VideoCapture(in_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every = max(1, int(round(src_fps / max(0.5, fps))))
        idxs = list(range(0, total, every))

        frames = []
        t_start = time.perf_counter()
        # Hold the GPU lock for the whole video pass so we don't interleave
        # frames with another client's image request.
        with _gpu_lock:
            model = get_model(task)
            _, predict = src.PREDICTORS[task]
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    continue
                vis, _ = predict(model, fr)
                h, w = vis.shape[:2]
                bar = np.full((28, w, 3), 30, np.uint8)
                cv2.putText(bar, f"{task.upper()}  t={fi/src_fps:5.2f}s",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
                frames.append(np.concatenate([vis, bar], axis=0))
        cap.release()
        if not frames:
            raise RuntimeError("no frames decoded from input video")

        h, w = frames[0].shape[:2]
        raw_path = CACHE_DIR / f"raw_{uuid.uuid4().hex}.mp4"
        out_path = CACHE_DIR / f"out_{uuid.uuid4().hex}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(raw_path), fourcc, max(2.0, fps), (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
        encode_h264(str(raw_path), str(out_path))
        raw_path.unlink(missing_ok=True)

        return JSONResponse({
            "video_url": f"/cached/{out_path.name}",
            "frames": len(frames),
            "wall_s": time.perf_counter() - t_start,
            "sample_fps": fps,
            "src_fps": src_fps,
        })
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(_error_payload(e, task), status_code=500)
    finally:
        try:
            if file is not None:
                file.file.close()
        except Exception:
            pass
        _free()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8511, log_level="info")
