"""Client for the Sapiens2 try-it FastAPI.

Usage as a library:
    from sapiens2_client import Sapiens2Client
    c = Sapiens2Client("http://10.0.62.159:8511")
    res  = c.predict("photo.jpg", task="pose")               # local path
    res  = c.predict("https://.../portrait.jpg", task="seg")  # URL
    res  = c.predict(bgr_numpy_array, task="pose")            # numpy / PIL / bytes

    # Batch — concurrent HTTP, server still serialises on one GPU.
    results = c.predict_batch(["a.jpg", "b.jpg", "c.jpg"], task="pose",
                              max_workers=4)

CLI:
    python sapiens2_client.py photo1.jpg photo2.jpg --task pose
    python sapiens2_client.py photo*.jpg --task seg --workers 4 --out out_dir/
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence, Union

import requests

# Optional deps — only needed if you pass numpy arrays / PIL images.
try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None

ImageLike = Union[str, Path, bytes, "np.ndarray", "Image.Image"]


class Sapiens2Client:
    def __init__(self, base_url: str = "http://localhost:18511",
                 timeout: float = 120.0, session: requests.Session | None = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ---------- low-level ----------
    def status(self) -> dict:
        r = self.session.get(f"{self.base}/api/status", timeout=10)
        r.raise_for_status()
        return r.json()

    def _to_jpeg_bytes(self, img: ImageLike, name_hint: str = "input.jpg") -> tuple[str, bytes]:
        """Coerce a path / URL / bytes / numpy / PIL into (filename, jpeg-bytes)."""
        # Path / URL
        if isinstance(img, (str, Path)):
            s = str(img)
            if s.startswith(("http://", "https://")):
                r = self.session.get(s, timeout=self.timeout)
                r.raise_for_status()
                return Path(s.split("?", 1)[0]).name or name_hint, r.content
            data = Path(s).read_bytes()
            return Path(s).name, data
        # Already bytes
        if isinstance(img, (bytes, bytearray)):
            return name_hint, bytes(img)
        # numpy
        if np is not None and isinstance(img, np.ndarray):
            if Image is None:
                raise RuntimeError("Pillow needed to encode numpy arrays")
            arr = img
            if arr.ndim == 2:
                pil = Image.fromarray(arr).convert("RGB")
            elif arr.shape[2] == 3:  # assume BGR (OpenCV convention)
                pil = Image.fromarray(arr[..., ::-1])
            elif arr.shape[2] == 4:
                pil = Image.fromarray(arr[..., [2, 1, 0]])
            else:
                raise ValueError(f"unsupported array shape {arr.shape}")
            buf = pil.tobytes  # placeholder
            import io
            b = io.BytesIO(); pil.save(b, format="JPEG", quality=92)
            return name_hint, b.getvalue()
        # PIL
        if Image is not None and isinstance(img, Image.Image):
            import io
            b = io.BytesIO(); img.convert("RGB").save(b, format="JPEG", quality=92)
            return name_hint, b.getvalue()
        raise TypeError(f"unsupported image type: {type(img)}")

    def predict(self, img: ImageLike, task: str = "pose") -> dict:
        """Run one image through /api/predict/image. Returns the parsed JSON."""
        name, data = self._to_jpeg_bytes(img)
        files = {"file": (name, data, "image/jpeg")}
        body  = {"task": task}
        t0 = time.time()
        r = self.session.post(f"{self.base}/api/predict/image",
                              data=body, files=files, timeout=self.timeout)
        elapsed = time.time() - t0
        # Even error responses are JSON for this API; raise only on transport.
        try:
            js = r.json()
        except Exception:
            r.raise_for_status()
            raise
        js["_client_wall_s"] = round(elapsed, 3)
        js["_http_status"]   = r.status_code
        return js

    # ---------- batch ----------
    def predict_batch(self, images: Sequence[ImageLike], task: str = "pose",
                      max_workers: int = 4, on_done=None) -> list[dict]:
        """Run a batch of images concurrently via ThreadPoolExecutor.

        Note: the server runs one inference at a time on a single GPU, so
        client-side concurrency mostly buys you upload/download overlap with
        compute. Throughput plateaus around max_workers ≈ 2–4.

        Returns results in the *same order* as input. If `on_done` is given,
        it's called as `on_done(idx, result)` as each finishes (out of order).
        """
        results: list[dict | None] = [None] * len(images)

        def worker(idx_img):
            idx, im = idx_img
            try:
                res = self.predict(im, task=task)
            except Exception as e:
                res = {"error": str(e), "type": type(e).__name__,
                       "task": task, "_idx": idx}
            return idx, res

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker, (i, im)) for i, im in enumerate(images)]
            for f in as_completed(futures):
                idx, res = f.result()
                results[idx] = res
                if on_done is not None:
                    on_done(idx, res)
        return results  # type: ignore[return-value]


# ---------- CLI ----------
def _summarise(d: dict) -> str:
    if "error" in d:
        return f"ERR ({d.get('type','?')}): {d['error']}"
    if d["task"] == "pose":
        return (f"pose  {d['input_size'][0]}x{d['input_size'][1]}  "
                f"{d['kpts_above_thr']}/{d['total_kpts']} kpts  "
                f"forward {d['forward_s']:.2f}s  wall {d['_client_wall_s']:.2f}s")
    if d["task"] == "seg":
        top = ", ".join(f"{c['name']}({c['px']:,}px)" for c in d["top_classes"][:3])
        return (f"seg   {d['input_size'][0]}x{d['input_size'][1]}  "
                f"fg {d['fg_pct']:.1f}%  top: {top}  "
                f"forward {d['forward_s']:.2f}s  wall {d['_client_wall_s']:.2f}s")
    return f"{d['task']}  {d['input_size']}  forward {d.get('forward_s', '?')}s"


def main():
    p = argparse.ArgumentParser(description="Sapiens2 API batch client")
    p.add_argument("images", nargs="+", help="image paths or URLs")
    p.add_argument("--task", default="pose",
                   choices=["pose", "seg", "normal", "pointmap"])
    p.add_argument("--api",  default="http://localhost:18511",
                   help="server URL (default tunneled localhost; use "
                        "http://10.0.62.159:8511 on intranet)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=None,
                   help="if set, write each response as <stem>.json here")
    args = p.parse_args()

    c = Sapiens2Client(args.api)
    s = c.status()
    print(f"server: GPU {s['gpu_free_gb']}/{s['gpu_total_gb']} GB free, "
          f"loaded={s['loaded_models']}, disabled={s['disabled_tasks']}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    def on_done(i, res):
        label = Path(str(args.images[i])).name
        print(f"  [{i+1}/{len(args.images)}] {label:40s}  {_summarise(res)}")
    results = c.predict_batch(args.images, task=args.task,
                              max_workers=args.workers, on_done=on_done)
    print(f"\ntotal: {len(results)} images in {time.time()-t0:.1f}s "
          f"(workers={args.workers})")

    if args.out:
        for src, res in zip(args.images, results):
            stem = Path(str(src)).stem
            light = {k: v for k, v in res.items()
                     if k not in ("input", "vis", "labels_png")}
            (args.out / f"{stem}_{args.task}.json").write_text(
                json.dumps(light, indent=2))
        print(f"wrote {len(results)} JSON files to {args.out}")


if __name__ == "__main__":
    main()
