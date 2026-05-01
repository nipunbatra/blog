"""Standalone Sapiens2 API client — single file, single dep.

For students / collaborators who only need to *use* the API. Drop this one
file into any folder, install requests + tqdm, and run.

    pip install requests tqdm

Set the URL once (env var or edit below):
    export SAPIENS2_API=http://10.0.62.159:8511     # on IITGN intranet
    # OR via SSH tunnel from off-campus:
    #   ssh -L 18511:localhost:8511 bhaskar.iitgn.ac.in
    #   export SAPIENS2_API=http://localhost:18511

Usage:
    # Single image, prints summary JSON
    python student_client.py photo.jpg --task pose

    # Folder of 7,000 images → JSONL output, with resume support
    python student_client.py /path/to/images/ \\
        --task pose --workers 4 --out results.jsonl

The output JSONL has one line per image. Each line is the full server
JSON plus a "src" field pointing to the input file.

Resume: re-running the same command skips images already in results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # noqa: E704

API = os.environ.get("SAPIENS2_API", "http://10.0.62.159:8511")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def predict(path: Path, task: str, timeout: float = 120.0) -> dict:
    """One HTTP POST. Returns the parsed JSON; on transport error wraps it."""
    with open(path, "rb") as f:
        try:
            r = requests.post(
                f"{API}/api/predict/image",
                data={"task": task},
                files={"file": (path.name, f, "image/jpeg")},
                timeout=timeout,
            )
            try:
                js = r.json()
            except Exception:
                return {"error": f"bad JSON (status {r.status_code})",
                        "raw": r.text[:300], "src": str(path)}
            js["src"] = str(path)
            js["_http_status"] = r.status_code
            return js
        except requests.RequestException as e:
            return {"error": str(e), "type": type(e).__name__, "src": str(path)}


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in EXTS)
    raise SystemExit(f"not a file or directory: {target}")


def already_done(out_path: Path) -> set[str]:
    """Read existing JSONL and return the set of `src` paths already processed."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if "src" in d and "error" not in d:
                    done.add(d["src"])
            except Exception:
                pass
    return done


def trim(d: dict) -> dict:
    """Drop the giant base64 image fields before writing — keeps JSONL small."""
    return {k: v for k, v in d.items() if k not in ("input", "vis", "labels_png")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path,
                   help="image file OR a directory (walked recursively)")
    p.add_argument("--task", default="pose",
                   choices=["pose", "seg", "normal", "pointmap"])
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent HTTP requests (server is single-GPU; 4 is plenty)")
    p.add_argument("--out", type=Path, default=Path("results.jsonl"),
                   help="JSONL output path; resume-aware on re-runs")
    p.add_argument("--api", default=API, help=f"server URL (default {API})")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on number of images (handy for a quick test)")
    args = p.parse_args()

    global API
    API = args.api

    # Sanity-check the server first.
    try:
        s = requests.get(f"{API}/api/status", timeout=10).json()
        print(f"server: {API}  GPU {s['gpu_free_gb']}/{s['gpu_total_gb']} GB free  "
              f"loaded={s['loaded_models']}  disabled={s['disabled_tasks']}")
    except Exception as e:
        sys.exit(f"can't reach {API}: {e}")

    images = collect_images(args.target)
    if args.limit:
        images = images[: args.limit]
    done = already_done(args.out)
    pending = [p for p in images if str(p) not in done]
    print(f"{len(images)} images total · {len(done)} already done · {len(pending)} to do")
    if not pending:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_ok = n_err = 0

    with open(args.out, "a") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict, p, args.task): p for p in pending}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"{args.task} via {args.workers}w"):
            d = f.result()
            if "error" in d:
                n_err += 1
            else:
                n_ok += 1
            fout.write(json.dumps(trim(d)) + "\n")
            fout.flush()

    dt = time.time() - started
    print(f"\ndone: ok={n_ok}  err={n_err}  total {dt:.0f}s  "
          f"throughput {n_ok/dt:.2f} img/s "
          f"(extrapolated to 7000: {7000/(n_ok/dt)/3600:.2f} h)")


if __name__ == "__main__":
    main()
