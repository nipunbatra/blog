"""Tiny client for the Sapiens2 try-it FastAPI.

Downloads a few random single-person images from Pexels (CC0), posts each
to /api/predict/image with task=pose (and once with task=seg), and prints
the parsed JSON response.

Run with the SSH tunnel up, e.g.
    ssh -fN -L 18511:localhost:8511 bhaskar.iitgn.ac.in
    python api_client_demo.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import requests

API = "http://localhost:18511"     # via SSH tunnel; set to http://10.0.62.159:8511 on intranet

IMAGES = [
    ("man_portrait_220453",   "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?w=800"),
    ("woman_portrait_1043471","https://images.pexels.com/photos/1043471/pexels-photo-1043471.jpeg?w=800"),
    ("woman_outdoors_3777943","https://images.pexels.com/photos/3777943/pexels-photo-3777943.jpeg?w=800"),
]


def fetch(url: str) -> bytes:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content


def predict(image_bytes: bytes, fname: str, task: str = "pose") -> dict:
    files = {"file": (fname, image_bytes, "image/jpeg")}
    data  = {"task": task}
    t0 = time.time()
    r = requests.post(f"{API}/api/predict/image", data=data, files=files, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    js = r.json()
    js["_client_wall_s"] = round(elapsed, 2)
    return js


def summarise_pose(d: dict) -> str:
    if "error" in d:
        return f"ERROR ({d['type']}): {d['error']}"
    kps = d["keypoints"]
    body = [k for k in kps if k["is_body"]]
    high = [k for k in kps if k["score"] >= 0.5]
    top = sorted(body, key=lambda k: -k["score"])[:5]
    lines = [
        f"  size       : {d['input_size'][0]}x{d['input_size'][1]}",
        f"  forward    : {d['forward_s']:.2f}s server  /  {d['_client_wall_s']:.2f}s wall",
        f"  kpts above : {d['kpts_above_thr']}/{d['total_kpts']} (score>=0.30 default)",
        f"  body kpts  : {sum(1 for k in body if k['score']>=0.5)}/17 above 0.5",
        f"  top body   : " + ", ".join(f"{k['name']}({k['score']:.2f})" for k in top),
    ]
    return "\n".join(lines)


def summarise_seg(d: dict) -> str:
    if "error" in d:
        return f"ERROR ({d['type']}): {d['error']}"
    top = d["top_classes"][:5]
    return (f"  size       : {d['input_size'][0]}x{d['input_size'][1]}\n"
            f"  forward    : {d['forward_s']:.2f}s server  /  {d['_client_wall_s']:.2f}s wall\n"
            f"  fg_pct     : {d['fg_pct']:.1f}%\n"
            f"  top classes: " + ", ".join(f"{c['name']}({c['px']:,}px)" for c in top))


def main():
    # Sanity check first.
    s = requests.get(f"{API}/api/status", timeout=5).json()
    print(f"server  GPU {s['gpu_free_gb']}/{s['gpu_total_gb']} GB free  "
          f"loaded={s['loaded_models']}  disabled={s['disabled_tasks']}\n")

    out_dir = Path("/tmp/sapiens2_api_demo")
    out_dir.mkdir(exist_ok=True)

    for name, url in IMAGES:
        print(f"--- {name} ---")
        print(f"  url        : {url}")
        try:
            blob = fetch(url)
        except Exception as e:
            print(f"  fetch err  : {e}\n"); continue
        # cache locally so we can re-run without re-downloading
        (out_dir / f"{name}.jpg").write_bytes(blob)

        # Pose
        try:
            pose = predict(blob, f"{name}.jpg", task="pose")
            print("[pose]")
            print(summarise_pose(pose))
        except Exception as e:
            print(f"  pose err   : {e}")
            continue

        # One seg run on the first image to demonstrate the other task.
        if name == IMAGES[0][0]:
            try:
                seg = predict(blob, f"{name}.jpg", task="seg")
                print("[seg]")
                print(summarise_seg(seg))
                # Persist the JSON for inspection (sans the giant image strings)
                light = {k: v for k, v in seg.items() if k not in ("input", "vis", "labels_png")}
                (out_dir / f"{name}_seg.json").write_text(json.dumps(light, indent=2))
                print(f"  trimmed JSON saved to {out_dir / f'{name}_seg.json'}")
            except Exception as e:
                print(f"  seg err   : {e}")
        print()


if __name__ == "__main__":
    main()
