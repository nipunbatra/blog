"""Sapiens2 try-it — Gradio rewrite with Image and Video tabs.

Same backend as the Streamlit app (src.py).  Two tabs:

  - Image  : pick a sample (or upload), run pose / seg / normal / pointmap
  - Video  : pick a clip (or upload), sample at N fps, overlay pose or seg
             on each frame, return as a downloadable mp4

Run on Bhaskar:
    cd /DATA/nipun.batra/sapiens2/app
    source ../venv/bin/activate
    export SAPIENS_CHECKPOINT_ROOT=/DATA/nipun.batra/sapiens2/checkpoints
    python gradio_app.py            # serves on 0.0.0.0:8512
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch

import src

SAMPLES = Path("/DATA/nipun.batra/sapiens2/samples")
VIDEOS = Path("/DATA/nipun.batra/sapiens2/videos")
TASK_NAMES = list(src.PREDICTORS.keys())  # pose / seg / normal / pointmap

IMAGE_SAMPLES = {
    "RGB": [
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


# ---------- model cache ---------------------------------------------------
_models: dict[str, object] = {}


def get_model(task: str):
    if task not in _models:
        loader, _ = src.PREDICTORS[task]
        _models[task] = loader()
    return _models[task]


# ---------- image helpers --------------------------------------------------
def cv2bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_cv2bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _pose_plot(image_rgb: np.ndarray, meta: dict):
    """Plotly figure: input image as background + scatter of keypoints
    with hover-name tooltip."""
    import plotly.graph_objects as go
    H, W = image_rgb.shape[:2]
    kp = np.array(meta["keypoints"])
    sc = np.array(meta["scores"])
    names = meta["names"]
    thr = meta.get("kpt_thr", 0.30)
    keep = (sc >= thr) & (kp[:, 0] >= 0) & (kp[:, 0] < W) & \
           (kp[:, 1] >= 0) & (kp[:, 1] < H)
    fig = go.Figure()
    fig.add_layout_image(dict(source=_to_data_uri(image_rgb),
                              x=0, y=0, sizex=W, sizey=H,
                              xref="x", yref="y", layer="below",
                              sizing="stretch"))
    # Skeleton lines
    skel_x, skel_y = [], []
    for a, b in src.SKELETON:
        if not (keep[a] and keep[b]):
            continue
        skel_x += [kp[a, 0], kp[b, 0], None]
        skel_y += [kp[a, 1], kp[b, 1], None]
    fig.add_trace(go.Scatter(x=skel_x, y=skel_y, mode="lines",
                             line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    # Keypoints (body markers larger)
    fig.add_trace(go.Scatter(
        x=kp[keep, 0], y=kp[keep, 1], mode="markers",
        marker=dict(size=[10 if i < 17 else 5 for i in np.where(keep)[0]],
                    color=sc[keep], colorscale="Viridis", cmin=0, cmax=1,
                    line=dict(color="black", width=0.5),
                    colorbar=dict(title="score", thickness=10)),
        text=[f"{names[i]} ({sc[i]:.2f})" for i in np.where(keep)[0]],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(visible=False, range=[0, W])
    fig.update_yaxes(visible=False, range=[H, 0], scaleanchor="x")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      height=520, paper_bgcolor="white", plot_bgcolor="white",
                      hovermode="closest")
    return fig


def _to_data_uri(rgb: np.ndarray) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _seg_annotations(image_rgb: np.ndarray, meta: dict):
    """Build a (base_image, [(mask_bool, label), ...]) tuple for
    gr.AnnotatedImage so each body-part class gets a hover tooltip."""
    labels = meta["labels"]
    class_names = meta["class_names"]
    out = []
    for cid in np.unique(labels):
        if cid == 0:
            continue
        name = class_names.get(int(cid), f"class_{cid}")
        out.append(((labels == cid).astype(np.uint8), name))
    return image_rgb, out


def run_image(image_rgb: np.ndarray, task: str):
    """Returns (image_out, pose_plot, seg_annot, info) where only the
    component relevant to the task is populated, the other two are None."""
    if image_rgb is None:
        return None, None, None, "Pick or upload an image first."
    bgr = src.to_bgr(rgb_to_cv2bgr(image_rgb))
    t0 = time.perf_counter()
    model = get_model(task)
    _, predict = src.PREDICTORS[task]
    vis_bgr, meta = predict(model, bgr)
    dt = time.perf_counter() - t0
    base_info = (f"**{task.upper()}** · {bgr.shape[1]}×{bgr.shape[0]} · "
                 f"forward {meta.get('forward_s', 0):.2f}s · total {dt:.1f}s")
    if task == "pose":
        info = (f"{base_info} · {meta['kpts_above_thr']}/308 keypoints above "
                f"{meta['kpt_thr']:.2f}\n\n_Hover any dot to see its keypoint "
                f"name and score._")
        return None, _pose_plot(image_rgb, meta), None, info
    if task == "seg":
        info = (f"{base_info} · foreground {meta['fg_pct']:.1f}% · "
                f"top: " + ", ".join(meta['class_names'].get(c, str(c)) +
                                     f"({n:,})" for c, n in meta['top_classes'][:4]) +
                "\n\n_Hover any region to see the body-part class name._")
        return None, None, _seg_annotations(image_rgb, meta), info
    # normal / pointmap
    return cv2bgr_to_rgb(vis_bgr), None, None, base_info


# ---------- video pipeline -------------------------------------------------
def encode_h264(src_path: str, dst_path: str):
    """Re-encode any mp4/avi to browser-friendly H.264."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", dst_path],
        check=True, capture_output=True,
    )


def run_video(video_path: str, task: str, sample_fps: float,
              progress=gr.Progress(track_tqdm=False)) -> Tuple[str, str]:
    if video_path is None:
        return None, "Pick or upload a video first."
    if task not in ("pose", "seg"):
        # normal / pointmap also work but the per-frame overlay isn't as useful
        # for the "what is this app doing" reading.  Allow them anyway.
        pass

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every_n = max(1, int(round(fps / max(0.5, sample_fps))))
    idxs = list(range(0, total, every_n))
    progress(0, desc=f"loading model · {len(idxs)} frames @ {sample_fps:.1f} fps")

    model = get_model(task)
    _, predict = src.PREDICTORS[task]

    # Read + infer + write
    frames_out = []
    t_load_start = time.perf_counter()
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        vis, _ = predict(model, frame)
        # Composite a small label band so playback shows time + task
        h, w = vis.shape[:2]
        bar = np.full((28, w, 3), 30, np.uint8)
        cv2.putText(bar, f"{task.upper()}  t={fi/fps:5.2f}s",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        frames_out.append(np.concatenate([vis, bar], axis=0))
        progress((i + 1) / len(idxs),
                 desc=f"frame {i+1}/{len(idxs)}  ({fi/fps:.1f}s)")
    cap.release()
    if not frames_out:
        return None, "No frames decoded."

    h, w = frames_out[0].shape[:2]
    tmpdir = tempfile.mkdtemp(prefix="sapiens2_vid_")
    raw_path = os.path.join(tmpdir, "raw.mp4")
    out_path = os.path.join(tmpdir, "out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(raw_path, fourcc, max(2.0, sample_fps), (w, h))
    for fr in frames_out:
        vw.write(fr)
    vw.release()
    encode_h264(raw_path, out_path)

    dt = time.perf_counter() - t_load_start
    info = (f"**{task.upper()} on {Path(video_path).name}** · "
            f"{len(frames_out)} frames @ {sample_fps:.1f} fps · "
            f"total {dt:.1f}s ({dt/len(frames_out):.2f}s/frame)")
    return out_path, info


# ---------- gallery thumbnails --------------------------------------------
def image_gallery_items():
    items = []
    for group, lst in IMAGE_SAMPLES.items():
        for fname, desc in lst:
            p = SAMPLES / fname
            if p.exists():
                items.append((str(p), f"{desc}\n[{group}]"))
    return items


def video_gallery_items():
    """Use a single still frame as the thumbnail, with the video path stashed
    in the caption (we resolve the actual video on click)."""
    items = []
    for fname, desc in VIDEO_SAMPLES:
        p = VIDEOS / fname
        if not p.exists():
            continue
        thumb_dir = Path("/tmp/sapiens2_video_thumbs")
        thumb_dir.mkdir(exist_ok=True)
        thumb = thumb_dir / (p.stem + ".jpg")
        if not thumb.exists():
            cap = cv2.VideoCapture(str(p))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 4))
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(str(thumb), fr)
            cap.release()
        items.append((str(thumb), f"{desc}\n→ {fname}"))
    return items


# ---------- gradio UI ------------------------------------------------------
CSS = """
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }
.tab-nav button { font-size: 1.05rem !important; padding: 10px 22px !important; }
"""


def build_ui():
    with gr.Blocks(title="Sapiens2 try-it") as demo:
        gr.Markdown(
            f"## Sapiens2 try-it · 1B models"
            f"\n_device = `{src.device()}` · GPUs = {torch.cuda.device_count()}_"
        )

        with gr.Tabs():
            # ---------- IMAGE TAB ----------
            with gr.Tab("Image"):
                with gr.Row():
                    img_task = gr.Radio(TASK_NAMES, value="pose", label="Task",
                                         interactive=True)
                with gr.Row():
                    img_gallery = gr.Gallery(
                        value=image_gallery_items(),
                        label="Pick a sample",
                        columns=6, height=320, object_fit="cover",
                        show_label=True, allow_preview=False,
                        elem_id="img_gallery",
                    )
                with gr.Row():
                    img_input = gr.Image(label="Input (or upload)", type="numpy",
                                          height=520)
                    with gr.Column():
                        img_output_image = gr.Image(label="Output (normal/pointmap)",
                                                     type="numpy", height=520,
                                                     visible=True)
                        img_output_pose = gr.Plot(
                            label="Pose · hover any keypoint for its name + score",
                            visible=True)
                        img_output_seg = gr.AnnotatedImage(
                            label="Seg · hover any region for its class name",
                            height=520, visible=True)
                img_run = gr.Button("Run on this image",
                                    variant="primary", size="lg")
                img_info = gr.Markdown()

                def pick_image(evt: gr.SelectData):
                    items = image_gallery_items()
                    path, _ = items[evt.index]
                    bgr = cv2.imread(path)
                    return cv2bgr_to_rgb(src.to_bgr(bgr))

                img_gallery.select(pick_image, None, img_input)
                img_run.click(
                    run_image, [img_input, img_task],
                    [img_output_image, img_output_pose, img_output_seg, img_info])

            # ---------- VIDEO TAB ----------
            with gr.Tab("Video"):
                with gr.Row():
                    vid_task = gr.Radio(["pose", "seg"], value="pose",
                                         label="Task")
                    vid_fps = gr.Slider(0.5, 4.0, value=1.0, step=0.5,
                                         label="Sample FPS",
                                         info="Frames per second to run "
                                              "inference on (lower = faster).")
                with gr.Row():
                    vid_gallery = gr.Gallery(
                        value=video_gallery_items(),
                        label="Pick a clip",
                        columns=5, height=240, object_fit="cover",
                        show_label=True, allow_preview=False,
                    )
                with gr.Row():
                    vid_input = gr.Video(label="Input video (or upload)",
                                          height=380)
                    vid_output = gr.Video(label="Output overlay video",
                                           height=380, autoplay=True, loop=True)
                vid_run = gr.Button("Run on this video", variant="primary",
                                    size="lg")
                vid_info = gr.Markdown()

                def pick_video(evt: gr.SelectData):
                    fname = VIDEO_SAMPLES[evt.index][0]
                    return str(VIDEOS / fname)

                vid_gallery.select(pick_video, None, vid_input)
                vid_run.click(run_video, [vid_input, vid_task, vid_fps],
                              [vid_output, vid_info])

            # ---------- ABOUT ----------
            with gr.Tab("About"):
                gr.Markdown(
                    """
**Models** — Sapiens2 1B for pose (308 keypoints), body-part segmentation
(29 classes, dome29 palette), surface normals, and pointmap depth.
All four loaded on demand and cached on the GPU between requests.

**Image samples** include:
- 3 RGB studio photos (in-distribution)
- 4 grayscale ThermEval thermals
- 6 FLIR iron-palette thermals (harder: distant subjects, occlusions, unusual poses)
- 2 Wikimedia CC thermals

**Video samples** are short Pexels CC0 clips of single men, fitness/yoga/walking.
Inference runs at the chosen sample FPS (default 1 fps), so a 22 s clip
processes in ≈30 s.

**Thermal note**: Sapiens2 was trained on RGB human imagery (Shutterstock /
RenderPeople).  Passing thermal IR through is honestly out-of-distribution.
Body outline → seg and gross posture → pose tend to survive; surface normals
and absolute depth do not.

**Hardware**: RTX A5000 (24 GB).  cuDNN is disabled because the bundled
cuDNN 9.1.9 cannot init against this driver; Sapiens2 is attention-dominated
so the native conv fallback costs ~10 % wall time.
                    """.strip()
                )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=8511,
        share=False,
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
        css=CSS,
        allowed_paths=[str(SAMPLES), str(VIDEOS),
                       "/tmp/sapiens2_video_thumbs"],
    )
