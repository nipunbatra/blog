"""Sapiens2 try-it app — pose / seg / normal / pointmap on RGB or thermal.

Run on Bhaskar:
    cd /DATA/nipun.batra/sapiens2/app
    source ../venv/bin/activate
    streamlit run app.py --server.port 8511 --server.address 0.0.0.0

Then on the IITGN network: open http://10.0.62.159:8511/
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

import src

st.set_page_config(page_title="Sapiens2 try-it", layout="wide")

SAMPLES_DIR = Path("/DATA/nipun.batra/sapiens2/samples")

SAMPLE_GROUPS = {
    "RGB (in-distribution)": [
        ("desk_worker.jpg", "desk worker, side view"),
        ("person1.jpg",     "man, frontal portrait"),
        ("person4.jpg",     "mountain climber"),
    ],
    "Thermal — grayscale (ThermEval)": [
        ("thermeval_head.png",                "head/shoulders close-up"),
        ("thermeval_standing.png",            "standing, full body"),
        ("thermeval_seated_arms.png",         "seated, arms crossed"),
        ("thermeval_seated_crosslegged.png",  "seated cross-legged"),
    ],
    "Thermal — FLIR iron palette (harder)": [
        ("therm_floor_crosslegged.jpg",  "cross-legged on floor"),
        ("therm_distant_stairs.jpg",     "distant subject on stairs"),
        ("therm_holding_object.jpg",     "cap + holding equipment"),
        ("therm_sunglasses_seated.jpg",  "sunglasses, seated"),
        ("therm_reclining.jpg",          "reclining"),
        ("therm_at_desk.jpg",            "at computer desk"),
    ],
    "Thermal — Wikimedia (creative commons)": [
        ("thermal_fae.jpg",    "Fae, mid-wavelength IR (CC-BY-SA 3.0)"),
        ("thermal_mensch.jpg", "Thermografie Mensch (CC-BY-SA 2.5)"),
    ],
}

TASK_BLURB = {
    "pose":     "308 keypoints (body + face + hands + feet). Body skeleton drawn over the input.",
    "seg":      "29-class body-part segmentation. Colored overlay (dome29 palette).",
    "normal":   "Surface normals as RGB (XYZ → R G B). Smooth = flat, color-banded = curvature.",
    "pointmap": "Per-pixel XYZ depth. Visualised as percentile-normalised turbo colormap.",
}


@st.cache_resource(show_spinner="Loading model on GPU…")
def get_model(task: str):
    loader, _ = src.PREDICTORS[task]
    return loader()


@st.cache_data(show_spinner=False)
def thumbnail(path: str, w: int = 220) -> np.ndarray:
    """Disk-thumbnail an image for the picker grid (cached on path+mtime)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = src.to_bgr(im)
    h = int(im.shape[0] * w / im.shape[1])
    return cv2.cvtColor(cv2.resize(im, (w, h)), cv2.COLOR_BGR2RGB)


def render_image(img: np.ndarray, cap: str = None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=cap, use_container_width=True)


# ---------- sidebar: just task + upload ----------
st.sidebar.title("Sapiens2 (1B)")
st.sidebar.caption(f"device = `{src.device()}`  ·  GPUs = {torch.cuda.device_count()}")
task = st.sidebar.selectbox("Task", list(src.PREDICTORS.keys()),
                            format_func=lambda t: t.upper())
st.sidebar.markdown(f"_{TASK_BLURB[task]}_")
st.sidebar.divider()
upload = st.sidebar.file_uploader("Upload your own (RGB or thermal)",
                                  type=["jpg", "jpeg", "png", "bmp", "tiff"])

# ---------- main ----------
st.title(f"Sapiens2 · {task.upper()}")

# ---------- thumbnail picker grid ----------
st.subheader("Pick a sample")
sample_choice = None
N_COLS = 6
for group, items in SAMPLE_GROUPS.items():
    st.markdown(f"**{group}**")
    cols = st.columns(N_COLS)
    for i, (fname, desc) in enumerate(items):
        col = cols[i % N_COLS]
        path = SAMPLES_DIR / fname
        with col:
            if path.exists():
                st.image(thumbnail(str(path)), use_container_width=True)
            if st.button(desc, key=f"btn_{fname}", use_container_width=True,
                         help=fname):
                sample_choice = path

st.divider()

# Resolve image source: most recent click wins; otherwise upload; otherwise default.
if sample_choice is not None:
    st.session_state["img_path"] = str(sample_choice)
    st.session_state["img_bytes"] = None
elif upload is not None:
    st.session_state["img_path"] = upload.name
    st.session_state["img_bytes"] = upload.getvalue()

if "img_path" not in st.session_state:
    st.info("Click a sample thumbnail above (or upload an image in the sidebar) to run.")
    st.stop()

img_path = st.session_state["img_path"]
img_bytes = st.session_state.get("img_bytes")

if img_bytes is not None:
    arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
else:
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if image is None:
    st.error(f"Could not read image: {img_path}")
    st.stop()

image = src.to_bgr(image)
H, W = image.shape[:2]
st.caption(f"`{img_path}`  ·  {W}×{H}")

col_in, col_out = st.columns(2)
with col_in:
    st.subheader("Input")
    render_image(image)

with col_out:
    st.subheader(f"{task.upper()} output")
    with st.spinner("Loading model + running forward pass…"):
        model = get_model(task)
        _, predict = src.PREDICTORS[task]
        vis, meta = predict(model, image)
    render_image(vis)

st.divider()
with st.expander("Run details"):
    st.json({"task": task, "checkpoint": str(src.CHECKPOINT[task]),
             "input_size": [W, H], **meta})

with st.expander("Notes — using thermal images"):
    st.markdown(
        "Sapiens2 was trained on **RGB** human imagery (Shutterstock / RenderPeople). "
        "Passing thermal IR through the RGB normalisation is honestly out-of-distribution.\n\n"
        "**What tends to survive:** body outline → seg, gross posture → pose. "
        "**What degrades fast:** fine joint locations, surface normals, depth scale.\n\n"
        "Drop your own LWIR/MWIR captures into "
        "`/DATA/nipun.batra/sapiens2/samples/` and add them to `SAMPLE_GROUPS` "
        "in `app.py` to expose them as thumbnails."
    )
