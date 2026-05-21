"""Rule-based sleep posture classification from 2D body keypoints.

Outputs:
   posture in {supine, prone, lateral_left, lateral_right, unknown}
   in-plane rotation angle (degrees, body axis vs vertical)
   yaw  (head turn left/right, degrees from frontal)
   pitch (chin up/down, degrees from horizontal — 2D approximation)

Inputs:
   kpts: ndarray (17, 2) — COCO-style body keypoints (nose, eyes, ears, ...)
   scores: ndarray (17,) — per-keypoint confidence (0-1)

Convention:
   Coordinate system: image pixel coords (y down).
   "left" / "right" refer to subject's left / right (not viewer's).

The classification cascade:
   1. Compute body axis: mid_hip -> mid_shoulder. Its angle vs image-vertical
      gives the in-plane rotation (theta_body).
   2. Look at the FACE block (nose, eyes, ears):
        - If all 5 high-confidence and roughly symmetric -> frontal head pose
          -> SUPINE (face visible to camera, subject lying face-up).
        - If face features have <50% mean confidence -> face hidden
          -> PRONE (lying face-down) or LATERAL with face down.
        - If only one ear is high-confidence -> lateral pose; the high-conf
          ear tells which side is up.
   3. Yaw (head turn) from the L_eye-R_eye line angle, normalised to the
      body axis.
   4. Pitch (head nod) from the (nose - mid_ear) projection length along
      body axis. Up-tilt and down-tilt produce different signs.
"""
import math
from dataclasses import dataclass
import numpy as np

# COCO body indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


@dataclass
class PostureReport:
    posture: str            # supine / prone / lateral_left / lateral_right / unknown
    theta_body: float       # in-plane rotation (deg). 0 = head-up, 90 = head-right
    yaw: float              # head turn (deg). + = subject's left, - = subject's right
    pitch: float            # chin up/down approximation (deg)
    confidence: float       # 0-1 — how trustworthy this report is
    face_visible: float     # mean face-block confidence (0-1)
    notes: str = ""


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle_deg(v):
    """angle (deg) of vector vs (0, -1) (image up), CCW positive."""
    return math.degrees(math.atan2(v[0], -v[1]))


def classify(kpts: np.ndarray, scores: np.ndarray, score_thresh=0.3) -> PostureReport:
    kp = np.asarray(kpts); sc = np.asarray(scores)
    face_block = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
    face_mean = float(sc[face_block].mean())
    notes = []

    # Body axis from hip -> shoulder (subject "up" direction).
    if sc[L_HIP] > score_thresh and sc[R_HIP] > score_thresh \
            and sc[L_SHOULDER] > score_thresh and sc[R_SHOULDER] > score_thresh:
        mh = mid(kp[L_HIP], kp[R_HIP])
        ms = mid(kp[L_SHOULDER], kp[R_SHOULDER])
        body_v = (ms[0] - mh[0], ms[1] - mh[1])
        theta_body = angle_deg(body_v)
        body_ok = True
    else:
        theta_body = float("nan")
        body_v = (0, -1)
        body_ok = False
        notes.append("body axis low-confidence")

    # Face visibility
    if face_mean < 0.2:
        posture = "prone"
        notes.append("face not visible -> prone (or lateral-face-down)")
        yaw = float("nan"); pitch = float("nan")
        return PostureReport(posture, theta_body, yaw, pitch,
                             confidence=0.6 if body_ok else 0.3,
                             face_visible=face_mean, notes="; ".join(notes))

    # If both ears are visible -> frontal head -> supine (assuming overhead camera)
    l_ear_ok = sc[L_EAR] > score_thresh
    r_ear_ok = sc[R_EAR] > score_thresh
    eye_ok = sc[L_EYE] > score_thresh and sc[R_EYE] > score_thresh

    if l_ear_ok and r_ear_ok and eye_ok:
        posture = "supine"
    elif l_ear_ok and not r_ear_ok:
        # subject's left ear visible from overhead camera -> subject is on his right side
        posture = "lateral_right"
    elif r_ear_ok and not l_ear_ok:
        posture = "lateral_left"
    else:
        posture = "unknown"
        notes.append("ear visibility ambiguous")

    # Yaw: angle of (left_eye -> right_eye) vector, normalised to body axis
    if eye_ok:
        eye_v = (kp[R_EYE][0] - kp[L_EYE][0], kp[R_EYE][1] - kp[L_EYE][1])
        eye_angle = angle_deg(eye_v)
        # If body is upright (theta_body ~ 0), eye_v should point right (eye_angle ~ 90)
        # for a frontal face. Deviation from 90 = yaw.
        yaw = eye_angle - 90 - theta_body if body_ok else float("nan")
        # Normalise to [-180, 180]
        if not math.isnan(yaw):
            while yaw > 180: yaw -= 360
            while yaw < -180: yaw += 360
    else:
        yaw = float("nan")

    # Pitch: project (nose - mid_ear) onto the body axis. + = chin down (nose
    # below ear line in body frame), - = chin up.
    if sc[NOSE] > score_thresh and (l_ear_ok or r_ear_ok):
        ears = []
        if l_ear_ok: ears.append(kp[L_EAR])
        if r_ear_ok: ears.append(kp[R_EAR])
        me = mid(ears[0], ears[-1]) if len(ears) > 1 else tuple(ears[0])
        nose_v = (kp[NOSE][0] - me[0], kp[NOSE][1] - me[1])
        # body axis as unit vector (subject "up")
        bn = math.hypot(body_v[0], body_v[1])
        if bn > 1e-6:
            ub = (body_v[0] / bn, body_v[1] / bn)
            proj = nose_v[0] * ub[0] + nose_v[1] * ub[1]
            # Convert projection magnitude relative to face size for an angle estimate
            # face size ~ inter-ear distance
            if len(ears) > 1:
                face_w = math.hypot(ears[0][0] - ears[-1][0], ears[0][1] - ears[-1][1])
            else:
                face_w = 30.0  # fallback
            if face_w > 0:
                pitch = math.degrees(math.atan2(proj, face_w / 2))
            else:
                pitch = float("nan")
        else:
            pitch = float("nan")
    else:
        pitch = float("nan")

    confidence = 0.9 if body_ok and face_mean > 0.5 and posture != "unknown" else 0.5
    return PostureReport(posture=posture, theta_body=theta_body, yaw=yaw,
                         pitch=pitch, confidence=confidence,
                         face_visible=face_mean,
                         notes="; ".join(notes) if notes else "")
