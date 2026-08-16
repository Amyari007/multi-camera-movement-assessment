"""
03_marker_triangulate.py — Reflective Marker Localization + Triangulation + Trunk Frame
=========================================================================================
Pipeline:
  1. Load mock_raw_frames.pkl (output of 01_mock_body_capture_win.py)
  2. For each frame, for each marker (wrist / elbow / shoulder / medial point):
       - Use MediaPipe landmark as an approximate ROI (rough u,v)
       - Within a small window around that ROI, find the brightest reflective
         blob centroid (sub-pixel) -> refined (u,v) for that marker, both cams
  3. Triangulate each marker's (u,v)_L / (u,v)_R into 3D using the Pi rig's
     fisheye stereo_calibration.toml (K, D, R, T)
  4. Build a trunk-local coordinate frame per frame via Gram-Schmidt from the
     3D marker positions (shoulder, elbow, wrist, medial point)
  5. Save a CSV (one row per frame) with raw 3D marker positions + the
     orthonormal trunk frame axes (origin + 3 unit vectors), ready for
     comparison against MoCap (same as Phase 1 alignment pipeline).

NOTE: This assumes you have a working `pi_fisheye_triangulate.py` already
(loads K/D/R/T from stereo_calibration.toml, confirmed baseline 78.749mm).
This script imports that loader directly so you don't duplicate calibration
logic — adjust the import below to match your actual filename/function names.

Usage:
    py -3.11 03_marker_triangulate.py
"""

import os, sys, pickle, csv
import numpy as np
import cv2

# ---- adjust this import to match your existing calibration loader ----
# Expected to expose: K_L, D_L, K_R, D_R, R (3x3), T (3x1), all as np arrays,
# already loaded from H:\dual_ov9281_parallel_calib_cz_30mm\stereo_calibration.toml
try:
    from pi_fisheye_triangulate import K_L, D_L, K_R, D_R, R as R_ST, T as T_ST
    print("✅ Loaded stereo calibration from pi_fisheye_triangulate.py")
except ImportError:
    print("⚠️  Could not import pi_fisheye_triangulate.py — edit the import "
          "block at the top of this script to point at your actual loader, "
          "or hardcode K_L/D_L/K_R/D_R/R_ST/T_ST below.")
    sys.exit(1)

DESKTOP = os.path.join(os.environ.get("OneDrive", os.path.expanduser("~")), "Desktop")
if not os.path.isdir(DESKTOP):
    DESKTOP = os.path.expanduser("~/Desktop")

PKL_IN   = os.path.join(DESKTOP, "mock_raw_frames.pkl")
CSV_OUT  = os.path.join(DESKTOP, "trunk_triangulated.csv")

# MediaPipe landmark indices used as rough ROI seeds for marker search
MARKERS = {
    "shoulder": 12,   # right_shoulder
    "elbow":    14,   # right_elbow
    "wrist":    16,   # right_wrist
    "medial":   24,   # right_hip  <-- PLACEHOLDER: confirm with your guide
                       # what "medial lo line" point actually is and swap
                       # this index, or replace with a manual click-based
                       # ROI if it isn't a standard MediaPipe landmark.
}

BLOB_WINDOW   = 25     # half-size (px) of search window around ROI
BLOB_THRESH   = 200    # brightness threshold for reflective marker (0-255)
MIN_BLOB_AREA = 3      # px^2, ignore noise specks


# ============================================================
# 1. SUB-PIXEL BLOB CENTROID NEAR A ROUGH ROI
# ============================================================
def refine_marker(gray, roi_xy, window=BLOB_WINDOW):
    """
    gray      : full grayscale frame
    roi_xy    : (u, v) rough estimate (e.g. from MediaPipe)
    Returns refined (u, v) as floats, or None if no blob found.
    """
    u0, v0 = int(roi_xy[0]), int(roi_xy[1])
    h, w = gray.shape
    x0, x1 = max(0, u0 - window), min(w, u0 + window)
    y0, y1 = max(0, v0 - window), min(h, v0 + window)
    if x1 <= x0 or y1 <= y0:
        return None

    patch = gray[y0:y1, x0:x1]
    _, mask = cv2.threshold(patch, BLOB_THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # pick the largest bright blob in the window
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_BLOB_AREA:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    cu = M["m10"] / M["m00"] + x0
    cv_ = M["m01"] / M["m00"] + y0
    return (cu, cv_)


def get_landmark_uv(landmarks, idx, w, h):
    """landmarks: result.pose_landmarks[0] from MediaPipe (list of NormalizedLandmark)"""
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


# ============================================================
# 2. FISHEYE UNDISTORT POINTS -> NORMALIZED RAYS
# ============================================================
def undistort_fisheye(uv, K, D):
    """uv: (N,1,2) float32 array of pixel coords -> undistorted normalized points (N,1,2)"""
    pts = np.array(uv, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(pts, K, D)  # returns normalized (x,y), no K applied
    return und


# ============================================================
# 3. TRIANGULATE ONE POINT PAIR
# ============================================================
P_L = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)   # left = world origin
P_R = np.hstack([R_ST, T_ST.reshape(3, 1)]).astype(np.float64)      # right relative to left

def triangulate_point(uv_L, uv_R):
    """uv_L, uv_R: (u,v) pixel tuples in their own camera. Returns 3D point (in left-cam frame, metres)."""
    und_L = undistort_fisheye([uv_L], K_L, D_L)[0, 0]  # normalized (x,y)
    und_R = undistort_fisheye([uv_R], K_R, D_R)[0, 0]

    pt_L = np.array([[und_L[0]], [und_L[1]]], dtype=np.float64)
    pt_R = np.array([[und_R[0]], [und_R[1]]], dtype=np.float64)

    X = cv2.triangulatePoints(P_L, P_R, pt_L, pt_R)
    X /= X[3]
    return X[:3, 0]  # (X, Y, Z) in metres, left-camera frame


# ============================================================
# 4. GRAM-SCHMIDT TRUNK FRAME
# ============================================================
def gram_schmidt_frame(p_shoulder, p_elbow, p_wrist, p_medial):
    """
    Build an orthonormal trunk-local frame.
    Convention (adjust to match your guide's definition):
      origin = shoulder
      x_axis = shoulder -> elbow   (primary segment direction)
      aux    = shoulder -> medial  (used to define the plane)
      y_axis = aux, orthogonalized against x_axis
      z_axis = x_axis  x  y_axis   (completes right-handed frame)
    wrist is kept as a 4th reference point for sanity-checking the triangle,
    not used to define the frame itself (avoids over-constraining with 4 pts).
    """
    origin = p_shoulder

    v1 = p_elbow - origin
    x_axis = v1 / np.linalg.norm(v1)

    v2 = p_medial - origin
    v2_proj = v2 - np.dot(v2, x_axis) * x_axis   # remove component along x_axis
    norm2 = np.linalg.norm(v2_proj)
    if norm2 < 1e-9:
        return None  # degenerate: medial point nearly collinear with x_axis
    y_axis = v2_proj / norm2

    z_axis = np.cross(x_axis, y_axis)

    return origin, x_axis, y_axis, z_axis


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(PKL_IN):
        print(f"❌ Not found: {PKL_IN}")
        sys.exit(1)

    with open(PKL_IN, "rb") as f:
        frames = pickle.load(f)
    print(f"📦 Loaded {len(frames)} frames from {PKL_IN}")

    # Re-run MediaPipe here only to get rough ROIs (or reuse stored detections
    # if you extend 01_ to save raw landmarks instead of just drawn points)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp

    MODEL_PATH = os.path.join(DESKTOP, "pose_landmarker_full.task")
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    lm_L = mp_vision.PoseLandmarker.create_from_options(opts)
    lm_R = mp_vision.PoseLandmarker.create_from_options(opts)

    rows_out = []
    skipped = 0

    for fr in frames:
        frameL, frameR = fr["frameL"], fr["frameR"]
        h, w = frameL.shape[:2]
        ts_ms = int(fr["timestamp"] * 1000)

        gL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        mp_imgL = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
        mp_imgR = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))
        resL = lm_L.detect_for_video(mp_imgL, ts_ms)
        resR = lm_R.detect_for_video(mp_imgR, ts_ms)

        if not resL.pose_landmarks or not resR.pose_landmarks:
            skipped += 1
            continue

        lmsL = resL.pose_landmarks[0]
        lmsR = resR.pose_landmarks[0]

        pts3d = {}
        ok = True
        for name, idx in MARKERS.items():
            roi_L = get_landmark_uv(lmsL, idx, w, h)
            roi_R = get_landmark_uv(lmsR, idx, w, h)

            ref_L = refine_marker(gL, roi_L) or roi_L  # fall back to MP estimate
            ref_R = refine_marker(gR, roi_R) or roi_R

            try:
                pts3d[name] = triangulate_point(ref_L, ref_R)
            except Exception:
                ok = False
                break

        if not ok:
            skipped += 1
            continue

        frame_out = {"frame_num": fr["frame_num"], "timestamp": fr["timestamp"]}
        for name, p in pts3d.items():
            frame_out[f"{name}_X"], frame_out[f"{name}_Y"], frame_out[f"{name}_Z"] = p

        frame_res = gram_schmidt_frame(pts3d["shoulder"], pts3d["elbow"],
                                        pts3d["wrist"], pts3d["medial"])
        if frame_res is not None:
            origin, xax, yax, zax = frame_res
            for label, vec in [("origin", origin), ("x_axis", xax),
                                ("y_axis", yax), ("z_axis", zax)]:
                frame_out[f"{label}_X"], frame_out[f"{label}_Y"], frame_out[f"{label}_Z"] = vec

        rows_out.append(frame_out)

    lm_L.close(); lm_R.close()

    if not rows_out:
        print("❌ No frames produced valid triangulation — check ROI/blob settings.")
        return

    fieldnames = list(rows_out[0].keys())
    with open(CSV_OUT, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerows(rows_out)

    print(f"\n✅ Wrote {len(rows_out)} frames -> {CSV_OUT}")
    print(f"⚠️  Skipped {skipped} frames (missing pose/triangulation failure)")


if __name__ == "__main__":
    main()
