"""
STEP 1 — STEREO CALIBRATION
============================
Reads checkerboard data from:
  H:/calib_cz30_dual_v2/
    cam0_frame.msgpack        (raw frames cam0)
    cam1_frame.msgpack        (raw frames cam1)
    cam0_timestamp.msgpack    (timestamps cam0)
    cam1_timestamp.msgpack    (timestamps cam1)
    chessb_corners_cam0_frame.pkl  (pre-detected corners cam0)
    chessb_corners_cam1_frame.pkl  (pre-detected corners cam1)
    stereo_calibration.toml   (read first — may already have K, D, R, T)

Outputs:
  calibration_output/
    stereo_params.npz    — K0, D0, K1, D1, R, T, E, F, R0, R1, P0, P1, Q, map0x, map0y, map1x, map1y
    calibration_report.txt
"""

import os, pickle, toml, cv2
import numpy as np
import msgpack

# ─── PATHS ────────────────────────────────────────────────────────────────────
CALIB_DIR   = r"H:\calib_cz30_dual_v2"
OUT_DIR     = "calibration_output"
os.makedirs(OUT_DIR, exist_ok=True)

TOML_PATH   = os.path.join(CALIB_DIR, "stereo_calibration.toml")
PKL_CAM0    = os.path.join(CALIB_DIR, "chessb_corners_cam0_frame.pkl")
PKL_CAM1    = os.path.join(CALIB_DIR, "chessb_corners_cam1_frame.pkl")
MSG_CAM0    = os.path.join(CALIB_DIR, "cam0_frame.msgpack")
MSG_CAM1    = os.path.join(CALIB_DIR, "cam1_frame.msgpack")
MSG_TS_CAM0 = os.path.join(CALIB_DIR, "cam0_timestamp.msgpack")
MSG_TS_CAM1 = os.path.join(CALIB_DIR, "cam1_timestamp.msgpack")

# ─── CHECKERBOARD CONFIG ──────────────────────────────────────────────────────
BOARD_COLS   = 13      # inner corners (squares - 1) horizontally
BOARD_ROWS   = 9       # inner corners vertically
SQUARE_SIZE  = 30.0    # mm per square (3 cm)

# ─── STEP 1A: Read TOML — check what's already computed ───────────────────────
print("=" * 60)
print("STEP 1A: Inspecting stereo_calibration.toml")
print("=" * 60)

toml_data = {}
if os.path.exists(TOML_PATH):
    toml_data = toml.load(TOML_PATH)
    print("TOML keys found:", list(toml_data.keys()))
else:
    print("WARNING: stereo_calibration.toml not found. Will compute from scratch.")

def extract_matrix(d, key):
    """Pull a matrix from toml dict if present."""
    if key in d:
        val = d[key]
        return np.array(val, dtype=np.float64)
    return None

# Try to load from TOML
K0_toml = extract_matrix(toml_data.get("cam0", {}), "camera_matrix")
D0_toml = extract_matrix(toml_data.get("cam0", {}), "dist_coeffs")
K1_toml = extract_matrix(toml_data.get("cam1", {}), "camera_matrix")
D1_toml = extract_matrix(toml_data.get("cam1", {}), "dist_coeffs")
R_toml  = extract_matrix(toml_data.get("stereo", {}), "R")
T_toml  = extract_matrix(toml_data.get("stereo", {}), "T")

toml_has_stereo = (R_toml is not None and T_toml is not None and
                   K0_toml is not None and K1_toml is not None)

if toml_has_stereo:
    print("\n✓ TOML has full stereo params (K, D, R, T). Will use as initial guess.")
else:
    print("\n  TOML does not have complete stereo params. Computing from corners.")

# ─── STEP 1B: Load corner detections ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1B: Loading pre-detected checkerboard corners")
print("=" * 60)

with open(PKL_CAM0, "rb") as f:
    corners_cam0_raw = pickle.load(f)  # list/dict of corners per frame
with open(PKL_CAM1, "rb") as f:
    corners_cam1_raw = pickle.load(f)

print(f"  cam0 corner entries: {len(corners_cam0_raw)}")
print(f"  cam1 corner entries: {len(corners_cam1_raw)}")

# Normalise: pkl may be dict {frame_idx: corners} or list
def normalise_corners(raw):
    if isinstance(raw, dict):
        return raw  # {frame_idx: corners_array or None}
    elif isinstance(raw, list):
        return {i: v for i, v in enumerate(raw)}
    else:
        raise ValueError(f"Unknown corner format: {type(raw)}")

corners_cam0 = normalise_corners(corners_cam0_raw)
corners_cam1 = normalise_corners(corners_cam1_raw)

# ─── STEP 1C: Build 3D object points and matched pairs ────────────────────────
print("\n" + "=" * 60)
print("STEP 1C: Matching valid frames (corners found in BOTH cameras)")
print("=" * 60)

# 3D board points (Z=0 plane)
objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []   # 3D world points
imgpoints0 = []  # 2D corners cam0
imgpoints1 = []  # 2D corners cam1

all_frames = sorted(set(corners_cam0.keys()) & set(corners_cam1.keys()))
img_size = None  # will detect from frames

for fid in all_frames:
    c0 = corners_cam0[fid]
    c1 = corners_cam1[fid]
    # Skip if detection failed in either
    if c0 is None or c1 is None:
        continue
    c0 = np.array(c0, dtype=np.float32).reshape(-1, 1, 2)
    c1 = np.array(c1, dtype=np.float32).reshape(-1, 1, 2)
    if c0.shape[0] != BOARD_ROWS * BOARD_COLS:
        continue
    if c1.shape[0] != BOARD_ROWS * BOARD_COLS:
        continue
    objpoints.append(objp)
    imgpoints0.append(c0)
    imgpoints1.append(c1)

print(f"  Valid paired frames: {len(objpoints)}")
if len(objpoints) < 10:
    print("  WARNING: fewer than 10 valid pairs — calibration may be inaccurate.")

# ─── Detect image size from first msgpack frame ───────────────────────────────
def load_msgpack_frames(path):
    """Load msgpack file → list of frames (numpy arrays)."""
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    frames = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (bytes, bytearray)):
                arr = np.frombuffer(item, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)
            elif isinstance(item, dict) and "frame" in item:
                arr = np.frombuffer(item["frame"], dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)
    elif isinstance(data, dict):
        # May be {frame_idx: bytes}
        for k in sorted(data.keys()):
            arr = np.frombuffer(data[k], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    return frames

def get_image_size_from_msgpack(path):
    """Get (width, height) from first decodable frame."""
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for item in unpacker:
            candidate = None
            if isinstance(item, (bytes, bytearray)):
                candidate = item
            elif isinstance(item, dict) and "frame" in item:
                candidate = item["frame"]
            elif isinstance(item, list) and len(item) > 0:
                candidate = item[0] if isinstance(item[0], (bytes, bytearray)) else None
            if candidate is not None:
                arr = np.frombuffer(candidate, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    return (w, h)
    return None

print("\n  Detecting image size from cam0 msgpack...")
img_size = get_image_size_from_msgpack(MSG_CAM0)
if img_size is None:
    print("  WARNING: Could not decode frame. Using fallback 1920x1080.")
    img_size = (1920, 1080)
else:
    print(f"  Image size: {img_size[0]}×{img_size[1]}")

# ─── STEP 1D: Run stereo calibration ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1D: Running stereo calibration")
print("=" * 60)

flags = (cv2.CALIB_RATIONAL_MODEL)
# If TOML had good individual cameras, use as fixed initial guess
if K0_toml is not None and D0_toml is not None and K1_toml is not None and D1_toml is not None:
    print("  Using TOML K/D as initial guess (FIX_INTRINSIC).")
    flags |= cv2.CALIB_FIX_INTRINSIC
    K0_init, D0_init = K0_toml, D0_toml
    K1_init, D1_init = K1_toml, D1_toml
else:
    print("  Estimating K/D from scratch (no TOML priors).")
    K0_init = np.eye(3, dtype=np.float64)
    D0_init = np.zeros((1, 8), dtype=np.float64)
    K1_init = np.eye(3, dtype=np.float64)
    D1_init = np.zeros((1, 8), dtype=np.float64)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

rms, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints0, imgpoints1,
    K0_init, D0_init,
    K1_init, D1_init,
    img_size,
    criteria=criteria,
    flags=flags
)

print(f"\n  ✓ Stereo RMS reprojection error: {rms:.4f} px")
print(f"  Baseline |T|: {np.linalg.norm(T):.4f} m  ({np.linalg.norm(T)*1000:.2f} mm)")

# ─── STEP 1E: Stereo rectification ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1E: Computing rectification maps")
print("=" * 60)

R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
    K0, D0, K1, D1,
    img_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0   # crop to valid pixels only
)

map0x, map0y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, img_size, cv2.CV_32FC1)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)

baseline_mm = np.linalg.norm(T) * 1000   # T is in metres (OptiTrack units)
fx = P0[0, 0]
print(f"  Rectified fx: {fx:.2f} px")
print(f"  Baseline: {baseline_mm:.2f} mm")
print(f"  Focal length × baseline (for depth): {fx * baseline_mm:.2f} px·mm")

# ─── STEP 1F: Save ────────────────────────────────────────────────────────────
out_npz = os.path.join(OUT_DIR, "stereo_params.npz")
np.savez(out_npz,
         K0=K0, D0=D0, K1=K1, D1=D1,
         R=R, T=T, E=E, F=F,
         R0=R0, R1=R1, P0=P0, P1=P1, Q=Q,
         map0x=map0x, map0y=map0y,
         map1x=map1x, map1y=map1y,
         img_size=np.array(img_size),
         rms=np.array(rms))

report_path = os.path.join(OUT_DIR, "calibration_report.txt")
with open(report_path, "w") as f:
    f.write(f"Stereo Calibration Report\n{'='*50}\n")
    f.write(f"RMS reprojection error: {rms:.6f} px\n")
    f.write(f"Valid paired frames used: {len(objpoints)}\n")
    f.write(f"Image size: {img_size[0]}x{img_size[1]}\n\n")
    f.write(f"Baseline: {np.linalg.norm(T)*1000:.4f} mm\n\n")
    f.write(f"K0 (cam0):\n{K0}\n\n")
    f.write(f"D0:\n{D0}\n\n")
    f.write(f"K1 (cam1/ref):\n{K1}\n\n")
    f.write(f"D1:\n{D1}\n\n")
    f.write(f"R (cam0→cam1):\n{R}\n\n")
    f.write(f"T (cam0→cam1):\n{T}\n\n")
    f.write(f"P0:\n{P0}\n\n")
    f.write(f"P1:\n{P1}\n\n")
    f.write(f"Q (disparity-to-depth):\n{Q}\n")

print(f"\n✓ Saved: {out_npz}")
print(f"✓ Saved: {report_path}")
print("\nSTEP 1 COMPLETE.")
