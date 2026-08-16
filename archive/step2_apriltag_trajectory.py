"""
STEP 2 — APRILTAG DETECTION & CAMERA TRAJECTORY
=================================================
Reads AprilTag run data from:
  H:/dual_cam_single_aprl_50mm_t0/
    cam0_frame.msgpack
    cam1_frame.msgpack
    cam0_timestamp.msgpack
    cam1_timestamp.msgpack
    <sync_csv>   — path set below (sync_value, cam0_frame, cam1_frame, etc.)

Also reads:
  calibration_output/stereo_params.npz   (from Step 1)

AprilTag setup (from system doc):
  - Tag family: tag36h11
  - ID 12 = primary (top-center, may be faded/damaged)
  - ID 14 = bottom-left backup
  - ID 20 = bottom-right backup
  - Tag face size: 48 mm (4.8 cm)
  - ID14–ID20 lateral distance: 98 mm
  - ID12 vertical offset above midpoint: 50 mm

Outputs:
  trajectory_output/
    cam0_trajectory.npy    — shape (N, 3) basis vectors in mm
    cam1_trajectory.npy
    cam0_timestamps.npy
    cam1_timestamps.npy
    cam0_frame_ids.npy
    cam1_frame_ids.npy
    detection_log.csv      — per-frame: which tags used, detection method
"""

import os, pickle, csv
import numpy as np
import pandas as pd
import msgpack
import cv2
from datetime import datetime, timedelta

# pupil-apriltags or dt-apriltags (install one on Pi)
try:
    from pupil_apriltags import Detector
    APRILTAG_BACKEND = "pupil"
except ImportError:
    try:
        import apriltag
        APRILTAG_BACKEND = "apriltag"
    except ImportError:
        raise ImportError("Install apriltag: pip install pupil-apriltags  OR  pip install apriltag")

# ─── PATHS ────────────────────────────────────────────────────────────────────
APRL_DIR    = r"H:\dual_cam_single_aprl_50mm_t0"
SYNC_CSV    = os.path.join(APRL_DIR, "sync_and_framerate.csv")  # adjust filename if different
CALIB_NPZ   = os.path.join("calibration_output", "stereo_params.npz")
OUT_DIR     = "trajectory_output"
os.makedirs(OUT_DIR, exist_ok=True)

MSG_CAM0    = os.path.join(APRL_DIR, "cam0_frame.msgpack")
MSG_CAM1    = os.path.join(APRL_DIR, "cam1_frame.msgpack")
MSG_TS_CAM0 = os.path.join(APRL_DIR, "cam0_timestamp.msgpack")
MSG_TS_CAM1 = os.path.join(APRL_DIR, "cam1_timestamp.msgpack")

# ─── APRILTAG CONFIG ──────────────────────────────────────────────────────────
TAG_SIZE_MM  = 48.0     # physical tag face size in mm
TAG_IDS      = [12, 14, 20]
# Geometry for reconstructing ID12 from ID14 + ID20
ID14_ID20_LATERAL_MM = 98.0   # horizontal distance between ID14 and ID20 centers
ID12_VERT_OFFSET_MM  = 50.0   # ID12 center is this far above midpoint of ID14–ID20

# MoCap sync window (from your system doc — adjust if different)
MOCAP_START_STR = "2026-06-23 16:31:23.721"  # capture start time from CSV header

# ─── STEP 2A: Load calibration ────────────────────────────────────────────────
print("=" * 60)
print("STEP 2A: Loading stereo calibration")
print("=" * 60)

cal = np.load(CALIB_NPZ)
K0, D0 = cal["K0"], cal["D0"]
K1, D1 = cal["K1"], cal["D1"]
print(f"  K0 fx={K0[0,0]:.2f}, fy={K0[1,1]:.2f}, cx={K0[0,2]:.2f}, cy={K0[1,2]:.2f}")
print(f"  K1 fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f}, cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f}")

# ─── STEP 2B: Parse sync CSV ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2B: Parsing sync CSV")
print("=" * 60)

sync_df = pd.read_csv(SYNC_CSV)
print(f"  Columns: {list(sync_df.columns)}")
print(f"  Total rows: {len(sync_df)}")

# Expected cols: timestamp, sync_value, cam0_frame, cam0_fps, cam1_frame, cam1_fps
sync_df = sync_df.sort_values("timestamp").reset_index(drop=True)
sync_df["sync_ffill"] = sync_df["sync_value"].ffill()

# MoCap overlap: start = capture start time
mocap_start = pd.Timestamp(MOCAP_START_STR)
sync_df["ts"] = pd.to_datetime(sync_df["timestamp"])

# Rows with sync=1 and within MoCap window
sync1 = sync_df[(sync_df["sync_ffill"] == 1) & (sync_df["ts"] >= mocap_start)].copy()

# Camera-specific rows
cam0_rows = sync1.dropna(subset=["cam0_frame"]).copy()
cam1_rows = sync1.dropna(subset=["cam1_frame"]).copy()
cam0_rows["cam0_frame"] = cam0_rows["cam0_frame"].astype(int)
cam1_rows["cam1_frame"] = cam1_rows["cam1_frame"].astype(int)

print(f"  sync=1 cam0 frames: {len(cam0_rows)}, cam1 frames: {len(cam1_rows)}")
print(f"  Sync=1 window: {sync1['ts'].min()} → {sync1['ts'].max()}")

sync_start_ts = sync1["ts"].min()
sync_end_ts   = sync1["ts"].max()

# ─── STEP 2C: Load frames from msgpack ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2C: Loading frames from msgpack (streaming)")
print("=" * 60)

def stream_msgpack_frames(path, frame_ids_needed):
    """
    Stream msgpack and yield (frame_idx, img) only for frame_ids in frame_ids_needed.
    Memory-efficient: doesn't load all frames at once.
    """
    needed = set(frame_ids_needed)
    results = {}
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=10 * 1024 * 1024 * 1024)
        idx = 0
        for item in unpacker:
            if idx in needed:
                raw_bytes = None
                if isinstance(item, (bytes, bytearray)):
                    raw_bytes = item
                elif isinstance(item, dict):
                    raw_bytes = item.get("frame") or item.get("data") or item.get("image")
                elif isinstance(item, list) and len(item) > 0:
                    raw_bytes = item[0] if isinstance(item[0], (bytes, bytearray)) else None
                if raw_bytes is not None:
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        results[idx] = img
            idx += 1
            if len(results) == len(needed):
                break  # got all we need
    return results

def load_msgpack_timestamps(path):
    """Load timestamps msgpack → dict {frame_idx: timestamp_value}."""
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    if isinstance(data, list):
        return {i: v for i, v in enumerate(data)}
    elif isinstance(data, dict):
        return data
    return {}

print("  Loading cam0 timestamps...")
ts_cam0 = load_msgpack_timestamps(MSG_TS_CAM0)
print("  Loading cam1 timestamps...")
ts_cam1 = load_msgpack_timestamps(MSG_TS_CAM1)

cam0_frame_ids = cam0_rows["cam0_frame"].tolist()
cam1_frame_ids = cam1_rows["cam1_frame"].tolist()

print(f"  Loading {len(cam0_frame_ids)} cam0 frames from msgpack...")
cam0_frames = stream_msgpack_frames(MSG_CAM0, cam0_frame_ids)
print(f"  Loading {len(cam1_frame_ids)} cam1 frames from msgpack...")
cam1_frames = stream_msgpack_frames(MSG_CAM1, cam1_frame_ids)
print(f"  Loaded cam0: {len(cam0_frames)}, cam1: {len(cam1_frames)}")

# ─── STEP 2D: Setup AprilTag detector ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2D: Setting up AprilTag detector")
print("=" * 60)

if APRILTAG_BACKEND == "pupil":
    detector = Detector(families="tag36h11",
                        nthreads=4,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25)
    print("  Using pupil_apriltags backend")
else:
    options = apriltag.DetectorOptions(families="tag36h11", nthreads=4)
    detector = apriltag.Detector(options)
    print("  Using apriltag backend")

# ─── STEP 2E: AprilTag detection + PnP per frame ──────────────────────────────
def detect_tags(img):
    """Return dict {tag_id: corners_4x2} from grayscale detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if APRILTAG_BACKEND == "pupil":
        detections = detector.detect(gray)
        return {d.tag_id: d.corners for d in detections}
    else:
        detections = detector.detect(gray)
        return {d.tag_id: d.corners for d in detections}

def corners_to_pnp_input(corners_2d, tag_size_mm):
    """
    Build 3D object points for a single tag (flat in XY plane, Z=0).
    Tag corners order (pupil/dt-apriltags): bottom-left, bottom-right, top-right, top-left
    """
    h = tag_size_mm / 2.0
    obj_pts = np.array([
        [-h,  h, 0],   # bottom-left
        [ h,  h, 0],   # bottom-right
        [ h, -h, 0],   # top-right
        [-h, -h, 0],   # top-left
    ], dtype=np.float64)
    img_pts = np.array(corners_2d, dtype=np.float64).reshape(4, 1, 2)
    return obj_pts.reshape(4, 1, 3), img_pts

def get_tag_center(corners_2d):
    """Mean of 4 corners → 2D center."""
    return np.mean(corners_2d, axis=0)

def solve_pnp(obj_pts, img_pts, K, D):
    """Run solvePnP → (rvec, tvec) or None."""
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        return None, None
    return rvec, tvec

def estimate_id12_from_14_20(corners_14, corners_20):
    """
    Geometrically reconstruct ID12 center (2D) from ID14 and ID20 corners.
    Returns a fake 'center point' in image coords — used only for centroid tracking,
    NOT for PnP (geometry is approximate).
    """
    c14 = get_tag_center(corners_14)
    c20 = get_tag_center(corners_20)
    midpoint = (c14 + c20) / 2.0
    # ID12 is above midpoint — estimate direction from orientation
    # Use perpendicular to ID14→ID20 vector, pointing upward in image
    vec = c20 - c14
    perp = np.array([-vec[1], vec[0]])  # rotate 90°
    perp_norm = perp / (np.linalg.norm(perp) + 1e-9)
    # Scale: pixel scale ≈ (c14-c20 distance) / 98mm * 50mm
    pixel_dist_14_20 = np.linalg.norm(c20 - c14)
    pixel_offset = pixel_dist_14_20 / ID14_ID20_LATERAL_MM * ID12_VERT_OFFSET_MM
    # Choose direction: perp pointing upward (smaller y in image = up)
    if perp_norm[1] > 0:
        perp_norm = -perp_norm  # flip so it points up in image
    estimated_center = midpoint + pixel_offset * perp_norm
    return estimated_center

def process_frame(img, K, D, method_log):
    """
    Detect AprilTags in frame, return (rvec, tvec, method_str) or (None, None, 'skip').
    
    Priority:
      1. ID 12 direct detection → PnP on ID12 corners
      2. ID 14 + ID 20 → PnP on combined object (more stable than geometric estimate)
      3. ID 14 alone or ID 20 alone → PnP on single tag
      4. None found → skip
    """
    detections = detect_tags(img)

    if 12 in detections:
        obj_pts, img_pts = corners_to_pnp_input(detections[12], TAG_SIZE_MM)
        rvec, tvec = solve_pnp(obj_pts, img_pts, K, D)
        if tvec is not None:
            method_log.append("id12_direct")
            return rvec, tvec, "id12_direct"

    # Try ID14 + ID20 together — build combined 3D object
    if 14 in detections and 20 in detections:
        # Combined object: ID14 at (-lateral/2, 0, 0), ID20 at (+lateral/2, 0, 0)
        # and ID12 estimate at (0, +vert_offset, 0) — but we don't use ID12 here
        h = TAG_SIZE_MM / 2.0
        lat = ID14_ID20_LATERAL_MM / 2.0

        # ID14 corners: centered at (-lat, 0, 0)
        obj14 = np.array([
            [-lat - h,  h, 0], [-lat + h,  h, 0],
            [-lat + h, -h, 0], [-lat - h, -h, 0]
        ], dtype=np.float64)
        # ID20 corners: centered at (+lat, 0, 0)
        obj20 = np.array([
            [ lat - h,  h, 0], [ lat + h,  h, 0],
            [ lat + h, -h, 0], [ lat - h, -h, 0]
        ], dtype=np.float64)

        obj_combined = np.vstack([obj14, obj20]).reshape(-1, 1, 3)
        img14 = np.array(detections[14], dtype=np.float64).reshape(4, 1, 2)
        img20 = np.array(detections[20], dtype=np.float64).reshape(4, 1, 2)
        img_combined = np.vstack([img14, img20])

        success, rvec, tvec = cv2.solvePnP(obj_combined, img_combined, K, D,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            method_log.append("id14+id20")
            return rvec, tvec, "id14+id20"

    # Single tag fallback
    for tid in [14, 20]:
        if tid in detections:
            obj_pts, img_pts = corners_to_pnp_input(detections[tid], TAG_SIZE_MM)
            rvec, tvec = solve_pnp(obj_pts, img_pts, K, D)
            if tvec is not None:
                method_log.append(f"id{tid}_only")
                return rvec, tvec, f"id{tid}_only"

    method_log.append("skip")
    return None, None, "skip"

# ─── STEP 2F: Process all cam0 frames ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2F: Processing cam0 frames")
print("=" * 60)

cam0_results = []   # list of (frame_id, timestamp, tvec) — raw, before basis transform
cam0_rvecs   = []
cam0_methods = []

for i, fid in enumerate(cam0_frame_ids):
    if fid not in cam0_frames:
        cam0_methods.append("no_frame")
        cam0_results.append((fid, None, None))
        continue
    img = cam0_frames[fid]
    rvec, tvec, method = process_frame(img, K0, D0, cam0_methods)
    cam0_results.append((fid, ts_cam0.get(fid), tvec))
    cam0_rvecs.append(rvec)
    if (i + 1) % 50 == 0:
        print(f"  cam0: {i+1}/{len(cam0_frame_ids)} frames processed")

print(f"\n  cam0 detection summary:")
for m in set(cam0_methods):
    print(f"    {m}: {cam0_methods.count(m)}")

# ─── STEP 2G: Process all cam1 frames ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2G: Processing cam1 frames")
print("=" * 60)

cam1_results = []
cam1_rvecs   = []
cam1_methods = []

for i, fid in enumerate(cam1_frame_ids):
    if fid not in cam1_frames:
        cam1_methods.append("no_frame")
        cam1_results.append((fid, None, None))
        continue
    img = cam1_frames[fid]
    rvec, tvec, method = process_frame(img, K1, D1, cam1_methods)
    cam1_results.append((fid, ts_cam1.get(fid), tvec))
    cam1_rvecs.append(rvec)
    if (i + 1) % 50 == 0:
        print(f"  cam1: {i+1}/{len(cam1_frame_ids)} frames processed")

print(f"\n  cam1 detection summary:")
for m in set(cam1_methods):
    print(f"    {m}: {cam1_methods.count(m)}")

# ─── STEP 2H: Compute basis vectors ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2H: Computing basis vectors (T_ref subtraction)")
print("=" * 60)

def compute_basis_vectors(results, rvecs):
    """
    basis = R^T @ (T_ref - T_frame)
    T_ref = tvec of first valid frame.
    Returns arrays: basis_vecs (N,3), frame_ids, timestamps, valid_mask
    """
    # Find T_ref: first valid frame
    T_ref = None
    R_ref = None
    for (fid, ts, tvec), rvec in zip(results, rvecs):
        if tvec is not None:
            T_ref = tvec.flatten()
            R_ref, _ = cv2.Rodrigues(rvec)
            print(f"    T_ref from frame {fid}: {T_ref}")
            break

    if T_ref is None:
        print("    ERROR: No valid detections found!")
        return None, None, None, None

    basis_list, fid_list, ts_list, valid_list = [], [], [], []
    for (fid, ts, tvec), rvec in zip(results, rvecs):
        fid_list.append(fid)
        ts_list.append(ts)
        if tvec is not None and rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            T = tvec.flatten()
            # Use R_ref (reference frame rotation) — consistent with system doc
            basis = R_ref.T @ (T_ref - T)  # in mm (tag size was mm → tvec is mm)
            basis_list.append(basis)
            valid_list.append(True)
        else:
            basis_list.append(np.array([np.nan, np.nan, np.nan]))
            valid_list.append(False)

    return (np.array(basis_list),
            np.array(fid_list),
            np.array(ts_list, dtype=object),
            np.array(valid_list))

print("  cam0:")
cam0_basis, cam0_fids, cam0_ts, cam0_valid = compute_basis_vectors(cam0_results, cam0_rvecs)
print("  cam1:")
cam1_basis, cam1_fids, cam1_ts, cam1_valid = compute_basis_vectors(cam1_results, cam1_rvecs)

# ─── STEP 2I: Save ────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "cam0_trajectory.npy"), cam0_basis)
np.save(os.path.join(OUT_DIR, "cam1_trajectory.npy"), cam1_basis)
np.save(os.path.join(OUT_DIR, "cam0_timestamps.npy"),  cam0_ts)
np.save(os.path.join(OUT_DIR, "cam1_timestamps.npy"),  cam1_ts)
np.save(os.path.join(OUT_DIR, "cam0_frame_ids.npy"),   cam0_fids)
np.save(os.path.join(OUT_DIR, "cam1_frame_ids.npy"),   cam1_fids)

# Detection log
log_rows = []
for (fid, ts, tvec), method in zip(cam0_results, cam0_methods):
    log_rows.append({"cam": 0, "frame": fid, "ts": ts, "method": method,
                     "valid": tvec is not None})
for (fid, ts, tvec), method in zip(cam1_results, cam1_methods):
    log_rows.append({"cam": 1, "frame": fid, "ts": ts, "method": method,
                     "valid": tvec is not None})
pd.DataFrame(log_rows).to_csv(os.path.join(OUT_DIR, "detection_log.csv"), index=False)

print(f"\n✓ cam0 valid: {cam0_valid.sum() if cam0_valid is not None else 0}/{len(cam0_frame_ids)}")
print(f"✓ cam1 valid: {cam1_valid.sum() if cam1_valid is not None else 0}/{len(cam1_frame_ids)}")
print("\nSTEP 2 COMPLETE.")
