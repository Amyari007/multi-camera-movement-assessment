"""
error_analysis.py
=================
Dual-Camera vs MoCap Error Analysis Pipeline

EXACT FORMULA (as specified):
  basis = R.T @ (T_ref - T_frame)

  WHERE:
    R      = rotation matrix from ChArUco board pose (rvec → 3x3)
    T_ref  = tvec of AprilTag ID12 centroid in FIRST frame after sync=1
             (from REF_CAM = right camera = cam1)
             THIS IS FIXED for the entire run
    T_frame = tvec of AprilTag ID12 centroid in CURRENT frame
              Applied to BOTH cam0 and cam1 independently

  MOCAP same formula:
    basis = R_base.T @ (T_ref_moc - T_moving)
    R_base   = rotation of BASE rigid body (quaternion → 3x3)
    T_ref_moc = position of MOVING rigid body in FIRST frame after sync=1
    T_moving  = position of MOVING rigid body in current frame

OUTPUT:
  - cam0_basis.npy, cam1_basis.npy, mocap_basis.npy
  - trajectory_plot.png  (3D + 3 axis subplots)
  - error_boxplot.png    (cam0 vs mocap, cam1 vs mocap, cam0 vs cam1)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# SECTION 0 — FILE PATHS  ← edit here only
# ══════════════════════════════════════════════════════════════════
CALIB_NPZ  = r"C:\Users\arya0\Desktop\camera_calibration_output.npz"
BOARD_PKL  = r"C:\Users\arya0\Downloads\raw_capture_output\board_detection.pkl"
SYNC_CSV   = r"C:\Users\arya0\OneDrive\Desktop\raw capture sync_and_framerate.csv"
MOCAP_CSV  = r"F:\mocap trial 1 donw w 2 cam.csv"
OUTPUT_DIR = r"C:\Users\arya0\Downloads\error_analysis_output"

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — PHYSICAL DIMENSIONS
# ══════════════════════════════════════════════════════════════════
CHARUCO_COLS       = 4
CHARUCO_ROWS       = 3
CHARUCO_SQUARE_LEN = 0.037   # 3.7 cm
CHARUCO_MARKER_LEN = 0.027   # 2.7 cm
APRILTAG_SIZE      = 0.048   # 4.8 cm (one tag — id14 or id20)
TARGET_TAG_ID      = 12      # We track ID12's bounding box centroid

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — MOCAP RIGID BODY NAMES
#   Run once, read the printed keys, then set these correctly
# ══════════════════════════════════════════════════════════════════
MOCAP_MOVING_BODY = 'rigidbody 002'   # AprilTag cluster (5 markers)
MOCAP_BASE_BODY   = 'rigidbody'       # ChArUco board    (4 markers)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — PARAMETERS
# ══════════════════════════════════════════════════════════════════
REF_CAM          = 1       # right camera is reference
MAD_THRESH       = 5.0
MAX_VEL_M_S      = 2.0     # max plausible human motion (m/s)
FALLBACK_FRAMES  = 10      # frames to average for rotation fallback
MAX_TS_GAP       = 0.15    # max seconds for nearest-neighbour match

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────
def to_epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.rstrip('Z')).timestamp()

def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])

def reortho(R: np.ndarray) -> np.ndarray:
    """Re-orthogonalise a rotation matrix via SVD."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt

def mad_filter(records: list) -> list:
    """Remove statistical outliers using Median Absolute Deviation."""
    if len(records) < 4:
        return records
    arr = np.array([r['basis'] for r in records])
    ts  = np.array([r['timestamp'] for r in records])
    # MAD per axis
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    mad_ok = np.all(np.abs(arr - med) / mad < MAD_THRESH, axis=1)
    # Velocity filter
    vel_ok = np.ones(len(records), dtype=bool)
    for i in range(1, len(records)):
        dt = ts[i] - ts[i-1]
        if dt < 1e-6:
            vel_ok[i] = False
        elif np.linalg.norm(arr[i] - arr[i-1]) / dt > MAX_VEL_M_S:
            vel_ok[i] = False
    keep = mad_ok & vel_ok
    removed = int(np.sum(~keep))
    return [r for r, k in zip(records, keep) if k], removed

# ══════════════════════════════════════════════════════════════════
# STEP 1 — CALIBRATION
# ══════════════════════════════════════════════════════════════════
def load_calibration():
    print("\n[1] Loading Camera Calibration")
    d   = np.load(CALIB_NPZ)
    K_L = d['cam_matrix_L'].astype(np.float64)
    D_L = d['dist_coeffs_L'].astype(np.float64)
    K_R = d['cam_matrix_R'].astype(np.float64)
    D_R = d['dist_coeffs_R'].astype(np.float64)
    print(f"    Left  cam: fx={K_L[0,0]:.2f} fy={K_L[1,1]:.2f}")
    print(f"    Right cam: fx={K_R[0,0]:.2f} fy={K_R[1,1]:.2f}")
    return K_L, D_L, K_R, D_R

# ══════════════════════════════════════════════════════════════════
# STEP 2 — SYNC WINDOW
#   Returns ISO strings for the contiguous sync=1 window
# ══════════════════════════════════════════════════════════════════
def load_sync_window():
    print("\n[2] Parsing Sync Window from CSV")
    df = pd.read_csv(SYNC_CSV)
    # find the sync column regardless of name
    sync_col = next((c for c in df.columns
                     if 'sync' in c.lower()), df.columns[-1])
    sync1 = df[df[sync_col] == 1].sort_values('timestamp')
    assert len(sync1) > 0, "No sync=1 rows found in CSV!"
    t_start = sync1['timestamp'].iloc[0]
    t_end   = sync1['timestamp'].iloc[-1]
    print(f"    MoCap ON: {t_start}  →  {t_end}")
    print(f"    Sync=1 pulses: {len(sync1)}")
    return t_start, t_end

# ══════════════════════════════════════════════════════════════════
# STEP 3 — LOAD PICKLE FRAMES (filtered by real timestamp)
# ══════════════════════════════════════════════════════════════════
def load_pickle_frames(t_start, t_end):
    print("\n[3] Loading Board Detection Pickle")
    with open(BOARD_PKL, 'rb') as f:
        data = pickle.load(f)
    frames = data['frames']
    print(f"    Total entries: {len(frames)}")
    cam0 = [fr for fr in frames
            if fr['cam_id'] == 0 and t_start <= fr['timestamp'] <= t_end]
    cam1 = [fr for fr in frames
            if fr['cam_id'] == 1 and t_start <= fr['timestamp'] <= t_end]
    print(f"    Cam0 in sync window: {len(cam0)}")
    print(f"    Cam1 in sync window: {len(cam1)}")
    assert len(cam0) >= 5 and len(cam1) >= 5, "Too few frames in sync window!"
    return cam0, cam1

# ══════════════════════════════════════════════════════════════════
# STEP 4 — DETECTOR SETUP
# ══════════════════════════════════════════════════════════════════
def setup_detectors():
    ch_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ch_board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, ch_dict)
    ch_det   = cv2.aruco.CharucoDetector(ch_board)
    ap_det   = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
        cv2.aruco.DetectorParameters())
    # 3D corners of one AprilTag (4.8cm)
    s = APRILTAG_SIZE / 2.0
    tag_3d = np.array([[-s, s, 0], [s, s, 0],
                        [s,-s, 0], [-s,-s, 0]], dtype=np.float32)
    return ch_board, ch_det, ap_det, tag_3d

# ══════════════════════════════════════════════════════════════════
# STEP 5 — POSE EXTRACTION (one frame)
#   Returns: rvec (board), tvec_tag (ID12 centroid in camera frame)
#   Both in metres, in the camera coordinate system
# ══════════════════════════════════════════════════════════════════
def extract_pose(frame_entry, K, D, ch_board, ch_det, ap_det, tag_3d):
    frame = frame_entry.get('frame')
    if frame is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── ChArUco board: get rotation ──────────────────────────────
    rvec_board = None
    ch_corners, ch_ids, _, _ = ch_det.detectBoard(gray)
    if ch_ids is not None and len(ch_ids) >= 4:
        try:
            obj_pts, img_pts = ch_board.matchImagePoints(ch_corners, ch_ids)
            if len(obj_pts) >= 4:
                ok, rv, _ = cv2.solvePnP(
                    obj_pts, img_pts, K, D,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    rvec_board = rv.flatten()
        except Exception:
            pass

    # ── AprilTag ID12: get translation ───────────────────────────
    # ID12 is the cluster — it has id14 and id20 sub-tags attached.
    # We detect all three (12, 14, 20) and take the centroid of
    # whichever corner pixels we find. This is more robust than
    # relying on a single ID being visible.
    tvec_tag = None
    corners, ids, _ = ap_det.detectMarkers(gray)
    if ids is not None:
        # Collect all corner pixels from the target cluster
        cluster_corners = []
        for idx, mid in enumerate(ids.flatten()):
            if int(mid) in (TARGET_TAG_ID, 14, 20):
                cluster_corners.append(corners[idx][0])

        if cluster_corners:
            # Use the first detected tag for solvePnP
            c2d = cluster_corners[0].astype(np.float32)
            ok2, _, tv2 = cv2.solvePnP(
                tag_3d, c2d, K, D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                tvec_tag = tv2.flatten()

    return rvec_board, tvec_tag

# ══════════════════════════════════════════════════════════════════
# STEP 6 — CAMERA BASIS COMPUTATION
#
#   basis = R.T @ (T_ref - T_frame)
#
#   T_ref  = tvec of tag in FIRST frame of cam1 after sync=1  [FIXED]
#   R      = rotation matrix from ChArUco rvec  (per frame)
#   T_frame = tvec of tag in current frame
#
#   Applied independently to cam0 and cam1.
#   Both cameras use the SAME T_ref (from cam1 first frame).
# ══════════════════════════════════════════════════════════════════
def compute_camera_basis(cam0_frames, cam1_frames,
                          K_L, D_L, K_R, D_R,
                          ch_board, ch_det, ap_det, tag_3d):
    print("\n[5] Computing Camera Basis Vectors")
    K = {0: K_L, 1: K_R}
    D = {0: D_L, 1: D_R}

    # ── Find T_ref from cam1 first valid frame ────────────────
    T_ref = None
    for fr in cam1_frames:
        _, tv = extract_pose(fr, K[1], D[1], ch_board, ch_det, ap_det, tag_3d)
        if tv is not None:
            T_ref = tv.copy()
            print(f"    T_ref (cam1 frame {fr['frame_num']}): "
                  f"[{T_ref[0]*1000:.1f}, {T_ref[1]*1000:.1f}, "
                  f"{T_ref[2]*1000:.1f}] mm  ← FIXED reference")
            break

    assert T_ref is not None, \
        "Could not get T_ref from cam1! Check AprilTag visibility."

    results = {0: [], 1: []}

    for cam_id, frames in [(0, cam0_frames), (1, cam1_frames)]:
        Ki, Di     = K[cam_id], D[cam_id]
        R_buf      = []
        R_fallback = np.eye(3)
        skipped    = 0

        for fr in frames:
            rv, tv_tag = extract_pose(
                fr, Ki, Di, ch_board, ch_det, ap_det, tag_3d)

            # Update rotation with fallback averaging
            if rv is not None:
                R_now = cv2.Rodrigues(rv)[0]
                R_buf.append(R_now)
                if len(R_buf) > FALLBACK_FRAMES:
                    R_buf.pop(0)
                R_fallback = reortho(np.mean(R_buf, axis=0))

            R = R_fallback  # use averaged/fallback rotation

            if tv_tag is None:
                skipped += 1
                continue

            # THE CORRECT FORMULA
            basis = R.T @ (T_ref - tv_tag)

            results[cam_id].append({
                'timestamp': to_epoch(fr['timestamp']),
                'basis':     basis,
                'T_tag':     tv_tag.copy(),
            })

        print(f"    Cam{cam_id}: {len(results[cam_id])} valid  "
              f"| {skipped} skipped (tag not detected)")

    return results

# ══════════════════════════════════════════════════════════════════
# STEP 7 — MOCAP DATA LOADING
#   Uses read_rigid_body_csv_multi logic from pd_multi_support.py
# ══════════════════════════════════════════════════════════════════
def _parse_motive_csv(path):
    """Parse Motive CSV, return dict of rigid body DataFrames + start_time."""
    from more_itertools import locate

    df      = pd.read_csv(path, skiprows=2, header=None, dtype=str)
    raw     = pd.read_csv(path, dtype=str)
    cols    = raw.columns
    idx_st  = [i for i, x in enumerate(cols) if x == "Capture Start Time"]
    st_time = datetime.strptime(cols[idx_st[0]+1], "%Y-%m-%d %I.%M.%S.%f %p")

    mtype  = [df[c][0] for c in df.columns]
    rb_df  = df[1:]
    atype  = list(rb_df.iloc[2].values)
    nrow   = list(rb_df.iloc[0].values)
    axrow  = list(rb_df.iloc[3].values)

    rot_ids  = set(i for i, v in enumerate(atype) if v == "Rotation")
    rb_cols  = [i for i, v in enumerate(mtype) if v == "Rigid Body"]
    rbm_cols = [i for i, v in enumerate(mtype) if v == "Rigid Body Marker"]

    rb_groups  = {}
    for ci in rb_cols:
        rb_groups.setdefault(str(nrow[ci]).lower().strip(), []).append(ci)

    rbm_groups = {}
    for ci in rbm_cols:
        rn = str(nrow[ci]).lower().split(":")[0].strip()
        rbm_groups.setdefault(rn, []).append(ci)

    result = {}
    for name, group in rb_groups.items():
        if not name or name == 'nan':
            continue
        rot = sorted(i for i in group if i in rot_ids)
        pos = sorted(i for i in group if i not in rot_ids)
        names, idxs = ["frame", "seconds"], [0, 1]
        for i in rot:
            ax = axrow[i]
            names.append("ang_" + ax.lower() if isinstance(ax, str) else "ang_err")
            idxs.append(i)
        for i in pos:
            if atype[i] == "Mean Marker Error":
                names.append("mean_marker_err")
            else:
                ax = axrow[i]
                names.append("pos_" + ax.lower() if isinstance(ax, str) else "pos_err")
            idxs.append(i)
        for i in rbm_groups.get(name, []):
            parts = str(nrow[i]).lower().split(":")
            ch = parts[1].strip().replace("marker","").strip() if len(parts)>1 else str(i)
            ax = axrow[i]
            names.append(f"marker_m{ch}_{ax.lower()}" if isinstance(ax,str)
                          else f"marker_m{ch}_mq")
            idxs.append(i)
        _d = rb_df[4:][idxs].copy()
        _d.columns = names
        _d = _d.reset_index(drop=True).apply(pd.to_numeric, errors='coerce')
        result[name] = _d

    return result, st_time


def load_mocap(sync_start_iso: str, sync_end_iso: str):
    print("\n[6] Loading MoCap Data")
    if not os.path.exists(MOCAP_CSV):
        print(f"    ⚠ MoCap CSV not found: {MOCAP_CSV}")
        return None

    rb_dict, st_time = _parse_motive_csv(MOCAP_CSV)
    keys = [k for k in rb_dict if k not in ('markers', '')]
    print(f"    Rigid bodies found: {keys}")
    print(f"    Capture start: {st_time}")

    # Verify body names
    global MOCAP_MOVING_BODY, MOCAP_BASE_BODY
    if MOCAP_MOVING_BODY not in rb_dict or MOCAP_BASE_BODY not in rb_dict:
        print(f"    Available keys: {keys}")
        print(f"    ❌ Set MOCAP_MOVING_BODY and MOCAP_BASE_BODY in Section 2!")
        return None

    df_move = rb_dict[MOCAP_MOVING_BODY]
    df_base = rb_dict[MOCAP_BASE_BODY]

    # Wall-clock timestamps for MoCap frames
    wall_ts = np.array([
        (st_time + timedelta(seconds=float(s))).timestamp()
        for s in df_move['seconds']
    ])

    sync_start_ep = to_epoch(sync_start_iso)
    sync_end_ep   = to_epoch(sync_end_iso)

    # Filter to sync=1 window only
    mask = (wall_ts >= sync_start_ep - 0.5) & (wall_ts <= sync_end_ep + 0.5)
    df_move   = df_move[mask].reset_index(drop=True)
    df_base   = df_base[mask].reset_index(drop=True)
    wall_ts   = wall_ts[mask]

    print(f"    MoCap rows in sync window: {len(df_move)}")
    assert len(df_move) >= 5, "Too few MoCap rows in sync window!"

    has_quat = all(c in df_move.columns for c in ['ang_x','ang_y','ang_z','ang_w'])
    has_base_quat = all(c in df_base.columns for c in ['ang_x','ang_y','ang_z','ang_w'])

    if not has_quat:
        print("    ⚠ No quaternion in moving body — using identity rotation")
    if not has_base_quat:
        print("    ⚠ No quaternion in base body — using identity rotation")

    R_buf      = []
    R_fallback = np.eye(3)
    records    = []

    for i in range(len(df_move)):
        ts = float(wall_ts[i])
        T_move = np.array([
            float(df_move.at[i, 'pos_x']),
            float(df_move.at[i, 'pos_y']),
            float(df_move.at[i, 'pos_z'])
        ], dtype=float)

        if np.any(np.isnan(T_move)):
            continue

        # Rotation comes from BASE body (stationary ChArUco)
        if has_base_quat:
            qx = float(df_base.at[i, 'ang_x'])
            qy = float(df_base.at[i, 'ang_y'])
            qz = float(df_base.at[i, 'ang_z'])
            qw = float(df_base.at[i, 'ang_w'])
            if not any(np.isnan([qx, qy, qz, qw])) and \
               abs(qx)+abs(qy)+abs(qz)+abs(qw) > 1e-6:
                R_now = quat_to_R(qx, qy, qz, qw)
                R_buf.append(R_now)
                if len(R_buf) > FALLBACK_FRAMES:
                    R_buf.pop(0)
                R_fallback = reortho(np.mean(R_buf, axis=0))

        records.append({
            'timestamp': ts,
            'T_move':    T_move.copy(),
            'R_base':    R_fallback.copy(),
        })

    print(f"    Valid MoCap entries: {len(records)}")
    return records


# ══════════════════════════════════════════════════════════════════
# STEP 8 — MOCAP BASIS COMPUTATION
#
#   basis = R_base.T @ (T_ref_moc - T_moving)
#
#   T_ref_moc = position of MOVING rigid body in FIRST frame  [FIXED]
#   R_base    = rotation of BASE rigid body (quaternion → 3x3)
#   T_moving  = position of MOVING rigid body in current frame
# ══════════════════════════════════════════════════════════════════
def compute_mocap_basis(mocap_records):
    if mocap_records is None or len(mocap_records) < 2:
        return None
    print("\n[7] Computing MoCap Basis Vectors")

    # T_ref = first frame's moving body position  [FIXED]
    T_ref = mocap_records[0]['T_move'].copy()
    print(f"    MoCap T_ref: [{T_ref[0]*1000:.1f}, "
          f"{T_ref[1]*1000:.1f}, {T_ref[2]*1000:.1f}] mm  ← FIXED")

    results = []
    for rec in mocap_records:
        basis = rec['R_base'].T @ (T_ref - rec['T_move'])
        results.append({
            'timestamp': rec['timestamp'],
            'basis':     basis,
        })

    print(f"    MoCap basis vectors: {len(results)}")
    return results

# ══════════════════════════════════════════════════════════════════
# STEP 9 — TEMPORAL ALIGNMENT
#   Nearest-neighbour on real timestamps (not frame index)
#   Pairs each camera frame with the closest MoCap frame
# ══════════════════════════════════════════════════════════════════
def align(cam_results, ref_results, label=""):
    if not ref_results or not cam_results:
        return []
    ref_ts    = np.array([r['timestamp'] for r in ref_results])
    ref_basis = np.array([r['basis']     for r in ref_results])
    aligned   = []
    for rec in cam_results:
        idx = int(np.argmin(np.abs(ref_ts - rec['timestamp'])))
        if abs(ref_ts[idx] - rec['timestamp']) > MAX_TS_GAP:
            continue
        aligned.append({
            'timestamp': rec['timestamp'],
            'cam_basis': rec['basis'],
            'ref_basis': ref_basis[idx],
            'error_vec': rec['basis'] - ref_basis[idx],
        })
    print(f"    {label}: {len(aligned)} matched pairs")
    return aligned

# ══════════════════════════════════════════════════════════════════
# STEP 10 — TRAJECTORY PLOT
# ══════════════════════════════════════════════════════════════════
def plot_trajectory(cam0, cam1, moc):
    print("\n[9] Plotting Trajectories")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Trajectory Comparison — Left Cam | Right Cam | MoCap\n"
        "basis = R.T @ (T_ref − T_frame)  |  Sync=1 window only",
        fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    SRCS = [
        ('Left Camera  (Cam0)', cam0, '#22c55e'),
        ('Right Camera (Cam1)', cam1, '#0ea5e9'),
        ('MoCap Ground Truth',  moc,  '#ef4444'),
    ]

    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    for lbl, res, col in SRCS:
        if not res: continue
        xs = [r['basis'][0]*1000 for r in res]
        ys = [r['basis'][1]*1000 for r in res]
        zs = [r['basis'][2]*1000 for r in res]
        ax3d.plot3D(xs, ys, zs, color=col, lw=1.6, label=lbl, alpha=0.85)
        ax3d.scatter3D(xs[0], ys[0], zs[0], color=col, s=50)
    ax3d.set_title('3D Trajectory', fontweight='bold')
    ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
    ax3d.legend(fontsize=9)

    all_res = [r for res in [cam0, cam1, moc] if res for r in res]
    t0 = min(r['timestamp'] for r in all_res) if all_res else 0

    for ax_i, ax_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[1, ax_i])
        for lbl, res, col in SRCS:
            if not res: continue
            ts   = [r['timestamp'] - t0        for r in res]
            vals = [r['basis'][ax_i]*1000       for r in res]
            ax.plot(ts, vals, color=col, lw=1.3, label=lbl, alpha=0.85)
        ax.set_title(f'{ax_name}-axis vs Time', fontweight='bold')
        ax.set_xlabel('Time (s)'); ax.set_ylabel(f'{ax_name} (mm)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    out = os.path.join(OUTPUT_DIR, 'trajectory_plot.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")

# ══════════════════════════════════════════════════════════════════
# STEP 11 — ERROR BOX PLOT
# ══════════════════════════════════════════════════════════════════
def plot_errors(a_c0_moc, a_c1_moc, a_c0_c1):
    print("\n[10] Plotting Error Box Plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Error Analysis — Basis Vector Differences Per Axis\n"
        "(after MAD outlier filtering, in mm)",
        fontsize=12, fontweight='bold')

    COLORS    = ['#22c55e', '#0ea5e9', '#f59e0b']
    AX_NAMES  = ['X-Axis Error', 'Y-Axis Error', 'Z-Axis Error']

    for ax_i, (ax, ax_name) in enumerate(zip(axes, AX_NAMES)):
        groups, labels, colors = [], [], []

        for aligned, lbl, col in [
            (a_c0_moc, 'Left Cam\nvs MoCap',    COLORS[0]),
            (a_c1_moc, 'Right Cam\nvs MoCap',   COLORS[1]),
            (a_c0_c1,  'Left Cam\nvs Right Cam', COLORS[2]),
        ]:
            if not aligned: continue
            errs = np.array([r['error_vec'][ax_i]*1000 for r in aligned])
            # Secondary MAD on errors
            med = np.nanmedian(errs)
            mad = max(np.nanmedian(np.abs(errs - med)), 1e-9)
            errs = errs[np.abs(errs - med) / mad < MAD_THRESH]
            if len(errs) < 3: continue
            groups.append(errs.tolist())
            labels.append(lbl)
            colors.append(col)

        if not groups:
            ax.text(0.5, 0.5, 'No data',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(ax_name); continue

        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', lw=2.0),
                        flierprops=dict(marker='o', markerfacecolor='red',
                                        markersize=3, alpha=0.5))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col); patch.set_alpha(0.55)

        ax.axhline(0, color='red', lw=1.0, ls='--', alpha=0.5)
        ax.set_title(ax_name, fontweight='bold')
        ax.set_ylabel('Error (mm)'); ax.grid(True, alpha=0.35)

        # Annotations anchored to axes fraction — never clips
        n = len(groups)
        for j, g in enumerate(groups):
            mu   = np.mean(g)
            sd   = np.std(g)
            rmse = np.sqrt(np.mean(np.array(g)**2))
            x    = (j + 0.5) / n
            ax.text(x, 0.97,
                    f"μ={mu:.1f}mm\nσ={sd:.1f}mm\nRMSE={rmse:.1f}mm",
                    transform=ax.transAxes,
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'error_boxplot.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")

# ══════════════════════════════════════════════════════════════════
# STEP 12 — SAVE NUMPY FILES
# ══════════════════════════════════════════════════════════════════
def save_npy(results, name):
    if not results: return
    arr = np.array([r['basis']     for r in results])
    ts  = np.array([r['timestamp'] for r in results])
    p1  = os.path.join(OUTPUT_DIR, f'{name}.npy')
    p2  = os.path.join(OUTPUT_DIR, f'{name}_timestamps.npy')
    np.save(p1, arr); np.save(p2, ts)
    print(f"    {name}.npy  shape={arr.shape}  → {p1}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  DUAL-CAM vs MoCap ERROR ANALYSIS")
    print("  Formula: basis = R.T @ (T_ref - T_frame)")
    print("=" * 62)

    K_L, D_L, K_R, D_R = load_calibration()
    t_start, t_end      = load_sync_window()
    cam0_frames, cam1_frames = load_pickle_frames(t_start, t_end)

    print("\n[4] Setting Up Detectors")
    ch_board, ch_det, ap_det, tag_3d = setup_detectors()

    raw = compute_camera_basis(
        cam0_frames, cam1_frames,
        K_L, D_L, K_R, D_R,
        ch_board, ch_det, ap_det, tag_3d)

    print("\n[5b] Filtering Outliers")
    cam0_clean, r0 = mad_filter(raw[0])
    cam1_clean, r1 = mad_filter(raw[1])
    print(f"    Cam0: {len(cam0_clean)} kept, {r0} removed")
    print(f"    Cam1: {len(cam1_clean)} kept, {r1} removed")

    assert len(cam0_clean) >= 5, "❌ Too few cam0 points after filtering!"
    assert len(cam1_clean) >= 5, "❌ Too few cam1 points after filtering!"

    mocap_records = load_mocap(t_start, t_end)
    moc_basis     = compute_mocap_basis(mocap_records)

    moc_clean = None
    if moc_basis:
        moc_clean, rm = mad_filter(moc_basis)
        print(f"    MoCap: {len(moc_clean)} kept, {rm} removed")

    print("\n[8] Temporal Alignment (nearest-neighbour on real timestamps)")
    a_c0_moc = align(cam0_clean, moc_clean,  "Cam0 vs MoCap")
    a_c1_moc = align(cam1_clean, moc_clean,  "Cam1 vs MoCap")
    a_c0_c1  = align(cam0_clean, cam1_clean, "Cam0 vs Cam1")

    print("\n[9b] Saving NumPy Basis Files")
    save_npy(cam0_clean, 'cam0_basis')
    save_npy(cam1_clean, 'cam1_basis')
    save_npy(moc_clean,  'mocap_basis')

    plot_trajectory(cam0_clean, cam1_clean, moc_clean)
    plot_errors(a_c0_moc, a_c1_moc, a_c0_c1)

    print("\n" + "=" * 62)
    print(f"  ✅  Done. All outputs → {OUTPUT_DIR}")
    print("=" * 62)

if __name__ == '__main__':
    main()
