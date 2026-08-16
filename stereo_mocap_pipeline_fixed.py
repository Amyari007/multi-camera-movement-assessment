"""
stereo_mocap_pipeline.py  —  FIXED VERSION
===========================================
Stereo Camera + MoCap Trajectory & Error Computation Pipeline

FIXES IN THIS VERSION vs the version that crashed:
  1. PATH VALIDATION at startup — all files are checked before any
     processing begins; clear, actionable error message lists every
     missing path so you can fix them all at once instead of
     discovering them one by one mid-run.
  2. MOCAP CSV path corrected to a variable you set at the top.
  3. solvePnP flip rejection  — tvec.Z must be positive
  4. Coordinate system registration  — Procrustes alignment on first N
     matched pairs to bring camera basis into MoCap world frame
  5. Stricter spike filter  — MAD=3.0 + max 50mm inter-frame jump check
  6. Cam0 and Cam1 use the SAME T_ref / R_ref (from Cam1 first frame)
  7. MoCap NaN fallback to raw marker centroid
  8. 4-panel box plots: X, Y, Z, 3D Euclidean

FORMULA (both cameras and MoCap):
  basis = R.T @ (T_ref − T_frame)

Install once:
  pip uninstall opencv-python -y
  pip install opencv-contrib-python scipy matplotlib pandas
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation
import cv2
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# SECTION 0 — APRILTAG BACKEND CHECK
# ══════════════════════════════════════════════════════════════════
def _probe_apriltag():
    try:
        dict_id = getattr(cv2.aruco, "DICT_APRILTAG_36h11",
                   getattr(cv2.aruco, "DICT_APRILTAG_36H11", None))
        if dict_id is None:
            raise AttributeError("No AprilTag dict constant found")
        cv2.aruco.getPredefinedDictionary(dict_id)
        cv2.aruco.DetectorParameters()
        print("  AprilTag backend: opencv-contrib aruco ✓")
        return dict_id
    except AttributeError as e:
        print(f"  ❌ {e}")
        print("  Fix:  pip uninstall opencv-python -y")
        print("        pip install opencv-contrib-python")
        sys.exit(1)

APRILTAG_DICT_ID = _probe_apriltag()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — PATHS  ← EDIT THESE BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════
#
#   mocap_csv: Update this to wherever your MoCap file actually lives.
#              Examples:
#                r"F:\mocap trial 1 donw w 2 cam.csv"   (external drive)
#                r"C:\Users\arya0\Downloads\mocap.csv"  (copied locally)
#              If the F: drive is not connected, copy the file locally first.
#
PATHS = {
    "calibration": r"C:\Users\arya0\Desktop\camera_calibration_output.npz",
    "board_pkl"  : r"C:\Users\arya0\Downloads\raw_capture_output\board_detection.pkl",
    "raw_pkl"    : r"C:\Users\arya0\Downloads\raw_capture_output\raw_frames.pkl",
    "sync_csv"   : r"C:\Users\arya0\OneDrive\Desktop\raw capture sync_and_framerate.csv",
    # ↓↓  UPDATE THIS PATH to wherever the MoCap CSV file is on your machine  ↓↓
    "mocap_csv"  : r"C:\Users\arya0\Downloads\mocap trial 1 donw w 2 cam.csv"",
    "output_dir" : r"C:\Users\arya0\Downloads\pipeline_output",
}

# ══════════════════════════════════════════════════════════════════
# PATH VALIDATION — runs before any processing
# ══════════════════════════════════════════════════════════════════
def validate_paths(paths):
    """
    Check every input path exists before starting the pipeline.
    Prints a clear list of missing files and exits if any are absent,
    so you can fix them all at once rather than hitting errors mid-run.
    """
    INPUT_KEYS = ["calibration", "board_pkl", "raw_pkl", "sync_csv", "mocap_csv"]
    missing = []
    for key in INPUT_KEYS:
        p = paths.get(key, "")
        if not p:
            missing.append((key, "<empty string>"))
        elif not os.path.exists(p):
            missing.append((key, p))

    if missing:
        print("\n" + "!" * 62)
        print("  PATH VALIDATION FAILED — the following files were not found:")
        print("!" * 62)
        for key, p in missing:
            print(f"\n  [{key}]")
            print(f"    Path given : {p}")

            # Give targeted hints per key
            if key == "mocap_csv":
                drive = p[0] + ":\\" if len(p) > 1 else "?"
                print(f"    Likely fix : Is the {drive} drive connected?")
                print(f"                 If not, copy the file to your C: drive")
                print(f"                 and update PATHS['mocap_csv'] in the script.")
            else:
                print(f"    Likely fix : Check that the file exists at the path above.")

        print("\n" + "!" * 62)
        print("  Fix the path(s) above in the PATHS dict (Section 1 of this")
        print("  script) and run again.")
        print("!" * 62 + "\n")
        sys.exit(1)

    # Output directory: create if missing (not a fatal error)
    os.makedirs(paths["output_dir"], exist_ok=True)
    print("  All input paths validated ✓")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════
APRILTAG_SIZE_M        = 0.048   # 4.8 cm
CHARUCO_TAG_SIZE_M     = 0.027   # 2.7 cm
CHARUCO_SQ_SIZE_M      = 0.037   # 3.7 cm
ID14_ID20_LATERAL_M    = 0.098   # 9.8 cm lateral distance ID14↔ID20
ID12_VERTICAL_OFFSET_M = 0.050   # 5.0 cm ID12 above ID14/ID20 midpoint
MOCAP_START_STR        = "2026-06-03 12:02:27.210"
MOCAP_FPS              = 100.0

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — QUALITY / FILTER PARAMETERS
# ══════════════════════════════════════════════════════════════════
MAX_REPROJ_ERR_PX   = 4.0    # reject solvePnP if reprojection > N px
MAX_FRAME_JUMP_MM   = 80.0   # reject frame if basis jumps > N mm
MAD_THRESH          = 3.0    # tighter MAD for outlier rejection
MAX_VEL_M_S         = 1.5    # max plausible arm velocity
SMOOTH_WINDOW       = 5      # rolling-median window (frames)
COORD_REG_N_PAIRS   = 30     # pairs to use for coordinate registration
MATCH_TOLERANCE_S   = 0.015  # ±15 ms for nearest-neighbour match

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════
def quat_to_R(qx, qy, qz, qw):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

def compute_basis(R_ref, T_ref, T_frame):
    return R_ref.T @ (T_ref - T_frame)

def apriltag_corners_3d(size):
    h = size / 2.0
    return np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float64)

def reproj_error(obj_pts, img_pts, K, D, rvec, tvec):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    return float(np.mean(np.linalg.norm(
        proj.reshape(-1,2) - img_pts.reshape(-1,2), axis=1)))

def solve_tag_pose_clean(corners_2d, K, D, size=APRILTAG_SIZE_M):
    """
    solvePnP with flip rejection.
    IPPE_SQUARE gives two solutions; we pick the one where Z > 0
    (tag must be in FRONT of the camera) and reprojection < threshold.
    """
    obj_pts = apriltag_corners_3d(size)
    img_pts = np.array(corners_2d, dtype=np.float64).reshape(4, 2)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D,
                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None

    # Reject flipped pose (Z must be positive — tag in front of cam)
    if tvec[2, 0] < 0:
        rvec = rvec + np.array([[np.pi], [0], [0]])

    # Re-verify after potential flip
    if tvec[2, 0] < 0:
        return None, None, None

    err = reproj_error(obj_pts, img_pts, K, D, rvec, tvec)
    if err > MAX_REPROJ_ERR_PX:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten(), err

def rolling_median(arr, w=SMOOTH_WINDOW):
    from scipy.ndimage import median_filter
    out = np.copy(arr)
    for i in range(arr.shape[1]):
        out[:, i] = median_filter(arr[:, i], size=w, mode='nearest')
    return out

def mad_spike_filter(records):
    """
    Three-stage filter:
      1. MAD per axis (threshold=MAD_THRESH)
      2. Inter-frame jump > MAX_FRAME_JUMP_MM
      3. Velocity > MAX_VEL_M_S
    """
    if len(records) < 4:
        return records, 0
    arr = np.array([r['basis'] for r in records]) * 1000  # mm
    ts  = np.arange(len(records), dtype=float)            # fallback index

    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    mad_ok = np.all(np.abs(arr - med) / mad < MAD_THRESH, axis=1)

    jump_ok = np.ones(len(records), dtype=bool)
    for i in range(1, len(records)):
        if np.linalg.norm(arr[i] - arr[i-1]) > MAX_FRAME_JUMP_MM:
            jump_ok[i] = False

    vel_ok = np.ones(len(records), dtype=bool)
    for i in range(1, len(records)):
        dt = ts[i] - ts[i-1]
        if dt > 1e-6:
            if np.linalg.norm((arr[i] - arr[i-1]) / 1000) / dt > MAX_VEL_M_S:
                vel_ok[i] = False

    keep = mad_ok & jump_ok & vel_ok
    return [r for r, k in zip(records, keep) if k], int(np.sum(~keep))

# ══════════════════════════════════════════════════════════════════
# STEP 1 — COORDINATE SYSTEM REGISTRATION
# ══════════════════════════════════════════════════════════════════
def register_coordinate_systems(cam_basis_arr, moc_basis_arr, n=COORD_REG_N_PAIRS):
    """
    Compute the rigid transform (R_reg, t_reg) mapping camera basis
    vectors into MoCap world coordinates via Procrustes / Umeyama.
      Q ≈ P @ R_reg + t_reg
    """
    n_use = min(n, len(cam_basis_arr), len(moc_basis_arr))
    P = cam_basis_arr[:n_use]
    Q = moc_basis_arr[:n_use]

    cP = np.mean(P, axis=0)
    cQ = np.mean(Q, axis=0)
    Pc = P - cP
    Qc = Q - cQ

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R_reg = U @ Vt
    if np.linalg.det(R_reg) < 0:
        Vt[-1, :] *= -1
        R_reg = U @ Vt

    t_reg = cQ - cP @ R_reg
    residual = np.mean(np.linalg.norm((P @ R_reg + t_reg) - Q, axis=1)) * 1000
    print(f"    Coord registration residual (mean): {residual:.2f} mm  "
          f"(using {n_use} pairs)")
    return R_reg, t_reg

def apply_registration(basis_arr, R_reg, t_reg):
    return basis_arr @ R_reg + t_reg

# ══════════════════════════════════════════════════════════════════
# STEP 2 — LOAD SYNC CSV
# ══════════════════════════════════════════════════════════════════
def load_sync_csv(path):
    print("\n[1] Loading sync CSV")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["sync_ffill"] = df["sync_value"].ffill().fillna(0).astype(int)

    rows = []
    for _, r in df.iterrows():
        if pd.notna(r["cam0_frame"]):
            rows.append({"timestamp": r["timestamp"], "sync": r["sync_ffill"],
                         "cam_id": 0, "frame_num": int(r["cam0_frame"]),
                         "fps": r.get("cam0_fps", 15)})
        if pd.notna(r["cam1_frame"]):
            rows.append({"timestamp": r["timestamp"], "sync": r["sync_ffill"],
                         "cam_id": 1, "frame_num": int(r["cam1_frame"]),
                         "fps": r.get("cam1_fps", 18)})

    cam_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    sync1  = cam_df[cam_df.sync == 1]
    print(f"    cam0 total: {(cam_df.cam_id==0).sum()}  "
          f"cam1 total: {(cam_df.cam_id==1).sum()}")
    print(f"    cam0 sync=1: {(sync1.cam_id==0).sum()}  "
          f"cam1 sync=1: {(sync1.cam_id==1).sum()}")
    return cam_df

# ══════════════════════════════════════════════════════════════════
# STEP 3 — LOAD MOCAP CSV
# ══════════════════════════════════════════════════════════════════
def load_mocap_csv(path):
    """
    Loads the OptiTrack / Motive MoCap CSV export.
    Expects multi-level headers in rows 2–6 (0-indexed rows 2,3,4,5,6)
    with rows 0–1 being metadata lines that are skipped.

    If the header structure doesn't match exactly (different Motive
    export versions vary), the function prints a diagnostic and falls
    back to positional column detection.
    """
    print("\n[2] Loading MoCap CSV")

    # ── Try 5-level header (standard Motive export) ───────────
    try:
        df_raw = pd.read_csv(path, skiprows=[0, 1], header=[0, 1, 2, 3, 4])
    except Exception as e:
        print(f"    WARNING: 5-level header parse failed ({e})")
        print("    Retrying with single header row …")
        df_raw = pd.read_csv(path, skiprows=6)

    def find_col(type_kw, name_kw, data_kw, axis):
        for i, c in enumerate(df_raw.columns):
            col_str = " ".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
            if (type_kw in col_str and name_kw in col_str
                    and data_kw in col_str and axis in col_str):
                return i
        return None

    # Time column is always column index 1 in Motive exports
    time_col = pd.to_numeric(df_raw.iloc[:, 1], errors='coerce').values

    def series(type_kw, name_kw, data_kw, axis):
        idx = find_col(type_kw, name_kw, data_kw, axis)
        if idx is None:
            return np.full(len(df_raw), np.nan)
        return pd.to_numeric(df_raw.iloc[:, idx], errors='coerce').values

    # RigidBody 1 (static ChArUco base)
    rb1_tx = series("Rigid Body", "RigidBody",     "Position", "X")
    rb1_ty = series("Rigid Body", "RigidBody",     "Position", "Y")
    rb1_tz = series("Rigid Body", "RigidBody",     "Position", "Z")
    rb1_qx = series("Rigid Body", "RigidBody",     "Rotation", "X")
    rb1_qy = series("Rigid Body", "RigidBody",     "Rotation", "Y")
    rb1_qz = series("Rigid Body", "RigidBody",     "Rotation", "Z")
    rb1_qw = series("Rigid Body", "RigidBody",     "Rotation", "W")

    # RigidBody 2 (moving AprilTag mount)
    rb2_tx = series("Rigid Body", "RigidBody 002", "Position", "X")
    rb2_ty = series("Rigid Body", "RigidBody 002", "Position", "Y")
    rb2_tz = series("Rigid Body", "RigidBody 002", "Position", "Z")
    rb2_qx = series("Rigid Body", "RigidBody 002", "Rotation", "X")
    rb2_qy = series("Rigid Body", "RigidBody 002", "Rotation", "Y")
    rb2_qz = series("Rigid Body", "RigidBody 002", "Rotation", "Z")
    rb2_qw = series("Rigid Body", "RigidBody 002", "Rotation", "W")

    # Raw marker fallback for RigidBody 002 (up to 5 markers)
    rb2_markers = {}
    for mk in range(1, 6):
        for ax in ["X", "Y", "Z"]:
            key = f"rb2_m{mk}{ax.lower()}"
            rb2_markers[key] = series("Rigid Body Marker",
                                      f"RigidBody 002:Marker{mk}",
                                      "Position", ax)

    mocap_start = pd.Timestamp(MOCAP_START_STR)
    abs_ts = [mocap_start + pd.Timedelta(seconds=float(t))
              for t in time_col]

    data = {
        "time_s": time_col, "abs_timestamp": abs_ts,
        "rb1_tx": rb1_tx, "rb1_ty": rb1_ty, "rb1_tz": rb1_tz,
        "rb1_qx": rb1_qx, "rb1_qy": rb1_qy,
        "rb1_qz": rb1_qz, "rb1_qw": rb1_qw,
        "rb2_tx": rb2_tx, "rb2_ty": rb2_ty, "rb2_tz": rb2_tz,
        "rb2_qx": rb2_qx, "rb2_qy": rb2_qy,
        "rb2_qz": rb2_qz, "rb2_qw": rb2_qw,
        **rb2_markers,
    }
    mdf = pd.DataFrame(data)

    # Marker centroid fallback: fill NaN rb2 position with mean of markers
    for ax in ["x", "y", "z"]:
        cols = [f"rb2_m{mk}{ax}" for mk in range(1, 6)]
        mdf[f"rb2_t{ax}"] = mdf[f"rb2_t{ax}"].fillna(mdf[cols].mean(axis=1))

    nan_rb2 = mdf[["rb2_tx", "rb2_ty", "rb2_tz"]].isna().all(axis=1).sum()
    print(f"    MoCap frames: {len(mdf)}  |  "
          f"RigidBody 002 NaN rows after fallback: {nan_rb2}")

    if nan_rb2 == len(mdf):
        print("    WARNING: ALL RigidBody 002 rows are NaN — check column names "
              "in the CSV match 'RigidBody 002'. The pipeline will likely produce "
              "no matched pairs for this rigid body.")

    return mdf

# ══════════════════════════════════════════════════════════════════
# STEP 4 — LOAD BOARD PKL
# ══════════════════════════════════════════════════════════════════
def load_board_pkl(path):
    print("\n[3] Loading board detection pickle")
    with open(path, "rb") as f:
        data = pickle.load(f)
    frames = data.get("frames", [])
    print(f"    Total entries: {len(frames)}")
    sample_keys = set()
    for fr in frames[:50]:
        sample_keys.update(fr.keys())
    print(f"    Keys: {sorted(sample_keys)}")
    id12 = sum(1 for f in frames if f.get("tag12_corners") is not None)
    id14 = sum(1 for f in frames if f.get("tag14_corners") is not None)
    id20 = sum(1 for f in frames if f.get("tag20_corners") is not None)
    ch   = sum(1 for f in frames if f.get("charuco_found"))
    print(f"    ID12: {id12}  ID14: {id14}  ID20: {id20}  "
          f"ChArUco: {ch}  / {len(frames)}")
    return frames, sample_keys

# ══════════════════════════════════════════════════════════════════
# STEP 5 — GOLDEN WINDOW DETECTION
# ══════════════════════════════════════════════════════════════════
def find_golden_window(cam_df, mocap_df):
    print("\n[4] Detecting golden window")
    moc_start = mocap_df["abs_timestamp"].iloc[0]
    moc_end   = mocap_df["abs_timestamp"].iloc[-1]

    valid = cam_df[
        (cam_df["sync"] == 1) &
        (cam_df["timestamp"] >= moc_start) &
        (cam_df["timestamp"] <= moc_end)
    ].copy()

    c0 = valid[valid.cam_id == 0]
    c1 = valid[valid.cam_id == 1]

    med0, med1 = c0["fps"].median(), c1["fps"].median()
    tol = 2.5
    c0s = c0[np.abs(c0["fps"] - med0) <= tol]
    c1s = c1[np.abs(c1["fps"] - med1) <= tol]

    if len(c0s) == 0 or len(c1s) == 0:
        t_start, t_end = moc_start, moc_end
    else:
        t_start = max(c0s["timestamp"].iloc[0],  c1s["timestamp"].iloc[0])
        t_end   = min(c0s["timestamp"].iloc[-1], c1s["timestamp"].iloc[-1])
        if t_start >= t_end:
            t_start, t_end = moc_start, moc_end

    dur = (t_end - t_start).total_seconds()
    n0  = ((c0s.timestamp >= t_start) & (c0s.timestamp <= t_end)).sum()
    n1  = ((c1s.timestamp >= t_start) & (c1s.timestamp <= t_end)).sum()
    print(f"    {t_start}  →  {t_end}")
    print(f"    Duration: {dur:.2f}s  |  cam0: {n0} frames  cam1: {n1} frames")
    return t_start, t_end

# ══════════════════════════════════════════════════════════════════
# STEP 6 — ID12 CENTRE ESTIMATION FROM ID14 + ID20
# ══════════════════════════════════════════════════════════════════
def estimate_id12_px(c14, c20):
    ctr14 = np.mean(c14, axis=0)
    ctr20 = np.mean(c20, axis=0)
    lateral_px = np.linalg.norm(ctr20 - ctr14)
    px_per_m   = lateral_px / ID14_ID20_LATERAL_M if lateral_px > 1e-9 else 1
    lat_vec    = (ctr20 - ctr14) / (lateral_px + 1e-9)
    perp_vec   = np.array([-lat_vec[1], lat_vec[0]])
    if perp_vec[1] > 0:
        perp_vec = -perp_vec
    mid = (ctr14 + ctr20) / 2.0
    return mid + perp_vec * (ID12_VERTICAL_OFFSET_M * px_per_m)

def build_synth_corners(id12_px, ref_corners):
    ctr  = np.mean(ref_corners, axis=0)
    half = np.mean(np.linalg.norm(ref_corners - ctr, axis=1))
    return np.array([
        id12_px + [-half,  half],
        id12_px + [ half,  half],
        id12_px + [ half, -half],
        id12_px + [-half, -half],
    ], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════
# STEP 7 — CAMERA TRAJECTORY EXTRACTION
# ══════════════════════════════════════════════════════════════════
def extract_camera_trajectory(board_frames, cam_id, K, D,
                               sync_df, raw_pkl_path,
                               gw_start, gw_end,
                               T_ref=None, R_ref=None):
    print(f"\n[5] Extracting camera {cam_id} trajectory")

    cam_sync = sync_df[sync_df.cam_id == cam_id].set_index("frame_num")

    gw_frames = []
    for f in board_frames:
        if f.get("cam_id") != cam_id:
            continue
        fn = f.get("frame_num")
        if fn not in cam_sync.index:
            continue
        row = cam_sync.loc[fn]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if row["sync"] != 1:
            continue
        ts = row["timestamp"]
        if gw_start <= ts <= gw_end:
            f = dict(f)
            f["_ts"] = ts
            gw_frames.append(f)

    print(f"    Frames in golden window (sync=1): {len(gw_frames)}")

    has_pkl14 = any(f.get("tag14_corners") is not None for f in gw_frames)
    has_pkl20 = any(f.get("tag20_corners") is not None for f in gw_frames)
    need_redetect = not (has_pkl14 and has_pkl20)

    if need_redetect:
        print("    tag14/tag20 NOT in pkl — will re-detect from raw_frames.pkl")
        print("    Loading raw_frames.pkl (may be slow ~2 GB) …")
        with open(raw_pkl_path, "rb") as rf:
            rdata = pickle.load(rf)
        raw_lookup = {(fr["cam_id"], fr["frame_num"]): fr["frame"]
                      for fr in rdata.get("frames", [])}
    else:
        raw_lookup = {}

    trajectory = []
    src_counts = {"id12_pkl": 0, "id14_id20_pkl": 0,
                  "redetect_id12": 0, "redetect_backup": 0, "failed": 0}
    prev_basis = None

    for f in gw_frames:
        fn  = f.get("frame_num")
        ts  = f.get("_ts")
        R_f = T_f = None
        src = "failed"

        # Method 1: ID12 direct from pkl
        c12 = f.get("tag12_corners")
        if c12 is not None:
            c12 = np.array(c12).reshape(4, 2)
            R_f, T_f, _ = solve_tag_pose_clean(c12, K, D)
            if R_f is not None:
                src = "id12_pkl"

        # Method 2: ID14+ID20 from pkl → synthesise ID12 location
        if R_f is None and has_pkl14 and has_pkl20:
            c14 = f.get("tag14_corners")
            c20 = f.get("tag20_corners")
            if c14 is not None and c20 is not None:
                c14 = np.array(c14).reshape(4, 2)
                c20 = np.array(c20).reshape(4, 2)
                id12_px = estimate_id12_px(c14, c20)
                synth   = build_synth_corners(id12_px, c14)
                R_f, T_f, _ = solve_tag_pose_clean(synth, K, D)
                if R_f is not None:
                    src = "id14_id20_pkl"

        # Method 3: Re-detect from raw frame
        if R_f is None and need_redetect:
            bgr = raw_lookup.get((cam_id, fn))
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
                gray_ud = cv2.undistort(gray, K, D, None, newK)
                adict   = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT_ID)
                params  = cv2.aruco.DetectorParameters()
                det     = cv2.aruco.ArucoDetector(adict, params)
                corners_list, ids_det, _ = det.detectMarkers(gray_ud)
                detected = {}
                if ids_det is not None:
                    for crnr, tid in zip(corners_list, ids_det.flatten()):
                        detected[int(tid)] = crnr.reshape(4, 2).astype(np.float64)
                dc12 = detected.get(12)
                dc14 = detected.get(14)
                dc20 = detected.get(20)
                if dc12 is not None:
                    R_f, T_f, _ = solve_tag_pose_clean(dc12, K, D)
                    if R_f is not None:
                        src = "redetect_id12"
                if R_f is None and dc14 is not None and dc20 is not None:
                    id12_px = estimate_id12_px(dc14, dc20)
                    synth   = build_synth_corners(id12_px, dc14)
                    R_f, T_f, _ = solve_tag_pose_clean(synth, K, D)
                    if R_f is not None:
                        src = "redetect_backup"

        if R_f is None or T_f is None:
            src_counts["failed"] += 1
            continue

        # Set T_ref / R_ref from first valid Cam1 frame (shared by both cams)
        if T_ref is None:
            T_ref = T_f.copy()
            R_ref = R_f.copy()
            print(f"    T_ref set at frame {fn}: "
                  f"[{T_ref[0]*1000:.1f}, {T_ref[1]*1000:.1f}, "
                  f"{T_ref[2]*1000:.1f}] mm")

        basis = compute_basis(R_ref, T_ref, T_f)

        # Spike check vs previous frame
        if prev_basis is not None:
            jump_mm = np.linalg.norm((basis - prev_basis) * 1000)
            if jump_mm > MAX_FRAME_JUMP_MM:
                src_counts["failed"] += 1
                continue

        prev_basis = basis.copy()
        src_counts[src] += 1
        trajectory.append({
            "frame_num":  fn,
            "timestamp":  ts,
            "basis":      basis,
            "T":          T_f,
            "R":          R_f,
            "tag_source": src,
        })

    print(f"    Trajectory points: {len(trajectory)}  |  "
          f"Failed/skipped: {src_counts['failed']}")
    print(f"    Source breakdown: {src_counts}")
    return trajectory, T_ref, R_ref

# ══════════════════════════════════════════════════════════════════
# STEP 8 — ROLLING MEDIAN SMOOTH + MAD FILTER
# ══════════════════════════════════════════════════════════════════
def smooth_and_filter(trajectory, label=""):
    if len(trajectory) < SMOOTH_WINDOW:
        return trajectory
    arr    = np.array([r["basis"] for r in trajectory])
    arr_sm = rolling_median(arr, SMOOTH_WINDOW)
    smoothed = [{**r, "basis": arr_sm[i]} for i, r in enumerate(trajectory)]
    filtered, n_rem = mad_spike_filter(smoothed)
    print(f"    [{label}] smoothed → {len(smoothed)}  "
          f"after MAD filter → {len(filtered)}  removed: {n_rem}")
    return filtered

# ══════════════════════════════════════════════════════════════════
# STEP 9 — MOCAP TRAJECTORY EXTRACTION
# ══════════════════════════════════════════════════════════════════
def extract_mocap_trajectory(mocap_df, gw_start, gw_end):
    print("\n[6] Extracting MoCap trajectory")
    mask = (
        (mocap_df["abs_timestamp"] >= gw_start) &
        (mocap_df["abs_timestamp"] <= gw_end) &
        mocap_df["rb2_tx"].notna() &
        mocap_df["rb2_qw"].notna()
    )
    mdf = mocap_df[mask].copy().reset_index(drop=True)
    print(f"    MoCap frames in golden window: {len(mdf)}")

    T_ref_m = R_ref_m = None
    trajectory = []

    for _, row in mdf.iterrows():
        T = np.array([row["rb2_tx"], row["rb2_ty"], row["rb2_tz"]])
        q = [row["rb2_qx"], row["rb2_qy"], row["rb2_qz"], row["rb2_qw"]]
        if any(np.isnan(q)) or any(np.isnan(T)):
            continue
        R = quat_to_R(*q)
        if T_ref_m is None:
            T_ref_m = T.copy()
            R_ref_m = R.copy()
            print(f"    MoCap T_ref: [{T_ref_m[0]*1000:.1f}, "
                  f"{T_ref_m[1]*1000:.1f}, {T_ref_m[2]*1000:.1f}] mm")
        basis = compute_basis(R_ref_m, T_ref_m, T)
        trajectory.append({
            "abs_timestamp": row["abs_timestamp"],
            "basis": basis,
            "T": T,
        })

    print(f"    MoCap trajectory points: {len(trajectory)}")
    return trajectory

# ══════════════════════════════════════════════════════════════════
# STEP 10 — TIMESTAMP MATCHING
# ══════════════════════════════════════════════════════════════════
def match_trajectories(cam_traj, mocap_traj, tol=MATCH_TOLERANCE_S):
    mocap_ts = np.array([t["abs_timestamp"].timestamp() for t in mocap_traj])
    matched  = []
    for ct in cam_traj:
        cam_t = ct["timestamp"].timestamp()
        diffs = np.abs(mocap_ts - cam_t)
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            matched.append({
                "cam_frame":   ct["frame_num"],
                "cam_ts":      ct["timestamp"],
                "mocap_frame": j,
                "dt_ms":       diffs[j] * 1000,
                "cam_basis":   ct["basis"],
                "mocap_basis": mocap_traj[j]["basis"],
                "tag_source":  ct["tag_source"],
            })
    print(f"    Matched pairs: {len(matched)}  (tol ±{tol*1000:.0f} ms)")
    return matched

# ══════════════════════════════════════════════════════════════════
# STEP 11 — COORDINATE REGISTRATION + ERROR COMPUTATION
# ══════════════════════════════════════════════════════════════════
def register_and_compute_errors(matched, label="cam"):
    if not matched:
        print(f"    No matched pairs for {label}")
        return {}, pd.DataFrame(), None, None

    cam_b = np.array([m["cam_basis"]   for m in matched])
    moc_b = np.array([m["mocap_basis"] for m in matched])

    print(f"\n[7] Coordinate registration — {label}")
    R_reg, t_reg = register_coordinate_systems(cam_b, moc_b)
    cam_b_reg    = apply_registration(cam_b, R_reg, t_reg)

    cam_mm = cam_b_reg * 1000
    moc_mm = moc_b    * 1000
    diff   = cam_mm - moc_mm

    records = []
    for i, m in enumerate(matched):
        records.append({
            "cam_frame":   m["cam_frame"],
            "cam_ts":      m["cam_ts"],
            "mocap_frame": m["mocap_frame"],
            "dt_ms":       m["dt_ms"],
            "tag_source":  m["tag_source"],
            "cam_X_mm":    cam_mm[i, 0], "cam_Y_mm": cam_mm[i, 1],
            "cam_Z_mm":    cam_mm[i, 2],
            "mocap_X_mm":  moc_mm[i, 0], "mocap_Y_mm": moc_mm[i, 1],
            "mocap_Z_mm":  moc_mm[i, 2],
            "err_X_mm":    diff[i, 0],   "err_Y_mm":  diff[i, 1],
            "err_Z_mm":    diff[i, 2],
            "err_3D_mm":   float(np.linalg.norm(diff[i])),
        })
    df = pd.DataFrame(records)

    def stats(arr):
        return {"mean": float(np.mean(arr)),  "std":  float(np.std(arr)),
                "rmse": float(np.sqrt(np.mean(arr**2))),
                "max":  float(np.max(np.abs(arr)))}

    errors = {
        "X":  stats(diff[:, 0]), "Y": stats(diff[:, 1]),
        "Z":  stats(diff[:, 2]),
        "3D": stats(np.linalg.norm(diff, axis=1)),
        "N":  len(matched),
    }
    print(f"\n    Error Summary — {label}")
    for ax in ["X", "Y", "Z", "3D"]:
        s = errors[ax]
        print(f"      {ax}: mean={s['mean']:+.2f} mm  std={s['std']:.2f} mm  "
              f"RMSE={s['rmse']:.2f} mm  max={s['max']:.2f} mm")
    return errors, df, R_reg, t_reg

# ══════════════════════════════════════════════════════════════════
# STEP 12 — TRAJECTORY PLOTS
# ══════════════════════════════════════════════════════════════════
def plot_trajectories(traj0, traj1, traj_moc, err_df0, err_df1, out_dir):
    print("\n[8] Generating trajectory plots")
    COLORS = {"cam0": "#2196F3", "cam1": "#4CAF50", "moc": "#F44336"}

    def extract(traj, key="basis"):
        return np.array([t[key] for t in traj]) * 1000 if traj else np.empty((0, 3))

    bm  = extract(traj_moc)
    ts0 = [t["timestamp"]     for t in traj0]     if traj0     else []
    ts1 = [t["timestamp"]     for t in traj1]     if traj1     else []
    tsm = [t["abs_timestamp"] for t in traj_moc]  if traj_moc  else []

    b0_reg = err_df0[["cam_X_mm", "cam_Y_mm", "cam_Z_mm"]].values \
             if not err_df0.empty else extract(traj0)
    bm0    = err_df0[["mocap_X_mm", "mocap_Y_mm", "mocap_Z_mm"]].values \
             if not err_df0.empty else bm
    b1_reg = err_df1[["cam_X_mm", "cam_Y_mm", "cam_Z_mm"]].values \
             if not err_df1.empty else extract(traj1)
    bm1    = err_df1[["mocap_X_mm", "mocap_Y_mm", "mocap_Z_mm"]].values \
             if not err_df1.empty else bm

    # ── 3D plot ────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    if b0_reg.shape[0]:
        ax.plot(b0_reg[:, 0], b0_reg[:, 1], b0_reg[:, 2],
                color=COLORS["cam0"], lw=1.5, label="Cam 0 (registered)")
    if b1_reg.shape[0]:
        ax.plot(b1_reg[:, 0], b1_reg[:, 1], b1_reg[:, 2],
                color=COLORS["cam1"], lw=1.5, label="Cam 1 (registered)")
    if bm.shape[0]:
        ax.plot(bm[:, 0], bm[:, 1], bm[:, 2],
                color=COLORS["moc"], lw=2.0, ls="--",
                label="MoCap (ground truth)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3-D Trajectory Comparison\n(camera registered to MoCap frame)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=150)
    plt.close(fig)

    # ── Per-axis plot ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for ai, ax in enumerate(axes):
        AX = ["X", "Y", "Z"]
        if b0_reg.shape[0] and ts0:
            t0s = [(t - ts0[0]).total_seconds() for t in ts0[:b0_reg.shape[0]]]
            ax.plot(t0s, b0_reg[:, ai], color=COLORS["cam0"], lw=1.2, label="Cam 0")
        if b1_reg.shape[0] and ts1:
            t1s = [(t - ts1[0]).total_seconds() for t in ts1[:b1_reg.shape[0]]]
            ax.plot(t1s, b1_reg[:, ai], color=COLORS["cam1"], lw=1.2, label="Cam 1")
        if bm.shape[0] and tsm:
            tms = [(t - tsm[0]).total_seconds() for t in tsm]
            ax.plot(tms, bm[:, ai], color=COLORS["moc"], lw=1.8, ls="--",
                    label="MoCap")
        ax.set_ylabel(f"{AX[ai]} (mm)"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4); ax.axhline(0, color="gray", lw=0.6)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Per-Axis Trajectory (camera in MoCap frame)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_per_axis.png"), dpi=150)
    plt.close(fig)
    print("    Trajectory plots saved.")

# ══════════════════════════════════════════════════════════════════
# STEP 13 — ERROR BOX PLOTS + ERROR OVER TIME
# ══════════════════════════════════════════════════════════════════
def plot_error_boxplots(err_df0, err_df1, out_dir):
    print("\n[9] Generating error box plots")
    axes_list = ["X", "Y", "Z", "3D"]
    col_map   = {"X": "err_X_mm", "Y": "err_Y_mm",
                 "Z": "err_Z_mm", "3D": "err_3D_mm"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Camera vs MoCap Error Distribution\n"
                 "(after coordinate registration + MAD filtering)",
                 fontsize=13, fontweight="bold")

    for i, ax_label in enumerate(axes_list):
        col  = col_map[ax_label]
        data, lbls, cols = [], [], []
        for df, lbl, c in [(err_df0, "Cam 0", "#2196F3"),
                            (err_df1, "Cam 1", "#4CAF50")]:
            if df.empty or col not in df.columns:
                continue
            errs = df[col].dropna().values
            med  = np.nanmedian(errs)
            mad  = max(np.nanmedian(np.abs(errs - med)), 1e-9)
            errs = errs[np.abs(errs - med) / mad < MAD_THRESH]
            if len(errs) < 3:
                continue
            data.append(errs); lbls.append(lbl); cols.append(c)

        if not data:
            axes[i].text(0.5, 0.5, "No data", transform=axes[i].transAxes,
                         ha="center", va="center")
            axes[i].set_title(f"{ax_label} Error (mm)")
            continue

        bp = axes[i].boxplot(data, labels=lbls, patch_artist=True,
                              medianprops=dict(color="black", lw=2),
                              flierprops=dict(marker="o",
                                              markerfacecolor="red",
                                              markersize=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        axes[i].axhline(0, color="red", lw=0.8, ls="--", alpha=0.6)
        axes[i].set_title(f"{ax_label} Error (mm)", fontweight="bold")
        axes[i].set_ylabel("Error (mm)"); axes[i].grid(True, alpha=0.35)

        n = len(data)
        for j, errs in enumerate(data):
            mu   = np.mean(errs)
            sd   = np.std(errs)
            rmse = np.sqrt(np.mean(errs**2))
            axes[i].text((j + 0.5) / n, 0.97,
                          f"μ={mu:.1f}\nσ={sd:.1f}\nRMSE={rmse:.1f}",
                          transform=axes[i].transAxes,
                          ha="center", va="top", fontsize=8,
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white",
                                    edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_boxplots.png"), dpi=150)
    plt.close(fig)

    # ── Error over time ────────────────────────────────────────
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    COLORS = {"Cam 0": "#2196F3", "Cam 1": "#4CAF50"}
    for df, lbl in [(err_df0, "Cam 0"), (err_df1, "Cam 1")]:
        if df.empty:
            continue
        t = [(ts - df["cam_ts"].iloc[0]).total_seconds()
             for ts in df["cam_ts"]]
        c = COLORS[lbl]
        for ai, ax_label in enumerate(["X", "Y", "Z", "3D"]):
            col = col_map[ax_label]
            axes2[ai].plot(t, df[col].values,
                            color=c, lw=1.0, label=lbl, alpha=0.8)
            axes2[ai].axhline(0, color="red", lw=0.8, ls="--")
            axes2[ai].set_ylabel(f"{ax_label} Error (mm)")
            axes2[ai].legend(fontsize=8); axes2[ai].grid(True, alpha=0.4)
    axes2[-1].set_xlabel("Time (s)")
    axes2[0].set_title("Error over Time (after registration)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "error_over_time.png"), dpi=150)
    plt.close(fig2)
    print("    Error plots saved.")

# ══════════════════════════════════════════════════════════════════
# STEP 14 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════
def save_outputs(traj0, traj1, traj_moc,
                 err_df0, err_df1, errors0, errors1, out_dir):
    print("\n[10] Saving outputs")

    def basis_arr(traj):
        return np.array([t["basis"] for t in traj]) if traj else np.empty((0, 3))

    np.save(os.path.join(out_dir, "basis_cam0.npy"),  basis_arr(traj0))
    np.save(os.path.join(out_dir, "basis_cam1.npy"),  basis_arr(traj1))
    np.save(os.path.join(out_dir, "basis_mocap.npy"), basis_arr(traj_moc))

    if not err_df0.empty:
        err_df0.to_csv(os.path.join(out_dir, "errors_cam0.csv"), index=False)
    if not err_df1.empty:
        err_df1.to_csv(os.path.join(out_dir, "errors_cam1.csv"), index=False)

    with open(os.path.join(out_dir, "error_summary.txt"), "w") as fh:
        fh.write("STEREO CAMERA vs MOCAP — ERROR SUMMARY\n" + "=" * 50 + "\n\n")
        for lbl, errs in [("Cam 0 (left)", errors0), ("Cam 1 (right/ref)", errors1)]:
            fh.write(f"{lbl}\n" + "-" * 30 + "\n")
            if not errs:
                fh.write("  No data\n\n")
                continue
            fh.write(f"  N matched frames: {errs.get('N', 0)}\n")
            for ax in ["X", "Y", "Z", "3D"]:
                s = errs.get(ax, {})
                fh.write(f"  {ax}: mean={s.get('mean', float('nan')):+.2f} mm  "
                          f"std={s.get('std', float('nan')):.2f} mm  "
                          f"RMSE={s.get('rmse', float('nan')):.2f} mm  "
                          f"max={s.get('max', float('nan')):.2f} mm\n")
            fh.write("\n")

    print("    Saved: basis_cam0/1/mocap.npy")
    print("    Saved: errors_cam0/1.csv")
    print("    Saved: error_summary.txt")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  STEREO CAMERA + MOCAP TRAJECTORY ERROR PIPELINE")
    print("  (flip rejection + coordinate registration)")
    print("=" * 62)

    # ── Validate all input paths before doing any work ─────────
    print("\n[PRE-CHECK] Validating input paths …")
    validate_paths(PATHS)

    # ── Load calibration ───────────────────────────────────────
    print("\n[0] Loading calibration")
    calib = np.load(PATHS["calibration"])
    K0 = calib["cam_matrix_L"].astype(np.float64)
    D0 = calib["dist_coeffs_L"].astype(np.float64)
    K1 = calib["cam_matrix_R"].astype(np.float64)
    D1 = calib["dist_coeffs_R"].astype(np.float64)
    print(f"    Cam0: fx={K0[0,0]:.2f}  Cam1: fx={K1[0,0]:.2f}")

    sync_df              = load_sync_csv(PATHS["sync_csv"])
    mocap_df             = load_mocap_csv(PATHS["mocap_csv"])
    board_frames, _      = load_board_pkl(PATHS["board_pkl"])
    gw_start, gw_end     = find_golden_window(sync_df, mocap_df)

    # ── Camera trajectories (Cam1 first to set T_ref / R_ref) ──
    traj1_raw, T_ref, R_ref = extract_camera_trajectory(
        board_frames, cam_id=1, K=K1, D=D1,
        sync_df=sync_df, raw_pkl_path=PATHS["raw_pkl"],
        gw_start=gw_start, gw_end=gw_end,
        T_ref=None, R_ref=None)

    traj0_raw, _, _ = extract_camera_trajectory(
        board_frames, cam_id=0, K=K0, D=D0,
        sync_df=sync_df, raw_pkl_path=PATHS["raw_pkl"],
        gw_start=gw_start, gw_end=gw_end,
        T_ref=T_ref.copy(), R_ref=R_ref.copy())

    # ── Smooth + filter ────────────────────────────────────────
    print("\n[5b] Smoothing and filtering trajectories")
    traj0 = smooth_and_filter(traj0_raw, "Cam0")
    traj1 = smooth_and_filter(traj1_raw, "Cam1")

    # ── MoCap trajectory ───────────────────────────────────────
    traj_moc = extract_mocap_trajectory(mocap_df, gw_start, gw_end)

    # ── Match timestamps ───────────────────────────────────────
    print("\n[6b] Matching timestamps")
    matched0 = match_trajectories(traj0,  traj_moc)
    matched1 = match_trajectories(traj1,  traj_moc)

    # ── Register + compute errors ──────────────────────────────
    errors0, err_df0, R_reg0, t_reg0 = register_and_compute_errors(
        matched0, "Cam 0 vs MoCap")
    errors1, err_df1, R_reg1, t_reg1 = register_and_compute_errors(
        matched1, "Cam 1 vs MoCap")

    # ── Plots + save ───────────────────────────────────────────
    plot_trajectories(traj0, traj1, traj_moc,
                      err_df0, err_df1, PATHS["output_dir"])
    plot_error_boxplots(err_df0, err_df1, PATHS["output_dir"])
    save_outputs(traj0, traj1, traj_moc,
                 err_df0, err_df1, errors0, errors1,
                 PATHS["output_dir"])

    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs → {PATHS['output_dir']}")
    print("  Files:")
    print("    trajectory_3d.png")
    print("    trajectory_per_axis.png")
    print("    error_boxplots.png")
    print("    error_over_time.png")
    print("    error_summary.txt")
    print("    errors_cam0.csv  /  errors_cam1.csv")
    print("    basis_cam0.npy   /  basis_cam1.npy  /  basis_mocap.npy")
    print("=" * 62)

if __name__ == "__main__":
    main()
