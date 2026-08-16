"""
STEP 3 — MOCAP TRAJECTORY PARSING
===================================
MoCap CSV structure (confirmed from your file):
  - Row 0: metadata (Format Version, Take Name, Capture Start Time, ...)
  - Rows 1–5: multi-header (Type, Name, ID, DataType, Axis)
  - Row 6 onwards: data

Single rigid body: 'aprl' (AprilTag mount)
  Col 0:  Frame
  Col 1:  Time (Seconds)
  Col 2:  Rot X (quaternion)
  Col 3:  Rot Y
  Col 4:  Rot Z
  Col 5:  Rot W
  Col 6:  Pos X (metres)
  Col 7:  Pos Y (metres)
  Col 8:  Pos Z (metres)
  Col 9:  Mean Marker Error
  Col 10–33: Rigid Body Markers 1–6 (X, Y, Z, Quality each)
  Col 34–51: Raw unassigned markers 1–6 (X, Y, Z each)

Capture Start Time: 2026-06-23 04:31:23.721 PM
100 fps → frame k → abs_time = capture_start + k * 0.01s

Outputs:
  trajectory_output/
    mocap_trajectory.npy     — shape (N,3) basis vectors in mm
    mocap_timestamps.npy     — absolute timestamps per frame
    mocap_frame_ids.npy
    mocap_raw.npy            — raw position X,Y,Z in metres (no transform)
    mocap_valid_mask.npy     — bool, True where rigid body was not NaN
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from datetime import datetime, timedelta

# ─── PATHS ────────────────────────────────────────────────────────────────────
MOCAP_CSV = r"H:\dual_cam_single_aprl_50mm_t0\dual_cam_single_aprl_50mm_t0.csv"
# OR if running from the directory where you saved the uploaded file:
# MOCAP_CSV = "dual_cam_single_aprl_50mm_t0.csv"

OUT_DIR = "trajectory_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── MOCAP CONSTANTS ──────────────────────────────────────────────────────────
MOCAP_FPS          = 100.0
MOCAP_START_STR    = "2026-06-23 04:31:23.721 PM"
MOCAP_START_FMT    = "%Y-%m-%d %I:%M:%S.%f %p"

# Sync window (from Step 2 sync CSV analysis — adjust if needed)
SYNC_START_STR     = None   # Set to "2026-06-23 16:XX:XX" after running Step 2
# If None, we use the full MoCap window

# ─── COLUMN INDICES (confirmed from CSV inspection) ───────────────────────────
COL_FRAME   = 0
COL_TIME    = 1
COL_ROT_X   = 2
COL_ROT_Y   = 3
COL_ROT_Z   = 4
COL_ROT_W   = 5
COL_POS_X   = 6
COL_POS_Y   = 7
COL_POS_Z   = 8
COL_ERR     = 9
# Rigid body marker columns (X, Y, Z, quality × 6)
MARKER_COLS = {
    1: (10, 11, 12),   # X, Y, Z of Marker1 (col 13 = quality)
    2: (14, 15, 16),
    3: (18, 19, 20),
    4: (22, 23, 24),
    5: (26, 27, 28),
    6: (30, 31, 32),
}

# ─── STEP 3A: Parse metadata ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 3A: Parsing MoCap CSV metadata")
print("=" * 60)

# Read row 0 as key-value pairs
meta_row = pd.read_csv(MOCAP_CSV, header=None, nrows=1, engine='python').iloc[0].tolist()
meta = {}
for i in range(0, len(meta_row) - 1, 2):
    k = meta_row[i]
    v = meta_row[i+1]
    if pd.notna(k):
        meta[k] = v

print(f"  Take Name: {meta.get('Take Name', 'unknown')}")
print(f"  Capture Start Time: {meta.get('Capture Start Time', 'unknown')}")
print(f"  Total Frames: {meta.get('Total Frames in Take', 'unknown')}")
print(f"  Capture FPS: {meta.get('Capture Frame Rate', 'unknown')}")

# Parse capture start time
capture_start_str = meta.get("Capture Start Time", MOCAP_START_STR)
# Handle format: "2026-06-23 04.31.23.721 PM" (dots instead of colons)
capture_start_str = capture_start_str.replace(".", ":", 2)  # first two dots → colons
# Now: "2026-06-23 04:31:23.721 PM" — but last segment is still milliseconds
# Fix: "2026-06-23 04:31:23.721 PM"
mocap_start_dt = datetime.strptime(capture_start_str, MOCAP_START_FMT)
print(f"  Parsed start: {mocap_start_dt}")

# ─── STEP 3B: Load data rows ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3B: Loading MoCap data")
print("=" * 60)

# Skip rows 0–5 (meta + 5 header rows), row 6 = column axis labels = another header skip
df = pd.read_csv(MOCAP_CSV, header=None, skiprows=6, engine='python',
                 on_bad_lines='skip')
df = df.apply(pd.to_numeric, errors='coerce')  # convert everything numeric

print(f"  Loaded {len(df)} frames, {df.shape[1]} columns")

# Rename key columns
df.columns = [f"c{i}" for i in range(df.shape[1])]
df = df.rename(columns={
    "c0": "frame",
    "c1": "time_s",
    "c2": "rot_x", "c3": "rot_y", "c4": "rot_z", "c5": "rot_w",
    "c6": "pos_x", "c7": "pos_y", "c8": "pos_z",
    "c9": "marker_err",
})

print(f"  Frame range: {df['frame'].min():.0f} → {df['frame'].max():.0f}")
print(f"  Time range:  {df['time_s'].min():.3f}s → {df['time_s'].max():.3f}s")

nan_frames = df["pos_x"].isna().sum()
print(f"  NaN rigid body frames: {nan_frames} / {len(df)}")

# ─── STEP 3C: Add absolute timestamps ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3C: Computing absolute timestamps")
print("=" * 60)

# Each frame k → abs_time = mocap_start + k * (1/fps)
df["abs_time"] = df["frame"].apply(
    lambda k: mocap_start_dt + timedelta(seconds=float(k) / MOCAP_FPS)
              if pd.notna(k) else pd.NaT
)
df["abs_time"] = pd.to_datetime(df["abs_time"])
print(f"  MoCap absolute start: {df['abs_time'].iloc[0]}")
print(f"  MoCap absolute end:   {df['abs_time'].iloc[-1]}")

# ─── STEP 3D: Fallback — use raw marker centroid when rigid body is NaN ───────
print("\n" + "=" * 60)
print("STEP 3D: Applying fallback centroids for NaN rigid body frames")
print("=" * 60)

pos_x = df["pos_x"].copy()
pos_y = df["pos_y"].copy()
pos_z = df["pos_z"].copy()

for i in df.index:
    if pd.isna(df.loc[i, "pos_x"]):
        # Compute centroid of all available raw markers
        xs, ys, zs = [], [], []
        for m, (cx, cy, cz) in MARKER_COLS.items():
            vx = df.loc[i, f"c{cx}"]
            vy = df.loc[i, f"c{cy}"]
            vz = df.loc[i, f"c{cz}"]
            if pd.notna(vx) and pd.notna(vy) and pd.notna(vz):
                xs.append(vx); ys.append(vy); zs.append(vz)
        if xs:
            pos_x[i] = np.mean(xs)
            pos_y[i] = np.mean(ys)
            pos_z[i] = np.mean(zs)

df["pos_x_filled"] = pos_x
df["pos_y_filled"] = pos_y
df["pos_z_filled"] = pos_z

fallback_count = df["pos_x"].isna().sum() - df["pos_x_filled"].isna().sum()
print(f"  Fallback centroid used for {fallback_count} frames")
still_nan = df["pos_x_filled"].isna().sum()
print(f"  Still NaN after fallback: {still_nan} frames")

# ─── STEP 3E: Compute MoCap basis vectors ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3E: Computing MoCap basis vectors")
print("=" * 60)

# T_ref = position at frame 0 (MoCap start = sync start)
ref_row = df.iloc[0]
T_ref = np.array([ref_row["pos_x_filled"], ref_row["pos_y_filled"], ref_row["pos_z_filled"]])

# R_ref = rotation at frame 0
qx, qy, qz, qw = ref_row["rot_x"], ref_row["rot_y"], ref_row["rot_z"], ref_row["rot_w"]
if not any(pd.isna([qx, qy, qz, qw])):
    R_ref = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: [x,y,z,w]
else:
    R_ref = np.eye(3)
    print("  WARNING: R_ref quaternion is NaN at frame 0 — using identity rotation")

print(f"  T_ref (m): {T_ref}")
print(f"  T_ref (mm): {T_ref * 1000}")

basis_list  = []
valid_mask  = []
frame_ids   = []
timestamps  = []
raw_pos     = []

for i, row in df.iterrows():
    fid = row["frame"]
    frame_ids.append(fid)
    timestamps.append(row["abs_time"])

    px = row["pos_x_filled"]
    py = row["pos_y_filled"]
    pz = row["pos_z_filled"]
    raw_pos.append([px, py, pz])

    qx = row["rot_x"]; qy = row["rot_y"]
    qz = row["rot_z"]; qw = row["rot_w"]

    if pd.isna(px) or pd.isna(py) or pd.isna(pz):
        basis_list.append([np.nan, np.nan, np.nan])
        valid_mask.append(False)
        continue

    T = np.array([px, py, pz])

    # Rotation: use R_ref (reference frame rotation for all frames, as per system doc)
    # basis = R_ref^T @ (T_ref - T) — positions in metres → convert to mm
    basis_m = R_ref.T @ (T_ref - T)
    basis_mm = basis_m * 1000.0   # metres → mm

    basis_list.append(basis_mm)
    valid_mask.append(True)

mocap_basis = np.array(basis_list)
mocap_valid = np.array(valid_mask)
mocap_fids  = np.array(frame_ids)
mocap_ts    = np.array(timestamps, dtype=object)
mocap_raw   = np.array(raw_pos)

print(f"\n  Valid MoCap frames: {mocap_valid.sum()} / {len(mocap_valid)}")
if mocap_valid.sum() > 0:
    vb = mocap_basis[mocap_valid]
    print(f"  Basis X range: {np.nanmin(vb[:,0]):.2f} → {np.nanmax(vb[:,0]):.2f} mm")
    print(f"  Basis Y range: {np.nanmin(vb[:,1]):.2f} → {np.nanmax(vb[:,1]):.2f} mm")
    print(f"  Basis Z range: {np.nanmin(vb[:,2]):.2f} → {np.nanmax(vb[:,2]):.2f} mm")

# ─── STEP 3F: Save ────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "mocap_trajectory.npy"),  mocap_basis)
np.save(os.path.join(OUT_DIR, "mocap_timestamps.npy"),  mocap_ts)
np.save(os.path.join(OUT_DIR, "mocap_frame_ids.npy"),   mocap_fids)
np.save(os.path.join(OUT_DIR, "mocap_valid_mask.npy"),  mocap_valid)
np.save(os.path.join(OUT_DIR, "mocap_raw.npy"),         mocap_raw)

print(f"\n✓ Saved mocap_trajectory.npy, mocap_timestamps.npy, mocap_valid_mask.npy")
print("\nSTEP 3 COMPLETE.")
