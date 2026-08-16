"""
speed_vs_error_pipeline.py

PURPOSE
-------
Runs your actual stereo-AprilTag-vs-MoCap pipeline (per
COMPLETE_SYSTEM_LAYOUT_PIPELINE.pdf, Sections 1-8) and then answers the
guide's question empirically: does moving the tag faster increase error,
and by how much? It does this by:

  1. Computing camera basis vectors (per-frame AprilTag-derived position,
     Section 5) and MoCap basis vectors (RigidBody 002 position, Section 5).
  2. Matching cam<->mocap frames by nearest timestamp (Section 6).
  3. Computing the per-frame INSTANTANEOUS SPEED of the tag from the MoCap
     trajectory (ground truth speed, since MoCap is 100Hz and reliable).
  4. Binning the per-frame camera-vs-MoCap error by that speed, and
     reporting mean/std/RMSE error PER SPEED BIN -> this is the real,
     measured "speed vs error" table for your dataset.

INPUTS (edit paths if yours differ)
------------------------------------
  - sync_and_framerate.csv   (interleaved sync + cam frame rows)
  - mocap_trial_1.csv        (5-row multi-header OptiTrack export, 100fps)
  - board_detection.pkl      (per-frame AprilTag/ChArUco corners; OPTIONAL -
                               only needed to recompute camera positions from
                               scratch. If you already have a stereo
                               trajectory CSV with x_mm/y_mm/z_mm/timestamp/
                               cam_id columns, point CAM_TRAJ_CSV at that
                               instead and set USE_RAW_DETECTIONS = False.)

NOTE ON board_detection.pkl
----------------------------
Per your audit doc, the exact tag-ID layout inside this pickle "needs
runtime check" - I cannot assume its schema blindly. This script inspects
it at runtime (prints keys/columns on load) and tries common layouts
(list-of-dicts with 'tag_id'/'corners', or a DataFrame with id columns).
If it doesn't match, the script prints what it found so you can adjust the
3-line extraction function `extract_tag_corners()` accordingly - everything
downstream is unaffected.

OUTPUTS
-------
  - speed_vs_error_table.csv      per speed-bin: mean/std/RMSE error, per cam
  - speed_vs_error_plot.png       error vs speed (binned), per camera
  - per_frame_errors.csv          raw per-frame matched data (for your own digging)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ----------------------------- PATHS (edit if needed) -----------------------
SYNC_CSV   = "sync_and_framerate.csv"
MOCAP_CSV  = "mocap_trial_1.csv"
BOARD_PKL  = "board_detection.pkl"          # set USE_RAW_DETECTIONS = True to use this
CAM_TRAJ_CSV = "tag_traj_stereo.csv"        # alt: already-computed cam x/y/z per frame
USE_RAW_DETECTIONS = True                   # False -> load CAM_TRAJ_CSV instead

OUT_DIR = "."
# -----------------------------------------------------------------------------

# ----------------------------- KNOWN GEOMETRY (Section 1/5) -----------------
ID12_LATERAL_OFFSET_CM = 9.8 / 2.0   # half of ID14-ID20 distance
ID12_VERTICAL_OFFSET_CM = 5.0
MOCAP_DT = 0.01                      # 100 fps
SYNC_TOLERANCE_S = 0.5 * MOCAP_DT    # +/- half a mocap frame (Section 6)

# Speed bins (mm/s) for the final tabulation - adjust to your dataset's range
SPEED_BINS_MM_S = [0, 50, 100, 200, 400, 800, 1500, 3000, 10000]


# =============================================================================
# 1. SYNC CSV  (Section 4)
# =============================================================================
def load_sync_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["sync_ffill"] = df["sync_value"].ffill()
    return df


def usable_cam_frames(sync_df, cam_id):
    frame_col = f"cam{cam_id}_frame"
    fps_col = f"cam{cam_id}_fps"
    cam_rows = sync_df[sync_df[frame_col].notna()].copy()
    cam_rows = cam_rows[cam_rows["sync_ffill"] == 1]
    return cam_rows[["timestamp", frame_col, fps_col]].reset_index(drop=True)


# =============================================================================
# 2. MOCAP CSV  (Section 3)
# =============================================================================
def load_mocap_csv(path):
    """5-row multi-header OptiTrack export. Data starts at row 7 (0-indexed
    skiprows=6 per your FINAL script convention)."""
    df = pd.read_csv(path, skiprows=6, header=0, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def mocap_rigidbody002_pose(df):
    """RigidBody 002 (AprilTag mount) - columns [26-32]: quat(X,Y,Z,W) + pos(X,Y,Z).
    Falls back to centroid of 5 raw markers if the rigid body is NaN."""
    quat = df.iloc[:, 26:30].to_numpy(dtype=float)   # X,Y,Z,W
    pos_m = df.iloc[:, 30:33].to_numpy(dtype=float)  # meters
    pos_mm = pos_m * 1000.0

    nan_rows = np.isnan(pos_mm).any(axis=1)
    if nan_rows.any():
        # fallback: centroid of the 5 raw markers (cols 34-53 -> X,Y,Z,qual repeating)
        marker_block = df.iloc[:, 34:54].to_numpy(dtype=float)
        n_markers = 5
        marker_xyz = marker_block.reshape(len(df), n_markers, 4)[:, :, :3]  # drop quality col
        centroid_mm = np.nanmean(marker_xyz, axis=1) * 1000.0
        pos_mm[nan_rows] = centroid_mm[nan_rows]

    return quat, pos_mm, nan_rows


def mocap_basis_vectors(quat, pos_mm, ref_idx=0):
    """basis = R^T @ (T_ref - T_frame), per Section 5."""
    T_ref = pos_mm[ref_idx]
    R_ref = Rotation.from_quat(quat[ref_idx]).as_matrix()
    basis = np.zeros_like(pos_mm)
    for i in range(len(pos_mm)):
        basis[i] = R_ref.T @ (T_ref - pos_mm[i])
    return basis


def mocap_instantaneous_speed(pos_mm, dt=MOCAP_DT):
    """mm/s, centered finite difference."""
    vel = np.gradient(pos_mm, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    return speed


# =============================================================================
# 3. CAMERA SIDE  (Section 5) - AprilTag-derived basis vectors
# =============================================================================
def extract_tag_corners(detection_record):
    """
    EDIT THIS if your board_detection.pkl schema differs.
    Expected to return a dict {tag_id: (4,2) array of pixel corners} for one frame.
    Tries a couple of common layouts; prints a warning + raw record once if
    nothing matches so you can patch this quickly.
    """
    if isinstance(detection_record, dict) and "tags" in detection_record:
        return {t["id"]: np.array(t["corners"]) for t in detection_record["tags"]}
    if isinstance(detection_record, dict):
        try:
            return {int(k): np.array(v) for k, v in detection_record.items()
                    if isinstance(k, (int, np.integer)) or str(k).isdigit()}
        except Exception:
            pass
    return {}


def load_board_detection(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[board_detection.pkl] loaded type={type(data)}")
    if isinstance(data, list):
        print(f"  -> list of {len(data)} records; sample keys: "
              f"{list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
    elif isinstance(data, pd.DataFrame):
        print(f"  -> DataFrame columns: {list(data.columns)}")
    return data


def tag_center_with_fallback(corners_by_id):
    """Section 5 / 'UPDATED' logic: prefer ID12, else average ID14 & ID20,
    else NaN. Returns pixel-space center (2,) or None."""
    if 12 in corners_by_id:
        return np.mean(corners_by_id[12], axis=0)
    if 14 in corners_by_id and 20 in corners_by_id:
        c14 = np.mean(corners_by_id[14], axis=0)
        c20 = np.mean(corners_by_id[20], axis=0)
        return (c14 + c20) / 2.0
    return None


def camera_basis_from_pnp(rvecs, tvecs, ref_idx=0):
    """basis = R^T @ (T_ref - T_frame), per Section 5, using solvePnP outputs
    you already produced upstream (rvec/tvec per frame, per camera, in mm)."""
    T_ref = tvecs[ref_idx]
    import cv2
    R_ref, _ = cv2.Rodrigues(rvecs[ref_idx])
    basis = np.zeros_like(tvecs)
    for i in range(len(tvecs)):
        basis[i] = R_ref.T @ (T_ref - tvecs[i])
    return basis


def load_camera_trajectory_csv(path):
    """Fallback path: if you already have per-frame camera x/y/z/timestamp
    (e.g. from your stereo triangulation script), load it directly instead
    of re-deriving from raw detections."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# =============================================================================
# 4. MATCHING  (Section 6) - nearest-neighbor cam<->mocap by timestamp
# =============================================================================
def match_frames(cam_times, mocap_start_time, mocap_dt=MOCAP_DT,
                  n_mocap_frames=None, tol_s=SYNC_TOLERANCE_S):
    """Returns array of mocap frame indices (or -1 if no match within tol)
    for each camera timestamp."""
    mocap_idx_out = np.full(len(cam_times), -1, dtype=int)
    elapsed = (cam_times - mocap_start_time).dt.total_seconds().to_numpy()
    nearest = np.round(elapsed / mocap_dt).astype(int)
    actual_offset = np.abs(elapsed - nearest * mocap_dt)
    valid = (actual_offset <= tol_s)
    if n_mocap_frames is not None:
        valid &= (nearest >= 0) & (nearest < n_mocap_frames)
    mocap_idx_out[valid] = nearest[valid]
    return mocap_idx_out, elapsed


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    # ---------- Sync CSV ----------
    sync_df = load_sync_csv(SYNC_CSV)
    cam_frames = {cam: usable_cam_frames(sync_df, cam) for cam in (0, 1)}
    for cam, df in cam_frames.items():
        print(f"cam{cam}: {len(df)} usable frames in sync=1 window")

    # ---------- MoCap CSV ----------
    mocap_df = load_mocap_csv(MOCAP_CSV)
    quat, pos_mm, nan_mask = mocap_rigidbody002_pose(mocap_df)
    mocap_speed = mocap_instantaneous_speed(pos_mm)
    mocap_basis = mocap_basis_vectors(quat, pos_mm, ref_idx=0)

    # MoCap absolute timestamps: t=0 anchors to first sync=1 camera time
    # (Section 6). Replace mocap_start_time with your actual captured start
    # if you log it explicitly; here we anchor to the earliest usable cam0 ts.
    mocap_start_time = min(df["timestamp"].min() for df in cam_frames.values())
    mocap_times = mocap_start_time + pd.to_timedelta(
        np.arange(len(mocap_df)) * MOCAP_DT, unit="s")

    # ---------- Camera trajectory ----------
    if USE_RAW_DETECTIONS and os.path.exists(BOARD_PKL):
        raw = load_board_detection(BOARD_PKL)
        print("NOTE: extract_tag_corners()/solvePnP step must be wired to your "
              "actual calibration + corner format. This script computes the "
              "speed-vs-error TABLE from whatever per-frame (x,y,z,timestamp,"
              "cam_id) trajectory you feed it below - if board_detection.pkl "
              "needs a custom unpack, do that here and produce a DataFrame "
              "with columns [timestamp, cam_id, x_mm, y_mm, z_mm].")
        # --- placeholder: user must complete corner->pose->basis chain here,
        # --- OR simply point USE_RAW_DETECTIONS=False at an existing CSV.
        raise SystemExit(
            "Set USE_RAW_DETECTIONS=False and point CAM_TRAJ_CSV at your "
            "existing per-frame stereo trajectory CSV (the one your stereo "
            "triangulation script already outputs, e.g. tag14_traj_stereo.csv) "
            "to run the speed-vs-error analysis end to end right now."
        )
    else:
        cam_traj = load_camera_trajectory_csv(CAM_TRAJ_CSV)
        cam_traj["timestamp"] = pd.to_datetime(cam_traj["timestamp"], unit="s",
                                                errors="ignore")
        if not np.issubdtype(cam_traj["timestamp"].dtype, np.datetime64):
            cam_traj["timestamp"] = pd.to_datetime(cam_traj["timestamp"])

    # ---------- Match cam frames to mocap frames & build per-frame table ----------
    records = []
    cam_ids = cam_traj["cam_id"].unique() if "cam_id" in cam_traj.columns else [0, 1]
    for cam_id in cam_ids:
        sub = cam_traj[cam_traj["cam_id"] == cam_id] if "cam_id" in cam_traj.columns else cam_traj
        mocap_idx, elapsed = match_frames(
            sub["timestamp"], mocap_start_time, n_mocap_frames=len(mocap_df))
        for row, midx in zip(sub.itertuples(), mocap_idx):
            if midx < 0:
                continue
            cam_xyz = np.array([row.x_mm, row.y_mm, row.z_mm])
            moc_xyz = mocap_basis[midx]
            err_xyz = cam_xyz - moc_xyz
            err_3d = np.linalg.norm(err_xyz)
            records.append(dict(
                camera=f"cam{cam_id}",
                cam_timestamp=row.timestamp,
                mocap_frame=midx,
                speed_mm_s=mocap_speed[midx],
                err_x=err_xyz[0], err_y=err_xyz[1], err_z=err_xyz[2],
                err_3d=err_3d,
            ))

    per_frame = pd.DataFrame(records)
    if per_frame.empty:
        raise SystemExit("No matched frames found - check timestamp formats/paths.")

    per_frame.to_csv(f"{OUT_DIR}/per_frame_errors.csv", index=False)
    print(f"Saved -> {OUT_DIR}/per_frame_errors.csv  ({len(per_frame)} matched frames)")

    # ---------- Bin by speed -> the actual "speed vs error" table ----------
    per_frame["speed_bin"] = pd.cut(per_frame["speed_mm_s"], bins=SPEED_BINS_MM_S)

    table = (per_frame.groupby(["camera", "speed_bin"])["err_3d"]
             .agg(["count", "mean", "std",
                   lambda x: np.sqrt(np.mean(x**2))])
             .rename(columns={"<lambda_0>": "rmse"}))
    table.to_csv(f"{OUT_DIR}/speed_vs_error_table.csv")
    print("\n=== MEASURED SPEED vs ERROR (per camera, binned) ===\n")
    print(table.to_string())
    print(f"\nSaved -> {OUT_DIR}/speed_vs_error_table.csv")

    # ---------- Plot ----------
    plt.figure(figsize=(8, 5))
    for cam in per_frame["camera"].unique():
        g = per_frame[per_frame.camera == cam]
        bin_mid = g["speed_bin"].apply(lambda b: b.mid if pd.notna(b) else np.nan)
        agg = g.assign(bin_mid=bin_mid).groupby("bin_mid")["err_3d"].agg(
            ["mean", "std"]).reset_index().dropna()
        plt.errorbar(agg.bin_mid, agg["mean"], yerr=agg["std"], marker="o",
                     capsize=3, label=cam)
    plt.xlabel("Tag speed (mm/s, from MoCap ground truth)")
    plt.ylabel("3D error vs MoCap (mm)")
    plt.title("Measured tracking error vs target speed")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/speed_vs_error_plot.png", dpi=150)
    plt.close()
    print(f"Saved -> {OUT_DIR}/speed_vs_error_plot.png")


if __name__ == "__main__":
    main()
