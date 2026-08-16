"""
mocap_match_stereo_FINAL.py

Run with:  py -3.11 mocap_match_stereo_FINAL.py

Inputs (edit ONLY these two paths if yours differ):
  - test5.csv              -> raw OptiTrack/Motive export (meters, 100 Hz)
  - tag14_traj_stereo.csv  -> your stereo triangulation output (mm)

Outputs (saved to OUT_DIR):
  - mocap_raw_standalone.png      -> full mocap capture, incl. static lead-in/out
  - mocap_matched_to_stereo.png   -> aligned mocap overlaid on stereo, with RMSE
  - mocap_standalone_matched.png  -> aligned mocap only (X/Y/Z vs time)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations, product

# ----------------------------- PATHS (edit if needed) -----------------------
RAW_MOCAP_CSV   = r"C:\Users\arya0\OneDrive\Desktop\test5.csv"
STEREO_CSV      = r"C:\Users\arya0\Desktop\tag14_traj_stereo.csv"
OUT_DIR         = r"C:\Users\arya0\Desktop"
# -----------------------------------------------------------------------------

# ----------------------------- KNOWN FILE LAYOUT -----------------------------
# test5.csv: 6 metadata rows, header on row 7 -> after skiprows=6 the position
# columns for the "apriltag" rigid body land on X.1 / Y.1 / Z.1 (meters).
MOCAP_FRAME_COL = "Frame"
MOCAP_TIME_COL  = "Time (Seconds)"
MOCAP_X_COL     = "X.1"
MOCAP_Y_COL     = "Y.1"
MOCAP_Z_COL     = "Z.1"
MOCAP_UNITS_TO_MM = 1000.0   # meters -> mm

MOTION_FRAME_MIN = 1500
MOTION_FRAME_MAX = 5500

# tag14_traj_stereo.csv columns (mm, epoch timestamp)
STEREO_FRAME_COL = "frame_num"
STEREO_TIME_COL  = "timestamp"   # epoch seconds, will be zeroed
STEREO_X_COL     = "x_mm"
STEREO_Y_COL     = "y_mm"
STEREO_Z_COL     = "z_mm"
# -----------------------------------------------------------------------------


def umeyama(src, dst, with_scale=True):
    """Similarity transform (R, c, t) minimizing ||dst - (c*R*src + t)||^2."""
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        c = np.trace(np.diag(D) @ S) / var_src
    else:
        c = 1.0

    t = mu_dst - c * R @ mu_src
    return R, c, t


def cross_corr_lag(t_ref, x_ref, t_query, x_query, max_lag_s=10.0, dt=0.01):
    """Lag (s) to shift x_query so it best aligns with x_ref."""
    grid = np.arange(min(t_ref.min(), t_query.min()),
                      max(t_ref.max(), t_query.max()), dt)
    ref_i = np.interp(grid, t_ref, x_ref, left=np.nan, right=np.nan)
    q_i = np.interp(grid, t_query, x_query, left=np.nan, right=np.nan)

    valid = ~np.isnan(ref_i) & ~np.isnan(q_i)
    ref_i = np.where(valid, ref_i, 0.0) - np.nanmean(ref_i[valid]) if valid.any() else ref_i
    q_i = np.where(valid, q_i, 0.0) - np.nanmean(q_i[valid]) if valid.any() else q_i

    corr = np.correlate(ref_i, q_i, mode="full")
    lags = (np.arange(-len(q_i) + 1, len(q_i))) * dt
    keep = np.abs(lags) <= max_lag_s
    best_idx = np.argmax(corr[keep])
    best_lag = lags[keep][best_idx]
    denom = (np.std(ref_i) * np.std(q_i) * len(ref_i) + 1e-9)
    best_corr = corr[keep][best_idx] / denom
    return best_lag, best_corr


def best_axis_permutation(mocap_xyz, stereo_xyz):
    """Try all 6 permutations x 8 signs (48 combos), Umeyama-fit each, keep lowest RMSE."""
    best = None
    for perm in permutations([0, 1, 2]):
        for signs in product([1, -1], repeat=3):
            permuted = mocap_xyz[:, perm] * np.array(signs)
            R, c, t = umeyama(permuted, stereo_xyz, with_scale=True)
            aligned = (c * (permuted @ R.T)) + t
            rmse = np.sqrt(np.mean(np.sum((aligned - stereo_xyz) ** 2, axis=1)))
            if best is None or rmse < best["rmse"]:
                best = dict(perm=perm, signs=signs, R=R, c=c, t=t,
                            rmse=rmse, aligned=aligned)
    return best


def main():
    # ---------- 1. Load raw mocap, full capture ----------
    df_raw = pd.read_csv(RAW_MOCAP_CSV, skiprows=6, header=0)
    df_raw.columns = [str(col).strip() for col in df_raw.columns]
    t_raw = df_raw[MOCAP_TIME_COL].to_numpy(dtype=float)
    t_raw = t_raw - t_raw[0]

    panel_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    colors = ["r", "g", "b"]
    raw_xyz_mm = df_raw[[MOCAP_X_COL, MOCAP_Y_COL, MOCAP_Z_COL]].to_numpy(dtype=float) * MOCAP_UNITS_TO_MM

    plt.figure(figsize=(9, 7))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_raw, raw_xyz_mm[:, i], color=colors[i])
        plt.ylabel(panel_labels[i])
        if i == 0:
            plt.title("MoCap raw trajectory (full capture, incl. static lead-in/out)")
    plt.xlabel("time (s)")
    plt.tight_layout()
    raw_out = f"{OUT_DIR}\\mocap_raw_standalone.png"
    plt.savefig(raw_out, dpi=150)
    plt.close()
    print(f"Saved -> {raw_out}")

    # ---------- 2. Crop to motion window ----------
    mask = (df_raw[MOCAP_FRAME_COL] >= MOTION_FRAME_MIN) & (df_raw[MOCAP_FRAME_COL] <= MOTION_FRAME_MAX)
    df_crop = df_raw[mask].reset_index(drop=True)
    t_crop = df_crop[MOCAP_TIME_COL].to_numpy(dtype=float)
    t_crop = t_crop - t_crop[0]
    mocap_xyz = df_crop[[MOCAP_X_COL, MOCAP_Y_COL, MOCAP_Z_COL]].to_numpy(dtype=float) * MOCAP_UNITS_TO_MM
    print(f"Cropped mocap motion window: {len(df_crop)} frames "
          f"({t_crop[0]:.2f}s to {t_crop[-1]:.2f}s)")

    # ---------- 3. Load stereo trajectory ----------
    df_stereo = pd.read_csv(STEREO_CSV)
    df_stereo.columns = [str(col).strip() for col in df_stereo.columns]
    t_stereo = df_stereo[STEREO_TIME_COL].to_numpy(dtype=float)
    t_stereo = t_stereo - t_stereo[0]
    stereo_xyz = df_stereo[[STEREO_X_COL, STEREO_Y_COL, STEREO_Z_COL]].to_numpy(dtype=float)
    print(f"Stereo trajectory: {len(df_stereo)} frames "
          f"({t_stereo[0]:.2f}s to {t_stereo[-1]:.2f}s)")

    # ---------- 4. Time alignment via SPEED cross-correlation ----------
    # Raw axis values can't be safely cross-correlated yet because we don't
    # know the axis permutation/sign. Motion *speed* (magnitude of velocity)
    # is invariant to axis choice/sign, so align on that instead - much more
    # robust than guessing an axis first.
    def speed(t, xyz):
        dt_ = np.diff(t)
        dt_[dt_ == 0] = 1e-6
        vel = np.diff(xyz, axis=0) / dt_[:, None]
        spd = np.linalg.norm(vel, axis=1)
        t_mid = (t[1:] + t[:-1]) / 2.0
        return t_mid, spd

    t_stereo_mid, speed_stereo = speed(t_stereo, stereo_xyz)
    t_crop_mid, speed_mocap = speed(t_crop, mocap_xyz)

    lag, corr_val = cross_corr_lag(t_stereo_mid, speed_stereo, t_crop_mid, speed_mocap,
                                    max_lag_s=40.0, dt=0.01)
    print(f"Time lag found: {lag:.3f} s (corr={corr_val:.2f}) via speed-profile cross-correlation")
    t_crop_aligned = t_crop + lag

    # ---------- 5. Overlap & resample stereo onto mocap-aligned timestamps ----------
    valid = (t_crop_aligned >= t_stereo.min()) & (t_crop_aligned <= t_stereo.max())
    common_t = t_crop_aligned[valid]
    if len(common_t) < 10:
        raise RuntimeError("Not enough time overlap after lag correction — "
                            "check frame rates / motion window / lag search range.")

    stereo_resampled = np.column_stack([
        np.interp(common_t, t_stereo, stereo_xyz[:, 0]),
        np.interp(common_t, t_stereo, stereo_xyz[:, 1]),
        np.interp(common_t, t_stereo, stereo_xyz[:, 2]),
    ])
    mocap_for_fit = mocap_xyz[valid]
    print(f"Overlapping samples used for fit: {len(common_t)}")

    # ---------- 6. Search best axis permutation/sign + similarity fit ----------
    best = best_axis_permutation(mocap_for_fit, stereo_resampled)
    aligned_mocap = best["aligned"]
    diff = aligned_mocap - stereo_resampled
    rmse_x = np.sqrt(np.mean(diff[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(diff[:, 1] ** 2))
    rmse_z = np.sqrt(np.mean(diff[:, 2] ** 2))

    print("\n--- BEST ALIGNMENT ---")
    print(f"Axis permutation : {best['perm']}")
    print(f"Signs            : {best['signs']}")
    print(f"Scale factor      : {best['c']:.4f}")
    print(f"RMSE 3D           : {best['rmse']:.2f} mm")
    print(f"RMSE X/Y/Z        : {rmse_x:.2f} / {rmse_y:.2f} / {rmse_z:.2f} mm")

    # ---------- 7. Plot matched mocap vs stereo ----------
    tt = common_t - common_t[0]
    plt.figure(figsize=(9, 7))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(tt, stereo_resampled[:, i], color="k", linewidth=1.5, label="Stereo")
        plt.plot(tt, aligned_mocap[:, i], color="m", linestyle="--", linewidth=1.2, label="MoCap (aligned)")
        plt.ylabel(panel_labels[i])
        if i == 0:
            plt.title(f"MoCap aligned to Stereo | RMSE 3D = {best['rmse']:.1f} mm")
            plt.legend(loc="upper right", fontsize=8)
    plt.xlabel("time (s)")
    plt.tight_layout()
    matched_out = f"{OUT_DIR}\\mocap_matched_to_stereo.png"
    plt.savefig(matched_out, dpi=150)
    plt.close()
    print(f"Saved -> {matched_out}")

    # ---------- 8. Standalone aligned mocap-only plot ----------
    plt.figure(figsize=(9, 7))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(tt, aligned_mocap[:, i], color=colors[i])
        plt.ylabel(panel_labels[i])
        if i == 0:
            plt.title("MoCap trajectory (cropped, time-aligned, axis-matched, scaled)")
    plt.xlabel("time (s)")
    plt.tight_layout()
    standalone_out = f"{OUT_DIR}\\mocap_standalone_matched.png"
    plt.savefig(standalone_out, dpi=150)
    plt.close()
    print(f"Saved -> {standalone_out}")


if __name__ == "__main__":
    main()
