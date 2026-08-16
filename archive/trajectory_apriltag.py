# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-contrib-python>=4.13.0.92", "matplotlib>=3.11.0"]
# ///
"""Mono AprilTag trajectory: estimate tag pose per camera via solvePnP.

Detects AprilTag (36h11) of a chosen id in each raw frame, solves its pose
independently in the LEFT and RIGHT cameras using intrinsics from
camera_calibration_full.npz, and plots x/y/z position over time per camera.

Pose = tag-center translation (tvec) in the respective camera frame, in mm.
"""
import argparse
import pickle

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALIB = "data/camera_calibration_full.npz"
RAW = "data/raw_frames.pkl"
TAG_ID = 14
TAG_MM = 50.0
DICT = cv2.aruco.DICT_APRILTAG_36H11


def tag_object_points(size_mm):
    """4 corners of a planar square tag centred at origin (TL,TR,BR,BL), z=0."""
    h = size_mm / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                    dtype=np.float32)


def detect_id(detector, gray, tag_id):
    """Return Nx2 corners of the requested tag id, or None."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.ravel()
    hits = np.where(ids == tag_id)[0]
    if not len(hits):
        return None
    return corners[hits[0]].reshape(4, 2).astype(np.float32)


def track_camera(name, frames, frame_key, K, dist, objp, detector, tag_id):
    """Estimate per-frame tag pose for one camera."""
    rows = []
    for fr in frames:
        gray = cv2.cvtColor(fr[frame_key], cv2.COLOR_BGR2GRAY)
        img_corners = detect_id(detector, gray, tag_id)
        if img_corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            objp, img_corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        x, y, z = tvec.ravel()
        rows.append((fr["frame_num"], fr["timestamp"], x, y, z))
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=np.float64)
    print(f"[{name}] tag {tag_id} poses: {len(arr)} / {len(frames)} frames")
    return arr


def plot_camera(name, arr, tag_id, out_path):
    """3 stacked subplots: x, y, z vs elapsed time."""
    if not len(arr):
        print(f"[{name}] no poses, skipping plot")
        return
    t = arr[:, 1] - arr[:, 1].min()
    labels = ("X (mm)", "Y (mm)", "Z (mm) depth")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col, lab, c in zip(axes, (2, 3, 4), labels, colors):
        ax.plot(t, arr[:, col], ".-", color=c, lw=1, ms=3)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"AprilTag {tag_id} trajectory — {name} camera (mono solvePnP)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[{name}] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-id", type=int, default=TAG_ID)
    ap.add_argument("--tag-mm", type=float, default=TAG_MM)
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()

    c = np.load(args.calib)
    KL, dL = c["cam_matrix_L"], c["dist_coeffs_L"]
    KR, dR = c["cam_matrix_R"], c["dist_coeffs_R"]

    with open(args.raw, "rb") as fh:
        frames = pickle.load(fh)
    print(f"loaded {len(frames)} frames | tag id={args.tag_id} | size={args.tag_mm}mm")

    objp = tag_object_points(args.tag_mm)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())

    left = track_camera("LEFT", frames, "frameL", KL, dL, objp, detector, args.tag_id)
    right = track_camera("RIGHT", frames, "frameR", KR, dR, objp, detector, args.tag_id)

    plot_camera("LEFT", left, args.tag_id, f"data/tag{args.tag_id}_traj_left.png")
    plot_camera("RIGHT", right, args.tag_id, f"data/tag{args.tag_id}_traj_right.png")

    hdr = "frame_num,timestamp,x_mm,y_mm,z_mm"
    np.savetxt(f"data/tag{args.tag_id}_traj_left.csv", left, delimiter=",",
               header=hdr, comments="", fmt=["%d", "%.6f", "%.4f", "%.4f", "%.4f"])
    np.savetxt(f"data/tag{args.tag_id}_traj_right.csv", right, delimiter=",",
               header=hdr, comments="", fmt=["%d", "%.6f", "%.4f", "%.4f", "%.4f"])
    print("saved CSVs")


if __name__ == "__main__":
    main()
