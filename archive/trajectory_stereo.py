# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-contrib-python>=4.13.0.92", "matplotlib>=3.11.0"]
# ///
"""Stereo AprilTag trajectory by triangulating the tag CENTRE.

For each frame where AprilTag (36h11) of the chosen id is seen in BOTH cameras,
the centre = mean of the 4 corners is taken in each image. The two centres
(u1,v1) and (u2,v2) are undistorted and triangulated with the stereo extrinsics
(R, T from camera_calibration_stereo.npz) to recover one metric 3D point in the
LEFT-camera frame. Plots x/y/z over time plus the 3D path.
"""
import argparse
import pickle

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

STEREO = "data/camera_calibration_stereo.npz"
RAW = "data/raw_frames.pkl"
TAG_ID = 14
DICT = cv2.aruco.DICT_APRILTAG_36H11


def tag_center(detector, gray, tag_id):
    """Mean of the 4 corners of the requested tag id, as (1,1,2) float32; else None."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.ravel()
    hits = np.where(ids == tag_id)[0]
    if not len(hits):
        return None
    c = corners[hits[0]].reshape(4, 2).mean(axis=0)
    return c.reshape(1, 1, 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-id", type=int, default=TAG_ID)
    ap.add_argument("--stereo", default=STEREO)
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()

    s = np.load(args.stereo)
    KL, dL = s["cam_matrix_L"], s["dist_coeffs_L"]
    KR, dR = s["cam_matrix_R"], s["dist_coeffs_R"]
    R, T = s["R"], s["T"]

    # Projection matrices in the LEFT-camera frame (metric units = those of T, metres).
    P_L = KL @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_R = KR @ np.hstack([R, T.reshape(3, 1)])

    with open(args.raw, "rb") as fh:
        frames = pickle.load(fh)
    print(f"loaded {len(frames)} frames | tag id={args.tag_id}")

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())

    rows = []
    for fr in frames:
        gL = cv2.cvtColor(fr["frameL"], cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fr["frameR"], cv2.COLOR_BGR2GRAY)
        cL = tag_center(detector, gL, args.tag_id)
        cR = tag_center(detector, gR, args.tag_id)
        if cL is None or cR is None:
            continue
        # Undistort centres back to pixel coords matching K[I|0] / K[R|T].
        uL = cv2.undistortPoints(cL, KL, dL, P=KL).reshape(2, 1)
        uR = cv2.undistortPoints(cR, KR, dR, P=KR).reshape(2, 1)
        Xh = cv2.triangulatePoints(P_L, P_R, uL, uR)
        X = (Xh[:3] / Xh[3]).ravel() * 1000.0  # metres -> mm
        rows.append((fr["frame_num"], fr["timestamp"], X[0], X[1], X[2]))

    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=np.float64)
    print(f"triangulated poses (both cams): {len(arr)} / {len(frames)} frames")

    np.savetxt(f"data/tag{args.tag_id}_traj_stereo.csv", arr, delimiter=",",
               header="frame_num,timestamp,x_mm,y_mm,z_mm", comments="",
               fmt=["%d", "%.6f", "%.4f", "%.4f", "%.4f"])

    # x/y/z vs time
    t = arr[:, 1] - arr[:, 1].min()
    labels = ("X (mm)", "Y (mm)", "Z (mm) depth")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col, lab, c in zip(axes, (2, 3, 4), labels, colors):
        ax.plot(t, arr[:, col], ".-", color=c, lw=1, ms=3)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"AprilTag {args.tag_id} stereo trajectory (triangulated centre, left-cam frame)")
    fig.tight_layout()
    fig.savefig(f"data/tag{args.tag_id}_traj_stereo.png", dpi=130)
    plt.close(fig)

    # 3D path
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(arr[:, 2], arr[:, 3], arr[:, 4], "-", color="#444", lw=0.8)
    p = ax.scatter(arr[:, 2], arr[:, 3], arr[:, 4], c=t, cmap="viridis", s=8)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    fig.colorbar(p, ax=ax, label="time (s)", shrink=0.6)
    ax.set_title(f"AprilTag {args.tag_id} 3D path (stereo)")
    fig.tight_layout()
    fig.savefig(f"data/tag{args.tag_id}_traj_stereo_3d.png", dpi=130)
    plt.close(fig)

    print(f"saved -> data/tag{args.tag_id}_traj_stereo.png / _3d.png / .csv")


if __name__ == "__main__":
    main()
