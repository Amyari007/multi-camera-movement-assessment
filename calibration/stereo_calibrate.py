# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-contrib-python>=4.13.0.92"]
# ///
"""Stereo calibration reusing fixed mono intrinsics.

Loads the mono intrinsics produced by calibrate_cameras.py
(camera_calibration_separate.npz), picks a few random frames where BOTH cameras
detected the 12x8 / 30 mm checkerboard, and runs cv2.stereoCalibrate with
CALIB_FIX_INTRINSIC to solve only the relative pose (R, T). Then rectifies and
saves extrinsics + projection matrices, matching camera_calibration_full.npz.
"""
import argparse
import pickle

import cv2
import numpy as np

PKL = "data/calibration_captured_data.pkl"
MONO = "data/camera_calibration_separate.npz"
OUT = "data/camera_calibration_stereo.npz"
PATTERN = (12, 8)
SQUARE_M = 0.03
N_FRAMES = 40
SEED = 7


def build_objp():
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    objp *= SQUARE_M
    return objp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=PKL)
    ap.add_argument("--mono", default=MONO)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=N_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    m = np.load(args.mono)
    KL, dL = m["cam_matrix_L"], m["dist_coeffs_L"]
    KR, dR = m["cam_matrix_R"], m["dist_coeffs_R"]
    img_size = tuple(int(v) for v in m["img_size"])  # (w, h)
    print(f"mono intrinsics loaded | img_size={img_size}")

    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)
    frames = data["captured_frames"]

    both = [f for f in frames
            if f.get("foundL") and f.get("foundR")
            and isinstance(f.get("cornersL"), np.ndarray)
            and isinstance(f.get("cornersR"), np.ndarray)]
    print(f"frames with BOTH detections: {len(both)}")

    n = min(args.n, len(both))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(both), size=n, replace=False)
    chosen = [both[i] for i in idx]
    print(f"using {n} stereo pairs (seed={args.seed})")

    objp = build_objp()
    objpoints = [objp for _ in chosen]
    ptsL = [c["cornersL"].astype(np.float32) for c in chosen]
    ptsR = [c["cornersR"].astype(np.float32) for c in chosen]

    rms, KL, dL, KR, dR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, ptsL, ptsR, KL, dL, KR, dR, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    baseline_mm = float(np.linalg.norm(T) * 1000.0)
    print(f"stereo RMS: {rms:.5f}  baseline: {baseline_mm:.3f} mm")
    print(f"T (m) = {T.ravel()}")

    R1, R2, P0, P1, Q, roiL, roiR = cv2.stereoRectify(
        KL, dL, KR, dR, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    np.savez(
        args.out,
        cam_matrix_L=KL, dist_coeffs_L=dL,
        cam_matrix_R=KR, dist_coeffs_R=dR,
        R=R, T=T, E=E, F=F, R1=R1, R2=R2, P0=P0, P1=P1, Q=Q,
        img_size=np.array(img_size), stereo_rms=rms, baseline_mm=baseline_mm,
        pattern_size=np.array(PATTERN), square_size_m=SQUARE_M,
        n_pairs=n, seed=args.seed,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
