# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "opencv-contrib-python>=4.13.0.92"]
# ///
"""Find which AprilTag family is used and how often id 14 appears (left cam)."""
import pickle
from collections import Counter
import cv2
import numpy as np

with open(r"C:\Users\arya0\Desktop\raw_frames.pkl", "rb") as f:
    frames = pickle.load(f)

apr = [fr for fr in frames if fr.get("phase") == "apriltag"]
print(f"apriltag-phase frames: {len(apr)}")

families = {
    "36h11": cv2.aruco.DICT_APRILTAG_36H11,
    "36h10": cv2.aruco.DICT_APRILTAG_36H10,
    "25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "16h5": cv2.aruco.DICT_APRILTAG_16H5,
}

sample = apr[::20][:8]  # ~8 spread-out frames
for name, fam in families.items():
    d = cv2.aruco.getPredefinedDictionary(fam)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    ids_all = Counter()
    for fr in sample:
        gray = cv2.cvtColor(fr["frameL"], cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None:
            ids_all.update(int(i) for i in ids.ravel())
    print(f"{name}: detected ids -> {dict(sorted(ids_all.items()))}")
