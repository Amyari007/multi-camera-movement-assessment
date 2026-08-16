"""
SCRIPT 2 — ChArUco World Frame + MediaPipe Pose Tracking
==========================================================
Loads raw_frames_cam0.pkl and raw_frames_cam1.pkl produced by Script 1,
then for every frame:

  1. Detects the ChArUco board (auto-scans common ArUco dictionaries).
  2. Estimates the board's pose → rvec / tvec in camera coordinates.
  3. Runs MediaPipe Pose to detect:
       - Left shoulder  (MP landmark 11)
       - Right shoulder (MP landmark 12)
       - Left elbow     (MP landmark 13)
       - Right elbow    (MP landmark 14)
       - Left wrist     (MP landmark 15)
       - Right wrist    (MP landmark 16)
       - Trunk center   (midpoint of shoulders, dropped ~10% of torso height
                         toward sternum — approximated from shoulder & hip landmarks)
  4. Saves everything to a single output .pkl per camera.

Output files (saved to C:\\Users\\arya0\\OneDrive\\Desktop):
  tracking_cam0.pkl
  tracking_cam1.pkl

Each pkl is a list of per-frame dicts:
  {
    "frame_num":      int,
    "timestamp":      str,
    "charuco_found":  bool,
    "rvec":           np.ndarray (3,1) or None,   # ChArUco pose
    "tvec":           np.ndarray (3,1) or None,
    "charuco_dict":   str or None,                # which dict was used
    "pose_found":     bool,
    "landmarks": {                                # pixel (x,y) + visibility
        "left_shoulder":  (x, y, vis),
        "right_shoulder": (x, y, vis),
        "left_elbow":     (x, y, vis),
        "right_elbow":    (x, y, vis),
        "left_wrist":     (x, y, vis),
        "right_wrist":    (x, y, vis),
        "trunk_center":   (x, y, vis),
    } or None
  }

Requirements:
  pip install opencv-contrib-python mediapipe numpy

Usage:
  python script2_tracking_worldframe.py
  (Edit the CONFIG section below if paths differ.)
"""

import cv2
import pickle
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SAVE_DIR       = r"C:\Users\arya0\OneDrive\Desktop"
RAW_CAM0       = os.path.join(SAVE_DIR, "raw_frames_cam0.pkl")
RAW_CAM1       = os.path.join(SAVE_DIR, "raw_frames_cam1.pkl")
CALIB_FILE     = os.path.join(SAVE_DIR, "camera_calibration_output.npz")

# ChArUco board geometry (from your system spec)
CHARUCO_COLS        = 4        # columns (squares)
CHARUCO_ROWS        = 3        # rows    (squares)
SQUARE_LENGTH_M     = 0.037    # 37 mm in metres
MARKER_LENGTH_M     = 0.027    # 27 mm in metres

# MediaPipe confidence thresholds
MP_DETECTION_CONF   = 0.5
MP_TRACKING_CONF    = 0.5

# Preview while processing (slows things down — set False for large datasets)
SHOW_PREVIEW        = True
PREVIEW_SCALE       = 0.6
# ───────────────────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
LM = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    # Used only to estimate trunk center:
    "left_hip":       23,
    "right_hip":      24,
}


# ─── Utility ───────────────────────────────────────────────────────────────────

def load_pkl(path: str):
    print(f"Loading {path} …")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  → {len(data)} frames")
    return data


def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved: {path}  ({size_mb:.1f} MB)")


def load_calibration(npz_path: str):
    """Load camera matrix and distortion coefficients."""
    if not os.path.exists(npz_path):
        print(f"[WARN] Calibration file not found: {npz_path}")
        print("       ChArUco pose estimation will be skipped.")
        return None, None
    data = np.load(npz_path)
    K    = data["camera_matrix"]
    dist = data["dist_coeffs"]
    print(f"Calibration loaded: K={K.shape}, dist={dist.shape}")
    return K, dist


# ─── ChArUco auto-detect ───────────────────────────────────────────────────────

# All dictionaries to try (in priority order — most common first)
ARUCO_DICTS = [
    ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
    ("DICT_4X4_50",         cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100",        cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_50",         cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100",        cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_50",         cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100",        cv2.aruco.DICT_6X6_100),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ("DICT_APRILTAG_16h5",  cv2.aruco.DICT_APRILTAG_16h5),
    ("DICT_APRILTAG_25h9",  cv2.aruco.DICT_APRILTAG_25h9),
]


class CharucoDetector:
    """
    Tries all ArUco dictionaries on the first N frames to find which one
    detects the board, then locks in that dictionary for the rest.
    """

    def __init__(self, cols, rows, square_m, marker_m, K, dist,
                 auto_detect_frames=30):
        self.cols    = cols
        self.rows    = rows
        self.sq_m    = square_m
        self.mk_m    = marker_m
        self.K       = K
        self.dist    = dist
        self.auto_n  = auto_detect_frames

        self._locked_dict_name  = None
        self._locked_board      = None
        self._locked_aruco_dict = None
        self._frames_tried      = 0

    @property
    def locked(self):
        return self._locked_dict_name is not None

    def _make_board(self, aruco_dict_obj):
        return cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            self.sq_m,
            self.mk_m,
            aruco_dict_obj
        )

    def detect(self, frame_bgr):
        """
        Returns (charuco_found, rvec, tvec, dict_name).
        rvec/tvec are None if pose estimation fails or calib not loaded.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if not self.locked and self._frames_tried < self.auto_n:
            self._frames_tried += 1
            result = self._try_all_dicts(gray)
            return result

        if self.locked:
            return self._detect_with(gray, self._locked_aruco_dict,
                                     self._locked_board,
                                     self._locked_dict_name)

        # Ran out of auto-detect frames without locking — keep trying
        return self._try_all_dicts(gray)

    def _try_all_dicts(self, gray):
        for name, dict_id in ARUCO_DICTS:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            board      = self._make_board(aruco_dict)
            found, rvec, tvec, _ = self._detect_with(gray, aruco_dict,
                                                      board, name)
            if found:
                if not self.locked:
                    print(f"[CharucoDetector] Auto-detected dictionary: {name}")
                    self._locked_dict_name  = name
                    self._locked_board      = board
                    self._locked_aruco_dict = aruco_dict
                return found, rvec, tvec, name
        return False, None, None, None

    def _detect_with(self, gray, aruco_dict, board, dict_name):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) < 1:
            return False, None, None, dict_name

        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
            gray, corners, ids)

        if charuco_corners is None or len(charuco_corners) < 4:
            return False, None, None, dict_name

        # Pose estimation (requires calibration)
        if self.K is None:
            return True, None, None, dict_name

        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < 4:
            return True, None, None, dict_name

        ret, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            return True, None, None, dict_name

        return True, rvec, tvec, dict_name

    def draw_overlay(self, frame_bgr, rvec, tvec):
        """Draw board axis on frame (for preview)."""
        if rvec is None or self.K is None:
            return frame_bgr
        out = frame_bgr.copy()
        cv2.drawFrameAxes(out, self.K, self.dist, rvec, tvec,
                          self.sq_m * 2)
        return out


# ─── MediaPipe Pose ─────────────────────────────────────────────────────────────

class PoseTracker:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=MP_DETECTION_CONF,
            min_tracking_confidence=MP_TRACKING_CONF,
        )

    def process(self, frame_bgr):
        """
        Returns dict of landmark pixel coords + visibility, or None.
        trunk_center = midpoint of shoulders shifted 20% toward hip midpoint
                       (approximates sternum / mid-chest area).
        """
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._pose.process(rgb)

        if not res.pose_landmarks:
            return None

        lm_list = res.pose_landmarks.landmark

        def get(idx):
            lm = lm_list[idx]
            return (lm.x * w, lm.y * h, lm.visibility)

        ls = get(LM["left_shoulder"])
        rs = get(LM["right_shoulder"])
        lh = get(LM["left_hip"])
        rh = get(LM["right_hip"])

        shoulder_mid = ((ls[0] + rs[0]) / 2,
                        (ls[1] + rs[1]) / 2)
        hip_mid      = ((lh[0] + rh[0]) / 2,
                        (lh[1] + rh[1]) / 2)

        # Trunk center ≈ 20% of the way from shoulder midpoint toward hip midpoint
        # This puts it roughly at the sternum / upper chest
        TRUNK_T = 0.20
        trunk_x = shoulder_mid[0] + TRUNK_T * (hip_mid[0] - shoulder_mid[0])
        trunk_y = shoulder_mid[1] + TRUNK_T * (hip_mid[1] - shoulder_mid[1])
        trunk_vis = min(ls[2], rs[2], lh[2], rh[2])

        return {
            "left_shoulder":  ls,
            "right_shoulder": rs,
            "left_elbow":     get(LM["left_elbow"]),
            "right_elbow":    get(LM["right_elbow"]),
            "left_wrist":     get(LM["left_wrist"]),
            "right_wrist":    get(LM["right_wrist"]),
            "trunk_center":   (trunk_x, trunk_y, trunk_vis),
        }

    def draw_overlay(self, frame_bgr, landmarks):
        if landmarks is None:
            return frame_bgr
        out = frame_bgr.copy()
        colors = {
            "left_shoulder":  (0, 255, 0),
            "right_shoulder": (0, 200, 0),
            "left_elbow":     (255, 128, 0),
            "right_elbow":    (200, 100, 0),
            "left_wrist":     (0, 128, 255),
            "right_wrist":    (0, 100, 200),
            "trunk_center":   (0, 0, 255),
        }
        # Draw skeleton lines
        skeleton = [
            ("left_shoulder",  "right_shoulder"),
            ("left_shoulder",  "left_elbow"),
            ("left_elbow",     "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow",    "right_wrist"),
            ("left_shoulder",  "trunk_center"),
            ("right_shoulder", "trunk_center"),
        ]
        for a, b in skeleton:
            pa = (int(landmarks[a][0]), int(landmarks[a][1]))
            pb = (int(landmarks[b][0]), int(landmarks[b][1]))
            cv2.line(out, pa, pb, (200, 200, 200), 2)

        for name, (x, y, vis) in landmarks.items():
            if vis > 0.3:
                cv2.circle(out, (int(x), int(y)), 7, colors[name], -1)
                cv2.putText(out, name.replace("_", " "),
                            (int(x) + 8, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[name], 1)
        return out

    def close(self):
        self._pose.close()


# ─── Per-camera processing ─────────────────────────────────────────────────────

def process_camera(cam_id: int, raw_frames, K, dist, save_path: str):
    charuco = CharucoDetector(CHARUCO_COLS, CHARUCO_ROWS,
                               SQUARE_LENGTH_M, MARKER_LENGTH_M,
                               K, dist)
    pose_tracker = PoseTracker()
    results      = []
    total        = len(raw_frames)

    print(f"\n[CAM{cam_id}] Processing {total} frames …")

    for i, entry in enumerate(raw_frames):
        frame_num, timestamp, frame_bgr = entry

        # 1. ChArUco detection
        charuco_found, rvec, tvec, dict_name = charuco.detect(frame_bgr)

        # 2. MediaPipe Pose
        landmarks = pose_tracker.process(frame_bgr)

        results.append({
            "frame_num":     frame_num,
            "timestamp":     timestamp,
            "charuco_found": charuco_found,
            "rvec":          rvec,
            "tvec":          tvec,
            "charuco_dict":  dict_name,
            "pose_found":    landmarks is not None,
            "landmarks":     landmarks,
        })

        # Progress + preview
        if i % 50 == 0 or i == total - 1:
            charuco_pct = sum(r["charuco_found"] for r in results) / len(results) * 100
            pose_pct    = sum(r["pose_found"]    for r in results) / len(results) * 100
            print(f"  [{i+1:>5}/{total}]  ChArUco: {charuco_pct:.0f}%  "
                  f"Pose: {pose_pct:.0f}%  "
                  f"Dict: {charuco.locked and charuco._locked_dict_name or 'searching'}")

        if SHOW_PREVIEW:
            disp = frame_bgr.copy()
            if charuco_found and rvec is not None:
                disp = charuco.draw_overlay(disp, rvec, tvec)
            disp = pose_tracker.draw_overlay(disp, landmarks)
            cv2.putText(disp, f"CAM{cam_id}  frame={frame_num}  ts={timestamp}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            if PREVIEW_SCALE != 1.0:
                disp = cv2.resize(disp, None,
                                  fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow(f"CAM{cam_id} — Tracking (Q to skip preview)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("  Preview disabled by user.")
                global SHOW_PREVIEW
                SHOW_PREVIEW = False

    pose_tracker.close()
    cv2.destroyAllWindows()

    charuco_detected = sum(r["charuco_found"] for r in results)
    pose_detected    = sum(r["pose_found"]    for r in results)
    print(f"\n[CAM{cam_id}] Summary:")
    print(f"  Total frames   : {total}")
    print(f"  ChArUco found  : {charuco_detected} ({charuco_detected/total*100:.1f}%)")
    print(f"  Pose found     : {pose_detected}    ({pose_detected/total*100:.1f}%)")
    print(f"  Dictionary used: {charuco._locked_dict_name or 'none'}")

    save_pkl(results, save_path)
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load calibration (optional — pose estimation skipped if missing)
    K, dist = load_calibration(CALIB_FILE)

    # Load raw frames
    raw_cam0 = load_pkl(RAW_CAM0)
    raw_cam1 = load_pkl(RAW_CAM1)

    out_cam0 = os.path.join(SAVE_DIR, "tracking_cam0.pkl")
    out_cam1 = os.path.join(SAVE_DIR, "tracking_cam1.pkl")

    # Process each camera
    process_camera(0, raw_cam0, K, dist, out_cam0)
    process_camera(1, raw_cam1, K, dist, out_cam1)

    print("\n=== All done ===")
    print(f"Output files:")
    print(f"  {out_cam0}")
    print(f"  {out_cam1}")
    print("\nEach pkl is a list of per-frame dicts with keys:")
    print("  frame_num, timestamp, charuco_found, rvec, tvec, charuco_dict,")
    print("  pose_found, landmarks {left_shoulder, right_shoulder, left_elbow,")
    print("  right_elbow, left_wrist, right_wrist, trunk_center}")


if __name__ == "__main__":
    main()
