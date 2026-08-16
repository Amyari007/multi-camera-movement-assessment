"""
01_mock_body_capture.py - Mock Stereo Capture for Body Tracking Check
======================================================================
PURPOSE: Quick sanity check — are both cameras capturing well? Is MediaPipe
         detecting the right landmarks? Is Charuco visible in both frames?

Records BOTH cameras simultaneously into a raw_frames.pkl file.
Charuco board is present throughout (world reference — no separate phase).
MediaPipe Pose runs live for visual feedback only (not saved).
No sync hardware needed.

Saves: mock_raw_frames.pkl
       mock_frame_log.csv

Usage on Pi:
    python3 01_mock_body_capture.py

Press ESC or Q to stop early.
"""

import cv2
import numpy as np
import os
import pickle
import time
import csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
DESKTOP        = os.path.expanduser("~/Desktop")
OUTPUT_PKL     = os.path.join(DESKTOP, "mock_raw_frames.pkl")
OUTPUT_CSV     = os.path.join(DESKTOP, "mock_frame_log.csv")

# Camera indices — adjust if Pi sees cameras differently
CAM_LEFT       = 0
CAM_RIGHT      = 1
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 800

# Recording duration (seconds). Press ESC/Q to stop early.
RECORD_DURATION = 30

# MediaPipe landmark indices we care about
# (used for live overlay only — no saving of MP keypoints in mock)
MP_LANDMARKS = {
    'nose':          0,
    'left_shoulder': 11,
    'right_shoulder':12,
    'left_elbow':    13,
    'right_elbow':   14,
    'left_wrist':    15,
    'right_wrist':   16,
    'left_hip':      23,
    'right_hip':     24,
}

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
try:
    import mediapipe as mp
    mp_pose     = mp.solutions.pose
    mp_drawing  = mp.solutions.drawing_utils
    MP_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError:
    MP_AVAILABLE = False
    print("⚠️  MediaPipe not installed — running without pose overlay.")
    print("   Install: pip install mediapipe")

# ============================================================
# CHARUCO SETUP (for live corner overlay)
# ============================================================
# Match your board: 12 cols x 8 rows, square 30mm, marker 22.5mm
# Dictionary: DICT_5X5_1000 — change if yours differs
try:
    ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    CHARUCO_BOARD = cv2.aruco.CharucoBoard(
        (12, 8),          # cols, rows of chessboard squares
        0.030,            # square side length in metres
        0.0225,           # marker side length in metres
        ARUCO_DICT
    )
    CHARUCO_PARAMS = cv2.aruco.DetectorParameters()
    CHARUCO_AVAILABLE = True
    print("✅ Charuco board configured (12x8, 30mm)")
except Exception as e:
    CHARUCO_AVAILABLE = False
    print(f"⚠️  Charuco setup failed: {e}")


def open_camera(idx, w, h):
    """Open a camera with given resolution."""
    # Try V4L2 first (Linux/Pi), fall back to default
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # reduce lag
    return cap


def detect_charuco_corners(gray):
    """Return (corners, ids) or (None, None)."""
    if not CHARUCO_AVAILABLE:
        return None, None
    try:
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, CHARUCO_PARAMS)
        m_corners, m_ids, _ = detector.detectMarkers(gray)
        if m_ids is None or len(m_ids) < 4:
            return None, None
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, CHARUCO_BOARD
        )
        if ret and ch_corners is not None and len(ch_corners) >= 4:
            return ch_corners, ch_ids
    except Exception:
        pass
    return None, None


def draw_pose_overlay(frame, results, color=(0, 255, 0)):
    """Draw selected MediaPipe landmarks on frame."""
    if results is None or not results.pose_landmarks:
        return frame
    h, w = frame.shape[:2]
    lms = results.pose_landmarks.landmark
    # Draw skeleton connections (full body)
    if MP_AVAILABLE:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
        )
    # Highlight our specific markers
    for name, idx in MP_LANDMARKS.items():
        lm = lms[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(frame, name[:3], (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return frame


def main():
    print("\n" + "="*70)
    print("MOCK BODY CAPTURE — Dual Camera Stereo Check")
    print("="*70)
    print(f"  Cameras: L={CAM_LEFT}, R={CAM_RIGHT}  |  {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Duration: {RECORD_DURATION}s  (press Q or ESC to stop early)")
    print(f"  Output: {OUTPUT_PKL}")
    print("="*70)

    # Open cameras
    print("\n📷 Opening cameras...")
    capL = open_camera(CAM_LEFT,  FRAME_WIDTH, FRAME_HEIGHT)
    capR = open_camera(CAM_RIGHT, FRAME_WIDTH, FRAME_HEIGHT)

    if not capL.isOpened():
        print(f"❌ Cannot open LEFT camera (index {CAM_LEFT})")
        return
    if not capR.isOpened():
        print(f"❌ Cannot open RIGHT camera (index {CAM_RIGHT})")
        return

    # Confirm actual resolution
    actual_w = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Cameras opened. Actual resolution: {actual_w}x{actual_h}")

    # MediaPipe pose instances (one per camera for independent inference)
    pose_L = pose_R = None
    if MP_AVAILABLE:
        pose_L = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_R = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # Storage
    all_frames = []
    frame_log  = []
    frame_count = 0

    print("\n⏳ Starting in 3 seconds... Stand in front of both cameras.")
    print("   Make sure Charuco board is also visible!\n")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)

    print("\n🟢 RECORDING — perform arm/trunk movements slowly.")
    print("   Charuco board should stay visible throughout.\n")

    record_start = time.time()

    while True:
        elapsed = time.time() - record_start
        if elapsed > RECORD_DURATION:
            print("\n⏹ Duration reached.")
            break

        # ---- Grab frames (as close in time as possible) ----
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        timestamp = time.time()

        if not retL or frameL is None:
            print("⚠️  Left camera dropped frame")
            continue
        if not retR or frameR is None:
            print("⚠️  Right camera dropped frame")
            continue

        # ---- Store raw frame (no annotation) ----
        all_frames.append({
            'frame_num': frame_count,
            'timestamp': timestamp,
            'frameL':    frameL.copy(),
            'frameR':    frameR.copy(),
            'phase':     'body_tracking',
        })
        frame_log.append({
            'frame_num': frame_count,
            'timestamp': timestamp,
            'elapsed':   round(elapsed, 4),
        })
        frame_count += 1

        # ---- Live display (every frame — no skipping on Pi, adjust if slow) ----
        dispL = frameL.copy()
        dispR = frameR.copy()

        # Charuco overlay
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        ch_L, _ = detect_charuco_corners(grayL)
        ch_R, _ = detect_charuco_corners(grayR)

        if ch_L is not None:
            cv2.polylines(dispL, [ch_L.astype(np.int32)], False, (0, 255, 255), 2)
            cv2.putText(dispL, f"Charuco OK ({len(ch_L)} pts)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(dispL, "Charuco NOT FOUND",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        if ch_R is not None:
            cv2.polylines(dispR, [ch_R.astype(np.int32)], False, (0, 255, 255), 2)
            cv2.putText(dispR, f"Charuco OK ({len(ch_R)} pts)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(dispR, "Charuco NOT FOUND",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # MediaPipe overlay
        if MP_AVAILABLE:
            rgbL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
            rgbR = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)
            resL = pose_L.process(rgbL)
            resR = pose_R.process(rgbR)
            dispL = draw_pose_overlay(dispL, resL, color=(0, 255, 0))
            dispR = draw_pose_overlay(dispR, resR, color=(0, 200, 255))

            lm_L = "Pose OK" if (resL.pose_landmarks is not None) else "No Pose"
            lm_R = "Pose OK" if (resR.pose_landmarks is not None) else "No Pose"
            cv2.putText(dispL, lm_L, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(dispR, lm_R, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # Labels
        for disp, label in [(dispL, "LEFT"), (dispR, "RIGHT")]:
            cv2.putText(disp, f"CAM {label} | Frame {frame_count} | {elapsed:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(disp, "Q/ESC = stop early",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Scale display to fit screen (both cams side by side)
        combined = np.hstack((dispL, dispR))
        disp_w = min(1280, combined.shape[1])
        scale  = disp_w / combined.shape[1]
        disp_h = int(combined.shape[0] * scale)
        combined_small = cv2.resize(combined, (disp_w, disp_h))

        cv2.imshow("Mock Capture — Left | Right", combined_small)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            print("\n⏹ Stopped by user.")
            break

    # ---- Save ----
    print(f"\n💾 Saving {len(all_frames)} frames to {OUTPUT_PKL} ...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_frames, f)
    print("   ✅ PKL saved.")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_num', 'timestamp', 'elapsed'])
        writer.writeheader()
        writer.writerows(frame_log)
    print(f"   ✅ Frame log saved: {OUTPUT_CSV}")

    # ---- Cleanup ----
    capL.release()
    capR.release()
    if pose_L: pose_L.close()
    if pose_R: pose_R.close()
    cv2.destroyAllWindows()

    fps = len(all_frames) / (time.time() - record_start)
    print("\n" + "="*70)
    print("📊 MOCK CAPTURE COMPLETE")
    print("="*70)
    print(f"   Total frames  : {len(all_frames)}")
    print(f"   Avg FPS       : {fps:.1f}")
    print(f"   Resolution    : {actual_w}x{actual_h}")
    print(f"   PKL file      : {OUTPUT_PKL}")
    print(f"   Frame log     : {OUTPUT_CSV}")
    print("\n✅ DONE — check the PKL with 02_inspect_mock.py")
    print("="*70)


if __name__ == "__main__":
    main()
