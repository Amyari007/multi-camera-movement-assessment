"""
SCRIPT 1 — Dual-Camera Raw Capture
====================================
Captures simultaneously from two USB cameras (one USB-A, one USB-C/COM7),
saves raw frames + sync/timestamp metadata to .pkl files.

Output files (saved to C:\\Users\\arya0\\OneDrive\\Desktop):
  raw_frames_cam0.pkl   — list of (frame_num, timestamp, frame_bgr)
  raw_frames_cam1.pkl   — list of (frame_num, timestamp, frame_bgr)
  sync_and_framerate.csv — interleaved rows: timestamp, cam0_frame, cam0_fps,
                           cam1_frame, cam1_fps  (same format as previous system)

Usage:
  python script1_raw_capture.py
  Press Q to stop recording.

Camera index assignment:
  OpenCV enumerates USB cameras. Run with --list-cameras first to see which
  index maps to which physical port. By default:
    CAM0_INDEX = 0   (USB-A port)
    CAM1_INDEX = 1   (USB-C/COM7 port)
  Swap if your system enumerates them differently.
"""

import cv2
import pickle
import csv
import time
import threading
import argparse
import os
from datetime import datetime
from collections import deque

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SAVE_DIR      = r"C:\Users\arya0\OneDrive\Desktop"
CAM0_INDEX    = 0          # USB-A port camera  (swap if needed)
CAM1_INDEX    = 1          # USB-C port camera  (swap if needed)
DISPLAY_SCALE = 0.5        # Preview window scale (reduce if too slow)
PREVIEW_ON    = True       # Show live preview (disable to maximise throughput)
# ───────────────────────────────────────────────────────────────────────────────


def list_cameras(max_index=6):
    """Print available camera indices."""
    print("Scanning camera indices …")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            status = "OK" if ret else "opened but no frame"
            print(f"  Index {i}: {status}")
            cap.release()
        else:
            print(f"  Index {i}: not available")


class CameraCapture:
    """Thread-safe per-camera capture worker."""

    def __init__(self, cam_id: int, index: int):
        self.cam_id   = cam_id
        self.index    = index
        self.frames   = []          # list of (frame_num, timestamp_str, frame_bgr)
        self.csv_rows = []          # list of dicts for CSV
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._fps_buf = deque(maxlen=30)
        self.last_frame = None      # for preview

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[CAM{self.cam_id}] ERROR: Cannot open camera index {self.index}")
            return

        # Request high resolution — camera will use nearest supported mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimise latency

        frame_num = 0
        prev_ts   = None

        print(f"[CAM{self.cam_id}] Started on index {self.index}")

        while not self._stop.is_set():
            ret, frame = cap.read()
            ts = time.time()

            if not ret:
                continue

            ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Compute instantaneous FPS
            if prev_ts is not None:
                dt = ts - prev_ts
                if dt > 0:
                    self._fps_buf.append(1.0 / dt)
            prev_ts = ts
            fps = sum(self._fps_buf) / len(self._fps_buf) if self._fps_buf else 0.0

            self.frames.append((frame_num, ts_str, frame.copy()))

            # CSV row keyed to this camera
            row = {
                "timestamp":  ts_str,
                "sync_value": "",
                f"cam{self.cam_id}_frame": frame_num,
                f"cam{self.cam_id}_fps":   round(fps, 2),
            }
            # Fill other camera columns as empty
            other = 1 - self.cam_id
            row[f"cam{other}_frame"] = ""
            row[f"cam{other}_fps"]   = ""
            self.csv_rows.append(row)

            self.last_frame = (frame, fps, frame_num)
            frame_num += 1

        cap.release()
        print(f"[CAM{self.cam_id}] Stopped. Captured {frame_num} frames.")


def merge_csv_rows(rows0, rows1):
    """
    Merge per-camera CSV rows into interleaved format sorted by timestamp.
    Mirrors the sync_and_framerate.csv format from the existing pipeline.
    """
    all_rows = rows0 + rows1
    all_rows.sort(key=lambda r: r["timestamp"])
    return all_rows


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Dual-camera raw capture")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List available camera indices and exit")
    parser.add_argument("--cam0", type=int, default=CAM0_INDEX,
                        help=f"Camera index for Cam0 (default {CAM0_INDEX})")
    parser.add_argument("--cam1", type=int, default=CAM1_INDEX,
                        help=f"Camera index for Cam1 (default {CAM1_INDEX})")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    cam0 = CameraCapture(cam_id=0, index=args.cam0)
    cam1 = CameraCapture(cam_id=1, index=args.cam1)

    print("Starting cameras … press Q in preview window (or Ctrl-C) to stop.\n")
    cam0.start()
    cam1.start()

    # ── Live preview loop (main thread) ────────────────────────────────────────
    try:
        while True:
            frames_to_show = []
            for cam in (cam0, cam1):
                if cam.last_frame is not None:
                    frame, fps, fnum = cam.last_frame
                    disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE) \
                           if DISPLAY_SCALE != 1.0 else frame.copy()
                    cv2.putText(disp,
                                f"CAM{cam.cam_id}  frame={fnum}  fps={fps:.1f}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    frames_to_show.append(disp)

            if PREVIEW_ON and frames_to_show:
                # Pad to same height before hstack
                if len(frames_to_show) == 2:
                    h0, h1 = frames_to_show[0].shape[0], frames_to_show[1].shape[0]
                    if h0 != h1:
                        diff = abs(h0 - h1)
                        pad  = (0, diff) if h0 < h1 else (diff, 0)
                        frames_to_show[0] = cv2.copyMakeBorder(
                            frames_to_show[0], *pad, 0, 0, cv2.BORDER_CONSTANT)
                combined = cv2.hconcat(frames_to_show)
                cv2.imshow("Dual Camera Preview (Q to quit)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping capture …")
        cam0.stop()
        cam1.stop()
        cv2.destroyAllWindows()

    # ── Save outputs ────────────────────────────────────────────────────────────
    print("\nSaving files …")

    raw0_path = os.path.join(SAVE_DIR, "raw_frames_cam0.pkl")
    raw1_path = os.path.join(SAVE_DIR, "raw_frames_cam1.pkl")
    csv_path  = os.path.join(SAVE_DIR, "sync_and_framerate.csv")

    save_pkl(cam0.frames, raw0_path)
    save_pkl(cam1.frames, raw1_path)

    merged = merge_csv_rows(cam0.csv_rows, cam1.csv_rows)
    fieldnames = ["timestamp", "sync_value",
                  "cam0_frame", "cam0_fps",
                  "cam1_frame", "cam1_fps"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Saved: {csv_path}")

    print(f"\nDone.  Cam0: {len(cam0.frames)} frames | Cam1: {len(cam1.frames)} frames")
    print(f"Files in: {SAVE_DIR}")


if __name__ == "__main__":
    main()
