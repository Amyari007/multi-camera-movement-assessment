"""Capture synced AprilTag-movement frames from two Pi cameras (Picamera2)
with sync state read from a Teensy over serial, mirroring the laptop rig's
Arduino-sync approach.

Run ON THE PI.

ASSUMPTIONS (confirm/edit before running):
- Two CSI cameras at Picamera2 indices 0 and 1.
- Teensy sends sync state as serial lines, one of:
    "1\n"  -> sync ON  (mocap/movement window active)
    "0\n"  -> sync OFF
  (same convention as your laptop Arduino sync_log.csv: sync=1 when active)
- Teensy enumerates as /dev/ttyACM0 (typical). Change TEENSY_PORT if not.
"""
import datetime
import pickle
import time

import cv2
import msgpack
import msgpack_numpy as m
import serial
from picamera2 import Picamera2

m.patch()

RESOLUTION = (1280, 800)   # match calibration resolution
DURATION_S = 40
OUT_DIR = r"/home/pi/dual_ov9281_apriltag_movement"
TEENSY_PORT = "/dev/ttyACM0"
TEENSY_BAUD = 115200


def open_cam(index, size):
    picam = Picamera2(camera_num=index)
    config = picam.create_video_configuration(
        main={"size": size, "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)  # warm up
    return picam


def read_sync_state(ser, last_state):
    """Non-blocking-ish read of latest sync byte from Teensy; keeps last
    known state if nothing new is waiting."""
    state = last_state
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line in ("0", "1"):
            state = int(line)
    return state


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Opening cameras...")
    cam0 = open_cam(0, RESOLUTION)
    cam1 = open_cam(1, RESOLUTION)

    print(f"Opening Teensy on {TEENSY_PORT}...")
    ser = serial.Serial(TEENSY_PORT, TEENSY_BAUD, timeout=0)
    time.sleep(2)  # let Teensy reset after serial open

    print(f"Starting capture in 3 seconds. Move the AprilTag for {DURATION_S}s.")
    time.sleep(3)
    print("CAPTURING - move the tag now!")

    frames0, frames1 = [], []
    ts0, ts1, sync0, sync1, idx0, idx1 = [], [], [], [], [], []

    sync_state = 0
    start = time.time()
    i = 0
    while (time.time() - start) < DURATION_S:
        f0 = cam0.capture_array()
        f1 = cam1.capture_array()
        now = datetime.datetime.now()
        sync_state = read_sync_state(ser, sync_state)

        frames0.append(cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY))
        frames1.append(cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY))
        ts0.append(now); ts1.append(now)
        idx0.append(i); idx1.append(i)
        sync0.append(sync_state); sync1.append(sync_state)
        i += 1

    cam0.stop()
    cam1.stop()
    ser.close()
    print(f"Captured {len(frames0)} synced frame pairs")

    for name, frames in (("cam0", frames0), ("cam1", frames1)):
        path = f"{OUT_DIR}/{name}_frame.msgpack"
        with open(path, "wb") as f:
            packer = msgpack.Packer(default=m.encode, use_bin_type=True)
            for fr in frames:
                f.write(packer.pack(fr))
        print(f"saved -> {path}")

    for name, idx, ts, sync in (("cam0", idx0, ts0, sync0), ("cam1", idx1, ts1, sync1)):
        path = f"{OUT_DIR}/{name}_apriltag_meta.pkl"
        with open(path, "wb") as f:
            pickle.dump({"frame_idx": idx, "timestamp": ts, "sync": sync}, f)
        print(f"saved -> {path}")

    print("\nDone. Next: run pi_probe_tags.py to confirm tag family/id, "
          "then pi_trajectory_stereo.py to triangulate.")


if __name__ == "__main__":
    main()
