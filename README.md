# Multi-Camera System for General-Purpose Assessment and Training of Movement Functions

A low-cost, dual-RGB-camera motion analysis pipeline for clinical movement assessment — developed at the Department of Bioengineering, Christian Medical College (CMC), Vellore, under Dr. Sivakumar Balasubramanian, Head of Department.

## Motivation

Clinical motion capture (MoCap) systems are accurate but cost several lakhs, require a dedicated calibration room, and can't move to where the patient is — a real barrier for tracking recovery in cases like post-stroke rehabilitation. This project asks: **how much of that capability can be recovered using consumer-grade RGB webcams, open-source software, and careful engineering?**

The project was structured around four objectives:

1. **Camera calibration and validation** — establish reliable intrinsic parameters (focal length, principal point, distortion) for each webcam via checkerboard calibration
2. **Stereo calibration and 3D reconstruction** — determine the geometric relationship between two cameras and triangulate the 3D position of a moving AprilTag marker, validated against an OptiTrack MoCap system as ground truth
3. **Trunk tracking** — use MediaPipe pose landmarks fed through the stereo reconstruction pipeline to track trunk motion across all three anatomical planes
4. **Raspberry Pi portable system** — study how the two-camera setup can be translated into a compact, low-cost, clinic- or home-deployable version

## Key Result

**27.6 mm 3D RMSE** between the stereo-reconstructed trajectory and OptiTrack ground truth, over a 56.55-second synchronized capture window. This confirms the system performs genuine 3D tracking (not just 2D projection) and that the calibration pipeline is sound — sufficient for coarse trunk/limb motion assessment, though not yet at sub-centimetre accuracy needed for fine joint-angle work like gait analysis.

## Objective 1 — Camera Calibration and Validation

- Intrinsic calibration performed on both webcams using a **13×9 checkerboard, 3×3 cm squares** (39×27 cm overall board)
- Image sets: 32 images (trial 1), 55 images (trial 2), 77 images (trial 3), captured at varying positions, angles, and distances
- Calibration via **Zhang's method**, implemented with `cv2.calibrateCamera()`, following `cv2.findChessboardCorners()` and sub-pixel refinement with `cv2.cornerSubPix()`
- Early attempts with visually similar images gave reprojection error >1.0 px (poorly conditioned system); more diverse viewpoints brought this down to a final validated **0.47–0.77 px**
- Extracted parameters: focal lengths (fx, fy), principal point (cx, cy), and distortion coefficients (k1, k2, p1, p2, k3), saved to `camera_calibration_output.npz` and carried through every downstream stage

## Objective 2 — Stereo Calibration, 3D Reconstruction & MoCap Validation

**Setup:** two USB webcams on tripods (Camera 0 = left, Camera 1 = right/reference), a 3D-printed white T-frame moving target with three AprilTag markers (ID 12 top-centre, ID 14 bottom-left, ID 20 bottom-right, each 4.8×4.8 cm), a static ChArUco reference board (4×3 grid, 2.7×2.7 cm tags, 14.8×11.1 cm total), and an OptiTrack infrared MoCap system tracking the same space.

**Stereo calibration:** `cv2.stereoCalibrate()` recovers rotation matrix **R** and translation vector **T** describing the physical relationship between the two camera origins — necessary because a single camera can't recover depth (Z = fB/d, where f = focal length, B = baseline, d = disparity). Images were rectified with `cv2.stereoRectify()` and `cv2.initUndistortRectifyMap()` so corresponding rows share the same epipolar line.

**Pose estimation:** per-frame 3D marker position computed via `cv2.solvePnP()` (ITERATIVE flag). AprilTag detection followed a fallback order: detect ID 12 directly → if missing, geometrically reconstruct ID 12's centre from ID 14 and ID 20 → if neither available, mark frame NaN. A common world-aligned reference frame was fixed to the first valid frame of Camera 1, with subsequent frames rotated into that frame via Rᵀ — applied identically to both cameras and to the MoCap side (OptiTrack RigidBody 002 position + quaternion-derived rotation).

**Synchronization:** cameras and OptiTrack run on independent clocks with no shared timestamp, so an Arduino Uno generated a binary sync signal (HIGH while MoCap recording, LOW otherwise), logged alongside camera frame numbers. The usable overlap window was **12:02:27.210 to 12:03:23.757 (56.55 s)**. Camera and MoCap frames were matched by nearest-neighbour timestamp with a **±5 ms tolerance** (half a MoCap frame interval at 100 fps).

**Result:** 3D RMSE of **27.6 mm** across the synchronized window, with the Z (depth) axis showing the largest variance — expected, since stereo depth error scales quadratically with distance and inversely with baseline.

## Objective 3 — Trunk Tracking and Joint Angle Estimation

MediaPipe Pose detects 33 body landmarks per camera frame; **7 were selected** for trunk assessment (both shoulders, both elbows, both wrists, trunk centre). Each landmark's left/right pixel position was triangulated via `cv2.triangulatePoints()` using the stereo projection matrices. Trunk orientation (pitch/flexion, yaw/lateral bending, roll/axial rotation) was derived from the shoulder-to-shoulder vector.

Two validation experiments were run:
- **Standing, composite motion** — combined side-to-side and front-back displacement, with a recorded side-to-side spread of **40.1 cm**
- **Seated, lateral trunk shifts** — angle-deviation-vs-frame plots showed all three rotational components responding correctly to the movement, confirming the decomposition was working

## Objective 4 — Raspberry Pi Portable System

The dual-webcam-and-laptop rig is a validation prototype, not a deployable clinical tool. The target form factor is a Raspberry Pi with two camera modules on a 3D-printed rigid frame — small enough for a physiotherapy room or a patient's home. Locking the cameras' relative geometry means stereo calibration is done **once** and reused indefinitely, without recalibrating each session.

Capture pipeline development used **Picamera2** and OpenCV on the Pi, reusing the Objectives 1–2 calibration workflow. Porting the MediaPipe tracking pipeline to the embedded hardware was initiated; full real-time inference on-device (balancing landmark accuracy against latency) is ongoing work.

## Methods Summary

- **Hardware**: 2× USB webcams (Camera 0 = left, Camera 1 = right/reference), 3D-printed AprilTag T-mount, ChArUco static reference board, OptiTrack infrared MoCap with reflective markers, Arduino Uno for sync signal generation
- **Calibration**: `cv2.findChessboardCorners()` → `cv2.cornerSubPix()` → `cv2.calibrateCamera()` (intrinsics); `cv2.stereoCalibrate()` (extrinsics, R/T); `cv2.stereoRectify()` + `cv2.initUndistortRectifyMap()` (rectification)
- **Pose estimation**: `cv2.solvePnP()` (ITERATIVE)
- **Trajectory sync**: sync CSV with forward-filled sync column, overlap-window filtering, nearest-neighbour timestamp matching (±5 ms)
- **Error metrics**: per-axis error (err_X/Y/Z, mm), 3D Euclidean error, mean signed error, standard deviation, RMSE — computed separately for Camera 0 vs MoCap and Camera 1 vs MoCap
- **Speed analysis**: instantaneous target speed from consecutive MoCap frames (distance / dt), correlated against tracking error to characterize error-vs-speed behaviour

## Tech Stack

Python · OpenCV · MediaPipe · NumPy · Raspberry Pi (Picamera2) · Arduino Uno · AprilTag · ChArUco · OptiTrack (validation ground truth)

## Repository Structure

```
multi-camera-movement-assessment/
├── calibration/        # Intrinsic + stereo calibration scripts, camera_calibration_output.npz
├── tracking/           # AprilTag detection, solvePnP pose estimation
├── sync/               # Arduino sync signal handling, timestamp alignment
├── trunk_tracking/      # MediaPipe landmark extraction + triangulation
├── analysis/            # Error computation, speed-vs-error analysis
├── raspberry_pi/        # Picamera2-based portable capture pipeline
└── README.md
```

*(Update this tree to match your actual folder/file names before pushing.)*

## Data & Privacy Note

This repository includes the processing pipeline and analysis code only. Raw MoCap exports, participant recordings, or subject-identifying data are not included, in line with data-sharing constraints from the CMC Vellore collaboration.

## Status

Objectives 1–3 are validated with quantitative results as above. Objective 4 (Raspberry Pi real-time deployment) is ongoing work.

## Acknowledgements

Project conducted at the Department of Bioengineering, Christian Medical College (CMC), Vellore, under the supervision of Dr. Sivakumar Balasubramanian, as part of the M.Tech Clinical Engineering programme, IIT Madras.

## License

MIT License *(update if a different license applies)*
