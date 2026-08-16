import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

frames = pickle.load(open(r"C:\Users\arya0\Desktop\raw_frames.pkl", "rb"))
calib = np.load(r"C:\Users\arya0\Desktop\camera_calibration_stereo.npz")

KL = calib["cam_matrix_L"]
DL = calib["dist_coeffs_L"]
KR = calib["cam_matrix_R"]
DR = calib["dist_coeffs_R"]
R = calib["R"]
T = calib["T"]
TAG_ID = 14

P_L = KL @ np.hstack([np.eye(3), np.zeros((3, 1))])
P_R = KR @ np.hstack([R, T.reshape(3, 1)])

dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())

apr = [f for f in frames if f.get("phase") == "apriltag"]
print(f"Processing {len(apr)} frames...")

rows = []
for fr in apr:
    gL = cv2.cvtColor(fr["frameL"], cv2.COLOR_BGR2GRAY)
    cL = det.detectMarkers(gL)
    if cL[1] is None:
        continue
    idsL = cL[1].ravel()
    if TAG_ID not in idsL:
        continue
    idxL = np.where(idsL == TAG_ID)[0][0]
    centerL = np.mean(cL[0][idxL].reshape(4, 2), axis=0).reshape(1, 1, 2).astype(np.float32)
    uL = cv2.undistortPoints(centerL, KL, DL, P=KL).reshape(2, 1)

    gR = cv2.cvtColor(fr["frameR"], cv2.COLOR_BGR2GRAY)
    cR = det.detectMarkers(gR)
    if cR[1] is None:
        continue
    idsR = cR[1].ravel()
    if TAG_ID not in idsR:
        continue
    idxR = np.where(idsR == TAG_ID)[0][0]
    centerR = np.mean(cR[0][idxR].reshape(4, 2), axis=0).reshape(1, 1, 2).astype(np.float32)
    uR = cv2.undistortPoints(centerR, KR, DR, P=KR).reshape(2, 1)

    Xh = cv2.triangulatePoints(P_L, P_R, uL, uR)
    X = (Xh[:3] / Xh[3]).ravel() * 1000
    rows.append((fr["frame_num"], fr["timestamp"], X[0], X[1], X[2]))

arr = np.array(rows)
print(f"Triangulated: {len(arr)} frames")
np.savetxt(r"C:\Users\arya0\Desktop\stereo_traj.csv", arr, delimiter=",", header="frame_num,timestamp,x_mm,y_mm,z_mm", comments="")

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
t = arr[:, 1] - arr[:, 1].min()
axes[0].plot(t, arr[:, 2], ".-", lw=1, color="red")
axes[0].set_ylabel("X (mm)")
axes[0].grid(True)
axes[1].plot(t, arr[:, 3], ".-", lw=1, color="green")
axes[1].set_ylabel("Y (mm)")
axes[1].grid(True)
axes[2].plot(t, arr[:, 4], ".-", lw=1, color="blue")
axes[2].set_ylabel("Z (mm)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)
plt.suptitle("Stereo Trajectory - Tag ID 14")
plt.tight_layout()
plt.savefig(r"C:\Users\arya0\Desktop\stereo_trajectory.png", dpi=130)
print("Saved: stereo_trajectory.png and stereo_traj.csv")
