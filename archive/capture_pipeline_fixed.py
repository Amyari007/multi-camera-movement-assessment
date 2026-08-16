"""
CAPTURE PIPELINE — World Frame (ChArUco, first 10 s) -> Body Tracking (press S)
===============================================================================
FIX vs previous version: ChArUco/ArUco detector now uses tuned
DetectorParameters (confirmed working via diagnose_charuco.py: 6/6 markers,
6/6 charuco corners). Previous version used OpenCV defaults, which are
tuned for large/sharp markers and were missing your smaller printed board.

Phase 1  : 10-second ChArUco world-frame window (both cams shown live).
           The ChArUco board defines the world origin + axes.
           Hold it still, clearly visible to both cams.

Phase 2  : Body tracking.  Window stays open so you can get into position.
           Press S  to START recording.
           Press Q  to STOP recording and save everything.

Saves
  world_frame.npz              — origin, R_world
  live_3d_output.pkl           — per-frame triangulated landmarks (world frame)
  trajectory_angle_output.pkl  — Euler angles + trunk trajectory
  teensy_sync_log.csv          — Teensy sync + cam-frame timestamps
  angle_deviation.png
  trunk_trajectory.png

Requirements
  pip install opencv-contrib-python mediapipe numpy matplotlib pyserial
"""

import os, csv, time, threading, collections, pickle
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import serial

# ═══════════════════════════════════════════════════════ CONFIG ══════════════
SAVE_DIR     = r"C:\Users\arya0\OneDrive\Desktop"
CAM0_INDEX   = 0        # left / cam0
CAM1_INDEX   = 1        # right / cam1
IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480

# ── Intrinsics from your checkerboard calibration ───────────────────────────
K0    = np.array([[461.81540924,   0.,         331.92022361],
                  [  0.,         463.35376794, 248.9576821 ],
                  [  0.,           0.,           1.        ]])
DIST0 = np.array([[-0.00592837, -0.07470545,  0.00268104, -0.00229303,  0.06809824]])

K1    = np.array([[468.56177787,   0.,         326.09298681],
                  [  0.,         469.80433513, 257.35515992],
                  [  0.,           0.,           1.        ]])
DIST1 = np.array([[ 0.0376792,  -0.10272102,  0.00527959, -0.00214092,  0.03904257]])

# ── ChArUco board (must match your printed board) ───────────────────────────
CHARUCO_SQUARES_X  = 4
CHARUCO_SQUARES_Y  = 3
CHARUCO_SQUARE_LEN = 0.037   # metres
CHARUCO_MARKER_LEN = 0.027   # metres
ARUCO_DICT_ID      = cv2.aruco.DICT_4X4_50
CHARUCO_LEGACY_PATTERN = False   # confirmed via diagnose_charuco.py: both
                                  # True/False detected 6/6 on your board,
                                  # so False (OpenCV's current default) is fine.
MIN_COMMON_CORNERS = 3           # lowered from 4 -- your board only has
                                  # 6 markers total, so don't demand more
                                  # common corners than realistic at an angle

WORLD_FRAME_SECS   = 10     # seconds to average for world frame

# ── Stereo extrinsics — filled in after stereoCalibrate runs live ────────────
STEREO_CALIB_FILE = os.path.join(SAVE_DIR, "stereo_calibration.npz")

# ── Output paths ────────────────────────────────────────────────────────────
WORLD_FILE    = os.path.join(SAVE_DIR, "world_frame.npz")
OUT_3D_FILE   = os.path.join(SAVE_DIR, "live_3d_output.pkl")
OUT_TRAJ_FILE = os.path.join(SAVE_DIR, "trajectory_angle_output.pkl")
SYNC_CSV_FILE = os.path.join(SAVE_DIR, "teensy_sync_log.csv")
ANGLE_PLOT    = os.path.join(SAVE_DIR, "angle_deviation.png")
TRAJ_PLOT     = os.path.join(SAVE_DIR, "trunk_trajectory.png")

# ── Teensy ───────────────────────────────────────────────────────────────────
TEENSY_PORT    = "COM5"
TEENSY_BAUD    = 115200
SERIAL_TIMEOUT = 0.01

# ── Tracking params ──────────────────────────────────────────────────────────
MP_DETECT_CONF = 0.5
MP_TRACK_CONF  = 0.5
VIS_THRESHOLD  = 0.5
SMOOTH_WINDOW  = 5
MAX_STEP_DEG   = 20.0
FPS_WINDOW     = 15
PREVIEW_SCALE  = 0.8

ORG_NAME  = "trunk_center"
XDIR_NAME = "right_shoulder"
ZDIR_NAME = "left_shoulder"
LANDMARKS_3D_NAMES = [
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "trunk_center",
]
LM = {"left_shoulder":11,"right_shoulder":12,"left_elbow":13,
      "right_elbow":14,"left_wrist":15,"right_wrist":16,
      "left_hip":23,"right_hip":24}
# ════════════════════════════════════════════════════════════════════════════


# ── Geometry helpers ─────────────────────────────────────────────────────────
def P(K, R, T):
    return K @ np.hstack([R, T])

def undistort(uv, K, dist):
    return cv2.undistortPoints(
        np.array([[uv]], dtype=np.float32), K, dist, P=K).reshape(2)

def triangulate(P0, P1, uv0, uv1):
    ph = cv2.triangulatePoints(P0, P1,
        np.array([[uv0[0]],[uv0[1]]], dtype=np.float64),
        np.array([[uv1[0]],[uv1[1]]], dtype=np.float64))
    return (ph[:3]/ph[3]).flatten()

def scale(frame):
    return cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)


# ── ChArUco setup — FIX: tuned detector params ───────────────────────────────
def make_tuned_aruco_params():
    """
    Confirmed via diagnose_charuco.py to detect 6/6 markers on your board
    (vs 0 with OpenCV's plain defaults, which assume larger/sharper markers).
    """
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 33
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def make_charuco():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, d)
    board.setLegacyPattern(CHARUCO_LEGACY_PATTERN)

    aruco_params = make_tuned_aruco_params()
    charuco_params = cv2.aruco.CharucoParameters()
    det = cv2.aruco.CharucoDetector(board, charuco_params, aruco_params)
    return board, det

def detect_charuco(gray, det):
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < MIN_COMMON_CORNERS:
        return None, None
    return corners, ids


# ── Stereo calibration (called once before world-frame phase) ─────────────────
def run_stereo_calibration(cap0, cap1):
    """
    Quick stereo capture phase: hold ChArUco board, SPACE=grab, Q=done (≥6).
    Uses your provided intrinsics as fixed; only estimates R, T.
    """
    print("\n[STEREO CALIB]  SPACE=grab pair  |  Q=done  (need ≥6 pairs)")
    board, det = make_charuco()
    obj_all, img0_all, img1_all = [], [], []
    count = 0
    board_corners_3d = board.getChessboardCorners()

    while True:
        r0, f0 = cap0.read(); r1, f1 = cap1.read()
        if not r0 or not r1: continue
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        cc0, ci0 = detect_charuco(g0, det)
        cc1, ci1 = detect_charuco(g1, det)
        d0, d1 = f0.copy(), f1.copy()
        if cc0 is not None: cv2.aruco.drawDetectedCornersCharuco(d0, cc0, ci0, (0,255,0))
        if cc1 is not None: cv2.aruco.drawDetectedCornersCharuco(d1, cc1, ci1, (0,255,0))
        both = cc0 is not None and cc1 is not None
        col  = (0,255,0) if both else (0,0,255)
        cv2.putText(d0, f"{'BOTH OK — SPACE' if both else 'waiting...'}", (8,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(d0, f"pairs: {count}", (8,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow("Stereo Calib — SPACE=grab, Q=done", cv2.hconcat([scale(d0), scale(d1)]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and both:
            m0 = {int(ci0[k]): cc0[k][0] for k in range(len(ci0))}
            m1 = {int(ci1[k]): cc1[k][0] for k in range(len(ci1))}
            common = sorted(set(m0)&set(m1))
            if len(common) < MIN_COMMON_CORNERS:
                print(f"  too few common corners ({len(common)}), move board"); continue
            obj_all.append(np.array([board_corners_3d[c] for c in common], dtype=np.float32))
            img0_all.append(np.array([m0[c] for c in common], dtype=np.float32).reshape(-1,1,2))
            img1_all.append(np.array([m1[c] for c in common], dtype=np.float32).reshape(-1,1,2))
            count += 1; print(f"  pair {count} captured ({len(common)} common corners)")
        elif key in (ord('q'), ord('Q')):
            if count < 6: print(f"  only {count} pairs — need ≥6"); continue
            break

    cv2.destroyAllWindows()
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj_all, img0_all, img1_all,
        K0.copy(), DIST0.copy(), K1.copy(), DIST1.copy(),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    print(f"  Stereo RMS: {rms:.4f} px   baseline: {np.linalg.norm(T)*100:.1f} cm")
    np.savez(STEREO_CALIB_FILE, K0=K0, dist0=DIST0, K1=K1, dist1=DIST1,
             R=R, T=T, image_size=np.array([IMAGE_WIDTH, IMAGE_HEIGHT]))
    print(f"  Saved: {STEREO_CALIB_FILE}")
    return R, T


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW  —  Phase 1 (world frame 10 s)  then  Phase 2 (body tracking)
# ═══════════════════════════════════════════════════════════════════════════

class TeensySyncLogger:
    HEADER = ["timestamp","sync_value","cam0_frame","cam0_fps","cam1_frame","cam1_fps"]
    def __init__(self, port, baud, csv_path, timeout=0.01):
        self._file   = open(csv_path,"w",newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._lock   = threading.Lock()
        self._running = False
        self._ser    = None
        self._last_sync = None
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout)
            print(f"Teensy on {port}")
        except Exception as e:
            print(f"Teensy WARNING: {e}  — continuing without sync.")
    def start(self):
        if not self._ser: return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self._running:
            try: raw = self._ser.readline().decode("utf-8",errors="ignore")
            except: break
            if not raw: continue
            sv = int(raw.strip()) if raw.strip() in ("0","1") else None
            if sv is None: continue
            ts = time.time()
            with self._lock:
                self._writer.writerow([ts,sv,"","","",""])
                self._file.flush()
                self._last_sync = ts
    def log_frame(self, fi, fps0, fps1):
        ts = time.time()
        with self._lock:
            self._writer.writerow([ts,"",fi,fps0,fi,fps1])
    def since_sync(self):
        return None if self._last_sync is None else time.time()-self._last_sync
    def stop(self):
        self._running = False
        if self._ser:
            try: self._ser.close()
            except: pass
        self._file.flush(); self._file.close()
        print(f"Saved: {SYNC_CSV_FILE}")

class FpsTracker:
    def __init__(self, w=15): self._t = collections.deque(maxlen=w)
    def tick(self): self._t.append(time.time())
    def fps(self):
        if len(self._t)<2: return float("nan")
        s = self._t[-1]-self._t[0]
        return float("nan") if s<=0 else (len(self._t)-1)/s

class PoseTracker:
    def __init__(self):
        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
            min_detection_confidence=MP_DETECT_CONF,
            min_tracking_confidence=MP_TRACK_CONF)
    def process(self, bgr):
        h,w = bgr.shape[:2]
        res = self._pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: return None
        ll = res.pose_landmarks.landmark
        def g(i): lm=ll[i]; return (lm.x*w,lm.y*h,lm.visibility)
        ls,rs = g(LM["left_shoulder"]),g(LM["right_shoulder"])
        lh,rh = g(LM["left_hip"]),g(LM["right_hip"])
        sm=((ls[0]+rs[0])/2,(ls[1]+rs[1])/2)
        hm=((lh[0]+rh[0])/2,(lh[1]+rh[1])/2)
        T=0.20
        return {
            "left_shoulder":ls,"right_shoulder":rs,
            "left_elbow":g(LM["left_elbow"]),"right_elbow":g(LM["right_elbow"]),
            "left_wrist":g(LM["left_wrist"]),"right_wrist":g(LM["right_wrist"]),
            "trunk_center":(sm[0]+T*(hm[0]-sm[0]), sm[1]+T*(hm[1]-sm[1]),
                            min(ls[2],rs[2],lh[2],rh[2])),
        }
    def draw(self, bgr, lm):
        if lm is None: return bgr
        out = bgr.copy()
        colors={"left_shoulder":(0,255,0),"right_shoulder":(0,200,0),
                "left_elbow":(255,128,0),"right_elbow":(200,100,0),
                "left_wrist":(0,128,255),"right_wrist":(0,100,200),"trunk_center":(0,0,255)}
        skel=[("left_shoulder","right_shoulder"),
              ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
              ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
              ("left_shoulder","trunk_center"),("right_shoulder","trunk_center")]
        for a,b in skel:
            cv2.line(out,(int(lm[a][0]),int(lm[a][1])),(int(lm[b][0]),int(lm[b][1])),(200,200,200),2)
        for nm,(x,y,v) in lm.items():
            if v>0.3: cv2.circle(out,(int(x),int(y)),7,colors[nm],-1)
        return out
    def close(self): self._pose.close()


def calc_rotmat(xdir, zdir, org):
    xdir,zdir,org = [np.asarray(v,dtype=np.float64).reshape(3,1) for v in (xdir,zdir,org)]
    v1=xdir-org; v2=zdir-org
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-3 or n2<1e-3: return None
    vx=v1/n1; vzcap=v2-(vx.T@v2)*vx; nz=np.linalg.norm(vzcap)
    if nz/n2<0.05: return None
    vz=vzcap/nz; vy=np.cross(vz.flatten(),vx.flatten()).reshape(3,1)
    return np.hstack((vx,vy,vz))

def smooth_lm(data, window=5):
    if window<=1: return data
    names=None
    for e in data:
        if e["landmarks_3d"]: names=list(e["landmarks_3d"].keys()); break
    if not names: return data
    n=len(data); S={nm:np.full((n,3),np.nan) for nm in names}
    for i,e in enumerate(data):
        if e["landmarks_3d"] is None: continue
        for nm in names: S[nm][i]=e["landmarks_3d"][nm]
    half=window//2; sm={nm:np.copy(a) for nm,a in S.items()}
    for nm in names:
        arr=S[nm]
        for i in range(n):
            w=arr[max(0,i-half):min(n,i+half+1)]; v=~np.isnan(w[:,0])
            if v.any(): sm[nm][i]=w[v].mean(axis=0)
    out=[]
    for i,e in enumerate(data):
        if e["landmarks_3d"] is None: out.append(e); continue
        out.append({"frame_idx":e["frame_idx"],"landmarks_3d":{nm:tuple(sm[nm][i]) for nm in names}})
    return out

def euler_xyz(R):
    sy=np.clip(-R[2,0],-1,1); y=np.arcsin(sy); cy=np.cos(y)
    if abs(cy)>1e-6: x=np.arctan2(R[2,1],R[2,2]); z=np.arctan2(R[1,0],R[0,0])
    else: x=np.arctan2(-R[1,2],R[1,1]); z=0.0
    return np.degrees([x,y,z])

def rot_angle(Ra,Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1)))

def compute_traj_angles(data):
    rots,traj,fidxs,skip={},{},{},0
    for e in data:
        idx,lm=e["frame_idx"],e["landmarks_3d"]
        if lm is None: continue
        R=calc_rotmat(lm[XDIR_NAME],lm[ZDIR_NAME],lm[ORG_NAME])
        if R is None: skip+=1; continue
        rots[idx]=R; traj[idx]=np.array(lm[ORG_NAME]); fidxs[idx]=idx
    if skip: print(f"Skipped {skip} degenerate frames")
    fidxs=sorted(fidxs.values())
    if not fidxs:
        return [], [], None
    # reject outliers
    kept=[fidxs[0]]; last=rots[fidxs[0]]
    for i in fidxs[1:]:
        if rot_angle(last,rots[i])>MAX_STEP_DEG: continue
        kept.append(i); last=rots[i]
    R0=rots[kept[0]]
    euler_log=[{"frame_idx":i,
                "angle_x_deg":euler_xyz(R0.T@rots[i])[0],
                "angle_y_deg":euler_xyz(R0.T@rots[i])[1],
                "angle_z_deg":euler_xyz(R0.T@rots[i])[2]} for i in kept]
    trunk_traj=[(i,traj[i]) for i in kept if i in traj]
    return euler_log, trunk_traj, kept[0]

def save_plots(euler_log, trunk_traj):
    idxs=[e["frame_idx"] for e in euler_log]
    plt.figure(figsize=(9,5))
    plt.plot(idxs,[e["angle_x_deg"] for e in euler_log],label="X pitch")
    plt.plot(idxs,[e["angle_y_deg"] for e in euler_log],label="Y yaw")
    plt.plot(idxs,[e["angle_z_deg"] for e in euler_log],label="Z roll")
    plt.xlabel("Frame"); plt.ylabel("deg"); plt.title("Trunk Angle Deviation")
    plt.legend(); plt.tight_layout(); plt.savefig(ANGLE_PLOT,dpi=150); plt.close()
    xs=[t[1][0]*100 for t in trunk_traj]; zs=[t[1][2]*100 for t in trunk_traj]
    plt.figure(figsize=(6,6))
    plt.plot(xs,zs,marker="o",markersize=2,linewidth=1)
    plt.scatter([xs[0]],[zs[0]],color="green",label="start",zorder=5)
    plt.scatter([xs[-1]],[zs[-1]],color="red",label="end",zorder=5)
    plt.xlabel("X cm"); plt.ylabel("Z cm"); plt.title("Trunk Trajectory (top-down)")
    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(TRAJ_PLOT,dpi=150); plt.close()
    print(f"Saved plots: {ANGLE_PLOT}  {TRAJ_PLOT}")


# ── World-frame detection ─────────────────────────────────────────────────────
def build_world_frame(cap0, cap1, P0, P1, board, det):
    """
    Stream both cams for WORLD_FRAME_SECS.  Detect ChArUco in both, triangulate
    corners, accumulate.  Returns R_world (3×3), world_origin (3,).
    """
    print(f"\n[WORLD FRAME]  Hold ChArUco board still — {WORLD_FRAME_SECS}s window starting now")
    board_corners_3d = board.getChessboardCorners()
    accum = {}
    start = time.time()
    good  = 0

    while True:
        elapsed = time.time()-start
        if elapsed >= WORLD_FRAME_SECS: break
        r0,f0=cap0.read(); r1,f1=cap1.read()
        if not r0 or not r1: continue
        g0=cv2.cvtColor(f0,cv2.COLOR_BGR2GRAY); g1=cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
        cc0,ci0=detect_charuco(g0,det); cc1,ci1=detect_charuco(g1,det)
        disp=f0.copy()
        if cc0 is not None: cv2.aruco.drawDetectedCornersCharuco(disp,cc0,ci0,(0,255,0))
        if cc0 is not None and cc1 is not None:
            m0={int(ci0[k]):cc0[k][0] for k in range(len(ci0))}
            m1={int(ci1[k]):cc1[k][0] for k in range(len(ci1))}
            common_ids = sorted(set(m0)&set(m1))
            for cid in common_ids:
                uv0=undistort(m0[cid],K0,DIST0); uv1=undistort(m1[cid],K1,DIST1)
                pt=triangulate(P0,P1,uv0,uv1)
                accum.setdefault(cid,[]).append(pt)
            if len(common_ids) >= MIN_COMMON_CORNERS:
                good+=1
        rem=max(0,WORLD_FRAME_SECS-elapsed)
        bw=int((elapsed/WORLD_FRAME_SECS)*300)
        cv2.rectangle(disp,(10,50),(310,68),(60,60,60),-1)
        cv2.rectangle(disp,(10,50),(10+min(bw,300),68),(0,200,100),-1)
        cv2.putText(disp,f"World frame: {rem:.1f}s  good:{good}",(10,45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,0),1)
        cv2.imshow("Dual Camera — World Frame Setup + Body Tracking",
                   cv2.hconcat([scale(disp), scale(f1)]))
        cv2.waitKey(1)

    avg={}
    for cid,pts in accum.items():
        if len(pts)>=3: avg[cid]=np.mean(pts,axis=0)
    if len(avg)<3:
        raise RuntimeError(
            f"Only {len(avg)} corners averaged (need >=3) — board not visible "
            f"to BOTH cameras enough during the 10s window. Re-run and hold "
            f"the board steadier/closer, equally visible to both cams."
        )

    sids=sorted(avg.keys())
    origin=avg[sids[0]]
    xvec=avg[sids[1]]-origin if len(sids)>1 else np.array([1,0,0],dtype=float)
    zvec=avg[sids[min(2,len(sids)-1)]]-origin if len(sids)>2 else np.array([0,0,1],dtype=float)
    xvec/=np.linalg.norm(xvec)
    zvec=zvec-np.dot(xvec,zvec)*xvec; zvec/=np.linalg.norm(zvec)
    yvec=np.cross(zvec,xvec)
    R_world=np.column_stack([xvec,yvec,zvec])
    np.savez(WORLD_FILE,origin=origin,R_world=R_world)
    print(f"  World frame saved: {WORLD_FILE}  ({len(avg)} corners, {good} good frames)")
    return R_world, origin


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR,exist_ok=True)

    cap0=cv2.VideoCapture(CAM0_INDEX,cv2.CAP_DSHOW)
    cap1=cv2.VideoCapture(CAM1_INDEX,cv2.CAP_DSHOW)
    for cap in (cap0,cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,IMAGE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Could not open cameras — check CAM0_INDEX / CAM1_INDEX")

    # ── Step 0: stereo extrinsics ─────────────────────────────────────────────
    if os.path.exists(STEREO_CALIB_FILE):
        d=np.load(STEREO_CALIB_FILE)
        R_stereo,T_stereo=d["R"],d["T"]
        print(f"Loaded stereo calib: {STEREO_CALIB_FILE}")
    else:
        R_stereo,T_stereo=run_stereo_calibration(cap0,cap1)

    P0=P(K0,np.eye(3),np.zeros((3,1)))
    P1=P(K1,R_stereo,T_stereo)
    board,det=make_charuco()

    # ── Step 1: world frame (first WORLD_FRAME_SECS of the session) ───────────
    print("\nPoint both cameras at the ChArUco board now.")
    print(f"Recording world frame for {WORLD_FRAME_SECS} seconds …")
    R_world,world_origin=build_world_frame(cap0,cap1,P0,P1,board,det)

    # ── Step 2: body tracking (S=start, Q=stop) ───────────────────────────────
    print("\n[BODY TRACKING]  Get into position.")
    print("  S = start recording  |  Q = stop + save")

    sync=TeensySyncLogger(TEENSY_PORT,TEENSY_BAUD,SYNC_CSV_FILE,SERIAL_TIMEOUT)
    sync.start()
    fps0=FpsTracker(FPS_WINDOW); fps1=FpsTracker(FPS_WINDOW)
    t0=PoseTracker(); t1=PoseTracker()

    results_3d=[]; frame_idx=0; recording=False; no_warn=False

    WIN="Dual Camera — World Frame Setup + Body Tracking"

    try:
        while True:
            r0,f0=cap0.read(); r1,f1=cap1.read()
            if not r0 or not r1: continue
            fps0.tick(); fps1.tick()
            lm0=t0.process(f0); lm1=t1.process(f1)

            lm3d=None
            if lm0 and lm1:
                chk=(ORG_NAME,XDIR_NAME,ZDIR_NAME)
                if min(min(lm0[n][2] for n in chk),min(lm1[n][2] for n in chk))>=VIS_THRESHOLD:
                    raw={}
                    for nm in LANDMARKS_3D_NAMES:
                        uv0=undistort(lm0[nm][:2],K0,DIST0)
                        uv1=undistort(lm1[nm][:2],K1,DIST1)
                        pt=triangulate(P0,P1,uv0,uv1)
                        raw[nm]=tuple(R_world.T@(pt-world_origin))
                    lm3d=raw

            if recording:
                results_3d.append({"frame_idx":frame_idx,"landmarks_3d":lm3d})
                sync.log_frame(frame_idx,fps0.fps(),fps1.fps())
                frame_idx+=1

            d0=t0.draw(f0,lm0); d1=t1.draw(f1,lm1)

            # status overlays
            phase_txt="● REC" if recording else "● STANDBY — press S to record"
            phase_col=(0,0,255) if recording else (0,200,200)
            cv2.putText(d0,phase_txt,(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,phase_col,2)
            cv2.putText(d1,"CAM1",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,0),1)
            if recording:
                cv2.putText(d0,f"frame {frame_idx}",(8,55),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),1)

            since=sync.since_sync()
            if since is None:     stxt,sc="TEENSY: no pulse",(0,0,255)
            elif since>2.0:
                stxt,sc=f"TEENSY: stale {since:.1f}s",(0,0,255)
                if not no_warn: print("WARNING: Teensy pulse missing >2s"); no_warn=True
            else:                 stxt,sc="TEENSY: live",(0,255,0); no_warn=False
            cv2.putText(d0,stxt,(8,78),cv2.FONT_HERSHEY_SIMPLEX,0.5,sc,1)
            cv2.putText(d0,"S=start  Q=stop+save",(8,IMAGE_HEIGHT-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

            cv2.imshow(WIN, cv2.hconcat([scale(d0),scale(d1)]))
            key=cv2.waitKey(1)&0xFF
            if key in (ord('s'),ord('S')) and not recording:
                recording=True; print("  Recording STARTED.")
            elif key in (ord('q'),ord('Q')):
                break
    finally:
        cap0.release(); cap1.release()
        t0.close(); t1.close()
        cv2.destroyAllWindows()
        sync.stop()

    n_total=len(results_3d)
    n_valid=sum(1 for r in results_3d if r["landmarks_3d"])
    print(f"\n{n_total} frames recorded, {n_valid} valid 3D.")
    if n_total==0: print("Nothing recorded — exiting."); return

    with open(OUT_3D_FILE,"wb") as f: pickle.dump(results_3d,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {OUT_3D_FILE}")

    data=smooth_lm(results_3d,SMOOTH_WINDOW)
    euler_log,trunk_traj,ref=compute_traj_angles(data)
    if not euler_log: print("No valid trajectory."); return

    out={"reference_frame":ref,"euler_angles":euler_log,"trunk_trajectory":trunk_traj}
    with open(OUT_TRAJ_FILE,"wb") as f: pickle.dump(out,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {OUT_TRAJ_FILE}")
    print("\nFirst 5 frames:")
    for e in euler_log[:5]:
        print(f"  frame {e['frame_idx']:>4}: X={e['angle_x_deg']:+.2f}° Y={e['angle_y_deg']:+.2f}° Z={e['angle_z_deg']:+.2f}°")
    save_plots(euler_log,trunk_traj)
    print("\n=== Done ===")

if __name__=="__main__":
    main()
