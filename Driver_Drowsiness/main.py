import cv2
import mediapipe as mp
from scipy.spatial import distance
import winsound
import csv
import os
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# FACE MESH SETUP
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

# ============================================================
# LANDMARK INDICES
# ============================================================
LEFT_EYE         = [33, 133, 159, 145]
RIGHT_EYE        = [362, 263, 386, 374]
UPPER_LIP        = 13
LOWER_LIP        = 14
NOSE_TIP         = 1
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]

# ============================================================
# THRESHOLDS & COUNTERS
# ============================================================
EAR_THRESHOLD             = 0.20
FRAME_THRESHOLD           = 20
YAWN_THRESHOLD            = 20
YAWN_FRAME_THRESHOLD      = 15
ATTENTION_THRESHOLD       = 50
ATTENTION_FRAME_THRESHOLD = 30
LOOK_DOWN_THRESHOLD       = 15
LOOK_DOWN_FRAME_THRESHOLD = 25

counter           = 0
blink_count       = 0
alarm_on          = False
eye_closed        = False
no_face_counter   = 0
yawn_counter      = 0
attention_counter = 0
attention_alarm   = False
attention_text    = "LOOKING CENTER"
look_down_counter = 0

# ============================================================
# FATIGUE SCORE
# ============================================================
fatigue_score    = 0.0
MAX_FATIGUE      = 100
FATIGUE_DECAY    = 0.02
DROWSY_WEIGHT    = 2.0
YAWN_WEIGHT      = 1.5
ATTENTION_WEIGHT = 1.0

# ============================================================
# CSV LOGGING
# ============================================================
LOG_FILE = "driver_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp","ear","mouth_distance","yawn_detected",
            "drowsy_detected","attention_status","fatigue_score",
            "blink_count","head_pitch","head_yaw"
        ])

def log_event(ear, mouth_dist, yawn, drowsy, attention, fatigue, blinks, pitch, yaw):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{ear:.3f}", f"{mouth_dist:.2f}", int(yawn), int(drowsy),
            attention, f"{fatigue:.1f}", blinks, f"{pitch:.1f}", f"{yaw:.1f}"
        ])

# ============================================================
# EVENT HISTORY
# ============================================================
event_history = deque(maxlen=12)

def add_event(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    event_history.appendleft(f"[{ts}] {msg}")

# ============================================================
# SCREENSHOT
# ============================================================
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def save_screenshot(frame, reason):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{reason}_{ts}.jpg")
    cv2.imwrite(path, frame)
    add_event(f"Screenshot: {reason}")

# ============================================================
# STATS
# ============================================================
stats = {
    "total_blinks"           : 0,
    "total_yawns"            : 0,
    "total_drowsy_alerts"    : 0,
    "total_attention_alerts" : 0,
    "session_start"          : time.time(),
    "ear_history"            : deque(maxlen=150),
    "fatigue_history"        : deque(maxlen=150),
}

# ============================================================
# AI FATIGUE PREDICTION
# ============================================================
def build_fatigue_model():
    np.random.seed(42)
    n          = 2000
    ear_v      = np.random.uniform(0.10, 0.40, n)
    yawn_v     = np.random.uniform(0, 40, n)
    blink_v    = np.random.randint(0, 40, n)
    pitch_v    = np.random.uniform(-30, 30, n)
    labels     = (
        (ear_v < 0.22).astype(int) * 3 +
        (yawn_v > 22).astype(int)  * 2 +
        (blink_v < 10).astype(int) +
        (np.abs(pitch_v) > 20).astype(int)
    )
    labels     = (labels >= 3).astype(int)
    X          = np.column_stack([ear_v, yawn_v, blink_v, pitch_v])
    scaler     = StandardScaler()
    X_s        = scaler.fit_transform(X)
    model      = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_s, labels)
    return model, scaler

ai_model, ai_scaler = build_fatigue_model()
ai_prediction = "ALERT"

def predict_fatigue(ear, mouth_dist, blinks, pitch):
    X   = np.array([[ear, mouth_dist, blinks, pitch]])
    X_s = ai_scaler.transform(X)
    return "FATIGUED" if ai_model.predict(X_s)[0] == 1 else "ALERT"

# ============================================================
# HEAD POSE
# ============================================================
face_3d_model = np.array([
    [0.0,    0.0,    0.0   ],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0],
    [0.0,  -330.0,  -65.0 ],
], dtype=np.float64)

head_pitch = 0.0
head_yaw   = 0.0

def estimate_head_pose(face_landmarks, w, h):
    face_2d = np.array([
        [int(face_landmarks.landmark[i].x * w),
         int(face_landmarks.landmark[i].y * h)]
        for i in HEAD_POSE_POINTS
    ], dtype=np.float64)
    fl         = w
    cam_matrix = np.array([[fl,0,w/2],[0,fl,h/2],[0,0,1]], dtype=np.float64)
    dist_mat   = np.zeros((4,1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_mat)
    if not ok:
        return 0.0, 0.0
    rmat, _    = cv2.Rodrigues(rvec)
    angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)
    return angles[0]*360, angles[1]*360

# ============================================================
# PHONE USAGE (MediaPipe Hands)
# ============================================================
mp_hands    = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
phone_detected = False

# ============================================================
# DASHBOARD PANEL RENDERER  (draws right-side panel as OpenCV image)
# ============================================================
PANEL_W = 420
DASH_BG  = (26, 26, 46)     # dark navy
C_CYAN   = (255, 220, 0)    # BGR cyan
C_WHITE  = (230, 230, 230)
C_GREEN  = (80, 220, 80)
C_RED    = (60, 60, 220)
C_ORANGE = (40, 160, 230)
C_YELLOW = (0, 220, 220)
C_GRAY   = (100, 100, 100)

graph_cache = None   # will hold the graph image
graph_timer = 0

def render_graph(ear_hist, fat_hist, gw, gh):
    """Render EAR + Fatigue graphs into an OpenCV image of size (gw, gh)."""
    dpi = 72
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(gw/dpi, gh/dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(hspace=0.55, left=0.12, right=0.97, top=0.88, bottom=0.12)

    for ax, data, color, title, thresh in [
        (ax1, list(ear_hist), "#00d4ff", "EAR over time", EAR_THRESHOLD),
        (ax2, list(fat_hist), "#ff8844", "Fatigue Score",  70),
    ]:
        ax.set_facecolor("#0f0f23")
        ax.plot(data, color=color, linewidth=1.2)
        ax.axhline(y=thresh, color="#ff4444", linestyle="--", linewidth=0.8)
        ax.set_title(title, color=color, fontsize=7, pad=3)
        ax.tick_params(colors="#888888", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

def build_dashboard(h, ear, mouth, fatigue, blinks, pitch, yaw,
                    ai_pred, total_blinks, total_yawns,
                    drowsy_alerts, attention_alerts, session_secs,
                    events, ear_hist, fat_hist):
    global graph_cache, graph_timer

    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = DASH_BG

    # ---- Title ----
    cv2.rectangle(panel, (0,0), (PANEL_W, 38), (40,40,70), -1)
    cv2.putText(panel, "DRIVER MONITORING", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, C_CYAN, 1, cv2.LINE_AA)

    # ---- Section helper ----
    def section_title(y, text):
        cv2.rectangle(panel, (0, y), (PANEL_W, y+22), (50,50,80), -1)
        cv2.putText(panel, text, (8, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)

    def metric(y, label, value, color=C_WHITE):
        cv2.putText(panel, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(panel, str(value), (170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # ---- Live Metrics ----
    section_title(42, " LIVE METRICS")
    fat_col  = C_GREEN if fatigue < 50 else C_ORANGE if fatigue < 75 else C_RED
    ai_col   = C_GREEN if ai_pred == "ALERT" else C_RED
    metric(78,  "EAR:",          f"{ear:.3f}",        C_GREEN if ear > EAR_THRESHOLD else C_RED)
    metric(98,  "Mouth:",        f"{mouth:.1f}")
    metric(118, "Fatigue:",      f"{fatigue:.1f}/100", fat_col)
    metric(138, "Blinks:",       str(blinks))
    metric(158, "Pitch:",        f"{pitch:.1f} deg",  C_YELLOW)
    metric(178, "Yaw:",          f"{yaw:.1f} deg",    C_YELLOW)
    metric(198, "AI Predict:",   ai_pred,             ai_col)
    metric(218, "Attention:",    attention_text,      C_CYAN)

    # Fatigue bar
    bar_x, bar_y = 10, 228
    bar_maxw     = PANEL_W - 20
    bar_fill     = int((fatigue / MAX_FATIGUE) * bar_maxw)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_maxw, bar_y+10), (50,50,50), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_fill,  bar_y+10), fat_col,    -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x+bar_maxw,  bar_y+10), C_GRAY,     1)

    # ---- Session Stats ----
    section_title(248, " SESSION STATS")
    hh = session_secs//3600
    mm = (session_secs%3600)//60
    ss = session_secs%60
    metric(272, "Total Blinks:",     str(total_blinks))
    metric(290, "Total Yawns:",      str(total_yawns))
    metric(308, "Drowsy Alerts:",    str(drowsy_alerts),    C_RED if drowsy_alerts else C_WHITE)
    metric(326, "Attention Alerts:", str(attention_alerts), C_ORANGE if attention_alerts else C_WHITE)
    metric(344, "Session Time:",     f"{hh:02d}:{mm:02d}:{ss:02d}")

    # ---- Event History ----
    section_title(358, " EVENT HISTORY")
    ev_y = 382
    for ev in list(events)[:7]:
        cv2.putText(panel, ev[:52], (8, ev_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 220, 160), 1, cv2.LINE_AA)
        ev_y += 16

    # ---- Graphs ----
    graph_area_top = ev_y + 8
    graph_h        = h - graph_area_top - 5
    if graph_h > 80:
        now = time.time()
        if graph_cache is None or (now - graph_timer) > 1.0:
            graph_cache = render_graph(ear_hist, fat_hist, PANEL_W, graph_h)
            graph_timer = now
        g = graph_cache
        # resize in case of mismatch
        if g.shape[0] != graph_h or g.shape[1] != PANEL_W:
            g = cv2.resize(g, (PANEL_W, graph_h))
        panel[graph_area_top:graph_area_top+graph_h, 0:PANEL_W] = g

    # Divider line between camera and panel
    cv2.line(panel, (0,0), (0, h), C_CYAN, 2)

    return panel


# ============================================================
# MAIN LOOP
# ============================================================
log_timer   = time.time()
LOG_INTERVAL = 1.0
frame_count  = 0
ear          = 0.0
mouth_distance = 0.0
yawn_detected  = False
drowsy_detected = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame      = cv2.flip(frame, 1)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _    = frame.shape
    frame_count += 1

    yawn_detected    = False
    drowsy_detected  = False

    # ---- Hand / phone detection ----
    hand_results   = hands_model.process(rgb_frame)
    phone_detected = False
    if hand_results.multi_hand_landmarks:
        for hlm in hand_results.multi_hand_landmarks:
            wrist = hlm.landmark[0]
            if wrist.y < 0.5 and 0.3 < wrist.x < 0.7:
                phone_detected = True

    # ---- Face mesh ----
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        no_face_counter += 1
        if no_face_counter > 50:
            cv2.putText(frame, "DRIVER NOT FOUND", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if no_face_counter == 51:
                add_event("Driver not found!")
                save_screenshot(frame, "no_face")
    else:
        no_face_counter = 0

        for face_landmarks in results.multi_face_landmarks:

            # HEAD POSE
            head_pitch, head_yaw = estimate_head_pose(face_landmarks, w, h)

            # LOOKING DOWN
            if head_pitch > LOOK_DOWN_THRESHOLD:
                look_down_counter += 1
            else:
                look_down_counter = 0
            if look_down_counter > LOOK_DOWN_FRAME_THRESHOLD:
                cv2.putText(frame, "LOOKING DOWN!", (40, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
                if look_down_counter == LOOK_DOWN_FRAME_THRESHOLD + 1:
                    add_event("Looking down detected")
                    winsound.Beep(1200, 500)

            # ATTENTION
            nose    = face_landmarks.landmark[NOSE_TIP]
            nose_x  = int(nose.x * w)
            center_x = w // 2
            cv2.circle(frame, (nose_x, int(nose.y*h)), 6, (0,255,255), -1)
            cv2.line(frame, (center_x,0), (center_x,h), (255,255,0), 2)

            if nose_x < center_x - ATTENTION_THRESHOLD:
                attention_text = "LOOKING LEFT";  attention_counter += 1
            elif nose_x > center_x + ATTENTION_THRESHOLD:
                attention_text = "LOOKING RIGHT"; attention_counter += 1
            else:
                attention_text = "LOOKING CENTER"
                attention_counter = 0; attention_alarm = False

            if attention_counter > ATTENTION_FRAME_THRESHOLD:
                cv2.putText(frame, "PAY ATTENTION!", (40,360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                if not attention_alarm:
                    winsound.Beep(1500, 1000)
                    attention_alarm = True
                    stats["total_attention_alerts"] += 1
                    add_event(f"Attention: {attention_text}")
                    save_screenshot(frame, "attention")

            # EAR
            def get_pts(idxs):
                return {i:(int(face_landmarks.landmark[i].x*w),
                           int(face_landmarks.landmark[i].y*h)) for i in idxs}
            lp = get_pts(LEFT_EYE);  rp = get_pts(RIGHT_EYE)
            l_ear = distance.euclidean(lp[159],lp[145]) / distance.euclidean(lp[33],lp[133])
            r_ear = distance.euclidean(rp[386],rp[374]) / distance.euclidean(rp[362],rp[263])
            ear   = (l_ear + r_ear) / 2.0

            # DROWSINESS
            if ear < EAR_THRESHOLD:
                counter += 1
                if not eye_closed: eye_closed = True
            else:
                if eye_closed:
                    blink_count += 1; stats["total_blinks"] += 1
                eye_closed = False; counter = 0; alarm_on = False

            if counter > FRAME_THRESHOLD:
                drowsy_detected = True
                cv2.putText(frame, "DROWSINESS DETECTED", (40,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                if not alarm_on:
                    winsound.Beep(1000, 1000); alarm_on = True
                    stats["total_drowsy_alerts"] += 1
                    add_event("Drowsiness detected!")
                    save_screenshot(frame, "drowsy")

            # YAWN
            ul = face_landmarks.landmark[UPPER_LIP]
            ll = face_landmarks.landmark[LOWER_LIP]
            mouth_distance = distance.euclidean(
                (int(ul.x*w), int(ul.y*h)), (int(ll.x*w), int(ll.y*h))
            )
            if mouth_distance > YAWN_THRESHOLD: yawn_counter += 1
            else:                               yawn_counter  = 0
            if yawn_counter > YAWN_FRAME_THRESHOLD:
                yawn_detected = True
                cv2.putText(frame, "YAWN DETECTED", (40,180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                if yawn_counter == YAWN_FRAME_THRESHOLD + 1:
                    stats["total_yawns"] += 1; add_event("Yawn detected")

            # FATIGUE SCORE
            fatigue_score = max(0, fatigue_score - FATIGUE_DECAY)
            if drowsy_detected: fatigue_score = min(MAX_FATIGUE, fatigue_score + DROWSY_WEIGHT)
            if yawn_detected:   fatigue_score = min(MAX_FATIGUE, fatigue_score + YAWN_WEIGHT)
            if attention_counter > ATTENTION_FRAME_THRESHOLD:
                fatigue_score = min(MAX_FATIGUE, fatigue_score + ATTENTION_WEIGHT)

            fat_col = (0,255,0) if fatigue_score < 50 else \
                      (0,165,255) if fatigue_score < 75 else (0,0,255)
            bar_w = int((fatigue_score/MAX_FATIGUE)*200)
            cv2.rectangle(frame,(20,220),(220,240),(50,50,50),-1)
            cv2.rectangle(frame,(20,220),(20+bar_w,240),fat_col,-1)
            cv2.rectangle(frame,(20,220),(220,240),(200,200,200),1)

            # AI PREDICTION
            if frame_count % 30 == 0:
                ai_prediction = predict_fatigue(ear, mouth_distance, blink_count, head_pitch)
            if ai_prediction == "FATIGUED":
                cv2.putText(frame, "AI: FATIGUED", (40,400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # HUD on camera
            cv2.putText(frame, f"EAR: {ear:.2f}",           (20,40),  cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.putText(frame, f"Blinks: {blink_count}",    (20,70),  cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(frame, f"Mouth: {mouth_distance:.1f}",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
            cv2.putText(frame, f"Fatigue:{fatigue_score:.0f}",(20,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,fat_col,2)
            cv2.putText(frame, attention_text,               (20,160), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            stats["ear_history"].append(ear)
            stats["fatigue_history"].append(fatigue_score)

    # Phone overlay
    if phone_detected:
        cv2.putText(frame, "PHONE USAGE DETECTED!", (40,520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        if frame_count % 60 == 0:
            add_event("Phone usage detected!")
            winsound.Beep(1800, 500)

    # ---- Build dashboard panel ----
    elapsed = int(time.time() - stats["session_start"])
    dashboard = build_dashboard(
        h, ear, mouth_distance, fatigue_score, blink_count,
        head_pitch, head_yaw, ai_prediction,
        stats["total_blinks"], stats["total_yawns"],
        stats["total_drowsy_alerts"], stats["total_attention_alerts"],
        elapsed, event_history,
        stats["ear_history"], stats["fatigue_history"]
    )

    # ---- Stitch camera + dashboard side by side ----
    combined = np.hstack([frame, dashboard])
    cv2.imshow("Driver Monitoring System", combined)

    # ---- CSV log ----
    if time.time() - log_timer >= LOG_INTERVAL:
        log_event(ear, mouth_distance, yawn_detected, drowsy_detected,
                  attention_text, fatigue_score, blink_count, head_pitch, head_yaw)
        log_timer = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_screenshot(combined, "manual")

cap.release()
cv2.destroyAllWindows()