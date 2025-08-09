# Optimized YOLO + MediaPipe (Hands + Elbows + Mouth) with conditional rotations + calibration
# pip install ultralytics opencv-python mediapipe

import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import math

# ---------- CONFIG ----------
MODEL_PATH   = "yolov10b.pt"  # or yolov8n/s/m/l/x.pt
CONF         = 0.10           # YOLO confidence
IMG_SIZE     = 768            # try 640 on CPU for more FPS; 832/960 on GPU
DEVICE       = "mps"          # 0 for CUDA, "cpu", or "mps" (Apple)
HALF         = False          # set True on CUDA for FP16
CAM_INDEX    = 0
FRAME_W      = 1280           # camera request (try 1280x720)
FRAME_H      = 720
TARGET_CLASS = "spoon"       # None = all classes
ELBOW_VIS_TH = 0.30
# Only used if 0° finds nothing:
SECONDARY_ANGLES = [90, 45]   # add/remove angles as desired
IOU_NMS      = 0.50
FPS_SMOOTH_N = 20
# Mouth proximity default scale (distance threshold = SCALE * ear_distance)
DEFAULT_MOUTH_SCALE = 0.8
# ----------------------------

# --- OpenCV runtime opts ---
cv2.setUseOptimized(True)

# --- MediaPipe setup (lighter) ---
mp_hands  = mp.solutions.hands
mp_pose   = mp.solutions.pose
mp_mesh   = mp.solutions.face_mesh
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def to_pixels(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------- geometry helpers ----------
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)

def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0: return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        if len(idxs) == 1: break
        rest = idxs[1:]
        mask = np.array([iou_xyxy(boxes[i], boxes[j]) < iou_thresh for j in rest])
        idxs = rest[mask]
    return keep

# ---------- rotation helpers (any angle) ----------
def rotate_with_matrix(frame, angle_deg):
    h, w = frame.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    invM = cv2.invertAffineTransform(M)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated, M, invM

def apply_affine_to_points(pts, M):
    pts_np = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    out = cv2.transform(pts_np, M)  # (N,1,2)
    return [(int(round(p[0][0])), int(round(p[0][1]))) for p in out]

def map_box_back_arbitrary(x1r, y1r, x2r, y2r, invM, W, H):
    corners_r = [(x1r, y1r), (x2r, y1r), (x2r, y2r), (x1r, y2r)]
    corners_o = apply_affine_to_points(corners_r, invM)
    xs = [x for x,_ in corners_o]; ys = [y for _,y in corners_o]
    bx1, by1, bx2, by2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return clamp_box(bx1, by1, bx2, by2, W, H)

# ---------- detection ----------
def detect_on_frame(model, frame_bgr, class_filter, conf, imgsz, device):
    """Returns list of (x1,y1,x2,y2, cls_id, score)."""
    h, w = frame_bgr.shape[:2]
    res = model.predict(
        source=frame_bgr, imgsz=imgsz, conf=conf, device=device,
        verbose=False, classes=class_filter
    )[0]
    out = []
    if res.boxes is None: return out
    for b in res.boxes:
        cls_id = int(b.cls[0]); score = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        out.append((x1, y1, x2, y2, cls_id, score))
    return out

def detect_conditional_rotations(model, frame_bgr, class_filter, conf, imgsz, device, angles, iou_merge):
    """Try 0° first, else fallback angles; map boxes back; NMS merge if needed."""
    H, W = frame_bgr.shape[:2]

    base = detect_on_frame(model, frame_bgr, class_filter, conf, imgsz, device)
    if base:
        return base

    all_boxes, all_scores, all_cls = [], [], []
    for ang in angles:
        rot, M, invM = rotate_with_matrix(frame_bgr, ang)
        res = model.predict(
            source=rot, imgsz=imgsz, conf=conf, device=device,
            verbose=False, classes=class_filter
        )[0]
        if res.boxes is None:
            continue
        Hr, Wr = rot.shape[:2]
        for b in res.boxes:
            cls_id = int(b.cls[0]); score = float(b.conf[0])
            xr1, yr1, xr2, yr2 = map(int, b.xyxy[0].cpu().numpy())
            xr1, yr1, xr2, yr2 = clamp_box(xr1, yr1, xr2, yr2, Wr, Hr)
            bx1, by1, bx2, by2 = map_box_back_arbitrary(xr1, yr1, xr2, yr2, invM, W, H)
            all_boxes.append([bx1, by1, bx2, by2])
            all_scores.append(score)
            all_cls.append(cls_id)

        # Optional early-exit after first successful angle:
        if len(all_boxes) > 0:
            break

    if not all_boxes:
        return []
    keep = nms(all_boxes, all_scores, iou_merge)
    return [(all_boxes[i][0], all_boxes[i][1], all_boxes[i][2], all_boxes[i][3],
             all_cls[i], float(all_scores[i])) for i in keep]

# ---------- main ----------
def main():
    # Load YOLO
    model = YOLO(MODEL_PATH)
    if HALF and DEVICE != "cpu":
        try:
            model.fuse()
        except Exception:
            pass
    names = model.names
    print("YOLO classes:", names)

    # Class filter => numeric id(s) for Ultralytics 'classes='
    class_filter = None
    if TARGET_CLASS:
        ids = [i for i, n in names.items() if str(n).lower() == TARGET_CLASS]
        if ids:
            class_filter = ids
        else:
            print(f'WARNING: class "{TARGET_CLASS}" not in model; running with all classes.')

    # Webcam (low-latency settings)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # MediaPipe (lighter configs)
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,  # 0 = lighter
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=0, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    face = mp_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # Calibration state
    mouth_scale = DEFAULT_MOUTH_SCALE     # distance threshold = mouth_scale * ear_distance
    last_near = False                     # debouncer for "at mouth" event

    # FPS smoother
    fps_deque = deque(maxlen=FPS_SMOOTH_N)
    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame"); break

            h, w = frame.shape[:2]

            # --- MediaPipe ALWAYS (original frame) ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(img_rgb)
            pose_results = pose.process(img_rgb)
            face_results = face.process(img_rgb)

            # Draw hands + get a right-hand index tip (if available)
            right_index_px = None
            if hand_results.multi_hand_landmarks:
                for hand_lms, handed in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    mp_draw.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    label = handed.classification[0].label  # "Left"/"Right"
                    if label == "Right":
                        li = hand_lms.landmark[8]  # index fingertip
                        right_index_px = to_pixels(li, w, h)

            # Pose elbows + ears
            left_ear_px = right_ear_px = None
            if pose_results.pose_landmarks:
                lms = pose_results.pose_landmarks.landmark

                le = lms[mp_pose.PoseLandmark.LEFT_ELBOW]
                if le.visibility >= ELBOW_VIS_TH:
                    x, y = to_pixels(le, w, h)
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

                re = lms[mp_pose.PoseLandmark.RIGHT_ELBOW]
                if re.visibility >= ELBOW_VIS_TH:
                    x, y = to_pixels(re, w, h)
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

                # Ears for head width (scale)
                le_ar = lms[mp_pose.PoseLandmark.LEFT_EAR]
                re_ar = lms[mp_pose.PoseLandmark.RIGHT_EAR]
                if le_ar.visibility > 0.3 and re_ar.visibility > 0.3:
                    left_ear_px  = to_pixels(le_ar, w, h)
                    right_ear_px = to_pixels(re_ar, w, h)
                    cv2.circle(frame, left_ear_px, 4, (255, 255, 0), -1)
                    cv2.circle(frame, right_ear_px, 4, (255, 255, 0), -1)
                    cv2.line(frame, left_ear_px, right_ear_px, (255, 255, 0), 2)

            # FaceMesh mouth center
            mouth_center_px = None
            if face_results.multi_face_landmarks:
                # draw lips only (lighter overlay)
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(thickness=2, circle_radius=1)
                )
                lmsf = face_results.multi_face_landmarks[0].landmark
                try:
                    upper_inner = lmsf[13]; lower_inner = lmsf[14]
                    ux, uy = to_pixels(upper_inner, w, h)
                    lx, ly = to_pixels(lower_inner, w, h)
                    mouth_center_px = ((ux + lx)//2, (uy + ly)//2)
                    cv2.circle(frame, mouth_center_px, 4, (0, 255, 255), -1)
                except IndexError:
                    pass

            # --- YOLO: try 0°, else conditional rotations ---
            detections = detect_conditional_rotations(
                model=model,
                frame_bgr=frame,
                class_filter=class_filter,
                conf=CONF,
                imgsz=IMG_SIZE,
                device=DEVICE,
                angles=SECONDARY_ANGLES,
                iou_merge=IOU_NMS
            )

            # Draw YOLO detections; pick top bottle for proximity logic
            top_bottle_center = None
            top_bottle_score = -1.0

            for (x1, y1, x2, y2, cls_id, score) in detections:
                name = names.get(cls_id, str(cls_id))
                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                y1t = max(0, y1 - th - 4)
                cv2.rectangle(frame, (x1, y1t), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                if score > top_bottle_score:
                    top_bottle_score = score
                    top_bottle_center = ((x1 + x2)//2, (y1 + y2)//2)

            # -------- Calibration + "bottle at mouth" trigger --------
            ear_dist = None
            if left_ear_px and right_ear_px:
                ear_dist = euclid(left_ear_px, right_ear_px)
                cv2.putText(frame, f"HeadW: {ear_dist:.0f}px  Scale: {mouth_scale:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

            # Press 'c' to calibrate using current RIGHT index fingertip -> mouth distance
            # (Stores mouth_scale = dist(hand, mouth) / ear_dist)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if ear_dist and mouth_center_px and right_index_px:
                    d = euclid(right_index_px, mouth_center_px)
                    if ear_dist > 1e-6:
                        mouth_scale = max(0.1, min(2.0, d / ear_dist))
                        print(f"[Calibrated] mouth_scale = {mouth_scale:.3f}")
                else:
                    print("Calibration needs visible ears, mouth, and right index fingertip.")

            # Compute proximity (bottle↔mouth)
            if ear_dist and mouth_center_px and top_bottle_center:
                thresh = mouth_scale * ear_dist
                d_bm = euclid(top_bottle_center, mouth_center_px)
                near = d_bm <= thresh
                # Debounce print
                if near and not last_near:
                    print("Bottle reached mouth!")
                last_near = near

                # Visualize threshold circle at mouth
                cv2.circle(frame, mouth_center_px, int(thresh), (0, 200, 255), 2)
                cv2.line(frame, mouth_center_px, top_bottle_center, (0, 200, 255), 2)

            # (Optional) also check hand→mouth
            if ear_dist and mouth_center_px and right_index_px:
                thresh = mouth_scale * ear_dist
                d_hm = euclid(right_index_px, mouth_center_px)
                cv2.line(frame, mouth_center_px, right_index_px, (180, 180, 255), 1)
                cv2.putText(frame, f"Hand-mouth: {d_hm:.0f}/{thresh:.0f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

            # FPS (smoothed)
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Optimized: YOLO + MediaPipe (Hands+Elbows+Mouth) + Calibration", frame)

            # standard quit
            if key == ord('q'):
                break

    finally:
        hands.close(); pose.close(); face.close()
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
