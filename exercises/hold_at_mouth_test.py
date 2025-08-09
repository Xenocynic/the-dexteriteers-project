import cv2
import time
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp

# -------- CONFIG --------
MODEL_PATH   = "yolov10b.pt"
CONF         = 0.50
IMG_SIZE     = 768
DEVICE       = "mps"
HALF         = False
CAM_INDEX    = 0
FRAME_W      = 1280
FRAME_H      = 720
FLIP_VIEW    = True
TARGET_CLASS = "bottle"
IOU_NMS      = 0.50
FPS_SMOOTH_N = 20

# Hold-specific config
HOLD_TIME_REQUIRED = 5.0  # seconds
DEFAULT_MOUTH_SCALE = 0.80  # starting distance threshold
MIN_SCALE, MAX_SCALE = 0.10, 2.50
POSITION_TOLERANCE = 0.2  # relative to head width

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def to_pixels(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def detect_on_frame(model, frame_bgr, class_filter, conf, imgsz, device):
    res = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, device=device,
                       verbose=False, classes=class_filter)[0]
    out = []
    if res.boxes is None: return out
    h, w = frame_bgr.shape[:2]
    for b in res.boxes:
        cls_id = int(b.cls[0]); score = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        out.append((x1, y1, x2, y2, cls_id, score))
    return out

def main():
    print("🦷 Hold at Mouth Test")
    print("Hold bottle at mouth level for 5 seconds")
    print("Press 'c' to calibrate, 's' to start test")

    # YOLO setup
    model = YOLO(MODEL_PATH)
    names = model.names
    class_filter = None
    if TARGET_CLASS:
        ids = [i for i, n in names.items() if str(n).lower() == TARGET_CLASS]
        if ids: class_filter = ids

    # Camera setup
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # MediaPipe setup
    face = mp_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                           refine_landmarks=True,
                           min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)

    # State tracking
    mouth_scale = DEFAULT_MOUTH_SCALE
    test_started = False
    hold_start = None
    initial_pos = None
    max_deviation = 0
    test_complete = False
    fps_deque = deque(maxlen=FPS_SMOOTH_N)
    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if FLIP_VIEW:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face and pose
            face_results = face.process(img_rgb)
            pose_results = pose.process(img_rgb)

            # Get ear positions for scale
            ear_dist = None
            if pose_results.pose_landmarks:
                lms = pose_results.pose_landmarks.landmark
                l_ear = lms[mp_pose.PoseLandmark.LEFT_EAR]
                r_ear = lms[mp_pose.PoseLandmark.RIGHT_EAR]
                l_ear_px = to_pixels(l_ear, w, h)
                r_ear_px = to_pixels(r_ear, w, h)
                ear_dist = euclid(l_ear_px, r_ear_px)
                cv2.line(frame, l_ear_px, r_ear_px, (0,255,255), 2)

            # Get mouth position
            mouth_center = None
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,255))
                )
                upper = landmarks[13]  # Upper lip
                lower = landmarks[14]  # Lower lip
                mouth_x = int((upper.x + lower.x) * w / 2)
                mouth_y = int((upper.y + lower.y) * h / 2)
                mouth_center = (mouth_x, mouth_y)
                cv2.circle(frame, mouth_center, 5, (0,255,255), -1)

            # Detect bottle
            bottle_center = None
            detections = detect_on_frame(model, frame, class_filter, CONF, IMG_SIZE, DEVICE)
            for (x1, y1, x2, y2, cls_id, score) in detections:
                name = names.get(cls_id, str(cls_id))
                if name.lower() == TARGET_CLASS:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    bottle_center = ((x1+x2)//2, (y1+y2)//2)

            # Test logic
            if test_started and not test_complete and bottle_center and mouth_center and ear_dist:
                thresh = mouth_scale * ear_dist
                dist = euclid(bottle_center, mouth_center)
                
                # Visualize threshold
                cv2.circle(frame, mouth_center, int(thresh), (0,200,255), 2)
                cv2.line(frame, mouth_center, bottle_center, (0,200,255), 2)
                
                if dist <= thresh:
                    if hold_start is None:
                        hold_start = time.time()
                        initial_pos = bottle_center
                        print("Position acquired - hold steady!")
                    else:
                        hold_time = time.time() - hold_start
                        
                        # Check stability
                        curr_deviation = euclid(bottle_center, initial_pos)
                        max_deviation = max(max_deviation, curr_deviation)
                        stability = max_deviation / ear_dist  # Normalize by head width
                        
                        # Draw timer
                        remaining = max(0, HOLD_TIME_REQUIRED - hold_time)
                        cv2.putText(frame, f"HOLD: {remaining:.1f}s", 
                                  (w//2-100, 100), cv2.FONT_HERSHEY_TRIPLEX,
                                  1.2, (0,255,0), 3)
                        
                        # Check completion
                        if hold_time >= HOLD_TIME_REQUIRED:
                            test_complete = True
                            passed = stability <= POSITION_TOLERANCE
                            
                            print("\n===== HOLD AT MOUTH TEST =====")
                            print(f"Hold Time: {hold_time:.1f}s")
                            print(f"Stability: {stability:.3f} (relative to head width)")
                            print("---------------------------")
                            if passed:
                                print("✅ TEST PASSED!")
                            else:
                                print("❌ TEST FAILED - Too much movement")
                            print("---------------------------")
                else:
                    if hold_start:
                        print("Position lost - realigning...")
                    hold_start = None
                    initial_pos = None

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not test_started:
                test_started = True
                print("\nTest started! Align bottle with mouth.")
            elif key == ord('c'):
                if bottle_center and mouth_center and ear_dist:
                    curr_dist = euclid(bottle_center, mouth_center)
                    mouth_scale = curr_dist / ear_dist
                    mouth_scale = max(MIN_SCALE, min(MAX_SCALE, mouth_scale))
                    print(f"\nCalibrated! New scale: {mouth_scale:.2f}")
            elif key == ord('r'):
                test_started = False
                hold_start = None
                initial_pos = None
                max_deviation = 0
                test_complete = False
                print("\nTest reset. Press 's' to start again.")
            elif key == ord('q'):
                break

            # Status display
            if not test_started:
                status = "Press 'c' to calibrate, 's' to start test"
            elif test_complete:
                status = "Test complete - press 'r' to retry"
            elif hold_start:
                status = "HOLDING - Keep steady!"
            else:
                status = "Align bottle with mouth"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0,200,200), 2)

            # FPS counter
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f"{fps_avg:.1f} FPS", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow("Hold at Mouth Test", frame)

    finally:
        face.close()
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()