from collections import Counter
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0            # default webcam
IMGSZ        = 480          # inference resolution (square)
CONF_THRES   = 0.60         # detection confidence threshold
IOU_THRES    = 0.45         # NMS IoU threshold
MODEL_PATH   = "yolov10n.pt"  # your model file
TARGET_FPS   = 5            # desired FPS
SNAP_DIR     = "snapshots"  # snapshot save dir
os.makedirs(SNAP_DIR, exist_ok=True)

MAX_IOU_THRESHOLD = 0.5     # threshold to consider same object

# ──────────────────────────────────────────────────────────────
# Load YOLO model (CPU only)
# ──────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH).to("cpu")

# ──────────────────────────────────────────────────────────────
# Initialize webcam
# ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check index/permissions.")

# ──────────────────────────────────────────────────────────────
# IoU function
# ──────────────────────────────────────────────────────────────
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0

# ──────────────────────────────────────────────────────────────
# Initialize variables
# ──────────────────────────────────────────────────────────────
fps_ema = 0.0
prev_time = time.time()
frame_idx = 0
session_cnt = Counter()
saved_instances = {}  # class_name -> list of saved boxes
min_frame_interval = 1.0 / TARGET_FPS

print("[INFO] Starting inference loop. Press 'q' to exit.")

# ──────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────
while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame grab failed. Exiting…")
        break

    result = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",
        half=False,
        stream=False,
        verbose=False,
    )[0]

    boxes = result.boxes
    detected_this_frame = False
    new_object_detected = False
    frame_classes_this_frame = set()

    if boxes is not None and len(boxes):
        clss = boxes.cls.cpu().numpy().astype(int)
        names = [model.names[c] for c in clss]
        counts = Counter(names)
        session_cnt.update(counts)
        detected_this_frame = True

        for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy().astype(int), clss, boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            cls_name = model.names[int(cls_id)]
            frame_classes_this_frame.add(cls_name)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if new object using IoU
            prev_boxes = saved_instances.get(cls_name, [])
            is_new = True
            for prev_box in prev_boxes:
                if compute_iou([x1, y1, x2, y2], prev_box) > MAX_IOU_THRESHOLD:
                    is_new = False
                    break

            if is_new:
                new_object_detected = True
                saved_instances.setdefault(cls_name, []).append([x1, y1, x2, y2])

        # Print detected object summary
        joined = " ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"Frame {frame_idx}: {joined}")

    # Save snapshot only for new object
    if new_object_detected:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for cls_name in frame_classes_this_frame:
            cls_folder = os.path.join(SNAP_DIR, cls_name)
            os.makedirs(cls_folder, exist_ok=True)
            snap_path = os.path.join(cls_folder, f"snap_{ts}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[SNAP] Saved {snap_path} for new object: {cls_name}")


    # FPS overlay
    curr_time = time.time()
    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv10 – Real‑Time Detection", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Frame-rate limiter
    elapsed = time.time() - loop_start
    sleep_time = max(0.0, min_frame_interval - elapsed)
    if sleep_time:
        time.sleep(sleep_time)

# ──────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("[INFO] Session complete. Total objects detected:")
for name, count in session_cnt.items():
    print(f"  {name}: {count}")