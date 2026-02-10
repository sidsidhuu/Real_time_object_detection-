import os
from datetime import datetime

import cv2
from ultralytics import YOLO


class DetectionService:
    def __init__(self, model_path="yolov10n.pt", device="cpu", output_dir=None):
        self.model_path = model_path
        self.device = device
        self.output_dir = output_dir or os.path.join("static", "snapshots")

        self.model = YOLO(self.model_path).to(self.device)
        self.session_name = None
        self.detection_history = {}
        self.latest_detections = []

        os.makedirs(self.output_dir, exist_ok=True)

    def start_session(self, session_name=None):
        cleaned = (session_name or "").strip().replace(" ", "_")
        if cleaned:
            self.session_name = cleaned
        else:
            self.session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self.detection_history = {}
        self.latest_detections = []
        return self.session_name

    def process_frame(self, frame):
        if not self.session_name:
            raise RuntimeError("Session not started.")

        results = self.model.predict(source=frame, device=self.device, stream=False, verbose=False)[0]
        boxes = results.boxes

        if boxes is not None and len(boxes):
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            for (box, cls_id, conf) in zip(xyxy, clss, confs):
                x1, y1, x2, y2 = box
                class_name = self.model.names[int(cls_id)]
                confidence = float(conf) * 100
                timestamp = self._timestamp_str()

                self._update_history(class_name, confidence, timestamp)
                self._draw_box(frame, x1, y1, x2, y2, class_name, confidence)
                self._save_snapshot(frame, class_name, timestamp)

        self._refresh_latest_detections()
        return frame

    def _timestamp_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _update_history(self, class_name, confidence, timestamp):
        if class_name not in self.detection_history:
            self.detection_history[class_name] = {
                "confidence": round(confidence, 2),
                "timestamp": timestamp,
            }
        elif confidence > self.detection_history[class_name]["confidence"]:
            self.detection_history[class_name]["confidence"] = round(confidence, 2)
            self.detection_history[class_name]["timestamp"] = timestamp

    def _draw_box(self, frame, x1, y1, x2, y2, class_name, confidence):
        color = (0, 165, 255)
        label = f"{class_name} {confidence:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _save_snapshot(self, frame, class_name, timestamp):
        class_dir = os.path.join(self.output_dir, self.session_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        safe_time = timestamp.replace(":", "_")
        snap_path = os.path.join(class_dir, f"{safe_time}.jpg")
        cv2.imwrite(snap_path, frame)

    def _refresh_latest_detections(self):
        self.latest_detections = []
        for idx, (cls, data) in enumerate(self.detection_history.items(), start=1):
            self.latest_detections.append(
                {
                    "index": idx,
                    "class_name": cls,
                    "confidence": data["confidence"],
                    "timestamp": data["timestamp"],
                }
            )
