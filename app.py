from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
import cv2
from datetime import datetime
from ultralytics import YOLO
import os

app = Flask(__name__)
app.secret_key = 'secret_key'

model = YOLO("yolov10n.pt").to("cpu")

camera = None
running = False
latest_detections = []
detection_history = {}
session_name = None

# ✅ Store snapshots inside static folder so they can be accessed in HTML
output_dir = os.path.join("static", "snapshots")
os.makedirs(output_dir, exist_ok=True)


def generate_frames():
    global running, latest_detections, detection_history, camera, session_name
    while running and camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            flash("[ERROR] Failed to read from camera.", "error")
            break

        results = model.predict(source=frame, device="cpu", stream=False, verbose=False)[0]
        boxes = results.boxes

        if boxes is not None and len(boxes):
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            for (box, cls_id, conf) in zip(xyxy, clss, confs):
                x1, y1, x2, y2 = box
                class_name = model.names[int(cls_id)]
                confidence = float(conf) * 100
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Track confidence and timestamp
                if class_name not in detection_history:
                    detection_history[class_name] = {
                        "confidence": round(confidence, 2),
                        "timestamp": timestamp
                    }
                elif confidence > detection_history[class_name]["confidence"]:
                    detection_history[class_name]["confidence"] = round(confidence, 2)

                # Draw box
                color = (0, 165, 255)
                label = f"{class_name} {confidence:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Save snapshot
                class_dir = os.path.join(output_dir, session_name, class_name)
                os.makedirs(class_dir, exist_ok=True)
                snap_path = os.path.join(class_dir, f"{timestamp.replace(':', '_')}.jpg")
                cv2.imwrite(snap_path, frame)

        # Update latest detections
        latest_detections = []
        for idx, (cls, data) in enumerate(detection_history.items(), start=1):
            latest_detections.append({
                "index": idx,
                "class_name": cls,
                "confidence": data["confidence"],
                "timestamp": data["timestamp"]
            })

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    global camera, running, detection_history, session_name
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            flash("[ERROR] Failed to open the camera.", "error")
            return redirect(url_for('index'))
        else:
            flash("[INFO] Camera started successfully.", "success")

    session_input = request.form.get("session_name")
    if session_input:
        session_name = session_input.strip().replace(" ", "_")
    else:
        session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    running = True
    detection_history = {}
    return redirect(url_for('detection', session=session_name))


@app.route('/detection')
def detection():
    return render_template('detection.html', session_name=session_name)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    return jsonify(detections=latest_detections)


@app.route('/stop', methods=['POST'])
def stop():
    global running, camera
    running = False
    flash("[INFO] Detection stopped. You can now view session snapshots.", "info")
    if camera and camera.isOpened():
        camera.release()
        flash("[INFO] Camera released.", "info")
    return redirect(url_for('detection', session=session_name))


@app.route('/snapshots')
def snapshots():
    session = request.args.get("session")
    session_dir = os.path.join(output_dir, session)
    snapshots = {}

    if not os.path.exists(session_dir):
        flash(f"[ERROR] Session '{session}' not found.", "error")
        return redirect(url_for('index'))

    for cls in os.listdir(session_dir):
        class_path = os.path.join(session_dir, cls)
        if os.path.isdir(class_path):
            images = [
                url_for('static', filename=f'snapshots/{session}/{cls}/{img}')
                for img in os.listdir(class_path)
                if img.lower().endswith(('.jpg', '.png'))
            ]
            snapshots[cls] = images

    return render_template('snapshots.html', session=session, snapshots=snapshots)


@app.route('/exit', methods=['POST'])
def exit_app():
    global running, camera
    running = False
    if camera and camera.isOpened():
        camera.release()
    os._exit(0)
    return "Exited"


if __name__ == '__main__':
    app.run(debug=True)
