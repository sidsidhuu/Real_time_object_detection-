from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
import cv2
import os
import re

from db import init_db, list_detections, list_sessions
from detector import DetectionService

app = Flask(__name__)
app.secret_key = "secret_key"


def _get_env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sanitize_session_name(value):
    if not value:
        return ""
    cleaned = value.strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", cleaned)
    return cleaned[:40]

camera = None
running = False
session_name = None

camera_index = _get_env_int("CAMERA_INDEX", 0)
model_path = os.environ.get("MODEL_PATH", "yolov10n.pt")
snapshot_dir = os.environ.get("SNAPSHOT_DIR", os.path.join("static", "snapshots"))
db_path = os.environ.get("DETECTION_DB", os.path.join("data", "detections.db"))
init_db(db_path)

detector = DetectionService(
    model_path=model_path,
    device="cpu",
    output_dir=snapshot_dir,
    db_path=db_path,
)


def generate_frames():
    global running, camera, session_name
    while running and camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            flash("[ERROR] Failed to read from camera.", "error")
            break

        try:
            frame = detector.process_frame(frame)
        except RuntimeError as exc:
            flash(f"[ERROR] {exc}", "error")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    global camera, running, session_name
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            flash("[ERROR] Failed to open the camera.", "error")
            return redirect(url_for('index'))
        else:
            flash("[INFO] Camera started successfully.", "success")

    session_input = request.form.get("session_name")
    cleaned = _sanitize_session_name(session_input)
    if session_input and not cleaned:
        flash("[ERROR] Session name must use letters, numbers, hyphens, or underscores.", "error")
        return redirect(url_for('index'))
    if session_input and cleaned != session_input.strip().replace(" ", "_"):
        flash("[INFO] Session name was normalized for safety.", "info")
    session_name = detector.start_session(cleaned)
    running = True
    return redirect(url_for('detection', session=session_name))


@app.route('/detection')
def detection():
    return render_template('detection.html', session_name=session_name)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    return jsonify(detections=detector.latest_detections)


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
    if not session:
        flash("[ERROR] Session is required to view snapshots.", "error")
        return redirect(url_for('index'))
    session_dir = os.path.join(detector.output_dir, session)
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


@app.route('/api/sessions')
def api_sessions():
    return jsonify(sessions=list_sessions(db_path=db_path))


@app.route('/api/detections')
def api_detections():
    session = request.args.get("session")
    if not session:
        return jsonify(error="session query parameter is required"), 400
    limit = _get_env_int("DETECTIONS_LIMIT", 100)
    return jsonify(detections=list_detections(session, limit=limit, db_path=db_path))


@app.route('/health')
def health():
    return jsonify(status="ok")


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
