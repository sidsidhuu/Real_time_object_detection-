from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
import cv2
import os

from detector import DetectionService

app = Flask(__name__)
app.secret_key = "secret_key"

camera = None
running = False
session_name = None

detector = DetectionService(
    model_path="yolov10n.pt",
    device="cpu",
    output_dir=os.path.join("static", "snapshots"),
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
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            flash("[ERROR] Failed to open the camera.", "error")
            return redirect(url_for('index'))
        else:
            flash("[INFO] Camera started successfully.", "success")

    session_input = request.form.get("session_name")
    session_name = detector.start_session(session_input)
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
