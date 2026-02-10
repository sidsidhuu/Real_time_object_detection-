# Real-Time Object Detection

Real-time object detection from a webcam, powered by YOLOv10 and served through a Flask web UI. Snapshots are saved per session and per class.

## Tech Stack

- Python
- Flask
- OpenCV
- Ultralytics YOLOv10
- HTML/CSS

## Project Structure

```
.
├── app.py
├── rt_object_detection.py
├── requirements.txt
├── templates/
│   ├── detection.html
│   ├── index.html
│   └── snapshots.html
├── static/
│   └── style.css
├── snapshots/                # optional local output (ignored)
└── yolov10n.pt                # download separately (ignored)
```

## Prerequisites

- Python 3.9+ recommended
- A webcam connected to the machine
- YOLOv10 model file (`yolov10n.pt`)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the YOLOv10 model file:

https://github.com/THU-MIG/yolov10/releases

Place `yolov10n.pt` in the project root.

## Run

```bash
python app.py
```

Open the app:

http://127.0.0.1:5000/

## Using the App

- Enter a session name (optional) and start detection
- The sidebar lists detected objects with confidence and time
- Stop detection to browse snapshots

## Snapshot Output

Snapshots are stored under:

```
static/snapshots/<session_name>/<class_name>/
```

Each image is named with a timestamp.
