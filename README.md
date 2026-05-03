<img src="header-hexvis.png" alt="Hex-Vision" width="50%">

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white) ![Status](https://img.shields.io/badge/status-alpha-ff7a18) ![Platform](https://img.shields.io/badge/platform-windows-0078D4?logo=windows&logoColor=white) ![Model](https://img.shields.io/badge/model-yolov8-111111) ![UI](https://img.shields.io/badge/ui-customtkinter-1f6feb)

Realtime object segmentation and tracking with a control-focused UI. Built around screen capture, YOLOv8 segmentation, and telemetry visuals for robotics-style workflows.

![Hex-Vision in action](hexvision-demo.gif)

## What it does

- Captures a user-defined RGB region and runs YOLOv8 segmentation.
- Displays live detections, threat matrix, and telemetry panels.
- Provides goal modes (avoid, follow, search) with configurable limits.
- Includes visual motor vector and predictive path widgets.
- Translates motion vector to mouse position for Chica Client controller app.

## Requirements

- Python 3.9+
- Windows 10/11 recommended
- A compatible GPU is optional but improves FPS

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Usage

1. Click "Set RGB Camera Zone" and drag to define the capture area.
2. (Optional) Set the depth and controller regions if you use them.
3. Choose a goal mode and target class.
4. Click "Start Vision" to begin tracking.
