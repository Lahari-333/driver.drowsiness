# Driver Monitoring System

## Overview
Driver Monitoring System is a real-time computer vision project developed using Python, OpenCV, and MediaPipe. The system monitors driver behavior through a webcam and detects signs of fatigue, drowsiness, yawning, and driver inattention. Audio alerts are generated to improve driver safety and reduce the risk of accidents caused by fatigue.

---

## Features

* Real-time face detection using MediaPipe Face Mesh
* Eye detection and Eye Aspect Ratio (EAR) calculation
* Drowsiness detection based on prolonged eye closure
* Audio alarm system for drowsiness alerts
* Yawn detection using mouth landmark analysis
* Driver attention monitoring (left, right, center tracking)
* Attention warning alerts for prolonged distraction
* Real-time facial landmark tracking
* Live webcam-based monitoring

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* SciPy
* Winsound
* Computer Vision
* Facial Landmark Detection

---

## Project Workflow

Webcam Input
→ Face Detection
→ Facial Landmark Extraction
→ Eye Detection & EAR Calculation
→ Drowsiness Detection
→ Yawn Detection
→ Attention Monitoring
→ Audio Alerts & Warnings

---

## Installation

1. Clone the repository

```bash
git clone <repository-link>
```

2. Install dependencies

```bash
pip install opencv-python mediapipe scipy
```

3. Run the project

```bash
python main.py
```

---

## Results

The system successfully performs:

* Drowsiness detection through eye closure analysis
* Yawn detection using mouth landmarks
* Driver attention monitoring
* Real-time alert generation through audio alarms

---

## Future Enhancements

* Machine Learning based fatigue prediction
* Driver fatigue score calculation
* Blink rate analysis
* Driver-not-found detection
* Data logging and analytics dashboard
* GUI-based monitoring dashboard
* Integration with deep learning models

---

## Author

Lahari Tummala

Computer Science Student | Machine Learning Enthusiast | OpenCV & AI Projects
