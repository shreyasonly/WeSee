<<<<<<< HEAD
# Face Recognition System

## Project Overview
This project is a Python-based face recognition and registration system designed for real-time use in attendance, access control, and group photo tagging scenarios. It leverages MediaPipe for face detection, custom embeddings for recognition, and DeepFace for age, gender, and emotion analysis. The system is optimized for Windows and provides a user-friendly GUI for both registration and recognition.

---

## Key Features
- **Real-time Video Registration**: Register new users via webcam with live quality feedback and auto-capture after 3 seconds of stable, high-quality detection.
- **Fallback Auto-Capture**: If quality remains below 60% for 15 seconds, the system auto-captures and saves with a `_fallback` suffix.
- **Manual Photo Upload**: Users can register by uploading a photo if webcam is not available.
- **Group Photo Recognition**: Detects and tags multiple faces in a single image with name, age, gender, and emotion.
- **Real-time Video Recognition**: Recognizes registered users live via webcam and displays their names on the video feed.
- **Unified Launcher**: A single GUI to access both registration and recognition systems.
- **Minimal Workspace**: Only essential files are kept for clarity and ease of use.

---

## Directory Structure
```
wesee/
  known_faces/           # Registered face images (named as <name>_main.jpg)
  output/                # Output images from recognition
  sample_images/         # Sample images for testing
  attendance_YYYY-MM-DD.csv # Attendance logs (auto-generated)
  face_system_launcher.py    # Main launcher GUI
  real_time_face_registration.py # Registration system
  real_time_video_recognition.py # Real-time recognition system
  enhanced_face_recognition.py   # Group photo recognition
  requirements.txt        # Python dependencies
  README.md               # This documentation
```

---

## Setup Instructions
1. **Install Python 3.8+** (recommended: 3.8 or 3.9 for best compatibility)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the launcher:**
   ```bash
   python face_system_launcher.py
   ```

---

## Usage
### Registration
- **Via Webcam:**
  - Open the launcher and select "Register New Face".
  - Enter the name and follow on-screen instructions.
  - The system will auto-capture after 3 seconds of stable, high-quality detection (>70%).
  - If quality is below 60% for 15 seconds, a fallback capture is saved.
- **Via Photo Upload:**
  - Use the "Upload Photo" option in the registration window.
  - Select a clear, front-facing image.

### Recognition
- **Real-time Video Recognition:**
  - Open the launcher and select "Start Recognition".
  - The system will display names of recognized users live on the video feed.
  - Attendance is logged in `attendance_YYYY-MM-DD.csv`.
- **Group Photo Recognition:**
  - Run `enhanced_face_recognition.py` and provide an image.
  - The output image will have faces outlined and tagged with name, age, gender, and emotion.

---

## Troubleshooting
- **Permission Denied on CSV:**
  - Ensure the attendance CSV is not open in Excel or another program.
  - Check file permissions and remove read-only status if set.
- **Camera Not Detected:**
  - Ensure your webcam is connected and not used by another application.
- **Recognition Accuracy:**
  - Use clear, well-lit images for registration.
  - Ensure faces are front-facing and unobstructed.
- **Suppressing Warnings:**
  - To hide TensorFlow/MediaPipe warnings, add these lines at the top of your scripts:
    ```python
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    ```

---

## Future Plans
- **Live Name Display in Video:**
  - Enhance real-time recognition to show names above faces as they appear in the camera feed.
- **Admin Dashboard:**
  - Add a dashboard for managing users and attendance logs.
- **Cloud Sync:**
  - Optionally sync attendance and face data to a secure cloud backend.

---

## Credits
Developed by Shreyas and contributors. Powered by MediaPipe, DeepFace, and OpenCV. 
=======
# WeSee
Facial Recognizing
>>>>>>> 0358e56112f012ea317b51ae11d8afc07188eae4
