# 📷 Cheating Detection System Using Computer Vision

This project is a real-time **cheating detection system** designed to monitor examinees using a webcam. It leverages **computer vision**, **deep learning**, and **facial landmark analysis** to detect suspicious behaviors such as:

- Presence of multiple persons
- Use of phones or books
- Absence or odd positioning of the face
- Lip movements indicating talking (mouth open)

---

## 🔧 Features

- **Real-Time Object Detection**: Detects people, phones, and books using SSD MobileNet V2.
- **Face Detection & Analysis**: Uses OpenCV and Dlib to detect and evaluate face positioning.
- **Lip Movement Monitoring**: Calibrates closed-lip position and detects when the user opens their mouth.
- **Cheating Score Calculation**: Computes a cheating likelihood score based on combined cues.
- **Live Plotting**: Displays a live graph of the cheating percentage over time.
- **User Calibration**: Calibrates mouth closed distance before monitoring begins.

---

## 🧠 Technologies Used

- Python
- OpenCV
- Dlib
- TensorFlow Hub
- SSD MobileNet V2
- Matplotlib
- NumPy

---

## 📦 Requirements

Before running the code, make sure the following dependencies are installed:

```bash
pip install opencv-python dlib tensorflow tensorflow-hub matplotlib numpy scipy
