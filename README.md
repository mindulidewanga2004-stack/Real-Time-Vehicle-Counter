# 🚦 Real-Time Vehicle Counter using YOLOv8

A real-time vehicle detection, tracking, and counting system built using **YOLOv8** and object tracking.

This project detects vehicles, tracks them with persistent IDs, and counts them using a line-crossing method. The system is optimized to run on CPU (Intel Mac) using frame resizing and frame skipping.

---

## 🔍 Features

- Vehicle detection (car, bus, truck, motorbike)
- Multi-object tracking with unique IDs
- Line-crossing based vehicle counting
- Counts only when vehicles reach a defined position in the frame (prevents double counting)
- CPU-optimized performance

---

## ⚙️ Current Challenges

- Handling occlusions in dense traffic
- Reducing missed detections
- Minimizing tracking ID switches
- Improving counting accuracy

---

## 🚀 Upcoming Features

- Web dashboard (live video stream)
- Real-time vehicle statistics
- Historical logs & analytics
- Improved tracking stability

---

## 🛠 Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Object Tracking (e.g., ByteTrack / SORT)

---

## 🎯 Goal

To develop a scalable real-time traffic monitoring system for smart traffic management and analytics.

