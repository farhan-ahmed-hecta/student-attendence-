# 🎓 CNN Live Face Attendance System

A high-performance, real-time facial recognition attendance system powered by **Deep Convolutional Neural Networks (CNN)**.

---

## 🌟 Key Upgrades & Features

1. **Facial Landmark Alignment (YuNet DNN)**:
   - Uses 5-point facial landmarks (eyes, nose, mouth corners) to mathematically align and normalize faces into a standard canonical pose.
   - Invariant to head tilts, minor angles, and distances from the camera.

2. **Dedicated Face CNN Feature Extractor (SFace)**:
   - Deep CNN trained specifically on millions of faces.
   - Extracts 128-dimensional L2-normalized facial embeddings.
   - Fast CPU inference (60+ FPS) with high verification accuracy.

3. **In-Memory Live Attendance (No Database Needed)**:
   - Displays real-time entries with exact timestamp (Date & Time).
   - Audio beep confirmation upon marking.
   - Cooldown duplicate prevention.
   - Optional 1-click CSV export (`attendance_records.csv`).

4. **Dual-Tab GUI Dashboard**:
   - **Tab 1: 📋 Live Attendance Log**: Shows who is currently present in the session with exact time.
   - **Tab 2: 👥 Registered Students Database**: Shows all enrolled students with sample counts, with individual and bulk delete buttons.

---

## 📁 Project Structure

```
d:\hccda project 3\
│
├── models/
│   ├── face_detection_yunet.onnx         # 5-point landmark face detector
│   ├── face_recognition_sface.onnx       # Deep SFace CNN face recognizer
│   └── student_embeddings.pkl            # Enrolled student feature vectors
│
├── dataset/                              # Aligned 112x112 face crop samples
│
├── config.py                             # Settings, paths & threshold tuning
├── face_detector.py                      # YuNet DNN detector with cyber HUD overlay
├── cnn_model.py                          # Deep CNN feature extractor & matcher
├── attendance_manager.py                 # Live session tracker & duplicate cooldown
│
├── app.py                                # Full Modern GUI Application (Recommended)
├── register_student.py                   # CLI: Student face registration
├── live_attendance.py                    # CLI: Standalone camera attendance
├── train_model.py                        # CLI: Dataset sync & re-training
└── test_pipeline.py                      # Automated test script
```

---

## 🚀 How to Run & Test

### 1. Launch the Desktop GUI
```powershell
python app.py
```

### 2. Register New Student
1. Enter **Student Full Name** and **Roll Number / ID**.
2. Click **"📸 Capture & Register Face"**.
3. Look at the camera. Slightly turn your head (left, right, up, smile) so it captures 25 diverse aligned face samples.
4. Once completed, the student will immediately appear in the **"👥 Registered Students Database"** tab!

### 3. Start Live Attendance
1. Click **"▶ START LIVE ATTENDANCE"**.
2. The webcam feed will start.
3. When the registered student faces the camera:
   - Face is detected and aligned.
   - **Green HUD box** displays student name, roll number, and match score.
   - Audio confirmation beep sounds.
   - **Live Attendance Log** table records entry with exact time!
