import cv2
import os
import sys
import time
import argparse
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel

def capture_student_faces(name, roll_no, num_samples=config.SAMPLES_PER_STUDENT, camera_index=config.CAMERA_INDEX):
    """
    Captures aligned face samples via webcam, extracts 128-D SFace CNN features,
    and registers student into the database.
    """
    name = name.strip()
    roll_no = str(roll_no).strip()

    if not name or not roll_no:
        print("[ERROR] Student name and roll number cannot be empty!")
        return False

    folder_name = f"{roll_no}_{name.replace(' ', '_')}"
    student_dir = os.path.join(config.DATASET_DIR, folder_name)
    os.makedirs(student_dir, exist_ok=True)

    print("\n[INFO] Initializing Deep Learning Face Detector & CNN Recognizer...")
    detector = FaceDetector()
    cnn = FaceCNNModel()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam (index {camera_index}). Check camera connection!")
        return False

    print("\n" + "=" * 62)
    print(f" REGISTERING STUDENT: {name} (Roll No: {roll_no})")
    print(f" Target Aligned Face Samples: {num_samples}")
    print(" Instructions: Look at the camera. Slowly tilt your head")
    print("               (slightly left, right, up, smile) for variation.")
    print(" Press 'q' at any time to cancel.")
    print("=" * 62 + "\n")

    captured_features = []
    sample_count = 0
    last_capture_time = 0
    capture_interval = 0.15

    cv2.namedWindow("Student Face Registration", cv2.WINDOW_AUTOSIZE)

    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        faces = detector.detect_faces(frame)

        # Header banner
        title_text = f"Registering: {name} | Roll: {roll_no}"
        progress_text = f"Samples: {sample_count}/{num_samples} ({int((sample_count/num_samples)*100)}%)"

        cv2.rectangle(frame, (10, 10), (w - 10, 70), (20, 20, 30), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 70), (0, 200, 255), 1)
        cv2.putText(frame, title_text, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, progress_text, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = int((w - 20) * (sample_count / num_samples))
        cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 10), (0, 255, 128), -1)
        cv2.rectangle(frame, (10, h - 25), (w - 10, h - 10), (100, 100, 100), 1)

        current_time = time.time()

        if len(faces) == 0:
            cv2.putText(frame, "No face detected. Please face the camera.", (25, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2, cv2.LINE_AA)
        elif len(faces) > 1:
            cv2.putText(frame, "Multiple faces detected! Only one person should be in frame.", (25, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            face_item = faces[0]
            bbox = face_item["bbox"]
            detector.draw_futuristic_box(frame, bbox, color=(0, 255, 128))

            if current_time - last_capture_time >= capture_interval:
                feat, aligned = cnn.extract_from_frame(frame, face_item["raw"])
                if feat is not None and aligned is not None:
                    captured_features.append(feat)
                    sample_count += 1
                    last_capture_time = current_time

                    # Save aligned crop
                    crop_filename = os.path.join(student_dir, f"sample_{sample_count:03d}.jpg")
                    cv2.imwrite(crop_filename, aligned)

        cv2.imshow("Student Face Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Registration cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if sample_count < num_samples:
        print(f"[WARN] Only {sample_count}/{num_samples} samples collected.")
        if sample_count == 0:
            return False

    print(f"\n[INFO] Saving {sample_count} aligned CNN embeddings for {name}...")
    success, msg = cnn.enroll_student(roll_no, name, captured_features)
    if success:
        print(f"[SUCCESS] {msg}")
        print("[READY] Student is now registered! You can start Live Attendance.")
        return True
    else:
        print(f"[ERROR] Enrollment failed: {msg}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Register a new student for CNN Face Attendance.")
    parser.add_argument("--name", type=str, help="Student Full Name")
    parser.add_argument("--roll", type=str, help="Student Roll Number / ID")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_STUDENT, help="Number of face samples")
    args = parser.parse_args()

    name = args.name
    roll_no = args.roll

    if not name:
        name = input("Enter Student Full Name: ").strip()
    if not roll_no:
        roll_no = input("Enter Student Roll Number / ID: ").strip()

    capture_student_faces(name, roll_no, num_samples=args.samples)

if __name__ == "__main__":
    main()
