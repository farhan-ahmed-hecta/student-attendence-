import os
import glob
import cv2
import numpy as np
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel

def train_or_rebuild_embeddings():
    """
    Scans dataset/ directory, extracts SFace 128-D CNN feature embeddings
    for all students, and updates models/student_embeddings.pkl.
    """
    print("=" * 65)
    print("      CNN FACE RECOGNITION - DATASET TRAINING & EMBEDDING")
    print("=" * 65)

    if not os.path.exists(config.DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {config.DATASET_DIR}")
        return False

    student_folders = [f for f in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, f))]

    if not student_folders:
        print("[WARN] No student data found in 'dataset/' directory.")
        print("Please run 'python register_student.py' or 'python app.py' to enroll students.")
        return False

    print(f"\n[INFO] Found {len(student_folders)} registered student folders.")
    detector = FaceDetector()
    cnn = FaceCNNModel()

    total_images_processed = 0
    enrolled_summary = []

    for folder in student_folders:
        folder_path = os.path.join(config.DATASET_DIR, folder)

        parts = folder.split("_", 1)
        if len(parts) == 2:
            roll_no = parts[0]
            name = parts[1].replace("_", " ")
        else:
            roll_no = folder
            name = folder

        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.png"))

        if not image_files:
            continue

        student_features = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # If image is already aligned 112x112, extract directly
            if img.shape[0] == 112 and img.shape[1] == 112:
                feat = cnn.extract_feature(img)
                student_features.append(feat)
            else:
                faces = detector.detect_faces(img)
                if faces:
                    feat, _ = cnn.extract_from_frame(img, faces[0]["raw"])
                    if feat is not None:
                        student_features.append(feat)

        if student_features:
            cnn.enroll_student(roll_no, name, student_features)
            total_images_processed += len(student_features)
            enrolled_summary.append((roll_no, name, len(student_features)))
            print(f" -> Enrolled: {name:<20} | Roll: {roll_no:<8} | Samples: {len(student_features)}")

    print("\n" + "-" * 65)
    print(" SUMMARY OF ENROLLED STUDENTS IN CNN DATABASE:")
    print("-" * 65)
    print(f"{'Roll No':<12} | {'Student Name':<25} | {'Face Samples':<12} | {'Status'}")
    print("-" * 65)
    for roll, name, count in enrolled_summary:
        print(f"{roll:<12} | {name:<25} | {count:<12} | Ready for Live")
    print("-" * 65)
    print(f"Total Students: {len(enrolled_summary)} | Total Samples: {total_images_processed}")
    print(f"Embeddings File Saved: {config.EMBEDDINGS_FILE}")
    print("[SUCCESS] CNN Model is ready! You can now launch live attendance.\n")
    return True

if __name__ == "__main__":
    train_or_rebuild_embeddings()
