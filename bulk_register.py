import os
import glob
import cv2
import argparse
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel

def register_from_folder(name, roll_no, source_folder):
    if not os.path.exists(source_folder):
        print(f"[ERROR] Source folder does not exist: {source_folder}")
        return False

    name = name.strip()
    roll_no = str(roll_no).strip()
    target_dir = os.path.join(config.DATASET_DIR, f"{roll_no}_{name.replace(' ', '_')}")
    os.makedirs(target_dir, exist_ok=True)

    detector = FaceDetector()
    cnn = FaceCNNModel()

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(source_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(source_folder, ext.upper())))

    if not image_paths:
        print(f"[WARN] No image files found in {source_folder}")
        return False

    print(f"\n[INFO] Found {len(image_paths)} images for {name} ({roll_no}). Extracting aligned CNN features...")

    valid_features = []
    saved_count = 0

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        faces = detector.detect_faces(img)
        if faces:
            # Pick highest score face
            best_face = max(faces, key=lambda f: f["score"])
            feat, aligned = cnn.extract_from_frame(img, best_face["raw"])
            if feat is not None and aligned is not None:
                saved_count += 1
                save_path = os.path.join(target_dir, f"sample_{saved_count:03d}.jpg")
                cv2.imwrite(save_path, aligned)
                valid_features.append(feat)

    if not valid_features:
        print("[ERROR] Could not extract any valid face features from photos.")
        return False

    print(f"[INFO] Successfully prepared {len(valid_features)} aligned face samples. Updating CNN embeddings...")
    success, msg = cnn.enroll_student(roll_no, name, valid_features)
    if success:
        print(f"[SUCCESS] {msg}")
        return True
    else:
        print(f"[ERROR] Failed to enroll: {msg}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk register a student from an existing folder of photos.")
    parser.add_argument("--name", type=str, required=True, help="Student Full Name")
    parser.add_argument("--roll", type=str, required=True, help="Student Roll Number")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing student photos")
    args = parser.parse_args()

    register_from_folder(args.name, args.roll, args.folder)

if __name__ == "__main__":
    main()
