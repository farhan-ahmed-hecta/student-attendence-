import numpy as np
import cv2
import os
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel
from attendance_manager import AttendanceManager

def test_full_system():
    print("[TEST 1] Testing FaceDetector (YuNet DNN + Haar Fallback)...")
    detector = FaceDetector()
    print("  -> FaceDetector successfully loaded models!")

    print("\n[TEST 2] Testing FaceCNNModel (Deep SFace CNN Feature Extractor)...")
    cnn = FaceCNNModel()
    dummy_aligned = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    feat = cnn.extract_feature(dummy_aligned)
    print(f"  -> Extracted SFace embedding shape: {feat.shape}, dtype: {feat.dtype}")
    assert feat.shape == (128,), f"Expected shape (128,), got {feat.shape}"
    norm = np.linalg.norm(feat)
    print(f"  -> Embedding L2-Norm: {norm:.4f}")
    assert np.isclose(norm, 1.0, atol=1e-3), "Embedding must be L2 normalized!"

    print("\n[TEST 3] Testing Fresh Student Enrollment...")
    test_roll = "101"
    test_name = "Farhan Test"
    # Generate 5 synthetic 128-D feature vectors representing slight angle variations
    base_feat = cnn.extract_feature(dummy_aligned)
    features_list = [base_feat + np.random.normal(0, 0.02, 128).astype(np.float32) for _ in range(5)]
    # Normalize them
    features_list = [f / np.linalg.norm(f) for f in features_list]

    ok, msg = cnn.enroll_student(test_roll, test_name, features_list)
    print(f"  -> Enrollment status: {ok}, Msg: {msg}")
    assert ok is True
    assert cnn.get_enrolled_count() == 1

    print("\n[TEST 4] Testing Face Matching...")
    # Matching same person with small variation
    query_feat = base_feat + np.random.normal(0, 0.03, 128).astype(np.float32)
    query_feat = query_feat / np.linalg.norm(query_feat)
    is_match, student, conf = cnn.match_face(query_feat)
    print(f"  -> Same person match: {is_match}, Student: {student['name']} ({student['roll_no']}), Confidence: {conf*100:.2f}%")
    assert is_match is True
    assert student["roll_no"] == test_roll

    # Matching completely random / orthogonal feature vector (different person)
    random_feat = np.random.randn(128).astype(np.float32)
    random_feat = random_feat / np.linalg.norm(random_feat)
    is_match_rnd, student_rnd, conf_rnd = cnn.match_face(random_feat)
    print(f"  -> Different person match: {is_match_rnd}, Name: {student_rnd['name']}, Confidence: {conf_rnd*100:.2f}%")
    assert is_match_rnd is False
    assert student_rnd["name"] == "Unknown"

    print("\n[TEST 5] Testing AttendanceManager (In-Memory Session & Cooldown)...")
    att = AttendanceManager(cooldown_seconds=60)
    new1, status1, rec1 = att.mark_attendance(test_roll, test_name, conf)
    print(f"  -> 1st Attendance Mark: Newly Marked={new1}, Status='{status1}', Time={rec1['time']}")
    assert new1 is True
    assert att.get_total_present() == 1

    new2, status2, rec2 = att.mark_attendance(test_roll, test_name, conf)
    print(f"  -> 2nd Attendance Mark (immediate): Newly Marked={new2}, Status='{status2}'")
    assert new2 is False
    assert att.get_total_present() == 1

    print("\n[TEST 6] Testing CSV Export...")
    test_csv_path = os.path.join(config.BASE_DIR, "test_attendance_export.csv")
    exp_ok, exp_msg = att.export_to_csv(test_csv_path)
    print(f"  -> Export status: {exp_ok}, Msg: {exp_msg}")
    assert exp_ok is True
    assert os.path.exists(test_csv_path)
    os.remove(test_csv_path)

    # Clean up test student from embeddings file so database remains clean for user
    cnn.student_database.clear()
    cnn.save_embeddings()

    print("\n" + "=" * 55)
    print(" >>> ALL PIPELINE TESTS PASSED 100% SUCCESSFULLY! <<< ")
    print("=" * 55)

if __name__ == "__main__":
    test_full_system()
