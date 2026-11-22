import cv2
import time
from datetime import datetime
import numpy as np
import config
from face_detector import FaceDetector
from cnn_model import FaceCNNModel
from attendance_manager import AttendanceManager

def run_live_attendance(camera_index=config.CAMERA_INDEX):
    print("=" * 65)
    print("      DEEP CNN LIVE FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 65)

    print("[INFO] Loading Deep Face Detector (YuNet)...")
    detector = FaceDetector()

    print("[INFO] Loading Deep Face CNN Recognizer (SFace)...")
    cnn = FaceCNNModel()
    enrolled_count = cnn.get_enrolled_count()

    if enrolled_count == 0:
        print("\n[WARN] No students found in database!")
        print("Please run 'python register_student.py' or 'python app.py' to enroll students first.")
        choice = input("Do you still want to start camera for testing Unknown detection? (y/n): ").strip().lower()
        if choice != 'y':
            return

    print(f"[INFO] Successfully loaded {enrolled_count} registered students.")
    print("[INFO] Initializing In-Memory Attendance Tracker (No database required)...")
    attendance_mgr = AttendanceManager()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera (index {camera_index}). Check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "-" * 65)
    print(" CONTROLS:")
    print("  [Q] : Quit live attendance")
    print("  [E] : Export today's live records to CSV")
    print("  [C] : Clear live session records")
    print("  [S] : Take a screenshot")
    print("-" * 65 + "\n")
    print("[STATUS] Camera started! Present face to detect live...")

    fps_time = time.time()
    fps = 0
    frame_count = 0

    cv2.namedWindow("CNN Live Attendance System", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        faces = detector.detect_faces(frame)

        for face_item in faces:
            bbox = face_item["bbox"]
            feat, aligned = cnn.extract_from_frame(frame, face_item["raw"])
            if feat is None:
                continue

            is_match, student, confidence = cnn.match_face(feat)

            if is_match:
                roll_no = student["roll_no"]
                name = student["name"]
                newly_marked, status_msg, record = attendance_mgr.mark_attendance(roll_no, name, confidence)

                if newly_marked:
                    print(f" -> [MARKED] Roll: {roll_no:<6} | Name: {name:<18} | Time: {record['time']} | Conf: {record['confidence']}")
                    detector.draw_hud(frame, bbox, name, roll_no, confidence, "PRESENT", is_marked=False)
                else:
                    detector.draw_hud(frame, bbox, name, roll_no, confidence, status_msg, is_marked=True)
            else:
                detector.draw_hud(frame, bbox, "Unknown", "N/A", confidence, "UNREGISTERED", is_marked=False)

        # Header banner
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%I:%M:%S %p")

        cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 25), -1)
        cv2.line(frame, (0, 55), (w, 55), (0, 200, 255), 2)

        cv2.putText(frame, "LIVE CNN ATTENDANCE SYSTEM", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2, cv2.LINE_AA)
        clock_text = f"{date_str}  |  {time_str}"
        (cw, _), _ = cv2.getTextSize(clock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, clock_text, (w - cw - 20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Recent entries drawer
        recent_records = attendance_mgr.get_records()[-4:]
        if recent_records:
            panel_w = 320
            panel_h = 30 + (len(recent_records) * 26)
            px1 = w - panel_w - 20
            py1 = 70
            px2 = w - 20
            py2 = py1 + panel_h

            sub_img = frame[py1:py2, px1:px2]
            dark_rect = np.full(sub_img.shape, (20, 20, 30), dtype=np.uint8)
            cv2.addWeighted(sub_img, 0.3, dark_rect, 0.7, 0, sub_img)
            frame[py1:py2, px1:px2] = sub_img
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 128), 1)

            cv2.putText(frame, "RECENT ATTENDANCE LOG", (px1 + 10, py1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)

            for i, rec in enumerate(reversed(recent_records)):
                entry_line = f"{rec['time']} | {rec['name'][:10]} ({rec['roll_no']})"
                cv2.putText(frame, entry_line, (px1 + 10, py1 + 44 + (i * 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom status bar
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 25), -1)
        cv2.line(frame, (0, h - 35), (w, h - 35), (80, 80, 90), 1)

        total_present = attendance_mgr.get_total_present()
        status_line = f"Enrolled: {enrolled_count}  |  Present Today: {total_present}  |  FPS: {fps}"
        cv2.putText(frame, status_line, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

        controls_line = "[Q] Quit   [E] Export CSV   [C] Clear Session"
        (ctrl_w, _), _ = cv2.getTextSize(controls_line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, controls_line, (w - ctrl_w - 20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("CNN Live Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[INFO] Exiting Live Attendance...")
            break
        elif key == ord('e'):
            ok, msg = attendance_mgr.export_to_csv()
            print(f"[EXPORT] {msg}")
        elif key == ord('c'):
            attendance_mgr.clear_session()
            print("[INFO] Live session attendance cleared.")
        elif key == ord('s'):
            snap_name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"[INFO] Saved screenshot: {snap_name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_attendance()
