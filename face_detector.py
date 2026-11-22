import cv2
import numpy as np
import os
import config

class FaceDetector:
    """
    High-precision Deep Learning Face Detector using YuNet ONNX,
    with automatic Haar Cascade fallback for edge cases.
    Detects faces and 5 facial landmarks (eyes, nose, mouth corners) for pose alignment.
    """
    def __init__(self, yunet_path=config.YUNET_MODEL_PATH, cascade_path=config.HAAR_CASCADE_PATH):
        self.yunet = None
        self.cascade = None

        # 1. Initialize YuNet DNN Face Detector
        if os.path.exists(yunet_path):
            try:
                self.yunet = cv2.FaceDetectorYN_create(
                    yunet_path,
                    "",
                    (320, 320),
                    score_threshold=0.60,
                    nms_threshold=0.30,
                    top_k=5000
                )
            except Exception as e:
                print(f"[WARN] Failed to load YuNet: {e}")

        # 2. Initialize Haar Cascade as fallback
        if os.path.exists(cascade_path):
            try:
                self.cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                pass

    def detect_faces(self, frame):
        """
        Detects faces in BGR frame.
        Returns:
            list of dicts: [
                {
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'raw': face_row (15 elements array for alignCrop),
                    'score': float
                }, ...
            ]
        """
        h, w = frame.shape[:2]
        results = []

        # Try YuNet first (Primary Deep Learning Detector)
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)

            if faces is not None and len(faces) > 0:
                for face in faces:
                    bx = max(0, int(face[0]))
                    by = max(0, int(face[1]))
                    bw = min(w - bx, int(face[2]))
                    bh = min(h - by, int(face[3]))
                    score = float(face[-1])

                    if bw > 25 and bh > 25:
                        results.append({
                            "bbox": (bx, by, bw, bh),
                            "raw": face,
                            "score": score
                        })
                return results

        # Fallback to Haar Cascade if YuNet found nothing
        if self.cascade is not None and not self.cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            haar_faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (hx, hy, hw, hh) in haar_faces:
                # Synthetic raw face format for alignment fallback
                raw_dummy = np.array([
                    hx, hy, hw, hh,
                    hx + hw * 0.3, hy + hh * 0.35,  # Right eye
                    hx + hw * 0.7, hy + hh * 0.35,  # Left eye
                    hx + hw * 0.5, hy + hh * 0.55,  # Nose
                    hx + hw * 0.35, hy + hh * 0.75, # Right mouth
                    hx + hw * 0.65, hy + hh * 0.75, # Left mouth
                    0.80
                ], dtype=np.float32)
                results.append({
                    "bbox": (int(hx), int(hy), int(hw), int(hh)),
                    "raw": raw_dummy,
                    "score": 0.80
                })

        return results

    def draw_futuristic_box(self, frame, bbox, color=(0, 255, 128), line_len=22, thickness=2):
        """Draws sleek cyber corner brackets around face."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        t = thickness + 1
        ll = min(line_len, w // 4, h // 4)

        # Top-Left
        cv2.line(frame, (x, y), (x + ll, y), color, t)
        cv2.line(frame, (x, y), (x, y + ll), color, t)

        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - ll, y), color, t)
        cv2.line(frame, (x + w, y), (x + w, y + ll), color, t)

        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + ll, y + h), color, t)
        cv2.line(frame, (x, y + h), (x, y + h - ll), color, t)

        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - ll, y + h), color, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - ll), color, t)

    def draw_hud(self, frame, bbox, name, roll_no, confidence, status, is_marked=False):
        """Draws HUD banner with student info and attendance status."""
        x, y, w, h = bbox

        if name == "Unknown":
            box_color = (0, 70, 255)       # Orange-Red for Unknown
        elif is_marked:
            box_color = (255, 200, 0)      # Cyan for Already Marked
        else:
            box_color = (0, 255, 100)      # Neon Green for Present

        self.draw_futuristic_box(frame, bbox, color=box_color)

        # Labels
        conf_pct = int(min(1.0, max(0.0, confidence)) * 100)
        header_text = f"{name} ({conf_pct}%)" if name != "Unknown" else "Unknown Face"
        if roll_no and roll_no != "N/A":
            sub_text = f"Roll: {roll_no} | {status}"
        else:
            sub_text = f"Status: {status}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thick = 1

        (tw1, th1), _ = cv2.getTextSize(header_text, font, font_scale, text_thick)
        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 0.45, 1)
        banner_w = max(w, max(tw1, tw2) + 16)
        banner_h = th1 + th2 + 18

        by1 = max(0, y - banner_h - 4)
        by2 = by1 + banner_h
        bx1 = x
        bx2 = min(frame.shape[1], x + banner_w)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (25, 25, 35), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)

        cv2.putText(frame, header_text, (bx1 + 8, by1 + th1 + 4), font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)
        cv2.putText(frame, sub_text, (bx1 + 8, by1 + th1 + th2 + 10), font, 0.45, box_color, 1, cv2.LINE_AA)
