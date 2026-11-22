import os
import pickle
import numpy as np
import cv2
import config

class FaceCNNModel:
    """
    State-of-the-art Deep CNN Face Recognition System (SFace Architecture).
    Uses Deep Residual Convolutional Neural Network specifically trained on faces.
    Features:
      - 5-point facial landmark alignment (invariant to head tilt / angle)
      - 128-dimensional L2-normalized deep face embeddings
      - Angular margin Cosine matching
    """
    def __init__(self, sface_path=config.SFACE_MODEL_PATH, embeddings_path=config.EMBEDDINGS_FILE):
        self.embeddings_path = embeddings_path
        self.threshold = config.SIMILARITY_THRESHOLD
        self.student_database = {}

        if not os.path.exists(sface_path):
            raise FileNotFoundError(f"SFace model file not found at: {sface_path}")

        self.recognizer = cv2.FaceRecognizerSF_create(sface_path, "")
        self.load_embeddings()

    def align_face(self, frame, raw_face):
        """
        Aligns the face using detected facial landmarks (eyes, nose, mouth)
        and crops to standard 112x112 canonical pose.
        """
        try:
            aligned = self.recognizer.alignCrop(frame, raw_face)
            return aligned
        except Exception as e:
            print(f"[WARN] Face alignment failed: {e}")
            return None

    def extract_feature(self, aligned_face):
        """Extracts 128-D L2-normalized feature vector from an aligned face crop."""
        feat = self.recognizer.feature(aligned_face)
        # Ensure flat shape (128,) and float32
        feat = np.squeeze(feat).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def extract_from_frame(self, frame, raw_face):
        """Convenience method: aligns face from frame and extracts 128-D feature vector."""
        aligned = self.align_face(frame, raw_face)
        if aligned is None:
            return None, None
        feat = self.extract_feature(aligned)
        return feat, aligned

    def enroll_student(self, roll_no, name, features_list, aligned_images=None):
        """
        Enrolls a new student using extracted 128-D SFace feature vectors.
        """
        if not features_list:
            return False, "No face features provided."

        roll_no = str(roll_no).strip()
        features_arr = np.array(features_list, dtype=np.float32)

        # Normalize features
        norms = np.linalg.norm(features_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features_arr = features_arr / norms

        # Compute centroid vector
        mean_feat = np.mean(features_arr, axis=0)
        c_norm = np.linalg.norm(mean_feat)
        centroid = (mean_feat / c_norm) if c_norm > 0 else mean_feat

        self.student_database[roll_no] = {
            "name": name,
            "roll_no": roll_no,
            "features": features_arr,
            "centroid": centroid,
            "samples": len(features_list)
        }

        self.save_embeddings()
        return True, f"Successfully enrolled {name} (Roll: {roll_no}) with {len(features_list)} aligned CNN samples."

    def match_face(self, feature):
        """
        Compares extracted 128-D face feature against all enrolled students.
        Returns:
            is_match (bool)
            student_info (dict): {'roll_no': ..., 'name': ...}
            confidence (float): 0.0 to 1.0
        """
        if not self.student_database or feature is None:
            return False, {"roll_no": "N/A", "name": "Unknown"}, 0.0

        feat = feature.flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        best_roll = None
        best_score = -1.0

        for roll, data in self.student_database.items():
            # 1. Cosine similarity against each enrolled sample
            sample_sims = np.dot(data["features"], feat)
            # Take top 3 average or max
            top_k_sim = float(np.mean(np.sort(sample_sims)[-3:])) if len(sample_sims) >= 3 else float(np.max(sample_sims))

            # 2. Similarity against centroid
            centroid_sim = float(np.dot(data["centroid"], feat))

            # Combined score
            combined_score = 0.65 * top_k_sim + 0.35 * centroid_sim

            if combined_score > best_score:
                best_score = combined_score
                best_roll = roll

        # Convert SFace score (typically 0.35 - 0.75) to intuitive 0% - 100% confidence
        # Baseline noise is ~0.25, threshold is 0.40, strong match is >= 0.65
        conf_pct = min(1.0, max(0.0, (best_score - 0.25) / 0.45))

        if best_roll is not None and best_score >= self.threshold:
            student = self.student_database[best_roll]
            return True, {"roll_no": student["roll_no"], "name": student["name"]}, conf_pct
        else:
            return False, {"roll_no": "N/A", "name": "Unknown"}, conf_pct

    def save_embeddings(self, filepath=None):
        path = filepath or self.embeddings_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.student_database, f)

    def load_embeddings(self, filepath=None):
        path = filepath or self.embeddings_path
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.student_database = pickle.load(f)
                return True
            except Exception as e:
                print(f"[WARN] Error loading embeddings: {e}")
                self.student_database = {}
                return False
        return False

    def get_enrolled_count(self):
        return len(self.student_database)

    def get_enrolled_list(self):
        return [
            {"roll_no": d["roll_no"], "name": d["name"], "samples": d.get("samples", len(d.get("features", [])))}
            for d in self.student_database.values()
        ]
