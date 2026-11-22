import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "student_embeddings.pkl")
CNN_CLASSIFIER_FILE = os.path.join(MODELS_DIR, "cnn_face_classifier.keras")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "attendance_records.csv")

# Ensure required directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
IMAGE_SIZE = (112, 112)          # Standard SFace aligned face dimensions (112x112)
EMBEDDING_DIM = 128              # SFace deep CNN embedding dimension
SAMPLES_PER_STUDENT = 25         # Number of face crops to capture per student
SIMILARITY_THRESHOLD = 0.40      # SFace Cosine similarity threshold (OpenCV standard benchmark: 0.363)
ATTENDANCE_COOLDOWN_SEC = 300    # 5 minutes cooldown before re-marking same student (session memory)
CAMERA_INDEX = 0                 # Default webcam device index
