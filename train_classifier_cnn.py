import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import config

os.environ["KERAS_HOME"] = os.path.join(config.BASE_DIR, ".keras")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def build_custom_face_cnn(num_classes, input_shape=(128, 128, 3)):
    """
    Constructs a Deep Convolutional Neural Network (CNN) architecture
    tailored for face classification tasks.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", name="student_output")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def load_dataset_arrays():
    """Loads all student face images and labels into numpy arrays for training."""
    if not os.path.exists(config.DATASET_DIR):
        return None, None, []

    student_folders = sorted([f for f in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, f))])
    if not student_folders:
        return None, None, []

    images = []
    labels = []
    class_names = []

    for idx, folder in enumerate(student_folders):
        class_names.append(folder)
        folder_path = os.path.join(config.DATASET_DIR, folder)
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (img_rgb.shape[0], img_rgb.shape[1]) != config.IMAGE_SIZE:
                    img_rgb = cv2.resize(img_rgb, config.IMAGE_SIZE)
                images.append(img_rgb.astype("float32") / 255.0)
                labels.append(idx)

    if not images:
        return None, None, []

    return np.array(images, dtype="float32"), np.array(labels, dtype="int32"), class_names

def train_and_evaluate(epochs=15, batch_size=8):
    """
    Trains the CNN model, evaluates performance, and saves:
      1. cnn_face_classifier.keras
      2. training_performance_curve.png
    """
    print("=" * 65)
    print("        CUSTOM CNN FACE CLASSIFIER - TRAINING & EVALUATION")
    print("=" * 65)

    X, y, class_names = load_dataset_arrays()
    if X is None or len(class_names) < 2:
        print("\n[WARN] At least 2 registered students with face images are required to train a CNN classifier!")
        print(f"Current student classes found: {len(class_names) if class_names else 0}")
        print("Please register at least 2 students first using 'python register_student.py'.")
        return False

    print(f"\n[INFO] Dataset Loaded: {len(X)} images across {len(class_names)} students.")
    print("Classes:")
    for i, c in enumerate(class_names):
        print(f"  [{i}] {c}")

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split train / validation (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\n[INFO] Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print("[INFO] Building Custom CNN Architecture...")
    model = build_custom_face_cnn(num_classes=len(class_names), input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3))
    model.summary()

    print(f"\n[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save trained model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.CNN_CLASSIFIER_FILE)
    print(f"\n[SUCCESS] Trained CNN Classifier saved to: {config.CNN_CLASSIFIER_FILE}")

    # Plot Accuracy & Loss Graphs for viva/presentation
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="#00e5ff", linewidth=2)
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", color="#ff007f", linewidth=2)
    plt.title("CNN Model Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss", color="#00e5ff", linewidth=2)
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", color="#ff007f", linewidth=2)
    plt.title("CNN Model Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    graph_path = os.path.join(config.BASE_DIR, "training_performance_curve.png")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=200)
    plt.close()
    print(f"[GRAPH] Training performance curves saved to: {graph_path}")

    return True

if __name__ == "__main__":
    train_and_evaluate(epochs=15)

