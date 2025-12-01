from collections import deque
import time
import os

import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle
from picamera2 import Picamera2

# Optional: system metrics (CPU/RAM)
try:
    import psutil
    PSUTIL_AVAILABLE = True
    PROCESS = psutil.Process(os.getpid())
except ImportError:
    PSUTIL_AVAILABLE = False
    PROCESS = None

# ============== CONFIG ==============
THRESHOLD = 0.7       # confidence threshold
MODEL_PATH = 'lstm_hand_relative_to_nose_polar.h5'
LABEL_PATH = 'label_encoder_words.pkl'

FRAME_SIZE = (640, 480)  # change to (960, 540), (1280, 720) etc. for experiments
# ====================================

# === LOAD MODEL & LABEL ENCODER ===
model = load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Input shape: (None, timesteps, features)
_, MODEL_SEQ_LEN, MODEL_FEATURES = model.input_shape
print(f"[INFO] Model expects sequences of length {MODEL_SEQ_LEN} with {MODEL_FEATURES} features.")

# === MEDIAPIPE HOLISTIC ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,              # faster model on Pi
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === PICAMERA2 SETUP ===
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}
)
picam2.configure(preview_config)
picam2.start()

# Sliding buffer of features
buffer = deque(maxlen=MODEL_SEQ_LEN)

# Last printed prediction to avoid console spam
last_printed = None

# ============== METRICS ==============
start_time = None
frame_count = 0

pred_count = 0
pred_time_total = 0.0  # seconds

cpu_samples = []
mem_samples = []
# =====================================


def extract_hand_features_relative_to_nose(results):
    """
    Same features as in dataset:
        dx, dy, r, angle for each right-hand landmark relative to nose.
    Returns list[float] of length MODEL_FEATURES or None.
    """
    if not results.pose_landmarks or not results.right_hand_landmarks:
        return None

    pose_lm = results.pose_landmarks.landmark
    hand_lm = results.right_hand_landmarks.landmark

    nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
    nose_x, nose_y = nose.x, nose.y

    features = []
    for lm in hand_lm:
        dx = lm.x - nose_x
        dy = lm.y - nose_y
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        features.extend([dx, dy, r, angle])

    if len(features) != MODEL_FEATURES:
        # Should not happen if training & feature logic match
        return None

    return features


try:
    print("[INFO] Starting sign recognition. Press Ctrl+C to stop.")
    print(f"[INFO] Frame size: {FRAME_SIZE}")
    print(f"[INFO] psutil available: {PSUTIL_AVAILABLE}")

    while True:
        if start_time is None:
            start_time = time.time()

        # Capture frame from camera (RGB)
        frame_rgb = picam2.capture_array()
        frame_count += 1

        # Mediapipe expects RGB
        results = holistic.process(frame_rgb)

        # Extract features for this frame
        features = extract_hand_features_relative_to_nose(results)

        if features is not None:
            buffer.append(features)
        else:
            # Lost hand/pose -> clear sequence
            buffer.clear()

        current_prediction = "NO_SIGN"

        # When buffer is full, run prediction
        if len(buffer) == MODEL_SEQ_LEN:
            x_input = np.asarray(buffer, dtype=np.float32).reshape(1, MODEL_SEQ_LEN, MODEL_FEATURES)

            # Measure prediction time
            t0 = time.time()
            probs = model.predict(x_input, verbose=0)[0]
            t1 = time.time()

            pred_time = t1 - t0
            pred_time_total += pred_time
            pred_count += 1

            idx = np.argmax(probs)
            max_prob = probs[idx]
            class_name = label_encoder.inverse_transform([idx])[0]

            # Only count as sign if confident and not NO_SIGN
            if max_prob >= THRESHOLD and class_name != 'NO_SIGN':
                current_prediction = class_name
            else:
                current_prediction = "NO_SIGN"

        # Print only when prediction changes
        if current_prediction != last_printed:
            print(f"[PRED] {current_prediction}")
            last_printed = current_prediction

        # Sample CPU/RAM every N frames (if psutil is available)
        if PSUTIL_AVAILABLE and frame_count % 30 == 0:
            cpu = PROCESS.cpu_percent(interval=None)  # percent of one logical CPU
            mem = PROCESS.memory_info().rss / (1024 * 1024)  # MB
            cpu_samples.append(cpu)
            mem_samples.append(mem)

except KeyboardInterrupt:
    print("\n[INFO] Stopping...")

finally:
    picam2.stop()
    holistic.close()

    if start_time is not None:
        total_time = time.time() - start_time
    else:
        total_time = 0.0

    print("\n===== RUNTIME METRICS =====")
    print(f"Total runtime: {total_time:.2f} s")
    print(f"Total frames processed: {frame_count}")

    if total_time > 0 and frame_count > 0:
        fps = frame_count / total_time
        print(f"Average FPS (capture + Mediapipe + model): {fps:.2f}")

    if pred_count > 0:
        avg_pred_time = pred_time_total / pred_count
        print(f"\nTotal predictions: {pred_count}")
        print(f"Average prediction time per call: {avg_pred_time * 1000:.2f} ms")
        print(f"Average predictions per second: {pred_count / total_time:.2f}")
    else:
        print("\nNo predictions were made (buffer was never full).")

    if PSUTIL_AVAILABLE and cpu_samples:
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        avg_mem = sum(mem_samples) / len(mem_samples)
        print("\nSystem load (process level):")
        print(f"Average CPU usage: {avg_cpu:.2f} %")
        print(f"Max CPU usage: {max_cpu:.2f} %")
        print(f"Average RAM usage: {avg_mem:.2f} MB")
    elif not PSUTIL_AVAILABLE:
        print("\npsutil not installed: CPU/RAM metrics not collected.")
    else:
        print("\nNo CPU/RAM samples were collected.")
