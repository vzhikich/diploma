# measure_fps_with_model_real_pipeline.py

from collections import deque
from picamera2 import Picamera2
import cv2
import time
import numpy as np
from keras.models import load_model
import mediapipe as mp
import math

# ==== CONFIG ====
MODEL_PATH = "lstm_hand_relative_to_nose_polar.h5"

# List of resolutions to test
RESOLUTIONS = [
    (640, 480),
    (960, 540),
    (1280, 720),
    # add more if you want
]

TEST_DURATION = 10.0  # seconds to measure per resolution

# ==== LOAD MODEL ====
print(f"[INFO] Loading model from '{MODEL_PATH}' ...")
model = load_model(MODEL_PATH)

# Model input shape: (None, seq_len, features)
_, SEQ_LEN, MODEL_FEATURES = model.input_shape
print(f"[INFO] Model expects sequences of shape: (batch, {SEQ_LEN}, {MODEL_FEATURES})")

# Warm-up prediction (to avoid first-call overhead in timing)
dummy_seq = np.zeros((1, SEQ_LEN, MODEL_FEATURES), dtype=np.float32)
_ = model.predict(dummy_seq, verbose=0)
print("[INFO] Model warm-up prediction done.")

# Sliding buffer for features
buffer = deque(maxlen=SEQ_LEN)

# ==== MEDIAPIPE HOLISTIC SETUP ====
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,            # faster on Raspberry Pi
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==== FEATURE EXTRACTION FUNCTION ====
def extract_hand_features_relative_to_nose(results):
    """
    Extract features for one frame:
      For each right-hand landmark:
        dx, dy, r, angle relative to the nose.
    Returns: list[float] or None if pose/right hand not detected.
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
        r = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)
        features.extend([dx, dy, r, angle])

    # Safety check: must match model's feature size
    if len(features) != MODEL_FEATURES:
        # You can print warning here if you want
        return None

    return features


# ==== CAMERA SETUP ====
picam2 = Picamera2()

results_summary = []

for width, height in RESOLUTIONS:
    print(f"\n[INFO] Testing resolution {width}x{height} ...")

    # Configure camera for this resolution
    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # small warm-up for camera

    buffer.clear()
    frames = 0
    start_time = time.time()

    while True:
        frame_rgb = picam2.capture_array()  # Picamera2 gives RGB
        frames += 1

        # Run MediaPipe on RGB frame
        results_mp = holistic.process(frame_rgb)

        # Extract real features from landmarks
        feats = extract_hand_features_relative_to_nose(results_mp)
        if feats is not None:
            buffer.append(feats)

            # When sequence full -> run real model prediction
            if len(buffer) == SEQ_LEN:
                x_input = np.asarray(buffer, dtype=np.float32).reshape(
                    1, SEQ_LEN, MODEL_FEATURES
                )
                _ = model.predict(x_input, verbose=0)

        # Stop condition: TEST_DURATION seconds
        elapsed = time.time() - start_time
        if elapsed >= TEST_DURATION:
            break

    picam2.stop()

    total_time = time.time() - start_time
    fps = frames / total_time if total_time > 0 else 0.0

    print(
        f"[RESULT] Resolution {width}x{height} -> "
        f"time: {total_time:.2f} s, frames: {frames}, FPS: {fps:.2f}"
    )

    results_summary.append((width, height, total_time, frames, fps))

# ==== PRINT SUMMARY TABLE ====
print("\n================= SUMMARY =================")
print("Resolution\t\tTime (s)\tFrames\tFPS")
for (w, h, t, f, fps) in results_summary:
    res_str = f"{w}x{h}"
    # align output a bit
    if len(res_str) < 8:
        res_str += "\t"
    print(f"{res_str}\t{t:.2f}\t\t{f}\t{fps:.2f}")

print("===========================================")
print("[INFO] Benchmark finished.")
