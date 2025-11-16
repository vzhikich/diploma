from collections import deque
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle
from picamera2 import Picamera2

THRESHOLD = 0.7       # confidence threshold
MODEL_PATH = 'lstm_hand_relative_to_nose_polar.h5'
LABEL_PATH = 'label_encoder_words.pkl'

# === LOAD MODEL & LABEL ENCODER ===
model = load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Input shape: (None, timesteps, features)
_, MODEL_SEQ_LEN, MODEL_FEATURES = model.input_shape
print(f"Model expects sequences of length {MODEL_SEQ_LEN} with {MODEL_FEATURES} features.")

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
FRAME_SIZE = (640, 480)  # you can try (960, 540) / (1280, 720) if it's fast enough
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}
)
picam2.configure(preview_config)
picam2.start()

# Sliding buffer of features
buffer = deque(maxlen=MODEL_SEQ_LEN)

# Last printed prediction to avoid spamming the console
last_printed = None


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
    print("Starting sign recognition. Press Ctrl+C to stop.")
    while True:
        # Get frame from picamera2 (RGB)
        frame_rgb = picam2.capture_array()

        # Mediapipe expects RGB
        results = holistic.process(frame_rgb)

        # Extract features for this frame
        features = extract_hand_features_relative_to_nose(results)

        if features is not None:
            buffer.append(features)
        else:
            # Lost hand/pose → clear sequence
            buffer.clear()

        current_prediction = "NO_SIGH"

        # When buffer is full, run prediction
        if len(buffer) == MODEL_SEQ_LEN:
            x_input = np.asarray(buffer, dtype=np.float32).reshape(1, MODEL_SEQ_LEN, MODEL_FEATURES)
            probs = model.predict(x_input, verbose=0)[0]
            idx = np.argmax(probs)
            max_prob = probs[idx]
            class_name = label_encoder.inverse_transform([idx])[0]

            # Only count as sign if confident and not NO_SIGN
            if max_prob >= THRESHOLD and class_name != 'NO_SIGN':
                current_prediction = class_name
            else:
                current_prediction = "NO_SIGH"

        # Print only when prediction changes
        if current_prediction != last_printed:
            print("Prediction:", current_prediction)
            last_printed = current_prediction

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    holistic.close()
