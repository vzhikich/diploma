from collections import deque
import cv2
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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# === PICAMERA2 SETUP ===
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()

# Sliding buffer
buffer = deque(maxlen=MODEL_SEQ_LEN)


def extract_hand_features_relative_to_nose(results, frame_bgr):
    """
    Same features as dataset:
        dx, dy, r, angle for each right-hand landmark relative to nose.
    Draws nose on frame.
    """
    if not results.pose_landmarks or not results.right_hand_landmarks:
        return None

    pose_lm = results.pose_landmarks.landmark
    hand_lm = results.right_hand_landmarks.landmark

    nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
    nose_x, nose_y = nose.x, nose.y

    h, w, _ = frame_bgr.shape
    nose_px = int(nose_x * w)
    nose_py = int(nose_y * h)
    cv2.circle(frame_bgr, (nose_px, nose_py), 5, (0, 0, 255), -1)

    features = []
    for lm in hand_lm:
        dx = lm.x - nose_x
        dy = lm.y - nose_y
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        features.extend([dx, dy, r, angle])

    return features  # length 84


while True:
    # Default prediction for this frame
    display_text = "NO"

    # Get frame from picamera2 (RGB), convert to BGR for OpenCV display
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    img_rgb = frame_rgb  # Mediapipe expects RGB
    results = holistic.process(img_rgb)

    features = extract_hand_features_relative_to_nose(results, frame_bgr)

    if features is not None:
        if len(features) != MODEL_FEATURES:
            print(f"Feature size mismatch: got {len(features)}, expected {MODEL_FEATURES}")
        else:
            buffer.append(features)

        # Optional: draw right-hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
    else:
        # Lost hand/pose → clear sequence
        buffer.clear()

    # Predict when we have a full sequence
    if len(buffer) == MODEL_SEQ_LEN:
        x_input = np.asarray(buffer, dtype=np.float32).reshape(1, MODEL_SEQ_LEN, MODEL_FEATURES)
        probs = model.predict(x_input, verbose=0)[0]
        idx = np.argmax(probs)
        max_prob = probs[idx]
        class_name = label_encoder.inverse_transform([idx])[0]

        # Only show sign if confident and not NO_SIGN
        if max_prob >= THRESHOLD and class_name != 'NO_SIGN':
            display_text = class_name
        else:
            display_text = "NO"

    cv2.putText(frame_bgr, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign word recognition (Picamera2)', frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

picam2.stop()
cv2.destroyAllWindows()
