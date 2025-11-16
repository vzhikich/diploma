from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle

THRESHOLD = 0.7       # confidence threshold to accept a sign

# === LOAD MODEL & LABEL ENCODER ===
model = load_model('lstm_hand_relative_to_nose_polar.h5')
with open('label_encoder_words.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Get expected input shape from model: (None, timesteps, features)
input_shape = model.input_shape  # e.g. (None, 30, 84)
_, MODEL_SEQ_LEN, MODEL_FEATURES = input_shape
print(f"Model expects sequences of length {MODEL_SEQ_LEN} with {MODEL_FEATURES} features.")

# === MEDIAPIPE HOLISTIC ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(0)

# Sliding buffer for last MODEL_SEQ_LEN frames
buffer = deque(maxlen=MODEL_SEQ_LEN)


def extract_hand_features_relative_to_nose(results, frame):
    """
    Extract 84 features for one frame:
    For each of 21 right-hand landmarks we compute:
        dx = x - nose_x
        dy = y - nose_y
        r  = sqrt(dx^2 + dy^2)
        angle = atan2(dy, dx)
    Nose acts as origin.
    Returns list[float] of length 84, or None if pose/right hand not detected.
    Also draws nose on the frame.
    """
    if not results.pose_landmarks or not results.right_hand_landmarks:
        return None

    pose_lm = results.pose_landmarks.landmark
    hand_lm = results.right_hand_landmarks.landmark

    # Nose as origin
    nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
    nose_x, nose_y = nose.x, nose.y

    # Draw nose on frame
    h, w, _ = frame.shape
    nose_px = int(nose_x * w)
    nose_py = int(nose_y * h)
    cv2.circle(frame, (nose_px, nose_py), 5, (0, 0, 255), -1)

    features = []
    for lm in hand_lm:
        dx = lm.x - nose_x
        dy = lm.y - nose_y
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        features.extend([dx, dy, r, angle])

    # 21 landmarks * 4 values = 84
    return features


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Default prediction for this frame
    current_prediction = "NO"

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    # Extract features for this frame
    features = extract_hand_features_relative_to_nose(results, frame)

    if features is not None:
        # Safety: check feature size
        if len(features) != MODEL_FEATURES:
            print(f"Feature size mismatch: got {len(features)}, expected {MODEL_FEATURES}")
        else:
            buffer.append(features)

        # Optional: draw right-hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
    else:
        # Lost hand/pose → clear sequence and stay "NO"
        buffer.clear()

    # When buffer full, run prediction
    if len(buffer) == MODEL_SEQ_LEN:
        x_input = np.asarray(buffer, dtype=np.float32).reshape(1, MODEL_SEQ_LEN, MODEL_FEATURES)
        probs = model.predict(x_input, verbose=0)[0]
        idx = np.argmax(probs)
        max_prob = probs[idx]

        if max_prob >= THRESHOLD:
            current_prediction = label_encoder.inverse_transform([idx])[0]
        # else keep "NO"

    # Show prediction
    cv2.putText(frame, current_prediction, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign word recognition (polar nose-relative)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
