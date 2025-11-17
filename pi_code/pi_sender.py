from collections import deque
import socket
import time

import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle
from picamera2 import Picamera2

# ========= CONFIG =========
THRESHOLD = 0.7
MODEL_PATH = "lstm_hand_relative_to_nose_polar.h5"
LABEL_PATH = "label_encoder_words.pkl"

SERVER_HOST = "192.168.0.102"   # <-- PUT YOUR PC's LAN IP HERE
SERVER_PORT = 5005              # must match sign_receiver.py
RECONNECT_DELAY = 5.0           # seconds

# ========= LOAD MODEL & LABELS =========
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

_, MODEL_SEQ_LEN, MODEL_FEATURES = model.input_shape
print(f"Model expects sequences of length {MODEL_SEQ_LEN} with {MODEL_FEATURES} features.")

# ========= MEDIAPIPE HOLISTIC =========
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,          # faster on Pi
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========= PICAMERA2 =========
picam2 = Picamera2()
FRAME_SIZE = (640, 480)
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}
)
picam2.configure(preview_config)
picam2.start()

# Sliding buffer of features
buffer = deque(maxlen=MODEL_SEQ_LEN)
last_sent_sign = None   # last sign sent to PC


def extract_hand_features_relative_to_nose(results):
    """
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
        return None

    return features


def connect_to_server():
    """Try to connect to PC server; return socket or None."""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((SERVER_HOST, SERVER_PORT))
            s.settimeout(None)
            print(f"Connected to server {SERVER_HOST}:{SERVER_PORT}")
            return s
        except OSError as e:
            print(f"Could not connect to server: {e}. Retrying in {RECONNECT_DELAY} sec...")
            s.close()
            time.sleep(RECONNECT_DELAY)


def send_sign(sock, sign_name):
    """Send sign name + newline over socket; handle errors."""
    msg = (sign_name + "\n").encode("utf-8")
    try:
        sock.sendall(msg)
        return True
    except OSError as e:
        print(f"Send failed: {e}")
        try:
            sock.close()
        except OSError:
            pass
        return False


def main():
    global last_sent_sign

    print("Starting sign recognition loop. Press Ctrl+C to stop (if running manually).")
    sock = connect_to_server()

    try:
        while True:
            # Capture frame (RGB)
            frame_rgb = picam2.capture_array()

            # Run Mediapipe
            results = holistic.process(frame_rgb)

            # Extract frame features
            features = extract_hand_features_relative_to_nose(results)

            if features is not None:
                buffer.append(features)
            else:
                buffer.clear()

            current_sign = "NO_SIGN"

            # Predict if we have a full sequence
            if len(buffer) == MODEL_SEQ_LEN:
                x_input = np.asarray(buffer, dtype=np.float32).reshape(
                    1, MODEL_SEQ_LEN, MODEL_FEATURES
                )
                probs = model.predict(x_input, verbose=0)[0]
                idx = np.argmax(probs)
                max_prob = probs[idx]
                class_name = label_encoder.inverse_transform([idx])[0]

                if max_prob >= THRESHOLD and class_name != "NO_SIGN":
                    current_sign = class_name
                else:
                    current_sign = "NO_SIGN"

            # Only send when sign changes
            if current_sign != last_sent_sign:
                print("Detected sign:", current_sign)
                # try to send; if fails, reconnect
                if not send_sign(sock, current_sign):
                    sock = connect_to_server()
                    # best effort re-send after reconnect
                    send_sign(sock, current_sign)
                last_sent_sign = current_sign

    except KeyboardInterrupt:
        print("Stopping by user request...")
    finally:
        picam2.stop()
        holistic.close()
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
