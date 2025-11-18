from collections import deque
import threading
import time
import socket

import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle
from picamera2 import Picamera2

# ========= CONFIG =========
THRESHOLD = 0.7
MODEL_PATH = "lstm_hand_relative_to_nose_polar.h5"
LABEL_PATH = "label_encoder_words.pkl"

SERVER_HOST = "0.0.0.0"   # listen on all interfaces
SERVER_PORT = 5005        # you will forward this port on your router

FRAME_SIZE = (640, 480)   # camera resolution

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
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}
)
picam2.configure(preview_config)
picam2.start()

# ========= SHARED STATE =========
buffer = deque(maxlen=MODEL_SEQ_LEN)
current_sign = "NO"      # last detected sign
current_lock = threading.Lock()


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


def detection_loop():
    """Runs on a separate thread: updates current_sign continuously."""
    global current_sign

    print("Detection loop started.")
    try:
        while True:
            # Capture frame (RGB)
            frame_rgb = picam2.capture_array()

            # Run Mediapipe
            results = holistic.process(frame_rgb)

            # Extract per-frame features
            features = extract_hand_features_relative_to_nose(results)

            if features is not None:
                buffer.append(features)
            else:
                buffer.clear()

            new_sign = "NO"

            # Predict if we have enough frames
            if len(buffer) == MODEL_SEQ_LEN:
                x_input = np.asarray(buffer, dtype=np.float32).reshape(
                    1, MODEL_SEQ_LEN, MODEL_FEATURES
                )
                probs = model.predict(x_input, verbose=0)[0]
                idx = np.argmax(probs)
                max_prob = probs[idx]
                class_name = label_encoder.inverse_transform([idx])[0]

                if max_prob >= THRESHOLD and class_name != "NO_SIGN":
                    new_sign = class_name
                else:
                    new_sign = "NO"

            # Update shared state
            with current_lock:
                if new_sign != current_sign:
                    current_sign = new_sign
                    print("Detected sign:", current_sign)

            # tiny sleep to avoid 100% busy loop
            time.sleep(0.01)

    except Exception as e:
        print("Detection loop error:", e)


def client_handler(conn, addr):
    """Send sign updates to a single client."""
    print("Client connected:", addr)
    last_sent = None
    try:
        while True:
            with current_lock:
                sign = current_sign
            if sign != last_sent:
                msg = (sign + "\n").encode("utf-8")
                conn.sendall(msg)
                last_sent = sign
            time.sleep(0.1)  # send updates at ~10 Hz max

    except (BrokenPipeError, ConnectionResetError, OSError):
        print("Client disconnected:", addr)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def server_loop():
    """Accepts clients and spawns handler threads."""
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((SERVER_HOST, SERVER_PORT))
        s.listen(5)

        while True:
            conn, addr = s.accept()
            t = threading.Thread(
                target=client_handler, args=(conn, addr), daemon=True
            )
            t.start()


def main():
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    try:
        server_loop()
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        picam2.stop()
        holistic.close()


if __name__ == "__main__":
    main()
