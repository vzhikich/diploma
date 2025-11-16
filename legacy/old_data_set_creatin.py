import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# === CONFIG ===
WORDS = ['HELLO']        # your gesture "words"
NUM_SEQUENCES = 50       # sequences per word
SEQ_LEN = 30             # frames per sequence

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_hand_features_relative_to_nose(results):
    """
    Extract 84 features for one frame:
    For each of 21 right-hand landmarks we compute:
        dx = x - nose_x
        dy = y - nose_y
        r  = sqrt(dx^2 + dy^2)
        angle = atan2(dy, dx)
    Returns list[float] of length 84, or None if pose/right hand not detected.
    """
    if not results.pose_landmarks or not results.right_hand_landmarks:
        return None

    pose_lm = results.pose_landmarks.landmark
    hand_lm = results.right_hand_landmarks.landmark

    # Nose as origin
    nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
    nose_x, nose_y = nose.x, nose.y

    features = []
    for lm in hand_lm:
        dx = lm.x - nose_x
        dy = lm.y - nose_y
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        features.extend([dx, dy, r, angle])

    # 21 landmarks * 4 values = 84
    return features

all_sequences = []
all_labels = []

for word in WORDS:
    for seq_idx in range(NUM_SEQUENCES):
        print(f"Get ready for '{word}', sequence {seq_idx+1}/{NUM_SEQUENCES}")
        time.sleep(2)  # a moment to prepare

        sequence = []

        while len(sequence) < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            # --- draw nose on the frame (red dot) ---
            if results.pose_landmarks:
                pose_lm = results.pose_landmarks.landmark
                nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
                h, w, _ = frame.shape
                nose_px = int(nose.x * w)
                nose_py = int(nose.y * h)
                cv2.circle(frame, (nose_px, nose_py), 5, (0, 0, 255), -1)

            # Extract features for this frame
            features = extract_hand_features_relative_to_nose(results)
            if features is not None:
                sequence.append(features)

            # Optional: draw right hand for feedback
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            cv2.putText(
                frame,
                f"{word} seq {seq_idx+1}/{NUM_SEQUENCES} frame {len(sequence)}/{SEQ_LEN}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            cv2.imshow('Recording', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort everything
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        if len(sequence) == SEQ_LEN:
            all_sequences.append(sequence)
            all_labels.append(word)
        else:
            print("Sequence too short, skipping...")

cap.release()
cv2.destroyAllWindows()

X = np.asarray(all_sequences, dtype=np.float32)  # shape: (N, SEQ_LEN, 84)
y = np.asarray(all_labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

with open('sequences_hand_relative_to_nose_polar.pickle', 'wb') as f:
    pickle.dump({'data': X, 'labels': y}, f)

print("Saved to sequences_hand_relative_to_nose_polar.pickle")
