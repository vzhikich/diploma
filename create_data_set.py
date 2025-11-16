import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# === CONFIG ===
WORDS = ['HELLO', 'NO_SIGN', 'YES', 'NO']   # put all words you *might* want
NUM_SEQUENCES = 50                    # sequences per class (per run)
SEQ_LEN = 30                          # frames per sequence (will sync with existing dataset if appending)

DATA_PATH = 'sequences_hand_relative_to_nose_polar.pickle'

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def extract_hand_features_relative_to_nose(results, frame):
    """
    Extract 84 features for one frame:
    For each of 21 right-hand landmarks we compute:
        dx = x - nose_x
        dy = y - nose_y
        r  = sqrt(dx^2 + dy^2)
        angle = atan2(dy, dx)
    Nose acts as origin.
    Also draws the nose on the frame.
    Returns list[float] of length 84, or None if pose/right hand not detected.
    """
    if not results.pose_landmarks or not results.right_hand_landmarks:
        return None

    pose_lm = results.pose_landmarks.landmark
    hand_lm = results.right_hand_landmarks.landmark

    # Nose as origin
    nose = pose_lm[mp_holistic.PoseLandmark.NOSE]
    nose_x, nose_y = nose.x, nose.y

    # Draw nose on frame (red dot)
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


# === LOAD OR INIT DATASET ===
all_sequences = []
all_labels = []

words_to_record = WORDS[:]  # will maybe filter this later

if os.path.exists(DATA_PATH):
    print(f"Found existing dataset: {DATA_PATH}")
    choice = input("Type 'n' to create NEW dataset (overwrite), or 'a' to APPEND only NEW words [n/a]: ").strip().lower()

    if choice.startswith('a'):
        # Append mode: load existing data
        with open(DATA_PATH, 'rb') as f:
            data_dict = pickle.load(f)
        existing_X = data_dict['data']
        existing_y = data_dict['labels']

        print("Existing X shape:", existing_X.shape)
        print("Existing y shape:", existing_y.shape)
        existing_classes = np.unique(existing_y)
        print("Existing classes:", existing_classes)

        # Sync SEQ_LEN with existing data
        existing_seq_len = existing_X.shape[1]
        if SEQ_LEN != existing_seq_len:
            print(f"WARNING: Config SEQ_LEN={SEQ_LEN} but existing dataset uses SEQ_LEN={existing_seq_len}.")
            print("Using existing SEQ_LEN from dataset.")
            SEQ_LEN = existing_seq_len

        # Initialize lists with existing data
        all_sequences = list(existing_X)   # list of (SEQ_LEN, 84)
        all_labels = list(existing_y)

        # Filter WORDS: record only those not already present
        existing_set = set(existing_classes.tolist())
        words_to_record = [w for w in WORDS if w not in existing_set]

        if not words_to_record:
            print("All words in WORDS already exist in the dataset. Nothing new to record.")
            cap.release()
            cv2.destroyAllWindows()
            # Save back the untouched dataset just in case
            X = np.asarray(all_sequences, dtype=np.float32)
            y = np.asarray(all_labels)
            print("Final X shape:", X.shape)
            print("Final y shape:", y.shape)
            print("Classes in dataset:", np.unique(y))
            with open(DATA_PATH, 'wb') as f:
                pickle.dump({'data': X, 'labels': y}, f)
            print(f"Dataset unchanged and saved to '{DATA_PATH}'")
            raise SystemExit

        print("Will record ONLY new words:", words_to_record)

    else:
        print("Starting NEW dataset (existing file will be overwritten).")
        # all_sequences, all_labels remain empty
else:
    print("No existing dataset found. Starting new.")


# === RECORD NEW SEQUENCES (ONLY FOR NEW WORDS) ===
for word in words_to_record:
    for seq_idx in range(NUM_SEQUENCES):
        print(f"Get ready for '{word}', sequence {seq_idx+1}/{NUM_SEQUENCES}")
        # Tips:
        #  - For HELLO: perform your HELLO sign during these frames
        #  - For NO_SIGN: idle / random / no hand
        time.sleep(2)

        sequence = []

        while len(sequence) < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            features = extract_hand_features_relative_to_nose(results, frame)
            if features is not None:
                sequence.append(features)

            # Optional: draw right hand
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
            print(f"Sequence for '{word}' #{seq_idx+1} too short, skipping...")

cap.release()
cv2.destroyAllWindows()

# === SAVE MERGED DATASET ===
X = np.asarray(all_sequences, dtype=np.float32)  # shape: (N, SEQ_LEN, 84)
y = np.asarray(all_labels)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Classes in dataset:", np.unique(y))

with open(DATA_PATH, 'wb') as f:
    pickle.dump({'data': X, 'labels': y}, f)

print(f"Saved merged dataset to '{DATA_PATH}'")
