import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from picamera2 import Picamera2

# === CONFIG ===
WORDS = ['HELLO', 'NO_SIGN', 'YES']   # all possible classes you want
NUM_SEQUENCES = 50                    # sequences per class (per run)
SEQ_LEN = 30                          # frames per sequence (will sync to existing DS if appending)

FRAME_SIZE = (800, 800)

DATA_PATH = 'sequences_hand_relative_to_nose_polar.pickle'

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# === Picamera2 setup ===
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}
)
picam2.configure(preview_config)
picam2.start()


def extract_hand_features_relative_to_nose(results, frame_bgr):
    """
    Extract 84 features for one frame:
    For each of 21 right-hand landmarks:
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

    # Draw nose on BGR frame
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

    # 21 landmarks * 4 values = 84
    return features


# === LOAD OR INIT DATASET ===
all_sequences = []
all_labels = []
words_to_record = WORDS[:]  # may filter based on existing dataset

if os.path.exists(DATA_PATH):
    print(f"Found existing dataset: {DATA_PATH}")
    choice = input(
        "Type 'n' to create NEW dataset (overwrite), or 'a' to APPEND only NEW words [n/a]: "
    ).strip().lower()

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

        # Sync SEQ_LEN with existing dataset
        existing_seq_len = existing_X.shape[1]
        if SEQ_LEN != existing_seq_len:
            print(f"WARNING: Config SEQ_LEN={SEQ_LEN} but dataset uses {existing_seq_len}.")
            print("Using existing SEQ_LEN from dataset.")
            SEQ_LEN = existing_seq_len

        # Keep existing data
        all_sequences = list(existing_X)
        all_labels = list(existing_y)

        # Determine which words are new
        existing_set = set(existing_classes.tolist())
        words_to_record = [w for w in WORDS if w not in existing_set]

        if not words_to_record:
            print("All WORDS already exist in the dataset. Nothing new to record.")
            X = np.asarray(all_sequences, dtype=np.float32)
            y = np.asarray(all_labels)
            print("Final X shape:", X.shape)
            print("Final y shape:", y.shape)
            print("Classes in dataset:", np.unique(y))
            with open(DATA_PATH, 'wb') as f:
                pickle.dump({'data': X, 'labels': y}, f)
            picam2.stop()
            cv2.destroyAllWindows()
            raise SystemExit

        print("Will record ONLY new words:", words_to_record)

    else:
        print("Starting NEW dataset (existing file will be overwritten).")
        # all_sequences, all_labels stay empty
else:
    print("No existing dataset found. Starting new.")


# === RECORD NEW SEQUENCES FOR NEW WORDS ONLY ===
for word in words_to_record:
    for seq_idx in range(NUM_SEQUENCES):
        print(f"Get ready for '{word}', sequence {seq_idx+1}/{NUM_SEQUENCES}")
        # HELLO: perform sign; NO_SIGN: neutral/random/no gesture; YES: your YES sign, etc.
        time.sleep(2)

        sequence = []

        while len(sequence) < SEQ_LEN:
            # Picamera2 returns RGB
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            img_rgb = frame_rgb  # Mediapipe wants RGB
            results = holistic.process(img_rgb)

            features = extract_hand_features_relative_to_nose(results, frame_bgr)
            if features is not None:
                sequence.append(features)

            # Optional: draw right-hand landmarks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            cv2.putText(
                frame_bgr,
                f"{word} seq {seq_idx+1}/{NUM_SEQUENCES} frame {len(sequence)}/{SEQ_LEN}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            cv2.imshow('Recording', frame_bgr)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                picam2.stop()
                cv2.destroyAllWindows()
                raise SystemExit

        if len(sequence) == SEQ_LEN:
            all_sequences.append(sequence)
            all_labels.append(word)
        else:
            print(f"Sequence for '{word}' #{seq_idx+1} too short, skipping...")

picam2.stop()
cv2.destroyAllWindows()

# === SAVE MERGED DATASET ===
X = np.asarray(all_sequences, dtype=np.float32)  # (N, SEQ_LEN, 84)
y = np.asarray(all_labels)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Classes in dataset:", np.unique(y))

with open(DATA_PATH, 'wb') as f:
    pickle.dump({'data': X, 'labels': y}, f)

print(f"Saved merged dataset to '{DATA_PATH}'")
