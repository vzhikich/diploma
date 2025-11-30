import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ================== CONFIG ==================
DATA_PATH = 'sequences_hand_relative_to_nose_polar.pickle'

# Architecture: 'A' (baseline), 'B' (extended), 'C' (lightweight)
ARCH = 'A'

# Split scheme: '80_20', '70_15_15', '60_20_20'
SPLIT_SCHEME = '70_15_15'

EPOCHS = 50
BATCH_SIZE = 16
RANDOM_STATE = 42
# ============================================


def make_splits(X, y, scheme, random_state=42):
    """
    Create train/val/test splits according to the chosen scheme.
    - '80_20'      -> train, test (no explicit val)
    - '70_15_15'   -> train, val, test
    - '60_20_20'   -> train, val, test
    """
    if scheme == '80_20':
        # 80% train, 20% test, no explicit validation set
        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=True,
            stratify=y,
            random_state=random_state
        )
        x_val, y_val = None, None

    elif scheme == '70_15_15':
        # First take out 15% for test, then 15% of remaining for val
        x_temp, x_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.15,
            shuffle=True,
            stratify=y,
            random_state=random_state
        )
        # We want val = 15% of original N => 0.15 / 0.85 of remaining
        val_size = 0.15 / (1.0 - 0.15)  # ≈ 0.1765
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp,
            test_size=val_size,
            shuffle=True,
            stratify=y_temp,
            random_state=random_state + 1
        )

    elif scheme == '60_20_20':
        # First take out 20% for test, then 20% of remaining for val
        x_temp, x_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.20,
            shuffle=True,
            stratify=y,
            random_state=random_state
        )
        # We want val = 20% of original N => 0.20 / 0.80 of remaining
        val_size = 0.20 / (1.0 - 0.20)  # = 0.25
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp,
            test_size=val_size,
            shuffle=True,
            stratify=y_temp,
            random_state=random_state + 1
        )
    else:
        raise ValueError(f"Unknown split scheme: {scheme}")

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_lstm_model(arch, seq_len, features, num_classes):
    """
    Build LSTM model according to architecture:
    A - baseline (medium size)
    B - extended (bigger, more capacity)
    C - lightweight (smaller, faster)
    """
    if arch == 'A':
        lstm1_units = 64
        lstm2_units = 32
        dropout_rate = 0.2
    elif arch == 'B':
        lstm1_units = 96
        lstm2_units = 48
        dropout_rate = 0.3
    elif arch == 'C':
        lstm1_units = 32
        lstm2_units = 16
        dropout_rate = 0.2
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = Sequential()
    model.add(LSTM(lstm1_units, return_sequences=True,
                   input_shape=(seq_len, features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm2_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============ LOAD DATASET ============
with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

X = data_dict['data']      # shape: (N, SEQ_LEN, FEATURES)
y = data_dict['labels']    # shape: (N,)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique classes:", np.unique(y))

SEQ_LEN = X.shape[1]
FEATURES = X.shape[2]

# ============ TRAIN / VAL / TEST SPLIT ============
x_train, x_val, x_test, y_train, y_val, y_test = make_splits(
    X, y, SPLIT_SCHEME, random_state=RANDOM_STATE
)

print(f"Split scheme: {SPLIT_SCHEME}")
print("Train shape:", x_train.shape, "Labels:", y_train.shape)
if x_val is not None:
    print("Val shape:", x_val.shape, "Labels:", y_val.shape)
else:
    print("Val set: None (80/20 scheme)")
print("Test shape:", x_test.shape, "Labels:", y_test.shape)

# ============ ENCODE LABELS ============
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)

if y_val is not None:
    y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

y_train_cat = to_categorical(y_train_enc)
if y_val is not None:
    y_val_cat = to_categorical(y_val_enc)
y_test_cat = to_categorical(y_test_enc)

num_classes = y_train_cat.shape[1]
print("Num classes:", num_classes)
print("Classes order:", label_encoder.classes_)

# ============ BUILD MODEL ============
print(f"Building LSTM model for architecture '{ARCH}' ...")
model = build_lstm_model(ARCH, SEQ_LEN, FEATURES, num_classes)
model.summary()

# ============ TRAIN ============
print("Starting training...")

if x_val is not None:
    history = model.fit(
        x_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val_cat)
    )
else:
    # 80/20 scheme: no explicit validation set
    history = model.fit(
        x_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

# ============ EVALUATE (ACCURACY) ============
loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy (architecture {ARCH}, split {SPLIT_SCHEME}): {acc * 100:.2f}%")

# ============ DETAILED METRICS: TP, FP, FN, PRECISION, RECALL, F1 ============
print("\n[INFO] Computing detailed classification metrics on test set...")

# Predict class probabilities and convert to class indices
y_pred_proba = model.predict(x_test, verbose=0)
y_pred_enc = np.argmax(y_pred_proba, axis=1)

# Confusion matrix (in encoded label space)
cm = confusion_matrix(y_test_enc, y_pred_enc)
classes = label_encoder.classes_
num_classes = len(classes)

print("\nConfusion matrix (rows = true, cols = predicted):")
print(cm)

print("\nPer-class metrics:")
print("Class\tTP\tFP\tFN\tPrecision\tRecall\t\tF1")

for i, cls_name in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    # TN is not used for precision/recall/F1, але можна порахувати за потреби:
    # TN = cm.sum() - TP - FP - FN

    # Safe computations with zero-division checks
    prec_den = TP + FP
    rec_den = TP + FN

    precision = TP / prec_den if prec_den > 0 else 0.0
    recall = TP / rec_den if rec_den > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print(f"{cls_name}\t{TP}\t{FP}\t{FN}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")

# (Optional) Full sklearn classification_report for convenience
print("\nSklearn classification report:")
print(classification_report(y_test_enc, y_pred_enc, target_names=classes))

# ============ SAVE MODEL & LABEL ENCODER ============
model_name = f"lstm_hand_relative_to_nose_polar_arch_{ARCH}.h5"
label_enc_name = "label_encoder_words.pkl"

model.save(model_name)
with open(label_enc_name, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\nModel saved to '{model_name}'")
print(f"Label encoder saved to '{label_enc_name}'")
