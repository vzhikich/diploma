import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# === LOAD DATASET ===
with open('sequences_hand_relative_to_nose_polar.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = data_dict['data']      # (N, SEQ_LEN, 84)
y = data_dict['labels']    # (N,)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))

SEQ_LEN = X.shape[1]
FEATURES = X.shape[2]      # should be 84

# === TRAIN / TEST SPLIT ===
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y   # now we have at least 2 classes
)

# === ENCODE LABELS ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

num_classes = y_train_cat.shape[1]
print("Num classes:", num_classes)

# === BUILD LSTM MODEL ===
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, FEATURES)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === TRAIN ===
model.fit(
    x_train, y_train_cat,
    epochs=50,
    batch_size=16,
    validation_data=(x_test, y_test_cat)
)

# === EVALUATE ===
loss, acc = model.evaluate(x_test, y_test_cat)
print(f"Test accuracy: {acc * 100:.2f}%")

# === SAVE MODEL & LABEL ENCODER ===
model.save('lstm_hand_relative_to_nose_polar.h5')
with open('label_encoder_words.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model saved to 'lstm_hand_relative_to_nose_polar.h5'")
print("Label encoder saved to 'label_encoder_words.pkl'")
