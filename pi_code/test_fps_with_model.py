# test_fps_with_model.py
from collections import deque
from picamera2 import Picamera2
import cv2
import time
import numpy as np
from keras.models import load_model

# === НАЛАШТУВАННЯ ===

# Список роздільностей, які хочеш перевірити
RESOLUTIONS = [
    (640, 480),
    (960, 540),
    (1280, 720),
    # додай свої варіанти
]

TEST_DURATION = 5.0  # скільки секунд міряти FPS для кожної роздільної здатності

MODEL_PATH = "lstm_hand_relative_to_nose_polar.h5"  # шлях до твоєї моделі

# === ЗАВАНТАЖЕННЯ МОДЕЛІ ===
print(f"Завантажуємо модель з {MODEL_PATH} ...")
model = load_model(MODEL_PATH)

# input_shape: (None, SEQ_LEN, FEATURES)
_, SEQ_LEN, FEATURES = model.input_shape
print(f"Модель очікує послідовності форми: (batch, {SEQ_LEN}, {FEATURES})")

# Буфер для "послідовності ознак"
buffer = deque(maxlen=SEQ_LEN)

# === КАМЕРА ===
picam2 = Picamera2()

for width, height in RESOLUTIONS:
    print(f"\n=== Тестуємо {width}x{height} з запуском LSTM на фоні ===")

    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # короткий прогрів камери

    # очистити буфер перед новим тестом
    buffer.clear()

    frames = 0
    t0 = time.time()

    while True:
        frame = picam2.capture_array()
        frames += 1

        # ---- ІМІТАЦІЯ РОБОТИ LSTM-МОДЕЛІ ----
        # Створюємо "фейковий" вектор ознак одного кадру
        # (можна зробити np.random.randn(...) замість нулів — різниці для навантаження майже немає)
        dummy_features = np.zeros((FEATURES,), dtype=np.float32)
        buffer.append(dummy_features)

        # Коли буфер заповнений — робимо передбачення
        if len(buffer) == SEQ_LEN:
            x_input = np.asarray(buffer, dtype=np.float32).reshape(1, SEQ_LEN, FEATURES)
            _ = model.predict(x_input, verbose=0)

        # ---- ПРЕВ'Ю (можна вимкнути для чистого вимірювання) ----
        cv2.imshow("Camera preview", frame)
        key = cv2.waitKey(1) & 0xFF
        # ESC → перервати поточний тест раніше
        if key == 27:
            break

        # Достатньо кадрів для цього тесту
        if time.time() - t0 >= TEST_DURATION:
            break

    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Роздільність {width}x{height}: {fps:.2f} FPS (за {elapsed:.2f} с)")

    picam2.stop()

cv2.destroyAllWindows()
print("\nТест завершено.")
