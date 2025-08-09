# To switch the backend in Keras 3, set the environment variable before importing keras:
# Example for TensorFlow backend:
#     export KERAS_BACKEND="tensorflow"
# Example for PyTorch backend:
#     export KERAS_BACKEND="torch"
# Example for JAX backend:
#     export KERAS_BACKEND="jax"

import os
import time
import csv
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import keras
from keras import layers, models, callbacks

# === Configuration ===
print(f"Keras version: {keras.__version__}")
print(f"Active backend: {keras.backend.backend()}")

TRAINING_PATH = "/path/to/training_data"
MODEL_SAVE_PATH = "/path/to/output_model.keras"
TF_EXPORT_PATH = "/path/to/saved_model_directory"
TRAINING_LOG_PATH = "/path/to/training_log.csv"
RESIZE_TO = (315, 344)
EPOCHS = 25
BATCH_SIZE = 1
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# === Custom training logger ===
class TrainingLogger(callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "epoch_time_sec", "learning_rate"])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        lr = float(self.model.optimizer.learning_rate.numpy())
        row = [epoch + 1, logs.get("loss"), logs.get("val_loss"), epoch_time, lr]
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Epoch {epoch + 1} log: {row}")


# === Dataset loading ===
def create_dataset(directory_path, resize_to):
    resize_width, resize_height = resize_to
    batch_names = [
        os.path.join(directory_path, name)
        for name in sorted(os.listdir(directory_path))
        if os.path.isdir(os.path.join(directory_path, name)) and not name.startswith('.')
    ]
    dataset = []
    for batch_path in batch_names:
        files = sorted([f for f in os.listdir(batch_path) if f.endswith(".hf5")])
        if len(files) < 36:
            continue
        crn_batch = np.zeros((36, resize_height, resize_width))
        try:
            for idx, raster in enumerate(files[:36]):
                fn = os.path.join(batch_path, raster)
                with h5py.File(fn, 'r') as img:
                    original = np.array(img["image1"]["image_data"]).astype(np.uint8)
                    resized = Image.fromarray(original).resize((resize_width, resize_height), resample=Image.BILINEAR)
                    normalized = np.array(resized).astype(np.float32) / 255.0
                    crn_batch[idx] = normalized
        except Exception:
            continue
        dataset.append(crn_batch)

    dataset = np.expand_dims(np.array(dataset), axis=-1)
    return dataset[:, :18], dataset[:, 18:]


print("Loading data...")
x, y = create_dataset(TRAINING_PATH, resize_to=RESIZE_TO)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=SEED)


# === Model definition (functional API) ===
input_shape = (18, 344, 315, 1)
inputs = keras.Input(shape=input_shape)
x = layers.ConvLSTM2D(64, (7, 7), padding="same", return_sequences=True, activation=None)(inputs)
x = layers.LeakyReLU(0.01)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(64, (5, 5), padding="same", return_sequences=True, activation=None)(x)
x = layers.LeakyReLU(0.01)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, activation=None)(x)
x = layers.LeakyReLU(0.01)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(64, (1, 1), padding="same", return_sequences=True, activation=None)(x)
x = layers.LeakyReLU(0.01)(x)
out = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)

model = models.Model(inputs, out)

model.compile(
    loss="binary_crossentropy",
    optimizer="adadelta",
    metrics=[keras.metrics.MeanSquaredError(name="mse")]
)

# === Training ===
print("Starting training...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[TrainingLogger(TRAINING_LOG_PATH)]
)

# === Save model in Keras format ===
model.save(MODEL_SAVE_PATH)
print("Model saved (.keras):", MODEL_SAVE_PATH)

# === Export as TensorFlow SavedModel ===
model.export(TF_EXPORT_PATH)
print("TensorFlow SavedModel export completed:", TF_EXPORT_PATH)
