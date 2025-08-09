import os
import time
import csv
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import h5py

print("Active Keras backend:", keras.backend.backend())

# ==== Configuration ====
MODEL_PATH = "/path/to/model.keras"
VALIDATION_PATH = "/path/to/validation_data"
SAVE_CSV_PATH = "/path/to/validation_results.csv"
RESIZE_TO = (315, 344)

# ==== GPU setup ====
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU enabled:", gpus[0])
    except RuntimeError as e:
        print("Could not set GPU memory growth:", e)
else:
    print("No GPU found – using CPU.")

# ==== Load model ====
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ==== Load validation data ====
def create_dataset_from_raw(directory_path, resize_to):
    resize_width, resize_height = resize_to
    batch_names = sorted(
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    )
    dataset = []
    for batch in batch_names:
        files = sorted(f for f in os.listdir(batch) if f.lower().endswith(".hf5"))[:36]
        if len(files) < 36:
            continue
        crn_batch = np.zeros((36, resize_height
