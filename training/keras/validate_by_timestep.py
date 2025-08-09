import os
import csv
import numpy as np
import keras
from PIL import Image
import h5py

# ==== Paths ====
MODEL_PATH = "/path/to/model.keras"
VALIDATION_PATH = "/path/to/validation_data"
SAVE_CSV_PATH = "/path/to/error_by_timestep.csv"
RESIZE_TO = (315, 344)

# ==== Load model ====
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ==== Load validation data ====
def load_validation_data(directory_path, resize_to):
    resize_width, resize_height = resize_to
    X, y = [], []
    for folder in sorted(os.listdir(directory_path)):
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".hf5")])[:36]
        if len(files) < 36:
            continue
        sequence = np.zeros((36, resize_height, resize_width), dtype=np.float32)
        for i, file in enumerate(files):
            with h5py.File(os.path.join(folder_path, file), "r") as hf:
                data = np.array(hf["image1"]["image_data"]).astype(np.uint8)
                resized = Image.fromarray(data).resize((resize_width, resize_height), resample=Image.BILINEAR)
                sequence[i] = np.asarray(resized, dtype=np.float32) / 255.0
        X.append(sequence[:18])
        y.append(sequence[18:])
    X = np.expand_dims(np.array(X), axis=-1)
    y = np.expand_dims(np.array(y), axis=-1)
    return X, y

print("Loading validation data...")
X_val, y_val = load_validation_data(VALIDATION_PATH, RESIZE_TO)
print("Validation data shape:", X_val.shape)

# ==== Calculate errors per timestep ====
rmse_list = []
mae_list = []

preds = model.predict(X_val, batch_size=1, verbose=0)  # shape: (N, 18, H, W, 1)
for t in range(18):
    pred_t = preds[:, t, :, :, 0]
    true_t = y_val[:, t, :, :, 0]
    rmse = np.sqrt(np.mean((pred_t - true_t) ** 2))
    mae = np.mean(np.abs(pred_t - true_t))
    rmse_list.append(rmse)
    mae_list.append(mae)

# ==== Save CSV ====
os.makedirs(os.path.dirname(SAVE_CSV_PATH) or ".", exist_ok=True)
with open(SAVE_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestep", "RMSE", "MAE"])
    for i in range(18):
        writer.writerow([i + 1, rmse_list[i], mae_list[i]])

print("Per-timestep errors saved to:", SAVE_CSV_PATH)
