import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from Palma_model import MultiLayerConvLSTM, HDF5SequenceDataset

# ==== Parameters ====
MODEL_PATH = "/path/to/model.pth"
VALIDATION_PATH = "/path/to/validation_data"
CSV_OUT_PATH = "/path/to/error_by_timestep.csv"
RESIZE_TO = (315, 344)
BATCH_SIZE = 1

# ==== Setup ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==== Data ====
print("Loading validation dataset...")
val_dataset = HDF5SequenceDataset(VALIDATION_PATH, RESIZE_TO)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataset size:", len(val_dataset))

# ==== Model ====
print("Loading model...")
model = MultiLayerConvLSTM(use_sigmoid=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ==== Per-timestep errors ====
rmse_accum = [[] for _ in range(18)]
mae_accum = [[] for _ in range(18)]

print("Running validation...")
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(DEVICE)            # (B, 18, 1, H, W)
        y_batch = y_batch.to(DEVICE)            # (B, 18, 1, H, W)
        preds = model(x_batch)                   # (B, T, H, W, 1)
        preds = preds.permute(0, 1, 4, 2, 3)     # -> (B, T, 1, H, W)

        for t in range(18):
            pred_t = preds[:, t, 0].cpu().numpy()
            true_t = y_batch[:, t, 0].cpu().numpy()

            rmse = np.sqrt(np.mean((pred_t - true_t) ** 2))
            mae = np.mean(np.abs(pred_t - true_t))
            rmse_accum[t].append(rmse)
            mae_accum[t].append(mae)

rmse_mean = [float(np.mean(r)) if len(r) else float("nan") for r in rmse_accum]
mae_mean = [float(np.mean(m)) if len(m) else float("nan") for m in mae_accum]

os.makedirs(os.path.dirname(CSV_OUT_PATH) or ".", exist_ok=True)
with open(CSV_OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestep", "RMSE", "MAE"])
    for t in range(18):
        writer.writerow([t + 1, rmse_mean[t], mae_mean[t]])

print("Per-timestep errors written to:", CSV_OUT_PATH)
