import os
import random
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from Palma_model import MultiLayerConvLSTM

# ==== Configuration ====
MODEL_PATH = "/path/to/model.pth"
DATA_PATH = "/path/to/validation_data"
RESIZE_TO = (315, 344)        # (width, height)
MAX_BATCHES = 50
MIN_GT_MEAN = 0.01
SAVE_PATH = "/path/to/prediction_vs_groundtruth.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_valid_sample(directory_path, resize_to, min_gt_mean=MIN_GT_MEAN):
    """Return one (X, Y) sample where ground-truth mean exceeds threshold."""
    resize_w, resize_h = resize_to
    batch_names = [
        os.path.join(directory_path, name)
        for name in sorted(os.listdir(directory_path))
        if os.path.isdir(os.path.join(directory_path, name)) and not name.startswith(".")
    ]
    random.shuffle(batch_names)

    for batch_path in batch_names[:MAX_BATCHES]:
        files = sorted(f for f in os.listdir(batch_path) if f.endswith(".hf5"))
        if len(files) < 36:
            continue

        crn_batch = np.zeros((36, resize_h, resize_w), dtype=np.float32)
        try:
            for idx, raster in enumerate(files[:36]):
                fn = os.path.join(batch_path, raster)
                with h5py.File(fn, "r") as img:
                    arr = np.array(img["image1"]["image_data"]).astype(np.uint8)
                    resized = Image.fromarray(arr).resize((resize_w, resize_h), resample=Image.BILINEAR)
                    crn_batch[idx] = np.asarray(resized, dtype=np.float32) / 255.0
        except Exception:
            continue

        if crn_batch[18:].mean() > min_gt_mean:
            crn_batch = np.expand_dims(crn_batch, axis=1)  # (36, 1, H, W)
            return crn_batch[:18], crn_batch[18:]          # X (18,1,H,W), Y (18,1,H,W)

    raise ValueError("No suitable batch with sufficient rainfall found.")

def visualize_prediction(Y_true, Y_pred, save_path=None):
    """Plot 18 ground-truth vs. prediction frames side-by-side (2 x 18)."""
    # Ensure shapes: (T, H, W)
    if Y_true.ndim == 4 and Y_true.shape[1] == 1:
        Y_true = Y_true[:, 0]
    if Y_pred.ndim == 4 and Y_pred.shape[-1] == 1:
        Y_pred = Y_pred[..., 0]

    vmin = min(float(Y_true.min()), float(Y_pred.min()))
    vmax = max(float(Y_true.max()), float(Y_pred.max()))

    fig, axes = plt.subplots(2, 18, figsize=(24, 4))
    for i in range(18):
        axes[0, i].imshow(np.squeeze(Y_true[i]), cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].imshow(np.squeeze(Y_pred[i]), cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Ground Truth", fontsize=10)
    axes[1, 0].set_ylabel("Prediction", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print("Visualization saved to:", save_path)
    else:
        plt.show()

def main():
    print("Loading model...")
    model = MultiLayerConvLSTM(use_sigmoid=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    print("Selecting a validation sample...")
    X, Y_true = create_valid_sample(DATA_PATH, RESIZE_TO)  # (18,1,H,W) each
    input_tensor = torch.from_numpy(X).unsqueeze(0).float().to(DEVICE)  # (1,18,1,H,W)

    print("Running inference...")
    with torch.no_grad():
        Y_pred = model(input_tensor).squeeze(0).cpu().numpy()  # (T,H,W,1)

    visualize_prediction(Y_true, Y_pred, save_path=SAVE_PATH)

if __name__ == "__main__":
    main()
