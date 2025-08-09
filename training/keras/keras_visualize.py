import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import random
from keras import models

# ==== Configuration ====
MODEL_PATH = "/path/to/model.keras"
DATA_PATH = "/path/to/validation_data"
RESIZE_TO = (315, 344)  # (width, height)
MAX_BATCHES = 50
MIN_GT_MEAN = 0.01
SAVE_PATH = "/path/to/prediction_vs_groundtruth.png"

def create_valid_sample(directory_path, resize_to, min_gt_mean=MIN_GT_MEAN):
    """Return one (X, Y) sample where GT mean exceeds threshold."""
    resize_width, resize_height = resize_to
    batch_names = [
        os.path.join(directory_path, name)
        for name in sorted(os.listdir(directory_path))
        if os.path.isdir(os.path.join(directory_path, name)) and not name.startswith(".")
    ]
    random.shuffle(batch_names)

    for batch_path in batch_names[:MAX_BATCHES]:
        files = sorted([f for f in os.listdir(batch_path) if f.endswith(".hf5")])
        if len(files) < 36:
            continue

        crn_batch = np.zeros((36, resize_height, resize_width), dtype=np.float32)
        try:
            for idx, raster in enumerate(files[:36]):
                fn = os.path.join(batch_path, raster)
                with h5py.File(fn, "r") as img:
                    original = np.array(img["image1"]["image_data"]).astype(np.uint8)
                    resized = Image.fromarray(original).resize((resize_width, resize_height), resample=Image.BILINEAR)
                    crn_batch[idx] = np.asarray(resized, dtype=np.float32) / 255.0
        except Exception:
            continue

        if crn_batch[18:].mean() > min_gt_mean:
            crn_batch = np.expand_dims(crn_batch, axis=-1)
            return crn_batch[:18], crn_batch[18:]  # X, Y

    raise ValueError("No suitable batch found.")

def visualize_prediction(X, Y_true, Y_pred, save_path=None):
    """Display GT and prediction for 18 frames side-by-side."""
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
    model = models.load_model(MODEL_PATH)

    print("Selecting validation batch...")
    X, Y_true = create_valid_sample(DATA_PATH, RESIZE_TO)

    print("Running inference...")
    Y_pred = model.predict(np.expand_dims(X, axis=0), verbose=0)[0]

    visualize_prediction(X, Y_true, Y_pred, save_path=SAVE_PATH)

if __name__ == "__main__":
    main()
