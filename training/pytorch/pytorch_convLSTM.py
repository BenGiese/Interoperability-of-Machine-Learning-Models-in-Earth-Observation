import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import random
import csv
import torch.nn.functional as F


# ==== Dataset ====
class HDF5SequenceDataset(Dataset):
    def __init__(self, directory_path, resize_to):
        self.resize_to = resize_to
        self.sequences = []

        batch_names = sorted([
            os.path.join(directory_path, name)
            for name in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, name))
        ])

        for batch in batch_names:
            files = sorted([
                f for f in os.listdir(batch)
                if f.lower().endswith((".hf5", ".h5", ".hdf5"))
            ])[:36]

            sequence = []
            for raster in files:
                fn = os.path.join(batch, raster)
                try:
                    with h5py.File(fn, 'r') as img:
                        original_image = np.array(img["image1"]["image_data"]).astype(np.uint8)
                        resized = Image.fromarray(original_image).resize(self.resize_to, resample=Image.BILINEAR)
                        normalized = np.array(resized).astype(np.float32) / 255.0
                        sequence.append(normalized)
                except Exception:
                    continue
            if len(sequence) == 36:
                self.sequences.append(np.expand_dims(np.array(sequence), axis=1))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = sequence[:18]
        y = sequence[18:]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ==== ConvLSTM Cell ====
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


# ==== ConvLSTM Block with optional BatchNorm ====
class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, batch_norm=True):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.use_bn = batch_norm
        self.relu = nn.LeakyReLU(0.01)
        if self.use_bn:
            self.bn = nn.BatchNorm3d(hidden_dim, momentum=0.99, eps=0.001)

    def forward(self, x):  # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        h_t = torch.zeros((b, self.cell.conv.out_channels // 4, h, w), device=x.device)
        c_t = torch.zeros_like(h_t)
        outputs = []
        for time in range(t):
            h_t, c_t = self.cell(x[:, time], h_t, c_t)
            out = self.relu(h_t)
            outputs.append(out.unsqueeze(1))
        out = torch.cat(outputs, dim=1)  # (B, T, C, H, W)
        if self.use_bn:
            out = out.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            out = self.bn(out)
            out = out.permute(0, 2, 1, 3, 4)
        return out


# ==== Full Multi-layer ConvLSTM Model ====
class MultiLayerConvLSTM(nn.Module):
    def __init__(self, use_sigmoid=True):
        super().__init__()
        self.layer1 = ConvLSTMBlock(1, 64, kernel_size=7)
        self.layer2 = ConvLSTMBlock(64, 64, kernel_size=5)
        self.layer3 = ConvLSTMBlock(64, 64, kernel_size=3)
        self.layer4 = ConvLSTMBlock(64, 64, kernel_size=1, batch_norm=False)
        self.final_conv3d = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x_seq):
        out = self.layer1(x_seq)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        out = self.final_conv3d(out)
        out = self.activation(out)
        return out.permute(0, 2, 3, 4, 1)  # Match Keras output format (B, T, H, W, 1)


# ==== Weight Initialization ====
def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ==== Training function ====
def train_model(model, model_save_path, log_csv_path):
    dataset = HDF5SequenceDataset(TRAINING_PATH, RESIZE_TO)
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(indices) * 0.1)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices))

    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001, rho=0.95, eps=1e-7)

    with open(log_csv_path, mode='w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Epoch", "TrainLoss", "ValLoss", "EpochTime", "LearningRate"])

        for epoch in range(EPOCHS):
            start_time = time.time()
            model.train()
            total_train_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                output = model(x_batch)
                y_batch = y_batch.permute(0, 2, 1, 3, 4).permute(0, 2, 3, 4, 1)
                loss = F.binary_cross_entropy(output, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                    output = model(x_val)
                    y_val = y_val.permute(0, 2, 1, 3, 4).permute(0, 2, 3, 4, 1)
                    val_loss = F.binary_cross_entropy(output, y_val)
                    total_val_loss += val_loss.item()

            mean_train_loss = total_train_loss / len(train_loader)
            mean_val_loss = total_val_loss / len(val_loader)
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            writer.writerow([epoch + 1, mean_train_loss, mean_val_loss, epoch_time, current_lr])

            if epoch == EPOCHS - 1:
                torch.save(model.state_dict(), model_save_path)

            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")


if __name__ == "__main__":
    # ==== Parameters ====
    TRAINING_PATH = "/path/to/training_data"
    MODEL_SAVE_PATH = "/path/to/output_model.pth"
    LOG_CSV_PATH = "/path/to/training_log.csv"
    RESIZE_TO = (315, 344)
    EPOCHS = 25
    BATCH_SIZE = 1
    SEED = 42

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Train
    model = MultiLayerConvLSTM(use_sigmoid=True).to(DEVICE)
    for m in model.modules():
        init_weights_xavier(m)
    train_model(model, MODEL_SAVE_PATH, LOG_CSV_PATH)
