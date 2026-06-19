import os
import json
import random
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    csv_path: str = "tdcr_combined_dataset_sweeps/combined_log.csv"
    output_dir: str = "tdcr_mlp_runs/run_tip_only"

    use_tracking_ok_only: bool = True
    validation_split: float = 0.2
    random_seed: int = 42
    last_n_samples: int = 5000

    batch_size: int = 32
    num_epochs: int = 400
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5

    hidden_dims = (256, 256, 128)
    dropout = 0.05

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    early_stopping_patience: int = 40

    model_name: str = "tip_only_mlp.pt"
    scalers_name: str = "scalers.pkl"
    config_name: str = "train_config.json"


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StandardScalerNP:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class TDCRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256, 128), dropout=0.05):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_dataset():
    df = pd.read_csv(CFG.csv_path)

    if CFG.use_tracking_ok_only:
        df = df[df["tracking_ok"] == 1].copy()

    df = df.dropna(subset=["points_json"]).copy()
    df = df[df["points_json"].astype(str).str.len() > 2].copy()

    if len(df) > CFG.last_n_samples:
        df = df.tail(CFG.last_n_samples).reset_index(drop=True)
        print(f"Using last {CFG.last_n_samples} samples")
    else:
        print(f"Using all {len(df)} samples")

    tip_xs, tip_ys = [], []
    for _, row in df.iterrows():
        pts = np.array(json.loads(row["points_json"]), dtype=np.float32)
        tip_xs.append(pts[-1][0])
        tip_ys.append(pts[-1][1])

    tip_xs = np.array(tip_xs)
    tip_ys = np.array(tip_ys)

    valid_mask = (
        (np.abs(tip_xs - np.nanmean(tip_xs)) < 3.0 * np.nanstd(tip_xs)) &
        (np.abs(tip_ys - np.nanmean(tip_ys)) < 3.0 * np.nanstd(tip_ys))
    )
    df = df[valid_mask].reset_index(drop=True)

    X, Y = [], []
    for _, row in df.iterrows():
        x = np.array([
            row["spool_pos_m0"],
            row["spool_pos_m1"],
            row["spool_pos_m2"],
            row["spool_pos_m3"],
        ], dtype=np.float32)

        pts = np.array(json.loads(row["points_json"]), dtype=np.float32)
        tip = pts[-1].astype(np.float32)

        X.append(x)
        Y.append(tip)

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    metadata = {
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
    }
    return X, Y, metadata


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_count = 0.0, 0
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), Y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        total_count += X_b.size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds, all_targets = [], []
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        preds = model(X_b)
        loss = criterion(preds, Y_b)
        total_loss += loss.item() * X_b.size(0)
        total_count += X_b.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(Y_b.cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    return avg_loss, rmse


def main():
    set_seed(CFG.random_seed)
    ensure_dir(CFG.output_dir)

    print("Loading dataset...")
    X, Y, metadata = load_dataset()
    print(f"Loaded {metadata['num_samples']} samples")

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = int(n * CFG.validation_split)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    x_scaler = StandardScalerNP().fit(X_train)
    y_scaler = StandardScalerNP().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    Y_train_s = y_scaler.transform(Y_train)
    X_val_s = x_scaler.transform(X_val)
    Y_val_s = y_scaler.transform(Y_val)

    train_loader = DataLoader(TDCRDataset(X_train_s, Y_train_s),
                              batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers)
    val_loader = DataLoader(TDCRDataset(X_val_s, Y_val_s),
                            batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers)

    model = MLPRegressor(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_dims=CFG.hidden_dims,
        dropout=CFG.dropout,
    ).to(CFG.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.num_epochs, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, CFG.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, CFG.device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{CFG.num_epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"rmse={val_rmse:.6f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": metadata["input_dim"],
                "output_dim": metadata["output_dim"],
                "hidden_dims": CFG.hidden_dims,
                "dropout": CFG.dropout,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }, os.path.join(CFG.output_dir, CFG.model_name))
        else:
            patience_counter += 1

        if patience_counter >= CFG.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch} | Best val loss: {best_val_loss:.6f}")

    with open(os.path.join(CFG.output_dir, CFG.scalers_name), "wb") as f:
        pickle.dump({
            "x_scaler_mean": x_scaler.mean,
            "x_scaler_std": x_scaler.std,
            "y_scaler_mean": y_scaler.mean,
            "y_scaler_std": y_scaler.std,
        }, f)


if __name__ == "__main__":
    main()