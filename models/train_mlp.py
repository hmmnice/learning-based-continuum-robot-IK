import os
import json
import math
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
    output_dir: str = "tdcr_mlp_runs/run_sweep"

    use_tracking_ok_only: bool = True
    normalize_shape_relative_to_base: bool = True
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

    model_name: str = "tdcr_mlp.pt"
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

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_points_json(points_json_str: str):
    return json.loads(points_json_str)

def flatten_points(points):
    return np.array(points, dtype=np.float32).reshape(-1)

def normalize_points_relative_to_base(points):
    pts = np.array(points, dtype=np.float32)
    base = pts[0].copy()
    pts[:, 0] -= base[0]
    pts[:, 1] -= base[1]
    return pts

def load_dataset_from_csv(cfg: Config):
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    required_columns = [
        "spool_pos_m0", "spool_pos_m1", "spool_pos_m2", "spool_pos_m3",
        "tracking_ok", "points_json",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if cfg.use_tracking_ok_only:
        df = df[df["tracking_ok"] == 1].copy()

    df = df.dropna(subset=["points_json"]).copy()
    df = df[df["points_json"].astype(str).str.len() > 2].copy()

    if len(df) == 0:
        raise ValueError("No valid rows after filtering.")

    if cfg.last_n_samples is not None and len(df) > cfg.last_n_samples:
        df = df.tail(cfg.last_n_samples).reset_index(drop=True)
        print(f"Using last {cfg.last_n_samples} samples (most recent data)")
    else:
        print(f"Using all {len(df)} samples (fewer than last_n_samples limit)")

    tip_xs, tip_ys = [], []
    for _, row in df.iterrows():
        try:
            pts = np.array(json.loads(row["points_json"]), dtype=np.float32)
            tip_xs.append(pts[-1][0])
            tip_ys.append(pts[-1][1])
        except Exception:
            tip_xs.append(np.nan)
            tip_ys.append(np.nan)

    tip_xs = np.array(tip_xs)
    tip_ys = np.array(tip_ys)

    valid_mask = (
        (np.abs(tip_xs - np.nanmean(tip_xs)) < 3.0 * np.nanstd(tip_xs)) &
        (np.abs(tip_ys - np.nanmean(tip_ys)) < 3.0 * np.nanstd(tip_ys))
    )
    removed = int((~valid_mask).sum())
    df = df[valid_mask].reset_index(drop=True)
    print(f"Outlier filter removed {removed} samples, {len(df)} remain")

    x_list, y_list = [], []
    num_points_expected = None

    for _, row in df.iterrows():
        x = np.array([
            row["spool_pos_m0"], row["spool_pos_m1"],
            row["spool_pos_m2"], row["spool_pos_m3"],
        ], dtype=np.float32)

        points = parse_points_json(row["points_json"])

        if cfg.normalize_shape_relative_to_base:
            points = normalize_points_relative_to_base(points)

        if num_points_expected is None:
            num_points_expected = len(points)
        elif len(points) != num_points_expected:
            continue

        x_list.append(x)
        y_list.append(flatten_points(points))

    X = np.stack(x_list).astype(np.float32)
    Y = np.stack(y_list).astype(np.float32)

    metadata = {
        "num_samples":   int(len(X)),
        "input_dim":     int(X.shape[1]),
        "output_dim":    int(Y.shape[1]),
        "num_points":    int(num_points_expected),
        "last_n_samples": cfg.last_n_samples,
        "normalize_shape_relative_to_base": cfg.normalize_shape_relative_to_base,
    }

    return X, Y, metadata

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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_count = 0.0, 0
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), Y_b)
        loss.backward()
        optimizer.step()
        total_loss  += loss.item() * X_b.size(0)
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
        loss  = criterion(preds, Y_b)
        total_loss  += loss.item() * X_b.size(0)
        total_count += X_b.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(Y_b.cpu().numpy())
    avg_loss   = total_loss / max(total_count, 1)
    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    rmse       = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    return avg_loss, rmse

def main():
    set_seed(CFG.random_seed)
    ensure_dir(CFG.output_dir)

    print("Loading dataset...")
    X, Y, metadata = load_dataset_from_csv(CFG)

    print(f"Loaded {metadata['num_samples']} samples")
    print(f"Input dim: {metadata['input_dim']}  Output dim: {metadata['output_dim']}")
    print(f"Points per sample: {metadata['num_points']}")

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size      = int(n * CFG.validation_split)
    train_idx     = indices[val_size:]
    val_idx       = indices[:val_size]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    x_scaler = StandardScalerNP().fit(X_train)
    y_scaler = StandardScalerNP().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    Y_train_s = y_scaler.transform(Y_train)
    X_val_s   = x_scaler.transform(X_val)
    Y_val_s   = y_scaler.transform(Y_val)

    train_loader = DataLoader(TDCRDataset(X_train_s, Y_train_s),
                              batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers)
    val_loader   = DataLoader(TDCRDataset(X_val_s, Y_val_s),
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

    print(f"\nTraining on: {CFG.device}")
    print(model)

    best_val_loss    = float("inf")
    best_epoch       = -1
    patience_counter = 0
    history          = []

    for epoch in range(1, CFG.num_epochs + 1):
        train_loss         = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, CFG.device)
        current_lr         = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": float(train_loss),
            "val_loss": float(val_loss), "val_rmse_scaled": float(val_rmse),
            "lr": float(current_lr),
        })

        print(
            f"Epoch {epoch:03d}/{CFG.num_epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"rmse={val_rmse:.6f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim":        metadata["input_dim"],
                "output_dim":       metadata["output_dim"],
                "hidden_dims":      CFG.hidden_dims,
                "dropout":          CFG.dropout,
                "metadata":         metadata,
                "best_val_loss":    best_val_loss,
                "best_epoch":       best_epoch,
            }, os.path.join(CFG.output_dir, CFG.model_name))
        else:
            patience_counter += 1

        if patience_counter >= CFG.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch}  |  Best val loss: {best_val_loss:.6f}")

    with open(os.path.join(CFG.output_dir, CFG.scalers_name), "wb") as f:
        pickle.dump({
            "x_scaler_mean": x_scaler.mean,
            "x_scaler_std":  x_scaler.std,
            "y_scaler_mean": y_scaler.mean,
            "y_scaler_std":  y_scaler.std,
            "metadata":      metadata,
        }, f)

    save_json(os.path.join(CFG.output_dir, CFG.config_name), {
        "csv_path": CFG.csv_path, "last_n_samples": CFG.last_n_samples,
        "validation_split": CFG.validation_split, "batch_size": CFG.batch_size,
        "num_epochs": CFG.num_epochs, "learning_rate": CFG.learning_rate,
        "hidden_dims": list(CFG.hidden_dims), "dropout": CFG.dropout,
        "early_stopping_patience": CFG.early_stopping_patience,
        "metadata": metadata, "history": history,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
    })

    print(f"Model   → {os.path.join(CFG.output_dir, CFG.model_name)}")
    print(f"Scalers → {os.path.join(CFG.output_dir, CFG.scalers_name)}")
    print(f"Config  → {os.path.join(CFG.output_dir, CFG.config_name)}")

if __name__ == "__main__":
    main()