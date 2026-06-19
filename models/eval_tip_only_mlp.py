import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "tdcr_mlp_runs/run_tip_only/tip_only_mlp.pt"
SCALER_PATH = "tdcr_mlp_runs/run_tip_only/scalers.pkl"
CSV_PATH = "tdcr_combined_dataset_sweeps/combined_log.csv"

LAST_N_SAMPLES = 5000
N_EVAL = 200


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model = MLP(
    checkpoint["input_dim"],
    checkpoint["output_dim"],
    checkpoint["hidden_dims"],
    checkpoint["dropout"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(SCALER_PATH, "rb") as f:
    scalers = pickle.load(f)

x_mean = scalers["x_scaler_mean"]
x_std = scalers["x_scaler_std"]
y_mean = scalers["y_scaler_mean"]
y_std = scalers["y_scaler_std"]

df = pd.read_csv(CSV_PATH)
df = df[df["tracking_ok"] == 1].copy()
df = df.dropna(subset=["points_json"]).copy()
df = df[df["points_json"].astype(str).str.len() > 2].copy()

if len(df) > LAST_N_SAMPLES:
    df = df.tail(LAST_N_SAMPLES).reset_index(drop=True)

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

eval_df = df.sample(min(N_EVAL, len(df)), random_state=42).reset_index(drop=True)

errors = []
gt_tips = []
pred_tips = []

for _, row in eval_df.iterrows():
    x = np.array([
        row["spool_pos_m0"],
        row["spool_pos_m1"],
        row["spool_pos_m2"],
        row["spool_pos_m3"],
    ], dtype=np.float32)

    gt = np.array(json.loads(row["points_json"]), dtype=np.float32)[-1]

    x_scaled = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]

    pred = pred_scaled * y_std + y_mean

    err = float(np.linalg.norm(pred - gt))
    errors.append(err)

    gt_tips.append(gt)
    pred_tips.append(pred)

gt_tips = np.array(gt_tips)
pred_tips = np.array(pred_tips)

plt.figure(figsize=(7, 7))
plt.scatter(gt_tips[:, 0], gt_tips[:, 1], s=20, alpha=0.6, label="Ground Truth Tip")
plt.scatter(pred_tips[:, 0], pred_tips[:, 1], s=20, alpha=0.6, label="Predicted Tip")
plt.scatter([0], [0], c="red", s=100, label="Base")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Tip-Only NN: Predicted vs Ground Truth Tips")
plt.legend()
plt.show()

print("\nTip-Only NN Metrics")
print("-" * 30)
print(f"Samples evaluated: {len(eval_df)}")
print(f"Mean Tip Error:    {np.mean(errors):.3f} px")
print(f"Std Tip Error:     {np.std(errors):.3f} px")