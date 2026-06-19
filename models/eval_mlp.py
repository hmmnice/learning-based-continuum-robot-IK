import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH  = "tdcr_mlp_runs/run_sweep/tdcr_mlp.pt"
SCALER_PATH = "tdcr_mlp_runs/run_sweep/scalers.pkl"
CSV_PATH    = "tdcr_combined_dataset_sweeps/combined_log.csv"
LAST_N_SAMPLES = 5000 
N_EVAL         = 200    
SAMPLES_TO_PLOT = 5     

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

input_dim  = checkpoint["input_dim"]
output_dim = checkpoint["output_dim"]
hidden_dims = checkpoint["hidden_dims"]
dropout     = checkpoint["dropout"]

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

model = MLP(input_dim, output_dim, hidden_dims, dropout)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(SCALER_PATH, "rb") as f:
    scalers = pickle.load(f)

x_mean = scalers["x_scaler_mean"]
x_std  = scalers["x_scaler_std"]
y_mean = scalers["y_scaler_mean"]
y_std  = scalers["y_scaler_std"]

df = pd.read_csv(CSV_PATH)
df = df[df["tracking_ok"] == 1].copy()
df = df.dropna(subset=["points_json"]).copy()
df = df[df["points_json"].astype(str).str.len() > 2].copy()

if len(df) > LAST_N_SAMPLES:
    df = df.tail(LAST_N_SAMPLES).reset_index(drop=True)
    print(f"Using last {LAST_N_SAMPLES} samples")

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
df = df[valid_mask].reset_index(drop=True)
print(f"After outlier filter: {len(df)} samples")

def parse_points(s):
    return np.array(json.loads(s), dtype=np.float32)

def predict(row):
    x = np.array([
        row["spool_pos_m0"], row["spool_pos_m1"],
        row["spool_pos_m2"], row["spool_pos_m3"],
    ], dtype=np.float32)

    x_scaled = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]

    pred = (pred_scaled * y_std + y_mean).reshape(-1, 2)
    return pred

def align_to_base(pts):
    return pts - pts[0].copy()

def tip_error(pred, gt):
    return float(np.linalg.norm(pred[-1] - gt[-1]))

def shape_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def pearson(pred, gt):
    a, b = pred.flatten(), gt.flatten()
    return float(np.corrcoef(a, b)[0, 1])

eval_df = df.sample(min(N_EVAL, len(df)), random_state=42).reset_index(drop=True)

tip_errors  = []
shape_rmses = []
pcc_scores  = []

all_gt   = []
all_pred = []

for i, row in eval_df.iterrows():
    gt   = parse_points(row["points_json"])
    pred = predict(row)

    gt   = align_to_base(gt)
    pred = align_to_base(pred)

    tip_errors.append(tip_error(pred, gt))
    shape_rmses.append(shape_rmse(pred, gt))
    pcc_scores.append(pearson(pred, gt))

    all_gt.append(gt)
    all_pred.append(pred)

plot_indices = np.random.choice(len(eval_df), SAMPLES_TO_PLOT, replace=False)

for idx in plot_indices:
    gt   = all_gt[idx]
    pred = all_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt[:,   0], gt[:,   1], "bo-", label="Ground Truth", markersize=4)
    ax.plot(pred[:, 0], pred[:, 1], "ro-", label="Prediction",   markersize=4)

    for k in range(len(gt)):
        ax.plot([gt[k,0], pred[k,0]], [gt[k,1], pred[k,1]], "k-", alpha=0.2, linewidth=0.8)

    ax.invert_yaxis()
    ax.axis("equal")
    ax.legend()
    ax.set_title(
        f"Sample {idx+1} | "
        f"tip_err={tip_errors[idx]:.1f}px | "
        f"rmse={shape_rmses[idx]:.1f}px"
    )
    ax.set_xlabel("x (relative to base)")
    ax.set_ylabel("y (relative to base)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

NUM_POINTS = len(all_gt[0])
per_point_errors = np.array([
    np.linalg.norm(all_pred[i] - all_gt[i], axis=1)
    for i in range(len(all_gt))
])

mean_pp = per_point_errors.mean(axis=0)
std_pp  = per_point_errors.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(NUM_POINTS), mean_pp, "b-o", label="Mean error")
ax.fill_between(range(NUM_POINTS), mean_pp - std_pp, mean_pp + std_pp, alpha=0.2)
ax.set_xlabel("Point index (0=base, 19=tip)")
ax.set_ylabel("Euclidean error (px)")
ax.set_title("Per-Point Error Along Robot Body")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

print("\nEvaluation Metrics")
print("-" * 40)
print(f"Samples evaluated:  {len(eval_df)}")
print(f"Mean Tip Error:     {np.mean(tip_errors):.3f} px")
print(f"Mean Shape RMSE:    {np.mean(shape_rmses):.3f} px")
print(f"Mean PCC:           {np.mean(pcc_scores):.6f}")

print(f"\n=== Per-Point Error Summary ===")
print(f"{'Point':<8} {'Mean':>8} {'Std':>8} {'Max':>8}")
print("-" * 36)
for i in range(NUM_POINTS):
    label = " ← base" if i == 0 else (" ← tip" if i == NUM_POINTS - 1 else "")
    print(f"{i:<8} {mean_pp[i]:>8.2f} {std_pp[i]:>8.2f} {per_point_errors[:,i].max():>8.2f}{label}")

print(f"\nOverall mean: {per_point_errors.mean():.2f}px")
print(f"Tip mean:     {per_point_errors[:,-1].mean():.2f}px")
print(f"Mid-body:     {per_point_errors[:,NUM_POINTS//2].mean():.2f}px")