import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

MODEL_PATH = "tdcr_mlp_runs/run_sweep/tdcr_mlp.pt"
SCALER_PATH = "tdcr_mlp_runs/run_sweep/scalers.pkl"
CSV_PATH = "tdcr_combined_dataset_sweeps/combined_log.csv"

LAST_N_SAMPLES = 5000
USE_TRACKING_OK_ONLY = True

NUM_POINTS = 20
RANDOM_SEED = 42

TARGET_TIP = np.array([60.0, -260.0], dtype=np.float32)

OBSTACLES = [
    {"center": np.array([0.0, -100.0], dtype=np.float32), "radius": 10.0},
    {"center": np.array([100.0, -175.0], dtype=np.float32), "radius": 12.0},
]

SAFETY_MARGIN = 5.0

N_RANDOM_SAMPLES = 1500
N_REFINEMENT_ITERS = 7
N_LOCAL_SAMPLES = 800

TIP_WEIGHT = 8.0
TARGET_FAIL_WEIGHT = 40.0
COLLISION_WEIGHT = 2000.0
CLEARANCE_WEIGHT = 120.0
SMOOTHNESS_WEIGHT = 2.0

USE_NEUTRAL_BIAS = True
USE_MEDIAN_START = False

SHOW_DATASET_TIPS = False
PLOT_WITHOUT_OBSTACLE_BASELINE = True

class FullMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                torch.nn.Linear(prev, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def parse_points(s):
    return np.array(json.loads(s), dtype=np.float32)

def load_filtered_dataset():
    df = pd.read_csv(CSV_PATH)

    if USE_TRACKING_OK_ONLY:
        df = df[df["tracking_ok"] == 1].copy()

    df = df.dropna(subset=["points_json"]).copy()
    df = df[df["points_json"].astype(str).str.len() > 2].copy()

    if len(df) > LAST_N_SAMPLES:
        df = df.tail(LAST_N_SAMPLES).reset_index(drop=True)
        print(f"Using last {LAST_N_SAMPLES} samples")

    tip_xs, tip_ys = [], []
    for _, row in df.iterrows():
        try:
            pts = parse_points(row["points_json"])
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
    return df

def get_actuation_array(df):
    return df[[
        "spool_pos_m0",
        "spool_pos_m1",
        "spool_pos_m2",
        "spool_pos_m3"
    ]].values.astype(np.float32)

def get_tip_array(df):
    tips = []
    for _, row in df.iterrows():
        pts = parse_points(row["points_json"])
        tips.append(pts[-1])
    return np.array(tips, dtype=np.float32)

def load_model_and_scalers():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = FullMLP(
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

    return model, x_mean, x_std, y_mean, y_std

def predict_shape(model, x, x_mean, x_std, y_mean, y_std):
    x = np.asarray(x, dtype=np.float32)
    x_scaled = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()[0]

    pts = (pred_scaled * y_std + y_mean).reshape(-1, 2).astype(np.float32)
    pts = pts - pts[0].copy()
    return pts

def point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a

    denom = np.dot(ab, ab)
    if denom < 1e-8:
        return float(np.linalg.norm(p - a))

    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))

def polyline_min_distance_to_circle_center(shape, obstacle_center):
    min_dist = float("inf")

    for pt in shape:
        d = np.linalg.norm(pt - obstacle_center)
        if d < min_dist:
            min_dist = float(d)

    for i in range(len(shape) - 1):
        a = shape[i]
        b = shape[i + 1]
        d = point_to_segment_distance(obstacle_center, a, b)
        if d < min_dist:
            min_dist = d

    return float(min_dist)

def min_clearance_to_single_obstacle(shape, obstacle_center, obstacle_radius):
    min_dist = polyline_min_distance_to_circle_center(shape, obstacle_center)
    return float(min_dist - obstacle_radius)

def minimum_clearance_all(shape, obstacles):
    min_clear = float("inf")
    for obs in obstacles:
        c = min_clearance_to_single_obstacle(shape, obs["center"], obs["radius"])
        if c < min_clear:
            min_clear = c
    return float(min_clear)

def violates_safety_any(shape, obstacles, safety_margin):
    for obs in obstacles:
        min_dist = polyline_min_distance_to_circle_center(shape, obs["center"])
        if min_dist < (obs["radius"] + safety_margin):
            return True
    return False

def tip_distance_cost(shape, target_tip):
    return float(np.linalg.norm(shape[-1] - target_tip))

def target_fail_penalty(shape, target_tip, threshold=45.0):
    d = np.linalg.norm(shape[-1] - target_tip)
    if d <= threshold:
        return 0.0
    return float((d - threshold) ** 2)

def collision_penalty_all(shape, obstacles, safety_margin):
    total = 0.0
    for obs in obstacles:
        center = obs["center"]
        radius = obs["radius"]
        dists = np.linalg.norm(shape - center[None, :], axis=1)
        clearance = dists - (radius + safety_margin)
        penalties = np.maximum(0.0, -clearance)
        total += float(np.sum(penalties ** 2))
    return float(total)

def soft_clearance_penalty_all(shape, obstacles, safety_margin):
    total = 0.0
    for obs in obstacles:
        center = obs["center"]
        radius = obs["radius"]
        dists = np.linalg.norm(shape - center[None, :], axis=1)
        desired = radius + safety_margin
        penalties = np.maximum(0.0, desired - dists)
        total += float(np.sum(penalties ** 2))
    return float(total)

def smoothness_cost(x, x_ref, x_range):
    normed = (x - x_ref) / np.maximum(x_range, 1.0)
    return float(np.sum(normed ** 2))

def total_cost(
    shape,
    x,
    target_tip,
    obstacles,
    safety_margin,
    x_ref,
    x_range,
    use_obstacle=True,
):

    if use_obstacle and violates_safety_any(shape, obstacles, safety_margin):
        return 1e12

    cost = TIP_WEIGHT * tip_distance_cost(shape, target_tip)
    cost += TARGET_FAIL_WEIGHT * target_fail_penalty(shape, target_tip, threshold=45.0)

    if use_obstacle:
        cost += COLLISION_WEIGHT * collision_penalty_all(
            shape, obstacles, safety_margin
        )
        cost += CLEARANCE_WEIGHT * soft_clearance_penalty_all(
            shape, obstacles, safety_margin
        )

    if USE_NEUTRAL_BIAS:
        cost += SMOOTHNESS_WEIGHT * smoothness_cost(x, x_ref, x_range)

    return float(cost)

def clip_to_bounds(x, x_min, x_max):
    return np.minimum(np.maximum(x, x_min), x_max)

def choose_start_configuration(df, target_tip):
    X = get_actuation_array(df)
    tips = get_tip_array(df)

    if USE_MEDIAN_START:
        return np.median(X, axis=0).astype(np.float32)

    dists = np.linalg.norm(tips - target_tip[None, :], axis=1)
    best_idx = int(np.argmin(dists))
    return X[best_idx].astype(np.float32)

def random_shooting_search(
    model,
    x_mean, x_std, y_mean, y_std,
    x_min, x_max,
    target_tip,
    obstacles,
    safety_margin,
    x_ref,
    use_obstacle=True
):
    x_range = x_max - x_min

    best_x = None
    best_shape = None
    best_cost = float("inf")

    for _ in range(N_RANDOM_SAMPLES):
        x = np.random.uniform(x_min, x_max).astype(np.float32)
        shape = predict_shape(model, x, x_mean, x_std, y_mean, y_std)
        c = total_cost(
            shape, x, target_tip,
            obstacles, safety_margin,
            x_ref, x_range,
            use_obstacle=use_obstacle
        )

        if c < best_cost:
            best_cost = c
            best_x = x.copy()
            best_shape = shape.copy()

    if best_x is None:
        raise RuntimeError("No feasible candidate found in global search.")

    current_sigma = 0.10 * x_range

    for _ in range(N_REFINEMENT_ITERS):
        for _ in range(N_LOCAL_SAMPLES):
            x = best_x + np.random.normal(loc=0.0, scale=current_sigma, size=4).astype(np.float32)
            x = clip_to_bounds(x, x_min, x_max).astype(np.float32)

            shape = predict_shape(model, x, x_mean, x_std, y_mean, y_std)
            c = total_cost(
                shape, x, target_tip,
                obstacles, safety_margin,
                x_ref, x_range,
                use_obstacle=use_obstacle
            )

            if c < best_cost:
                best_cost = c
                best_x = x.copy()
                best_shape = shape.copy()

        current_sigma *= 0.55

    return best_x, best_shape, best_cost

def plot_solution(
    obstacle_shape,
    obstacle_x,
    no_obstacle_shape,
    no_obstacle_x,
    target_tip,
    obstacles,
    safety_margin,
    dataset_tips=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    if dataset_tips is not None and SHOW_DATASET_TIPS:
        ax.scatter(dataset_tips[:, 0], dataset_tips[:, 1], s=8, alpha=0.15, label="Dataset tips")

    ax.plot(
        obstacle_shape[:, 0], obstacle_shape[:, 1],
        "ro-", linewidth=2, markersize=5, label="Obstacle-aware IK"
    )

    if no_obstacle_shape is not None:
        ax.plot(
            no_obstacle_shape[:, 0], no_obstacle_shape[:, 1],
            "bo--", linewidth=2, markersize=4, label="No-obstacle IK"
        )

    ax.scatter([0], [0], c="black", s=100, label="Base")
    ax.scatter([target_tip[0]], [target_tip[1]], c="purple", s=120, label="Target tip")

    for i, obs in enumerate(obstacles):
        center = obs["center"]
        radius = obs["radius"]

        circle = plt.Circle(
            center,
            radius,
            color="orange",
            alpha=0.30,
            label="Obstacle" if i == 0 else None
        )
        ax.add_patch(circle)

        safety_circle = plt.Circle(
            center,
            radius + safety_margin,
            color="orange",
            fill=False,
            linestyle="--",
            linewidth=2,
            label="Safety margin" if i == 0 else None
        )
        ax.add_patch(safety_circle)

    ax.set_title("Obstacle-Aware IK using Full-Body NN")
    ax.set_xlabel("x (robot frame)")
    ax.set_ylabel("y (robot frame)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nBest obstacle-aware actuation:")
    print(obstacle_x)

    if no_obstacle_x is not None:
        print("\nBest no-obstacle actuation:")
        print(no_obstacle_x)

def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    df = load_filtered_dataset()
    dataset_X = get_actuation_array(df)
    dataset_tips = get_tip_array(df)

    x_min = dataset_X.min(axis=0).astype(np.float32)
    x_max = dataset_X.max(axis=0).astype(np.float32)

    print("\nActuation bounds from dataset:")
    print("x_min =", x_min)
    print("x_max =", x_max)

    model, x_mean, x_std, y_mean, y_std = load_model_and_scalers()

    x_ref = choose_start_configuration(df, TARGET_TIP).astype(np.float32)

    print("\nStart / reference configuration:")
    print(x_ref)

    if PLOT_WITHOUT_OBSTACLE_BASELINE:
        best_x_noobs, best_shape_noobs, best_cost_noobs = random_shooting_search(
            model=model,
            x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
            x_min=x_min, x_max=x_max,
            target_tip=TARGET_TIP,
            obstacles=OBSTACLES,
            safety_margin=SAFETY_MARGIN,
            x_ref=x_ref,
            use_obstacle=False
        )
    else:
        best_x_noobs, best_shape_noobs, best_cost_noobs = None, None, None

    obs_ref = best_x_noobs.astype(np.float32) if best_x_noobs is not None else x_ref

    best_x_obs, best_shape_obs, best_cost_obs = random_shooting_search(
        model=model,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        x_min=x_min, x_max=x_max,
        target_tip=TARGET_TIP,
        obstacles=OBSTACLES,
        safety_margin=SAFETY_MARGIN,
        x_ref=obs_ref,
        use_obstacle=True
    )

    tip_err_obs = np.linalg.norm(best_shape_obs[-1] - TARGET_TIP)
    min_clear_obs = minimum_clearance_all(best_shape_obs, OBSTACLES)

    print("Obstacle-Aware IK Result ->>>")
    print(f"Target tip: {TARGET_TIP}")
    print("Obstacles:")
    for i, obs in enumerate(OBSTACLES):
        print(f"  {i+1}. center={obs['center']} radius={obs['radius']}")
    print(f"Best obstacle-aware tip: {best_shape_obs[-1]}")
    print(f"Tip distance to target:  {tip_err_obs:.3f}")
    print(f"Total cost:              {best_cost_obs:.3f}")
    print(f"Obstacle-aware minimum clearance: {min_clear_obs:.3f}")
    print(f"Required safety clearance:        {SAFETY_MARGIN:.3f}")

    if best_shape_noobs is not None:
        tip_err_noobs = np.linalg.norm(best_shape_noobs[-1] - TARGET_TIP)
        min_clear_noobs = minimum_clearance_all(best_shape_noobs, OBSTACLES)

        print("\nNo-obstacle baseline:")
        print(f"Best no-obstacle tip:    {best_shape_noobs[-1]}")
        print(f"Tip distance to target:  {tip_err_noobs:.3f}")
        print(f"Total cost:              {best_cost_noobs:.3f}")
        print(f"No-obstacle minimum clearance:    {min_clear_noobs:.3f}")

    plot_solution(
        obstacle_shape=best_shape_obs,
        obstacle_x=best_x_obs,
        no_obstacle_shape=best_shape_noobs,
        no_obstacle_x=best_x_noobs,
        target_tip=TARGET_TIP,
        obstacles=OBSTACLES,
        safety_margin=SAFETY_MARGIN,
        dataset_tips=dataset_tips,
    )

if __name__ == "__main__":
    main()