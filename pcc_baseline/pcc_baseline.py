import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "tdcr_combined_dataset_sweeps/combined_log.csv"
LAST_N_SAMPLES = 5000
N_EVAL = 200
SAMPLES_TO_PLOT = 5
NUM_POINTS = 20
ROBOT_LENGTH = None
CURVATURE_GAIN = 0.005
CURVATURE_CLIP = 0.003
BEND_SIGN = -1.0
ACTUATION_SCALE = 50000.0
PRINT_DEBUG = True

def parse_points(s):
    return np.array(json.loads(s), dtype=np.float32)

def align_to_base(pts):
    return pts - pts[0].copy()

def polyline_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def tip_error(pred, gt):
    return float(np.linalg.norm(pred[-1] - gt[-1]))

def shape_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def pearson(pred, gt):
    a, b = pred.flatten(), gt.flatten()
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def estimate_robot_length_from_dataset(df, max_samples=200):
    lengths = []

    sample_df = df.sample(min(max_samples, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        try:
            pts = parse_points(row["points_json"])
            pts = align_to_base(pts)
            lengths.append(polyline_length(pts))
        except Exception:
            continue

    if len(lengths) == 0:
        raise ValueError("Could not estimate.")

    return float(np.median(lengths))

def compute_delta_u(row):
    m0 = float(row["spool_pos_m0"])
    m1 = float(row["spool_pos_m1"])
    m2 = float(row["spool_pos_m2"])
    m3 = float(row["spool_pos_m3"])

    u_right = m0 + m2
    u_left = m1 + m3

    return u_left - u_right

def pcc_predict_from_actuation(
    row,
    num_points=NUM_POINTS,
    length=350.0,
    actuation_scale=ACTUATION_SCALE,
    curvature_gain=CURVATURE_GAIN,
    curvature_clip=CURVATURE_CLIP,
    bend_sign=BEND_SIGN,
):
   

    delta_u = compute_delta_u(row)
    delta_u_norm = delta_u / actuation_scale

    curvature = bend_sign * curvature_gain * delta_u_norm
    curvature = float(np.clip(curvature, -curvature_clip, curvature_clip))

    s_vals = np.linspace(0.0, length, num_points)

    if abs(curvature) < 1e-8:
        x = np.zeros_like(s_vals)
        y = -s_vals
    else:
        R = 1.0 / curvature
        x = R * (1.0 - np.cos(curvature * s_vals))
        y = -R * np.sin(curvature * s_vals)

    pts = np.stack([x, y], axis=1).astype(np.float32)
    return pts, delta_u, delta_u_norm, curvature

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

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

    if ROBOT_LENGTH is None:
        robot_length = estimate_robot_length_from_dataset(df, max_samples=300)
        print(f"Estimated ROBOT_LENGTH from dataset: {robot_length:.2f} px")
    else:
        robot_length = float(ROBOT_LENGTH)
        print(f"Using fixed ROBOT_LENGTH: {robot_length:.2f} px")

    eval_df = df.sample(min(N_EVAL, len(df)), random_state=42).reset_index(drop=True)

    tip_errors = []
    shape_rmses = []
    pcc_scores = []

    all_gt = []
    all_pred = []

    printed_debug = 0

    for _, row in eval_df.iterrows():
        gt = parse_points(row["points_json"])
        gt = align_to_base(gt)

        pred, delta_u, delta_u_norm, curvature = pcc_predict_from_actuation(
            row,
            num_points=NUM_POINTS,
            length=robot_length,
            actuation_scale=ACTUATION_SCALE,
            curvature_gain=CURVATURE_GAIN,
            curvature_clip=CURVATURE_CLIP,
            bend_sign=BEND_SIGN,
        )

        pred = align_to_base(pred)

        if PRINT_DEBUG and printed_debug < 10:
            print(
                f"delta_u={delta_u:9.1f} | "
                f"delta_u_norm={delta_u_norm:8.4f} | "
                f"curvature={curvature:8.6f}"
            )
            printed_debug += 1

        tip_errors.append(tip_error(pred, gt))
        shape_rmses.append(shape_rmse(pred, gt))
        pcc_scores.append(pearson(pred, gt))

        all_gt.append(gt)
        all_pred.append(pred)

    plot_indices = np.random.choice(
        len(eval_df),
        min(SAMPLES_TO_PLOT, len(eval_df)),
        replace=False
    )

    for idx in plot_indices:
        gt = all_gt[idx]
        pred = all_pred[idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(gt[:, 0], gt[:, 1], "bo-", label="Ground Truth", markersize=4)
        ax.plot(pred[:, 0], pred[:, 1], "go-", label="PCC Baseline", markersize=4)

        for k in range(len(gt)):
            ax.plot(
                [gt[k, 0], pred[k, 0]],
                [gt[k, 1], pred[k, 1]],
                "k-",
                alpha=0.2,
                linewidth=0.8
            )

        ax.axis("equal")
        ax.legend()
        ax.set_title(
            f"PCC Sample {idx+1} | "
            f"tip_err={tip_errors[idx]:.1f}px | "
            f"rmse={shape_rmses[idx]:.1f}px"
        )
        ax.set_xlabel("x (relative to base)")
        ax.set_ylabel("y (relative to base)")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    print("\nPCC Baseline Metrics")
    print("-" * 40)
    print(f"Samples evaluated:  {len(eval_df)}")
    print(f"Robot length used:  {robot_length:.3f} px")
    print(f"Mean Tip Error:     {np.mean(tip_errors):.3f} px")
    print(f"Mean Shape RMSE:    {np.mean(shape_rmses):.3f} px")
    print(f"Mean PCC:           {np.mean(pcc_scores):.6f}")

if __name__ == "__main__":
    main()