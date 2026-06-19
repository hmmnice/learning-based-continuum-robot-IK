import cv2
import csv
import json
import math
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import serial

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
SERIAL_TIMEOUT = 0.05
ACK_TIMEOUT = 3.0

RIGHT_MOTORS = [0, 2]
LEFT_MOTORS = [1, 3]

SPOOL_IN_SIGN = [1, -1, -1, -1]

SPOOL_MIN = [10204, 2947, 22355, 15083]
SPOOL_MAX = [100000, 100000, 100000, 100000]

INITIAL_SPOOL_POS = [10388, 4940, 26926, 15458]

PULL_GAIN = [1.0, 1.0, 1.0, 1.0]
RELEASE_GAIN = [0.40, 0.40, 0.40, 0.40]
DIR_CHANGE_COMP = [15, 15, 15, 15]

TARGET_VALID_SAMPLES = 2500
MAX_TOTAL_ACTIONS = 12000

RANDOM_SEED = 42
STATE_FILE = "tdcr_motor_state.json"

SMALL = 40
MED = 80
LARGE = 120
XLARGE = 160

RESET_RELEASE_1 = 120
RESET_RELEASE_2 = 60

SETTLE_TIME_SEC = 0.40
RANDOM_EXTRA_SETTLE = 0.10
RESET_SETTLE_SEC = 0.80

TRIM_STEP_OPTIONS = [12, 20, 28]

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

USE_CALIBRATION = True
CAMERA_MATRIX = np.array([
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

FLIP_HORIZONTAL = False
FLIP_VERTICAL = False
ROTATE_180 = True

USE_MANUAL_ROI_SELECTION = True
ROI_X = 200
ROI_Y = 100
ROI_W = 800
ROI_H = 500

INNER_TOP_IGNORE = 45
INNER_BOTTOM_IGNORE = 15
INNER_LEFT_IGNORE = 10
INNER_RIGHT_IGNORE = 10

DARK_PERCENTILE = 2.0
DARK_MARGIN = 12
BLUR_KERNEL = 5
OPEN_KERNEL = 3
CLOSE_KERNEL = 7
VERTICAL_CONNECT_KERNEL_W = 3
VERTICAL_CONNECT_KERNEL_H = 17

MIN_COMPONENT_AREA = 120
MAX_COMPONENT_AREA = 100000

MIN_SEGMENT_WIDTH = 5
MAX_SEGMENT_WIDTH = 120
MAX_X_JUMP_PER_ROW = 45
MAX_MISSED_ROWS = 35
MIN_PATH_POINTS = 20
SMOOTHING_WINDOW = 6
NUM_CENTERLINE_POINTS = 20

USE_CONTOUR_TIP_REFINEMENT = True
MAX_TIP_CORRECTION_DIST = 60.0
USE_TIP_TAPER = True
TIP_TAPER_POINTS = 3

OUTPUT_DIR = "tdcr_combined_dataset_sweeps"
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_log.csv")

SAVE_DEBUG_OVERLAY = True
SAVE_DEBUG_MASK = False
SHOW_WINDOWS = True
SHOW_DEBUG_WINDOWS = False

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

def get_timestamp_strings():
    now = datetime.now()
    return now.isoformat(), now.strftime("%Y%m%d_%H%M%S_%f")

def clamp_spool_state(pos):
    out = []
    for i in range(4):
        out.append(int(max(SPOOL_MIN[i], min(SPOOL_MAX[i], int(pos[i])))))
    return out

def connect_serial():
    ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(2.0)
    ser.reset_input_buffer()
    print("Connected to ESP32.")
    return ser

def wait_for_done(ser, timeout=ACK_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue

            if line:
                print("ESP32:", line)

            if line == "DONE":
                return True

        time.sleep(0.001)
    return False

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            pos = data.get("spool_pos", INITIAL_SPOOL_POS)
            if len(pos) == 4:
                pos = clamp_spool_state(pos)
                print(f"Loaded saved spool state: {pos}")
                return pos
        except Exception as e:
            print("Failed to load saved state:", e)

    print(f"Using initial spool state: {INITIAL_SPOOL_POS}")
    return INITIAL_SPOOL_POS.copy()

def save_state(spool_pos):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"spool_pos": clamp_spool_state(spool_pos)}, f, indent=2)
    except Exception as e:
        print("Failed to save state:", e)

class TDCRController:
    def __init__(self, ser):
        self.ser = ser
        self.spool_pos = load_state()
        self.last_direction = [0, 0, 0, 0]
        self.busy = False

    def semantic_to_raw_delta(self, motor_index, semantic_delta):
        return int(semantic_delta * SPOOL_IN_SIGN[motor_index])

    def pull_room(self, motor_index):
        return SPOOL_MAX[motor_index] - self.spool_pos[motor_index]

    def release_room(self, motor_index):
        return self.spool_pos[motor_index] - SPOOL_MIN[motor_index]

    def status_string(self):
        return (
            f"pos={self.spool_pos} | "
            f"pull_room={[self.pull_room(i) for i in range(4)]} | "
            f"release_room={[self.release_room(i) for i in range(4)]}"
        )

    def compensate(self, motor, val):
        if val == 0:
            return 0

        direction = 1 if val > 0 else -1

        if direction > 0:
            delta = int(round(val * PULL_GAIN[motor]))
        else:
            delta = int(round(val * RELEASE_GAIN[motor]))

        if self.last_direction[motor] != 0 and direction != self.last_direction[motor]:
            if direction > 0:
                delta += DIR_CHANGE_COMP[motor]
            else:
                delta -= DIR_CHANGE_COMP[motor]

        self.last_direction[motor] = direction
        return delta

    def apply_semantic_move(self, semantic_moves):
        if self.busy:
            return False, [0, 0, 0, 0], [0, 0, 0, 0]

        clamped_semantic = [0, 0, 0, 0]

        for i in range(4):
            requested = self.compensate(i, semantic_moves[i])
            new_pos = self.spool_pos[i] + requested

            if new_pos > SPOOL_MAX[i]:
                new_pos = SPOOL_MAX[i]
            if new_pos < SPOOL_MIN[i]:
                new_pos = SPOOL_MIN[i]

            clamped_semantic[i] = int(new_pos - self.spool_pos[i])

        if clamped_semantic == [0, 0, 0, 0]:
            return False, clamped_semantic, [0, 0, 0, 0]

        for i in range(4):
            self.spool_pos[i] += clamped_semantic[i]

        raw_moves = [
            self.semantic_to_raw_delta(i, clamped_semantic[i])
            for i in range(4)
        ]

        cmd = f"{raw_moves[0]},{raw_moves[1]},{raw_moves[2]},{raw_moves[3]}\n"

        try:
            self.busy = True
            self.ser.write(cmd.encode())
            ok = wait_for_done(self.ser, timeout=ACK_TIMEOUT)
            if not ok:
                print("WARNING: Timed out waiting for DONE from ESP32.")
        except Exception as e:
            print("Serial write failed:", e)
            return False, clamped_semantic, raw_moves
        finally:
            self.busy = False
            save_state(self.spool_pos)

        return True, clamped_semantic, raw_moves

    def bend_right(self, amount):
        semantic = [0, 0, 0, 0]
        for m in RIGHT_MOTORS:
            semantic[m] = amount
        for m in LEFT_MOTORS:
            semantic[m] = -amount
        return self.apply_semantic_move(semantic)

    def bend_left(self, amount):
        semantic = [0, 0, 0, 0]
        for m in LEFT_MOTORS:
            semantic[m] = amount
        for m in RIGHT_MOTORS:
            semantic[m] = -amount
        return self.apply_semantic_move(semantic)

    def bend_right_max(self):
        max_pull_right = min(self.pull_room(m) for m in RIGHT_MOTORS)
        max_release_left = min(self.release_room(m) for m in LEFT_MOTORS)
        feasible = int(min(max_pull_right, max_release_left))
        if feasible <= 0:
            return False, [0, 0, 0, 0], [0, 0, 0, 0]

        semantic = [0, 0, 0, 0]
        for m in RIGHT_MOTORS:
            semantic[m] = feasible
        for m in LEFT_MOTORS:
            semantic[m] = -feasible
        return self.apply_semantic_move(semantic)

    def bend_left_max(self):
        max_pull_left = min(self.pull_room(m) for m in LEFT_MOTORS)
        max_release_right = min(self.release_room(m) for m in RIGHT_MOTORS)
        feasible = int(min(max_pull_left, max_release_right))
        if feasible <= 0:
            return False, [0, 0, 0, 0], [0, 0, 0, 0]

        semantic = [0, 0, 0, 0]
        for m in LEFT_MOTORS:
            semantic[m] = feasible
        for m in RIGHT_MOTORS:
            semantic[m] = -feasible
        return self.apply_semantic_move(semantic)

    def release_all(self, amount):
        semantic = [-amount, -amount, -amount, -amount]
        return self.apply_semantic_move(semantic)

    def trim_motor_pull(self, motor_index, amount):
        semantic = [0, 0, 0, 0]
        semantic[motor_index] = amount
        return self.apply_semantic_move(semantic)

    def trim_motor_release(self, motor_index, amount):
        semantic = [0, 0, 0, 0]
        semantic[motor_index] = -amount
        return self.apply_semantic_move(semantic)

class MotionScheduler:
    def __init__(self):
        self.queue = deque()
        self.pattern_index = 0

    def enqueue_reset(self):
        self.queue.append(("reset_release", RESET_RELEASE_1, {}))
        self.queue.append(("reset_release", RESET_RELEASE_2, {}))

    def enqueue_left_sweep(self):
        for a in [SMALL, MED, LARGE, MED, SMALL]:
            self.queue.append(("bend_left", a, {}))

    def enqueue_right_sweep(self):
        for a in [SMALL, MED, LARGE, MED, SMALL]:
            self.queue.append(("bend_right", a, {}))

    def enqueue_trim_block(self):

        for _ in range(3):
            motor = random.randint(0, 3)
            mag = random.choice(TRIM_STEP_OPTIONS)
            self.queue.append(("trim_pull", mag, {"motor_index": motor}))

    def enqueue_extreme_pair(self):
        direction = random.choice(["left", "right"])
        if direction == "left":
            self.queue.append(("bend_left_max", 0, {}))
        else:
            self.queue.append(("bend_right_max", 0, {}))

    def refill(self):

        choice = self.pattern_index % 6
        self.pattern_index += 1

        if choice == 0:
            self.enqueue_reset()
            self.enqueue_left_sweep()
        elif choice == 1:
            self.enqueue_reset()
            self.enqueue_right_sweep()
        elif choice == 2:
            self.enqueue_reset()
            self.enqueue_left_sweep()
            self.enqueue_trim_block()
        elif choice == 3:
            self.enqueue_reset()
            self.enqueue_right_sweep()
            self.enqueue_trim_block()
        elif choice == 4:
            self.enqueue_reset()
            self.enqueue_trim_block()
        else:
            self.enqueue_reset()
            self.enqueue_extreme_pair()
            self.enqueue_reset()

    def next_action(self):
        if not self.queue:
            self.refill()
        return self.queue.popleft()

def maybe_undistort(frame):
    if not USE_CALIBRATION:
        return frame

    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h)
    )
    return cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_matrix)

def apply_view_transforms(frame):
    if ROTATE_180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    if FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)
    return frame

def select_roi_from_frame(frame):
    print("\nDraw a rectangle around the robot area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select Detection Area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Detection Area")
    cv2.waitKey(1)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        print("No ROI selected. Falling back to default ROI.")
        return ROI_X, ROI_Y, ROI_W, ROI_H

    return int(x), int(y), int(w), int(h)

def crop_roi(frame, roi_x, roi_y, roi_w, roi_h):
    h, w = frame.shape[:2]
    x1 = max(0, roi_x)
    y1 = max(0, roi_y)
    x2 = min(w, roi_x + roi_w)
    y2 = min(h, roi_y + roi_h)

    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0)

    return frame[y1:y2, x1:x2], (x1, y1)

def crop_inner_region(frame, left_ignore, right_ignore, top_ignore, bottom_ignore):
    h, w = frame.shape[:2]
    x1 = left_ignore
    x2 = w - right_ignore
    y1 = top_ignore
    y2 = h - bottom_ignore

    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0)

    return frame[y1:y2, x1:x2], (x1, y1)

def preprocess_darkest_mask(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    dark_level = np.percentile(gray, DARK_PERCENTILE)
    threshold_value = min(255, dark_level + DARK_MARGIN)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray <= threshold_value] = 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (VERTICAL_CONNECT_KERNEL_W, VERTICAL_CONNECT_KERNEL_H)
    )
    mask_connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel)

    mask_connected = cv2.erode(mask_connected, np.ones((3, 3), np.uint8), iterations=1)
    mask_connected = cv2.dilate(mask_connected, np.ones((3, 3), np.uint8), iterations=1)

    return gray, mask, mask_connected

def keep_darkest_center_component(mask, gray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape[:2]
    target_x = w / 2.0
    target_y = h * 0.82

    best_label = -1
    best_score = float("inf")

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA or area > MAX_COMPONENT_AREA:
            continue

        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[label]
        pixels = gray[labels == label]
        mean_intensity = float(np.mean(pixels))
        dist = math.hypot(cx - target_x, cy - target_y)

        aspect_bonus = 0.0
        if ww > 0:
            aspect_ratio = hh / ww
            aspect_bonus = -4.0 * min(aspect_ratio, 8.0)

        score = mean_intensity + 0.45 * dist - 0.005 * area + aspect_bonus

        if score < best_score:
            best_score = score
            best_label = label

    result = np.zeros_like(mask)
    if best_label != -1:
        result[labels == best_label] = 255

    return result

def get_row_segments(binary_row):
    xs = np.where(binary_row > 0)[0]
    if len(xs) == 0:
        return []

    segments = []
    start = xs[0]
    prev = xs[0]

    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            segments.append((int(start), int(prev)))
            start = x
            prev = x

    segments.append((int(start), int(prev)))
    return segments

def choose_start_segment(mask):
    h, w = mask.shape[:2]
    center_x = w / 2.0

    for y in range(h - 1, -1, -1):
        segments = get_row_segments(mask[y])
        valid = []
        for x1, x2 in segments:
            width = x2 - x1 + 1
            if MIN_SEGMENT_WIDTH <= width <= MAX_SEGMENT_WIDTH:
                cx = 0.5 * (x1 + x2)
                valid.append((abs(cx - center_x), cx, y, x1, x2))

        if valid:
            valid.sort(key=lambda t: t[0])
            _, cx, yv, x1, x2 = valid[0]
            return {
                "x_center": float(cx),
                "y": float(yv),
                "x_left": int(x1),
                "x_right": int(x2),
            }

    return None

def extract_centerline_tracking(mask):
    start = choose_start_segment(mask)
    if start is None:
        return []

    points = [(start["x_center"], start["y"])]
    prev_x = start["x_center"]
    missed_rows = 0
    start_y = int(start["y"]) - 1

    for y in range(start_y, -1, -1):
        segments = get_row_segments(mask[y])
        candidates = []
        for x1, x2 in segments:
            width = x2 - x1 + 1
            if not (MIN_SEGMENT_WIDTH <= width <= MAX_SEGMENT_WIDTH):
                continue

            cx = 0.5 * (x1 + x2)
            jump = abs(cx - prev_x)

            if jump <= MAX_X_JUMP_PER_ROW:
                candidates.append((jump, cx, x1, x2))

        if not candidates:
            missed_rows += 1
            if missed_rows > MAX_MISSED_ROWS:
                break
            continue

        missed_rows = 0
        candidates.sort(key=lambda t: t[0])

        _, cx, _, _ = candidates[0]
        points.append((float(cx), float(y)))
        prev_x = float(cx)

    return points

def smooth_centerline(points, window_size=7):
    if len(points) == 0:
        return []

    if len(points) < window_size or window_size < 3:
        return points

    if window_size % 2 == 0:
        window_size += 1

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)

    pad = window_size // 2
    xs_padded = np.pad(xs, (pad, pad), mode="edge")

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    xs_smooth = np.convolve(xs_padded, kernel, mode="valid")

    smoothed = list(zip(xs_smooth.tolist(), ys.tolist()))
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed

def find_tip_from_contour(mask, base_point):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if largest is None or len(largest) == 0:
        return None

    pts = largest[:, 0, :].astype(np.float32)
    bx, by = base_point

    dists = np.sqrt((pts[:, 0] - bx) ** 2 + (pts[:, 1] - by) ** 2)
    if len(dists) == 0:
        return None

    tip_idx = int(np.argmax(dists))
    tx, ty = pts[tip_idx]
    return (float(tx), float(ty))

def refine_tip_with_contour(robot_mask, resampled_points):
    if not USE_CONTOUR_TIP_REFINEMENT:
        return resampled_points

    if len(resampled_points) < 2:
        return resampled_points

    refined = list(resampled_points)
    base_pt = refined[0]
    current_tip = refined[-1]

    contour_tip = find_tip_from_contour(robot_mask, base_pt)
    if contour_tip is None:
        return refined

    correction_dist = math.hypot(contour_tip[0] - current_tip[0], contour_tip[1] - current_tip[1])
    if correction_dist > MAX_TIP_CORRECTION_DIST:
        return refined

    refined[-1] = contour_tip

    if USE_TIP_TAPER and len(refined) >= TIP_TAPER_POINTS + 1:
        end_idx = len(refined) - 1
        start_idx = max(0, end_idx - TIP_TAPER_POINTS)

        anchor = refined[start_idx]
        ax, ay = anchor
        tx, ty = refined[-1]

        num_steps = end_idx - start_idx
        if num_steps > 0:
            for i in range(1, num_steps + 1):
                alpha = i / float(num_steps)
                px = (1.0 - alpha) * ax + alpha * tx
                py = (1.0 - alpha) * ay + alpha * ty
                refined[start_idx + i] = (float(px), float(py))

    return refined

def resample_polyline(points, n_samples):
    if len(points) == 0:
        return []

    if len(points) == 1:
        return [points[0]] * n_samples

    cumulative = [0.0]
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        cumulative.append(cumulative[-1] + math.hypot(x2 - x1, y2 - y1))

    total_len = cumulative[-1]
    if total_len == 0:
        return [points[0]] * n_samples

    targets = np.linspace(0.0, total_len, n_samples)
    resampled = []

    seg_idx = 0
    for t in targets:
        while seg_idx < len(cumulative) - 2 and cumulative[seg_idx + 1] < t:
            seg_idx += 1

        d1 = cumulative[seg_idx]
        d2 = cumulative[seg_idx + 1]

        p1 = np.array(points[seg_idx], dtype=np.float32)
        p2 = np.array(points[seg_idx + 1], dtype=np.float32)

        if d2 == d1:
            p = p1
        else:
            alpha = (t - d1) / (d2 - d1)
            p = (1.0 - alpha) * p1 + alpha * p2

        resampled.append((float(p[0]), float(p[1])))

    return resampled

def draw_overlay(full_frame, roi_offset, roi_box, path, resampled_points):
    overlay = full_frame.copy()
    ox, oy = roi_offset
    roi_x, roi_y, roi_w, roi_h = roi_box

    cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)

    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        cv2.line(
            overlay,
            (int(round(x1 + ox)), int(round(y1 + oy))),
            (int(round(x2 + ox)), int(round(y2 + oy))),
            (0, 255, 0),
            2
        )

    for idx, (x, y) in enumerate(resampled_points):
        px = int(round(x + ox))
        py = int(round(y + oy))
        cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)

    if len(resampled_points) > 0:
        bx, by = resampled_points[0]
        tx, ty = resampled_points[-1]

        cv2.circle(overlay, (int(round(bx + ox)), int(round(by + oy))), 7, (0, 255, 255), 2)
        cv2.putText(
            overlay, "BASE",
            (int(round(bx + ox)) + 8, int(round(by + oy))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
        )

        cv2.circle(overlay, (int(round(tx + ox)), int(round(ty + oy))), 7, (255, 0, 255), 2)
        cv2.putText(
            overlay, "TIP",
            (int(round(tx + ox)) + 8, int(round(ty + oy))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA
        )

    return overlay

def convert_points_to_robot_frame(points_full):
    pts = np.array(points_full, dtype=np.float32)
    base_x, base_y = pts[0]

    pts_robot = np.zeros_like(pts)
    pts_robot[:, 0] = pts[:, 0] - base_x
    pts_robot[:, 1] = base_y - pts[:, 1]

    return pts_robot.tolist()

def capture_shape_sample(cap, roi_x, roi_y, roi_w, roi_h):
    ok, frame = cap.read()
    if not ok:
        return {
            "tracking_ok": False,
            "reason": "camera_read_failed",
            "overlay": None,
            "mask": None
        }

    frame = maybe_undistort(frame)
    frame = apply_view_transforms(frame)
    full_frame = frame.copy()

    roi_frame, roi_offset = crop_roi(frame, roi_x, roi_y, roi_w, roi_h)
    inner_frame, inner_offset = crop_inner_region(
        roi_frame,
        left_ignore=INNER_LEFT_IGNORE,
        right_ignore=INNER_RIGHT_IGNORE,
        top_ignore=INNER_TOP_IGNORE,
        bottom_ignore=INNER_BOTTOM_IGNORE
    )

    total_offset = (
        roi_offset[0] + inner_offset[0],
        roi_offset[1] + inner_offset[1]
    )

    gray, dark_mask, connected_mask = preprocess_darkest_mask(inner_frame)
    robot_mask = keep_darkest_center_component(connected_mask, gray)
    robot_mask_thin = cv2.erode(robot_mask, np.ones((3, 3), np.uint8), iterations=1)

    path = extract_centerline_tracking(robot_mask_thin)
    path = smooth_centerline(path, window_size=SMOOTHING_WINDOW)

    if len(path) < MIN_PATH_POINTS:
        overlay = draw_overlay(
            full_frame=full_frame,
            roi_offset=total_offset,
            roi_box=(roi_x, roi_y, roi_w, roi_h),
            path=path,
            resampled_points=[]
        )
        return {
            "tracking_ok": False,
            "reason": "path_too_short",
            "overlay": overlay,
            "mask": robot_mask,
            "num_path_points": len(path)
        }

    resampled = resample_polyline(path, NUM_CENTERLINE_POINTS)
    resampled = resampled[::-1]
    resampled = refine_tip_with_contour(robot_mask, resampled)

    points_full = [
        [float(x + total_offset[0]), float(y + total_offset[1])]
        for (x, y) in resampled
    ]

    points_robot = convert_points_to_robot_frame(points_full)

    base_x_img, base_y_img = points_full[0]
    tip_x_img, tip_y_img = points_full[-1]

    base_x_robot, base_y_robot = points_robot[0]
    tip_x_robot, tip_y_robot = points_robot[-1]

    overlay = draw_overlay(
        full_frame=full_frame,
        roi_offset=total_offset,
        roi_box=(roi_x, roi_y, roi_w, roi_h),
        path=path,
        resampled_points=resampled
    )

    if SHOW_WINDOWS:
        cv2.imshow("Overlay", overlay)
        if SHOW_DEBUG_WINDOWS:
            cv2.imshow("Robot Mask", robot_mask)
        cv2.waitKey(1)

    return {
        "tracking_ok": True,
        "reason": "",
        "overlay": overlay,
        "mask": robot_mask,

        "base_x": float(base_x_robot),
        "base_y": float(base_y_robot),
        "tip_x": float(tip_x_robot),
        "tip_y": float(tip_y_robot),

        "base_x_img": float(base_x_img),
        "base_y_img": float(base_y_img),
        "tip_x_img": float(tip_x_img),
        "tip_y_img": float(tip_y_img),

        "num_path_points": len(path),
        "num_resampled_points": len(resampled),

        "points_json": json.dumps(points_robot),

        "points_image_json": json.dumps(points_full),
    }

def open_combined_csv():
    ensure_dirs()
    file_exists = os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "sample_id",
            "timestamp_iso",
            "phase_name",
            "action_name",
            "action_value",
            "extra_info",
            "semantic_move_m0",
            "semantic_move_m1",
            "semantic_move_m2",
            "semantic_move_m3",
            "raw_move_m0",
            "raw_move_m1",
            "raw_move_m2",
            "raw_move_m3",
            "spool_pos_m0",
            "spool_pos_m1",
            "spool_pos_m2",
            "spool_pos_m3",
            "tracking_ok",
            "tracking_reason",

            "base_x",
            "base_y",
            "tip_x",
            "tip_y",

            "base_x_img",
            "base_y_img",
            "tip_x_img",
            "tip_y_img",

            "num_path_points",
            "num_resampled_points",
            "points_json",
            "points_image_json",
        ])

    return f, writer

def log_combined_sample(writer, sample_id, phase_name, action_name, action_value, extra_info,
                        semantic_move, raw_move, spool_pos, shape_data):
    timestamp_iso = datetime.now().isoformat()

    writer.writerow([
        sample_id,
        timestamp_iso,
        phase_name,
        action_name,
        action_value,
        json.dumps(extra_info),
        semantic_move[0],
        semantic_move[1],
        semantic_move[2],
        semantic_move[3],
        raw_move[0],
        raw_move[1],
        raw_move[2],
        raw_move[3],
        spool_pos[0],
        spool_pos[1],
        spool_pos[2],
        spool_pos[3],
        int(shape_data.get("tracking_ok", False)),
        shape_data.get("reason", ""),

        shape_data.get("base_x", ""),
        shape_data.get("base_y", ""),
        shape_data.get("tip_x", ""),
        shape_data.get("tip_y", ""),

        shape_data.get("base_x_img", ""),
        shape_data.get("base_y_img", ""),
        shape_data.get("tip_x_img", ""),
        shape_data.get("tip_y_img", ""),

        shape_data.get("num_path_points", ""),
        shape_data.get("num_resampled_points", ""),
        shape_data.get("points_json", ""),
        shape_data.get("points_image_json", ""),
    ])

def execute_action(controller, action_name, action_value, extra_info):
    if action_name == "bend_left":
        return controller.bend_left(action_value)

    if action_name == "bend_right":
        return controller.bend_right(action_value)

    if action_name == "bend_left_max":
        return controller.bend_left_max()

    if action_name == "bend_right_max":
        return controller.bend_right_max()

    if action_name == "reset_release":
        return controller.release_all(action_value)

    if action_name == "trim_pull":
        return controller.trim_motor_pull(extra_info["motor_index"], action_value)

    if action_name == "trim_release":
        return controller.trim_motor_release(extra_info["motor_index"], action_value)

    raise ValueError(f"Unknown action {action_name}")

def main():
    random.seed(RANDOM_SEED)
    ensure_dirs()

    ser = connect_serial()
    controller = TDCRController(ser)
    scheduler = MotionScheduler()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame from camera.")

    first_frame = maybe_undistort(first_frame)
    first_frame = apply_view_transforms(first_frame)

    if USE_MANUAL_ROI_SELECTION:
        roi_x, roi_y, roi_w, roi_h = select_roi_from_frame(first_frame)
    else:
        roi_x, roi_y, roi_w, roi_h = ROI_X, ROI_Y, ROI_W, ROI_H

    print(f"Using ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    csv_file, writer = open_combined_csv()


    print("TDCR SWEEP + RESET + TRIM DATA COLLECTION")
    print(f"Target valid samples: {TARGET_VALID_SAMPLES}")
    print(f"Max total actions:    {MAX_TOTAL_ACTIONS}")
    print(f"Current state:        {controller.status_string()}")

    valid_samples = 0
    total_actions = 0

    try:
        while valid_samples < TARGET_VALID_SAMPLES and total_actions < MAX_TOTAL_ACTIONS:
            action_name, action_value, extra_info = scheduler.next_action()
            total_actions += 1

            if action_name in ["bend_left", "bend_right"]:
                phase_name = "sweep"
            elif action_name == "reset_release":
                phase_name = "reset"
            elif "trim" in action_name:
                phase_name = "trim"
            elif "max" in action_name:
                phase_name = "extreme"
            else:
                phase_name = "other"

            success, semantic_move, raw_move = execute_action(
                controller,
                action_name,
                action_value,
                extra_info
            )

            if not success:
                print(f"[action {total_actions:04d}] skipped | action={action_name} | no feasible motion")
                time.sleep(0.15)
                continue

            if action_name == "reset_release":
                settle = RESET_SETTLE_SEC
            else:
                settle = SETTLE_TIME_SEC + random.uniform(0.0, RANDOM_EXTRA_SETTLE)
            time.sleep(settle)

            shape_data = capture_shape_sample(cap, roi_x, roi_y, roi_w, roi_h)

            if not shape_data.get("tracking_ok", False):
                print(
                    f"[action {total_actions:04d}] "
                    f"tracking failed | action={action_name} | reason={shape_data.get('reason', '')}"
                )
                continue

            valid_samples += 1

            log_combined_sample(
                writer=writer,
                sample_id=valid_samples,
                phase_name=phase_name,
                action_name=action_name,
                action_value=action_value,
                extra_info=extra_info,
                semantic_move=semantic_move,
                raw_move=raw_move,
                spool_pos=controller.spool_pos,
                shape_data=shape_data
            )
            csv_file.flush()

            timestamp_iso, frame_name = get_timestamp_strings()

            if SAVE_DEBUG_OVERLAY and shape_data.get("overlay") is not None:
                cv2.imwrite(
                    os.path.join(OVERLAY_DIR, f"{valid_samples:05d}_{frame_name}.png"),
                    shape_data["overlay"]
                )

            if SAVE_DEBUG_MASK and shape_data.get("mask") is not None:
                cv2.imwrite(
                    os.path.join(MASK_DIR, f"{valid_samples:05d}_{frame_name}.png"),
                    shape_data["mask"]
                )

            print(
                f"[valid {valid_samples:05d} | action {total_actions:05d}] "
                f"phase={phase_name:<7} "
                f"action={action_name:<14} "
                f"value={action_value:<3} "
                f"semantic={semantic_move} "
                f"state={controller.spool_pos} "
                f"tip=({shape_data.get('tip_x', 0):.1f},{shape_data.get('tip_y', 0):.1f})"
            )

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    finally:
        save_state(controller.spool_pos)
        csv_file.close()
        cap.release()
        ser.close()
        cv2.destroyAllWindows()

        print("\nCollection finished.")
        print(f"Valid tracked samples: {valid_samples}")
        print(f"Total actions tried:   {total_actions}")
        print(f"Final tracked spool state: {controller.spool_pos}")

if __name__ == "__main__":
    main()