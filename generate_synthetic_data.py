"""Generate synthetic MediaPipe-style hand landmarks for ASL letters A, B, C.

Since the machine has no webcam, this script creates realistic training data
by defining base hand poses for each letter and applying random augmentations
(noise, rotation, scale, translation) to produce varied samples.

Output: data/landmarks.csv  (columns: f0, f1, ..., f62, label)
"""

import csv
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import src.preprocessing
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import normalize_landmarks  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
NUM_FEATURES = 63
HEADER = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]

SAMPLES_PER_LETTER = 300
NOISE_STD = 0.015
ROTATION_MAX_DEG = 10
SCALE_MIN = 0.85
SCALE_MAX = 1.15

np.random.seed(42)


# ---------------------------------------------------------------------------
# Base landmark definitions (21 landmarks x 3 coords each)
# Coordinate system: x increases right, y increases DOWN (image coords),
# z is depth (toward camera is negative).
# All values roughly in [0, 1].
# ---------------------------------------------------------------------------

def _base_landmarks_A() -> np.ndarray:
    """Letter A: Closed fist with thumb extended to the side.

    All four fingers curled into the palm (tips near MCP joints).
    Thumb sticks up alongside the index finger.
    """
    lm = np.zeros((21, 3), dtype=np.float64)

    # Wrist
    lm[0] = [0.50, 0.80, 0.00]

    # --- THUMB (extended to side, pointing up) ---
    lm[1] = [0.38, 0.70, -0.02]   # CMC
    lm[2] = [0.33, 0.62, -0.03]   # MCP
    lm[3] = [0.31, 0.54, -0.03]   # IP
    lm[4] = [0.30, 0.47, -0.02]   # TIP  (up and to the left)

    # --- INDEX FINGER (curled) ---
    lm[5] = [0.42, 0.55, -0.01]   # MCP
    lm[6] = [0.41, 0.50, -0.04]   # PIP  (starts extending then curls)
    lm[7] = [0.42, 0.53, -0.06]   # DIP  (curled back down)
    lm[8] = [0.43, 0.56, -0.05]   # TIP  (near MCP level)

    # --- MIDDLE FINGER (curled) ---
    lm[9]  = [0.48, 0.53, -0.01]  # MCP
    lm[10] = [0.47, 0.48, -0.04]  # PIP
    lm[11] = [0.48, 0.51, -0.06]  # DIP
    lm[12] = [0.49, 0.54, -0.05]  # TIP  (near MCP level)

    # --- RING FINGER (curled) ---
    lm[13] = [0.54, 0.54, -0.01]  # MCP
    lm[14] = [0.53, 0.49, -0.04]  # PIP
    lm[15] = [0.54, 0.52, -0.06]  # DIP
    lm[16] = [0.55, 0.55, -0.05]  # TIP  (near MCP level)

    # --- PINKY (curled) ---
    lm[17] = [0.59, 0.56, -0.01]  # MCP
    lm[18] = [0.58, 0.52, -0.03]  # PIP
    lm[19] = [0.59, 0.55, -0.05]  # DIP
    lm[20] = [0.60, 0.57, -0.04]  # TIP  (near MCP level)

    return lm


def _base_landmarks_B() -> np.ndarray:
    """Letter B: Flat open hand, fingers extended straight up, thumb tucked.

    All four fingers pointing up with tips far above MCP joints.
    Thumb folded across the palm.
    """
    lm = np.zeros((21, 3), dtype=np.float64)

    # Wrist
    lm[0] = [0.50, 0.80, 0.00]

    # --- THUMB (tucked across palm) ---
    lm[1] = [0.38, 0.70, -0.02]   # CMC
    lm[2] = [0.36, 0.63, -0.03]   # MCP
    lm[3] = [0.39, 0.58, -0.04]   # IP   (folding inward)
    lm[4] = [0.42, 0.56, -0.04]   # TIP  (near index MCP base)

    # --- INDEX FINGER (fully extended up) ---
    lm[5] = [0.43, 0.55, -0.01]   # MCP
    lm[6] = [0.43, 0.44, -0.01]   # PIP
    lm[7] = [0.43, 0.34, -0.01]   # DIP
    lm[8] = [0.43, 0.25, -0.01]   # TIP  (far above MCP)

    # --- MIDDLE FINGER (fully extended up) ---
    lm[9]  = [0.48, 0.53, -0.01]  # MCP
    lm[10] = [0.48, 0.42, -0.01]  # PIP
    lm[11] = [0.48, 0.32, -0.01]  # DIP
    lm[12] = [0.48, 0.22, -0.01]  # TIP  (far above MCP)

    # --- RING FINGER (fully extended up) ---
    lm[13] = [0.53, 0.54, -0.01]  # MCP
    lm[14] = [0.53, 0.43, -0.01]  # PIP
    lm[15] = [0.53, 0.33, -0.01]  # DIP
    lm[16] = [0.53, 0.24, -0.01]  # TIP  (far above MCP)

    # --- PINKY (fully extended up) ---
    lm[17] = [0.58, 0.56, -0.01]  # MCP
    lm[18] = [0.58, 0.46, -0.01]  # PIP
    lm[19] = [0.58, 0.37, -0.01]  # DIP
    lm[20] = [0.58, 0.28, -0.01]  # TIP  (far above MCP)

    return lm


def _base_landmarks_C() -> np.ndarray:
    """Letter C: Curved hand like holding a cup.

    All fingers partially bent in a C-arc. Thumb opposing fingers, also curved.
    Tips are between fully extended and fully curled.
    """
    lm = np.zeros((21, 3), dtype=np.float64)

    # Wrist
    lm[0] = [0.50, 0.80, 0.00]

    # --- THUMB (curved, opposing fingers) ---
    lm[1] = [0.38, 0.70, -0.02]   # CMC
    lm[2] = [0.33, 0.63, -0.03]   # MCP
    lm[3] = [0.30, 0.56, -0.04]   # IP
    lm[4] = [0.30, 0.50, -0.05]   # TIP  (curved inward toward fingers)

    # --- INDEX FINGER (partially bent, C-curve) ---
    lm[5] = [0.43, 0.55, -0.01]   # MCP
    lm[6] = [0.42, 0.46, -0.02]   # PIP
    lm[7] = [0.43, 0.40, -0.04]   # DIP  (curving inward)
    lm[8] = [0.45, 0.38, -0.05]   # TIP  (medium height)

    # --- MIDDLE FINGER (partially bent, C-curve) ---
    lm[9]  = [0.48, 0.53, -0.01]  # MCP
    lm[10] = [0.48, 0.44, -0.02]  # PIP
    lm[11] = [0.49, 0.38, -0.04]  # DIP
    lm[12] = [0.51, 0.36, -0.05]  # TIP  (medium height)

    # --- RING FINGER (partially bent, C-curve) ---
    lm[13] = [0.54, 0.54, -0.01]  # MCP
    lm[14] = [0.54, 0.46, -0.02]  # PIP
    lm[15] = [0.55, 0.40, -0.04]  # DIP
    lm[16] = [0.57, 0.38, -0.05]  # TIP  (medium height)

    # --- PINKY (partially bent, C-curve) ---
    lm[17] = [0.59, 0.56, -0.01]  # MCP
    lm[18] = [0.59, 0.49, -0.02]  # PIP
    lm[19] = [0.60, 0.44, -0.04]  # DIP
    lm[20] = [0.62, 0.42, -0.05]  # TIP  (medium height)

    return lm


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _rotate_z(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate landmarks around the z-axis by *angle_deg* degrees."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ])
    # Rotate around the wrist (landmark 0) so the hand stays in place.
    center = landmarks[0].copy()
    rotated = (landmarks - center) @ rot.T + center
    return rotated


def augment(base: np.ndarray) -> np.ndarray:
    """Apply random noise, rotation, scale, and translation to base landmarks."""
    sample = base.copy()

    # 1. Gaussian noise
    sample += np.random.normal(0.0, NOISE_STD, sample.shape)

    # 2. Small random rotation around z-axis
    angle = np.random.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG)
    sample = _rotate_z(sample, angle)

    # 3. Random scale (uniform)
    scale = np.random.uniform(SCALE_MIN, SCALE_MAX)
    center = sample[0].copy()
    sample = (sample - center) * scale + center

    # 4. Random translation (keep roughly in 0-1 range)
    tx = np.random.uniform(-0.05, 0.05)
    ty = np.random.uniform(-0.05, 0.05)
    tz = np.random.uniform(-0.01, 0.01)
    sample += np.array([tx, ty, tz])

    return sample


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    base_poses = {
        "A": _base_landmarks_A(),
        "B": _base_landmarks_B(),
        "C": _base_landmarks_C(),
    }

    all_rows: list[list] = []

    for letter, base in base_poses.items():
        print(f"Generating {SAMPLES_PER_LETTER} samples for letter '{letter}' ...")
        for i in range(SAMPLES_PER_LETTER):
            sample = augment(base)
            # Normalize using the same pipeline as the real collector.
            features = normalize_landmarks(sample, handedness="Right")
            row = features.tolist() + [letter]
            all_rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{SAMPLES_PER_LETTER}")

    # Shuffle so the CSV isn't sorted by label
    np.random.shuffle(all_rows)

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} samples to {CSV_PATH}")
    print("Summary:")
    for letter in ("A", "B", "C"):
        count = sum(1 for row in all_rows if row[-1] == letter)
        print(f"  {letter}: {count} samples")


if __name__ == "__main__":
    main()
