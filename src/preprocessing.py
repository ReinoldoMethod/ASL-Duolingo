"""Normalize MediaPipe hand landmarks for ASL sign classification."""

import numpy as np


def normalize_landmarks(landmarks: np.ndarray, handedness: str) -> np.ndarray:
    """Normalize raw MediaPipe hand landmarks into a canonical feature vector.

    Args:
        landmarks: numpy array shape (21, 3) - raw x, y, z from MediaPipe.
        handedness: "Left" or "Right" from MediaPipe handedness classification.

    Returns:
        numpy array shape (63,) - normalized, flattened feature vector.
    """
    # 1. Translate so the wrist is at the origin (position-invariant).
    centered = landmarks - landmarks[0]

    # 2. Mirror left hands to canonical right-hand orientation.
    if handedness == "Left":
        centered[:, 0] = -centered[:, 0]

    # 3. Scale normalize by the max absolute value (size-invariant).
    max_abs = np.max(np.abs(centered))
    if max_abs > 0:
        centered = centered / max_abs

    # 4. Flatten to a 63-element vector.
    return centered.flatten()
