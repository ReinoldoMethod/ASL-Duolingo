"""ASL Data Collector -- record hand landmarks via webcam to CSV."""

import csv
import os
import sys

import cv2

from src.detector import HandDetector
from src.preprocessing import normalize_landmarks

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
VALID_LETTERS = ["A", "B", "C"]
NUM_FEATURES = 63
HEADER = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_existing_counts() -> dict:
    """Load sample counts from an existing CSV file."""
    counts = {letter: 0 for letter in VALID_LETTERS}
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[-1] in counts:
                    counts[row[-1]] += 1
    return counts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    detector = HandDetector()
    counts = load_existing_counts()
    new_rows: list[list] = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print("ASL Data Collector started. Press A, B, or C to record. Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam.")
            break

        landmarks, handedness, annotated = detector.detect(frame, is_rgb=False)

        # -- Draw instructions --
        cv2.putText(
            annotated,
            "Press A, B, or C to record that letter. Press Q to quit.",
            (10, 30),
            FONT,
            0.5,
            GREEN,
            1,
            cv2.LINE_AA,
        )

        # -- Draw sample counts --
        y_offset = 60
        for letter in VALID_LETTERS:
            cv2.putText(
                annotated,
                f"{letter}: {counts[letter]} samples",
                (10, y_offset),
                FONT,
                0.5,
                WHITE,
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        # -- Hand status --
        if landmarks is None:
            cv2.putText(
                annotated,
                "No hand detected",
                (10, y_offset + 10),
                FONT,
                0.6,
                RED,
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("ASL Data Collector", annotated)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key in (ord("q"), ord("Q")):
            break

        # Record a sample
        for letter in VALID_LETTERS:
            if key in (ord(letter.lower()), ord(letter)):
                if landmarks is None:
                    print(f"  No hand detected -- skipped recording for '{letter}'.")
                    break
                features = normalize_landmarks(landmarks, handedness)
                new_rows.append(features.tolist() + [letter])
                counts[letter] += 1
                print(f"  Recorded '{letter}' (total: {counts[letter]})")
                break

    # -- Cleanup --
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    # -- Save collected data --
    if new_rows:
        file_exists = os.path.isfile(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HEADER)
            writer.writerows(new_rows)
        print(f"\nSaved {len(new_rows)} new samples to {CSV_PATH}")
    else:
        print("\nNo new samples collected.")

    print("Summary:")
    for letter in VALID_LETTERS:
        print(f"  {letter}: {counts[letter]} total samples")


if __name__ == "__main__":
    main()
