"""Hand landmark detection module wrapping MediaPipe Hands."""

import numpy as np
import mediapipe as mp


class HandDetector:
    """Detects hand landmarks in video frames using MediaPipe Hands."""

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray, is_rgb: bool = True) -> tuple:
        """Detect hand landmarks in a video frame.

        Args:
            frame: numpy array (RGB from Gradio, or BGR from OpenCV).
            is_rgb: If True, frame is already RGB (Gradio webcam).
                    If False, frame is BGR (OpenCV) and will be converted.

        Returns:
            Tuple of (landmarks, handedness, annotated_frame):
                - landmarks: np.ndarray of shape (21, 3) or None if no hand detected.
                - handedness: "Left" or "Right" string, or None.
                - annotated_frame: Copy of frame with landmarks drawn on it.
        """
        if is_rgb:
            frame_rgb = frame
        else:
            frame_rgb = frame[:, :, ::-1]

        frame_rgb = np.ascontiguousarray(frame_rgb)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)

        annotated_frame = frame.copy()
        landmarks = None
        handedness = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract landmark coordinates as (21, 3) array
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )

            # Extract handedness label
            handedness = results.multi_handedness[0].classification[0].label

            # Draw landmarks on the annotated frame
            annotated_frame.flags.writeable = True
            self._mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )

        return landmarks, handedness, annotated_frame

    def release(self):
        """Release MediaPipe resources."""
        self._hands.close()
