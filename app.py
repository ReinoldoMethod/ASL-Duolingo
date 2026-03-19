"""ASL Alphabet Teacher — Gradio web application.

Entry point for both local running and HuggingFace Spaces deployment.
Provides three tabs: Free Sign, Learn, and Practice.
"""

import os
import random
import time

import cv2
import gradio as gr
import numpy as np

from src.detector import HandDetector
from src.preprocessing import normalize_landmarks
from src.classifier import ASLClassifier

# ---------------------------------------------------------------------------
# Global model / detector initialisation
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "asl_classifier.pkl")

detector = HandDetector()
classifier = None

try:
    classifier = ASLClassifier(MODEL_PATH)
except FileNotFoundError:
    print(
        f"[WARNING] Model not found at {MODEL_PATH}. "
        "Please run train_model.py first."
    )

# ---------------------------------------------------------------------------
# Hand-position descriptions for the Learn tab
# ---------------------------------------------------------------------------

LETTER_DESCRIPTIONS = {
    "A": "Make a fist with thumb resting on the side",
    "B": "Hold hand flat, fingers together pointing up, thumb tucked across palm",
    "C": "Curve hand into a C shape, like holding a cup",
}

LEARN_LETTERS = ["A", "B", "C"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _process_frame(frame: np.ndarray):
    """Run detection + classification on a single BGR frame.

    Returns (annotated_frame, letter_or_none, confidence_or_none).
    """
    if frame is None:
        return None, None, None

    landmarks, handedness, annotated = detector.detect(frame)

    if landmarks is None or classifier is None:
        return annotated, None, None

    features = normalize_landmarks(landmarks, handedness)
    letter, confidence = classifier.predict(features)
    return annotated, letter, confidence


def _draw_text(frame, text, position, font_scale=1.2, color=(0, 255, 0),
               thickness=3, bg_color=None):
    """Draw text with optional background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    if bg_color is not None:
        cv2.rectangle(frame, (x - 4, y - th - 8), (x + tw + 4, y + baseline + 4),
                       bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


# ===================================================================
# Tab 1 — Free Sign
# ===================================================================

def process_free_sign(frame: np.ndarray) -> np.ndarray:
    """Stream callback for the Free Sign tab."""
    if frame is None:
        return frame

    annotated, letter, confidence = _process_frame(frame)
    if annotated is None:
        return frame

    if classifier is None:
        _draw_text(annotated, "Model not loaded — run train_model.py",
                   (20, 40), font_scale=0.8, color=(0, 0, 255))
        return annotated

    if letter is not None and confidence is not None:
        label = f"{letter}  {confidence * 100:.0f}%"
        _draw_text(annotated, label, (20, 60), font_scale=2.0,
                   color=(0, 255, 0), thickness=4, bg_color=(0, 0, 0))
    else:
        _draw_text(annotated, "Show your hand to the camera",
                   (20, 50), font_scale=0.9, color=(0, 200, 255),
                   thickness=2, bg_color=(0, 0, 0))

    return annotated


# ===================================================================
# Tab 2 — Learn
# ===================================================================

def process_learn(frame: np.ndarray, current_index: int,
                  correct_time: float) -> tuple:
    """Stream callback for the Learn tab.

    State variables:
        current_index: index into LEARN_LETTERS (0-based)
        correct_time:  timestamp when the correct sign was first detected
                       (0.0 means not yet detected)

    Returns (annotated_frame, current_index, correct_time,
             status_html, progress_text).
    """
    total = len(LEARN_LETTERS)
    current_index = int(current_index)

    # Already completed all letters
    if current_index >= total:
        status = ("<span style='color:green; font-size:1.4em; font-weight:bold;'>"
                  "Congratulations! You learned A, B, C!</span>")
        progress = f"Letter {total}/{total}"
        if frame is not None:
            annotated, _, _ = _process_frame(frame)
            if annotated is not None:
                _draw_text(annotated, "Complete!", (20, 50),
                           font_scale=1.5, color=(0, 255, 0), thickness=3,
                           bg_color=(0, 0, 0))
                return annotated, current_index, correct_time, status, progress
        return frame, current_index, correct_time, status, progress

    target = LEARN_LETTERS[current_index]
    desc = LETTER_DESCRIPTIONS[target]
    progress = f"Letter {current_index + 1}/{total}"

    if frame is None:
        status = (f"<b>Sign the letter: <span style='font-size:1.6em;'>"
                  f"{target}</span></b><br>{desc}")
        return frame, current_index, correct_time, status, progress

    annotated, letter, confidence = _process_frame(frame)
    if annotated is None:
        annotated = frame

    if classifier is None:
        status = "<span style='color:red;'>Model not loaded — run train_model.py</span>"
        return annotated, current_index, correct_time, status, progress

    now = time.time()

    # Check if the user is signing the correct letter
    is_correct = (letter == target and confidence is not None
                  and confidence > 0.70)

    if is_correct:
        if correct_time == 0.0:
            correct_time = now

        elapsed = now - correct_time
        if elapsed >= 2.0:
            # Advance to next letter
            current_index += 1
            correct_time = 0.0
            if current_index >= total:
                status = ("<span style='color:green; font-size:1.4em; "
                          "font-weight:bold;'>"
                          "Congratulations! You learned A, B, C!</span>")
                progress = f"Letter {total}/{total}"
            else:
                next_target = LEARN_LETTERS[current_index]
                next_desc = LETTER_DESCRIPTIONS[next_target]
                status = (f"<b>Sign the letter: <span style='font-size:1.6em;'>"
                          f"{next_target}</span></b><br>{next_desc}")
                progress = f"Letter {current_index + 1}/{total}"
        else:
            status = ("<span style='color:green; font-size:1.3em; "
                      "font-weight:bold;'>Correct!</span>")
    else:
        correct_time = 0.0
        status = (f"<b>Sign the letter: <span style='font-size:1.6em;'>"
                  f"{target}</span></b><br>{desc}")

    # Draw overlay
    if letter is not None and confidence is not None:
        label = f"{letter}  {confidence * 100:.0f}%"
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        _draw_text(annotated, label, (20, 60), font_scale=2.0,
                   color=color, thickness=4, bg_color=(0, 0, 0))
    else:
        _draw_text(annotated, "Show your hand to the camera",
                   (20, 50), font_scale=0.9, color=(0, 200, 255),
                   thickness=2, bg_color=(0, 0, 0))

    return annotated, current_index, correct_time, status, progress


def reset_learn():
    """Reset the Learn tab state."""
    target = LEARN_LETTERS[0]
    desc = LETTER_DESCRIPTIONS[target]
    status = (f"<b>Sign the letter: <span style='font-size:1.6em;'>"
              f"{target}</span></b><br>{desc}")
    return 0, 0.0, status, f"Letter 1/{len(LEARN_LETTERS)}"


# ===================================================================
# Tab 3 — Practice
# ===================================================================

def _random_letter():
    return random.choice(LEARN_LETTERS)


def process_practice(frame: np.ndarray, target_letter: str, score: int,
                     attempts: int, streak: int, best_streak: int) -> tuple:
    """Stream callback for the Practice tab.

    Returns (annotated_frame, target_letter, score, attempts, streak,
             best_streak, score_text, streak_text, target_html, feedback_html).
    """
    score = int(score)
    attempts = int(attempts)
    streak = int(streak)
    best_streak = int(best_streak)

    score_text = f"{score}/{attempts}" if attempts > 0 else "0/0"
    streak_text = f"Current streak: {streak} | Best: {best_streak}"
    target_html = (f"<span style='font-size:2em; font-weight:bold;'>"
                   f"Sign: {target_letter}</span>")
    feedback = ""

    if frame is None:
        return (frame, target_letter, score, attempts, streak, best_streak,
                score_text, streak_text, target_html, feedback)

    annotated, letter, confidence = _process_frame(frame)
    if annotated is None:
        annotated = frame

    if classifier is None:
        feedback = "<span style='color:red;'>Model not loaded — run train_model.py</span>"
        return (annotated, target_letter, score, attempts, streak, best_streak,
                score_text, streak_text, target_html, feedback)

    if letter is not None and confidence is not None and confidence > 0.70:
        attempts += 1
        if letter == target_letter:
            score += 1
            streak += 1
            if streak > best_streak:
                best_streak = streak
            feedback = "<span style='color:green; font-weight:bold;'>Correct!</span>"
            target_letter = _random_letter()
            target_html = (f"<span style='font-size:2em; font-weight:bold;'>"
                           f"Sign: {target_letter}</span>")
        else:
            streak = 0
            feedback = (f"<span style='color:red;'>Detected <b>{letter}</b> — "
                        f"expected <b>{target_letter}</b></span>")

        score_text = f"{score}/{attempts}"
        streak_text = f"Current streak: {streak} | Best: {best_streak}"

    # Draw overlay
    if letter is not None and confidence is not None:
        label = f"{letter}  {confidence * 100:.0f}%"
        is_correct = letter == target_letter
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        _draw_text(annotated, label, (20, 60), font_scale=2.0,
                   color=color, thickness=4, bg_color=(0, 0, 0))
    else:
        _draw_text(annotated, "Show your hand to the camera",
                   (20, 50), font_scale=0.9, color=(0, 200, 255),
                   thickness=2, bg_color=(0, 0, 0))

    return (annotated, target_letter, score, attempts, streak, best_streak,
            score_text, streak_text, target_html, feedback)


def reset_practice():
    """Reset the Practice tab state."""
    letter = _random_letter()
    target_html = (f"<span style='font-size:2em; font-weight:bold;'>"
                   f"Sign: {letter}</span>")
    return (letter, 0, 0, 0, 0,
            "0/0",
            "Current streak: 0 | Best: 0",
            target_html,
            "")


# ===================================================================
# Gradio UI
# ===================================================================

with gr.Blocks(title="ASL Alphabet Teacher") as demo:
    gr.Markdown(
        "# ASL Alphabet Teacher\n"
        "Learn and practice American Sign Language letters (A, B, C) "
        "using your webcam and real-time hand detection."
    )

    # ---------------------------------------------------------------
    # Tab 1 — Free Sign
    # ---------------------------------------------------------------
    with gr.Tab("Free Sign"):
        gr.Markdown("Sign any letter and see what the model detects.")
        free_webcam = gr.Image(sources=["webcam"], streaming=True,
                               type="numpy", label="Webcam")

        free_webcam.stream(
            fn=process_free_sign,
            inputs=[free_webcam],
            outputs=[free_webcam],
        )

    # ---------------------------------------------------------------
    # Tab 2 — Learn
    # ---------------------------------------------------------------
    with gr.Tab("Learn"):
        gr.Markdown("Follow along to learn the ASL signs for A, B, and C.")

        # State variables
        learn_index = gr.State(value=0)
        learn_correct_time = gr.State(value=0.0)

        with gr.Row():
            with gr.Column(scale=2):
                learn_webcam = gr.Image(sources=["webcam"], streaming=True,
                                        type="numpy", label="Webcam")
            with gr.Column(scale=1):
                learn_status = gr.HTML(
                    value=(f"<b>Sign the letter: <span style='font-size:1.6em;'>"
                           f"A</span></b><br>{LETTER_DESCRIPTIONS['A']}")
                )
                learn_progress = gr.Textbox(value=f"Letter 1/{len(LEARN_LETTERS)}",
                                            label="Progress", interactive=False)
                learn_reset_btn = gr.Button("Reset", variant="secondary")

        learn_webcam.stream(
            fn=process_learn,
            inputs=[learn_webcam, learn_index, learn_correct_time],
            outputs=[learn_webcam, learn_index, learn_correct_time,
                     learn_status, learn_progress],
        )

        learn_reset_btn.click(
            fn=reset_learn,
            inputs=[],
            outputs=[learn_index, learn_correct_time,
                     learn_status, learn_progress],
        )

    # ---------------------------------------------------------------
    # Tab 3 — Practice
    # ---------------------------------------------------------------
    with gr.Tab("Practice"):
        gr.Markdown("Test yourself! Sign the letter shown and try to "
                     "build a streak.")

        # State variables
        practice_target = gr.State(value=_random_letter())
        practice_score = gr.State(value=0)
        practice_attempts = gr.State(value=0)
        practice_streak = gr.State(value=0)
        practice_best_streak = gr.State(value=0)

        with gr.Row():
            with gr.Column(scale=2):
                practice_webcam = gr.Image(sources=["webcam"], streaming=True,
                                           type="numpy", label="Webcam")
            with gr.Column(scale=1):
                practice_target_html = gr.HTML(
                    value="<span style='font-size:2em; font-weight:bold;'>"
                          "Sign: A</span>"
                )
                practice_feedback = gr.HTML(value="")
                practice_score_text = gr.Textbox(value="0/0", label="Score",
                                                 interactive=False)
                practice_streak_text = gr.Textbox(
                    value="Current streak: 0 | Best: 0",
                    label="Streak", interactive=False
                )
                practice_reset_btn = gr.Button("Reset Score",
                                               variant="secondary")

        practice_webcam.stream(
            fn=process_practice,
            inputs=[practice_webcam, practice_target, practice_score,
                    practice_attempts, practice_streak,
                    practice_best_streak],
            outputs=[practice_webcam, practice_target, practice_score,
                     practice_attempts, practice_streak,
                     practice_best_streak, practice_score_text,
                     practice_streak_text, practice_target_html,
                     practice_feedback],
        )

        practice_reset_btn.click(
            fn=reset_practice,
            inputs=[],
            outputs=[practice_target, practice_score, practice_attempts,
                     practice_streak, practice_best_streak,
                     practice_score_text, practice_streak_text,
                     practice_target_html, practice_feedback],
        )

    # ---------------------------------------------------------------
    # Footer tips
    # ---------------------------------------------------------------
    gr.Markdown(
        "---\n"
        "**Tips:** Face your palm toward the camera. "
        "Ensure good lighting. Hold hand at arm's length."
    )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
