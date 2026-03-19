"""ASL hand sign classifier using a trained scikit-learn model."""

from __future__ import annotations

import os

import joblib
import numpy as np


class ASLClassifier:
    """Loads a trained scikit-learn model and predicts ASL letters."""

    def __init__(self, model_path: str):
        """Load a joblib-serialized scikit-learn model from *model_path*.

        Raises
        ------
        FileNotFoundError
            If *model_path* does not point to an existing file.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please provide a valid path to a joblib-serialized scikit-learn model."
            )
        self._model = joblib.load(model_path)

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict the ASL letter for a normalised landmark vector.

        Parameters
        ----------
        features:
            numpy array of shape ``(63,)`` — the normalised landmark vector
            produced by ``preprocessing.normalize_landmarks``.

        Returns
        -------
        tuple[str, float]
            ``(letter, confidence)`` where *letter* is a string such as
            ``"A"`` and *confidence* is the maximum predicted probability
            (float between 0 and 1).
        """
        probabilities = self._model.predict_proba(features.reshape(1, -1))[0]
        best_index = int(np.argmax(probabilities))
        letter = self._model.classes_[best_index]
        confidence = float(probabilities[best_index])
        return letter, confidence

    @property
    def labels(self) -> list[str]:
        """Return the list of class labels the model was trained on."""
        return self._model.classes_.tolist()
