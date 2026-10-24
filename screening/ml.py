"""Learned tamper classifier on top of the forensic feature vector.

The shipped model is trained on SYNTHETIC cards (see train_model.py) so it is
a working demonstration of the approach. For production, retrain it on your
own labelled genuine / fraudulent submissions.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

FEATURES = ["ela_mean", "ela_outlier_ratio", "ela_max_z", "noise_cv", "noise_outlier_ratio",
            "copy_move_matches", "copy_move_consistent", "exif_present", "edited_software"]
# blur_var is deliberately excluded: on synthetic data it acts as a shortcut
# (tamper generators change global sharpness) and would not transfer to real scans.

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tamper_model.joblib")


def vectorise(feats: Dict[str, float]) -> np.ndarray:
    return np.array([[float(feats.get(k, 0.0)) for k in FEATURES]], dtype=np.float64)


class TamperModel:
    def __init__(self, path: str = MODEL_PATH):
        self.model = None
        if os.path.exists(path):
            try:
                import joblib
                self.model = joblib.load(path)
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, feats: Dict[str, float]) -> Optional[float]:
        if not self.available:
            return None
        return float(self.model.predict_proba(vectorise(feats))[0, 1])
