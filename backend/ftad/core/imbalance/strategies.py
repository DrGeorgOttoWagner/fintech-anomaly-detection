"""Class-imbalance treatments.

Samplers are inserted into an imbalanced-learn ``Pipeline`` *after* preprocessing, so
they run only during ``fit`` — i.e. on the training partition (or on training folds
inside cross-validation). Validation and test data are never resampled and keep the
original class prevalence.
"""

from __future__ import annotations

IMBALANCE_STRATEGIES = {
    "none": {
        "name": "No adjustment",
        "description": "Train on the observed class distribution.",
    },
    "class_weight": {
        "name": "Class weighting",
        "description": "Re-weights the loss so each class contributes equally; no records are added or removed.",
    },
    "undersample": {
        "name": "Random undersampling",
        "description": "Randomly discards majority-class training records. Fast, but throws away genuine information.",
    },
    "oversample": {
        "name": "Random oversampling",
        "description": "Duplicates minority-class training records. Keeps all data but can encourage overfitting to repeated cases.",
    },
    "smote": {
        "name": "SMOTE",
        "description": "Creates synthetic minority records by interpolating between neighbouring minority cases in feature space. "
        "Synthetic records are artificial and are never reported as observed cases.",
    },
}


def make_sampler(strategy: str, ratio: float, seed: int, n_minority: int):
    """Return an imblearn sampler or ``None``. ``ratio`` = minority/majority after resampling."""
    if strategy in ("none", "class_weight"):
        return None
    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    if strategy == "oversample":
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(sampling_strategy=ratio, random_state=seed)
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        if n_minority < 2:
            raise ValueError("SMOTE needs at least two minority records in the training data.")
        return SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=min(5, n_minority - 1))
    raise ValueError(f"Unknown imbalance strategy '{strategy}'.")
