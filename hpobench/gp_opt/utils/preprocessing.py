import random
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def train_val_split(
    X: np.array,
    y: np.array,
    train_split: float,
    normalize: bool = True,
    ordinal: bool = False,
    random_state: int = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Split data into training and validation sets with optional normalization.

    Supports both random and sequential splitting, with the option to normalize
    features based on the training set statistics.

    Args:
        X: Feature variables with shape (n_samples, n_features).
        y: Target variable with shape (n_samples,).
        train_split: Fraction of data to use for training, must be in [0, 1].
        normalize: Whether to normalize features using training set statistics.
        ordinal: Whether to split sequentially (for time-ordered data) or randomly.
        random_state: Random seed for reproducible splitting.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val) where:
        - X_train: Training features
        - y_train: Training targets
        - X_val: Validation features
        - y_val: Validation targets
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if ordinal:
        train_idx = list(range(len(X) - round(len(X) * (1 - train_split))))
        val_idx = list(range(len(X) - round(len(X) * (1 - train_split)), len(X)))
    else:
        train_idx = list(
            np.random.choice(len(X), round(len(X) * train_split), replace=False)
        )
        val_idx = list(np.setdiff1d(np.arange(len(X)), train_idx))

    X_val = X[val_idx, :]
    X_train = X[train_idx, :]

    y_val = y[val_idx]
    y_train = y[train_idx]

    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val
