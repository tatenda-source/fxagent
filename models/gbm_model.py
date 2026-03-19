import numpy as np
import lightgbm as lgb


class ForexGBM:
    """LightGBM model for forex log-return prediction.

    Captures non-linear feature interactions that LSTM may miss.
    Uses the last row of each sequence as a flat feature vector
    plus engineered lag features."""

    def __init__(self, params: dict = None):
        self.params = params or {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model = None
        self.num_rounds = 200

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert 3D LSTM sequences to 2D GBM features.

        Takes the last row (current features) plus statistical
        summaries of the sequence window."""
        batch, seq_len, n_feat = X.shape

        # Current features (last timestep)
        current = X[:, -1, :]  # (batch, n_feat)

        # Rolling statistics over the sequence window
        seq_mean = X.mean(axis=1)     # (batch, n_feat)
        seq_std = X.std(axis=1)       # (batch, n_feat)
        seq_min = X.min(axis=1)       # (batch, n_feat)
        seq_max = X.max(axis=1)       # (batch, n_feat)

        # Rate of change: last vs first in window
        seq_delta = X[:, -1, :] - X[:, 0, :]  # (batch, n_feat)

        # Recent momentum: last 5 vs previous 5
        if seq_len >= 10:
            recent = X[:, -5:, :].mean(axis=1)
            older = X[:, -10:-5, :].mean(axis=1)
            momentum = recent - older
        else:
            momentum = np.zeros_like(current)

        return np.hstack([current, seq_mean, seq_std, seq_min, seq_max, seq_delta, momentum])

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Train LightGBM on flattened sequence features."""
        X_flat = self._flatten_sequences(X_train)

        train_data = lgb.Dataset(X_flat, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and len(X_val) > 0:
            X_val_flat = self._flatten_sequences(X_val)
            val_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Direction accuracy on validation
        dir_acc = 0.0
        if X_val is not None and len(X_val) > 0:
            preds = self.model.predict(X_val_flat)
            dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))

        return {"direction_accuracy": dir_acc, "best_iteration": self.model.best_iteration}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log returns from sequences."""
        X_flat = self._flatten_sequences(X)
        return self.model.predict(X_flat)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
