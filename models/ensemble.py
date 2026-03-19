import numpy as np

from models.lstm_model import ForexLSTM, LSTMTrainer
from models.gbm_model import ForexGBM
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT


class EnsemblePredictor:
    """Ensemble combining LSTM (temporal patterns) + LightGBM (feature interactions).

    The final prediction is a weighted average. Weights are determined by
    each model's directional accuracy on validation data."""

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.lstm_model = ForexLSTM(input_size=num_features)
        self.lstm_trainer = LSTMTrainer(self.lstm_model, lr=LEARNING_RATE)
        self.gbm_model = ForexGBM()

        # Default equal weights — updated after training
        self.lstm_weight = 0.5
        self.gbm_weight = 0.5

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Train both models and calibrate ensemble weights from directional accuracy."""
        val_size = int(len(X_train) * VALIDATION_SPLIT)
        X_val = X_train[-val_size:] if val_size > 0 else None
        y_val = y_train[-val_size:] if val_size > 0 else None
        X_tr = X_train[:-val_size] if val_size > 0 else X_train
        y_tr = y_train[:-val_size] if val_size > 0 else y_train

        # Train LSTM
        lstm_result = self.lstm_trainer.train(
            X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
            X_val=X_val, y_val=y_val,
        )

        # Train GBM
        gbm_result = self.gbm_model.train(X_tr, y_tr, X_val, y_val)

        # Calibrate weights based on directional accuracy
        lstm_dir = lstm_result.get("direction_accuracy", 0.5)
        gbm_dir = gbm_result.get("direction_accuracy", 0.5)

        # Softmax-style weighting to avoid zero weights
        total = lstm_dir + gbm_dir
        if total > 0:
            self.lstm_weight = lstm_dir / total
            self.gbm_weight = gbm_dir / total
        else:
            self.lstm_weight = 0.5
            self.gbm_weight = 0.5

        return {
            "lstm_result": lstm_result,
            "gbm_result": gbm_result,
            "lstm_weight": self.lstm_weight,
            "gbm_weight": self.gbm_weight,
            "lstm_dir_acc": lstm_dir,
            "gbm_dir_acc": gbm_dir,
        }

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """Ensemble prediction with uncertainty from both model disagreement and MC dropout.

        Returns (mean_return, uncertainty) as numpy arrays."""
        # LSTM: MC dropout gives mean + std
        lstm_mean, lstm_std = self.lstm_trainer.predict_with_uncertainty(X)

        # GBM: single deterministic prediction
        gbm_pred = self.gbm_model.predict(X)

        # Weighted ensemble mean
        ensemble_mean = self.lstm_weight * lstm_mean + self.gbm_weight * gbm_pred

        # Uncertainty: combine MC dropout std + model disagreement
        model_disagreement = np.abs(lstm_mean - gbm_pred)
        ensemble_uncertainty = (
            self.lstm_weight * lstm_std +
            self.gbm_weight * model_disagreement * 0.5
        )

        return ensemble_mean, ensemble_uncertainty

    def predict_direction_confidence(self, X: np.ndarray) -> tuple:
        """Get direction and confidence from the ensemble.

        Returns (predicted_return, direction, confidence, uncertainty)."""
        mean_return, uncertainty = self.predict_with_uncertainty(X)

        pred_return = float(mean_return[0])
        pred_uncertainty = float(uncertainty[0])

        direction = "UP" if pred_return > 0 else "DOWN"

        # Confidence based on:
        # 1. Agreement between models (both say same direction)
        lstm_mean, _ = self.lstm_trainer.predict_with_uncertainty(X)
        gbm_pred = self.gbm_model.predict(X)

        lstm_dir = "UP" if float(lstm_mean[0]) > 0 else "DOWN"
        gbm_dir = "UP" if float(gbm_pred[0]) > 0 else "DOWN"
        models_agree = lstm_dir == gbm_dir

        # 2. Signal magnitude relative to uncertainty (signal-to-noise ratio)
        abs_return = abs(pred_return)
        snr = abs_return / (pred_uncertainty + 1e-8)

        # Confidence formula:
        # - Base: SNR scaled (0-1), higher SNR = more confident
        # - Bonus: +0.15 if both models agree on direction
        # - Floor: 0.10, Cap: 0.95
        base_conf = min(1.0, snr / 3.0)  # SNR of 3 = full confidence
        agreement_bonus = 0.15 if models_agree else 0.0
        confidence = base_conf + agreement_bonus

        return pred_return, direction, max(0.10, min(0.95, confidence)), pred_uncertainty

    def save(self, lstm_path: str, gbm_path: str):
        self.lstm_trainer.save(lstm_path)
        self.gbm_model.save(gbm_path)

    def load(self, lstm_path: str, gbm_path: str):
        self.lstm_trainer.load(lstm_path)
        self.gbm_model.load(gbm_path)
