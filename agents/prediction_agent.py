import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from agents.base_agent import BaseAgent
from models.lstm_model import ForexLSTM, LSTMTrainer
from models.model_utils import prepare_sequences, inverse_scale_close
from data.storage import Storage
from config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_VERSION


class PredictionAgent(BaseAgent):
    """Agent 3: Trains/loads attention-LSTM models and predicts next close price.
    Uses MC Dropout for uncertainty-based confidence scoring."""

    def __init__(self):
        super().__init__(name="PredictionAgent")
        self.storage = Storage()
        os.makedirs(MODEL_DIR, exist_ok=True)

    def run(self, input_data: dict) -> dict:
        analyzed_data = input_data["analyzed_data"]
        predictions = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._predict_pair, pair, df): pair
                for pair, df in analyzed_data.items()
            }
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    pred = future.result()
                    if pred:
                        predictions[pair] = pred
                except Exception as e:
                    self.logger.error(f"Prediction failed for {pair}: {e}")

        return {"predictions": predictions}

    def _predict_pair(self, pair: str, df) -> dict:
        X_train, X_test, y_train, y_test, scaler = prepare_sequences(df)

        if len(X_train) < 10:
            self.logger.warning(f"Not enough data to train for {pair}")
            return None

        num_features = X_train.shape[2]
        model = ForexLSTM(input_size=num_features)
        trainer = LSTMTrainer(model, lr=LEARNING_RATE)

        model_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}_lstm_v2.pt")

        if os.path.exists(model_path):
            try:
                trainer.load(model_path)
                self.logger.info(f"Loaded v2 model for {pair}")
            except Exception as e:
                self.logger.warning(f"Failed to load model for {pair}: {e}. Retraining...")
                os.remove(model_path)
                self._train_and_save(trainer, X_train, y_train, model_path, pair)
        else:
            self._train_and_save(trainer, X_train, y_train, model_path, pair)

        # Predict using the last available sequence
        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]

        # MC Dropout for uncertainty estimation
        mean_pred, std_pred = trainer.predict_with_uncertainty(last_sequence)
        pred_scaled = mean_pred[0]
        pred_uncertainty = std_pred[0]

        predicted_price = inverse_scale_close(pred_scaled, scaler, num_features)
        current_price = float(df["Close"].iloc[-1])

        direction = "UP" if predicted_price > current_price else "DOWN"
        change_pct = abs(predicted_price - current_price) / current_price
        confidence = self._compute_confidence(change_pct, pred_uncertainty)

        self.logger.info(
            f"{pair}: current={current_price:.5f}, predicted={predicted_price:.5f}, "
            f"direction={direction}, confidence={confidence:.2f}, "
            f"uncertainty={pred_uncertainty:.6f}"
        )

        self.storage.save_prediction({
            "pair": pair,
            "predicted_price": predicted_price,
            "prediction_horizon": "1d",
            "model_version": MODEL_VERSION,
        })

        return {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "direction": direction,
            "confidence": confidence,
            "change_pct": change_pct,
            "uncertainty": pred_uncertainty,
        }

    def _train_and_save(self, trainer, X_train, y_train, model_path, pair):
        self.logger.info(f"Training v2 model for {pair}...")
        result = trainer.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        trainer.save(model_path)
        final_train = result["train_losses"][-1]
        final_val = result["val_losses"][-1] if result["val_losses"] else "N/A"
        self.logger.info(
            f"Model saved for {pair}. "
            f"Stopped epoch: {result['stopped_epoch']}, "
            f"Train loss: {final_train:.6f}, Val loss: {final_val}"
        )

    @staticmethod
    def _compute_confidence(change_pct: float, uncertainty: float) -> float:
        """Compute confidence from price change magnitude and MC dropout uncertainty.

        Low uncertainty + large change = high confidence.
        High uncertainty = low confidence regardless of change."""
        signal_strength = min(1.0, change_pct * 15)
        uncertainty_penalty = min(1.0, uncertainty / 0.03)
        confidence = signal_strength * (1.0 - uncertainty_penalty)
        return max(0.05, min(0.95, confidence))
