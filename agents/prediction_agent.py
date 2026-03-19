import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from agents.base_agent import BaseAgent
from models.ensemble import EnsemblePredictor
from models.model_utils import prepare_sequences, log_return_to_price
from data.storage import Storage
from config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_VERSION


class PredictionAgent(BaseAgent):
    """Agent 3: Trains/loads LSTM+GBM ensemble and predicts next-day log returns.

    V3 improvements:
    - Predicts log returns (stationary) instead of raw price
    - Ensemble: LSTM (temporal) + LightGBM (feature interactions)
    - Directional loss penalizes wrong-direction predictions
    - Confidence from SNR + model agreement (not broken threshold)
    """

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
        result = prepare_sequences(df)
        X_train, X_test, y_train, y_test, scaler, meta = result

        if len(X_train) < 10:
            self.logger.warning(f"Not enough data to train for {pair}")
            return None

        num_features = X_train.shape[2]
        ensemble = EnsemblePredictor(num_features)

        lstm_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}_lstm_v3.pt")
        gbm_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}_gbm_v3.txt")

        if os.path.exists(lstm_path) and os.path.exists(gbm_path):
            try:
                ensemble.load(lstm_path, gbm_path)
                self.logger.info(f"Loaded v3 ensemble for {pair}")
            except Exception as e:
                self.logger.warning(f"Failed to load model for {pair}: {e}. Retraining...")
                self._safe_remove(lstm_path)
                self._safe_remove(gbm_path)
                self._train_and_save(ensemble, X_train, y_train, lstm_path, gbm_path, pair)
        else:
            self._train_and_save(ensemble, X_train, y_train, lstm_path, gbm_path, pair)

        # Predict using the last available sequence
        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]

        pred_return, direction, confidence, uncertainty = ensemble.predict_direction_confidence(
            last_sequence
        )

        current_price = meta["last_close"]
        predicted_price = log_return_to_price(pred_return, current_price)
        change_pct = abs(predicted_price - current_price) / current_price

        self.logger.info(
            f"{pair}: current={current_price:.5f}, predicted={predicted_price:.5f}, "
            f"return={pred_return:.6f}, direction={direction}, "
            f"confidence={confidence:.2f}, uncertainty={uncertainty:.6f}"
        )

        self.storage.save_prediction({
            "pair": pair,
            "predicted_price": predicted_price,
            "prediction_horizon": "1d",
            "model_version": "ensemble_v3",
        })

        return {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "direction": direction,
            "confidence": confidence,
            "change_pct": change_pct,
            "predicted_return": pred_return,
            "uncertainty": uncertainty,
        }

    def _train_and_save(self, ensemble, X_train, y_train, lstm_path, gbm_path, pair):
        self.logger.info(f"Training v3 ensemble for {pair}...")
        result = ensemble.train(X_train, y_train)

        ensemble.save(lstm_path, gbm_path)

        lstm_r = result["lstm_result"]
        final_train = lstm_r["train_losses"][-1]
        final_val = lstm_r["val_losses"][-1] if lstm_r["val_losses"] else "N/A"

        self.logger.info(
            f"Ensemble saved for {pair}. "
            f"LSTM dir_acc: {result['lstm_dir_acc']:.1%}, "
            f"GBM dir_acc: {result['gbm_dir_acc']:.1%}, "
            f"weights: LSTM={result['lstm_weight']:.2f} GBM={result['gbm_weight']:.2f}, "
            f"stopped_epoch: {lstm_r['stopped_epoch']}, "
            f"train_loss: {final_train:.6f}, val_loss: {final_val}"
        )

    @staticmethod
    def _safe_remove(path: str):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
