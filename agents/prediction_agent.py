import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from agents.base_agent import BaseAgent
from models.ensemble import EnsemblePredictor
from models.model_utils import prepare_sequences, log_return_to_price
from data.storage import Storage
from config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_VERSION

MAX_PREDICTION_RETURN = 0.10
MODEL_STALENESS_DAYS = 7


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

        models_exist = os.path.exists(lstm_path) and os.path.exists(gbm_path)

        if models_exist and self._model_is_stale(lstm_path):
            self.logger.warning(f"Model for {pair} is >{MODEL_STALENESS_DAYS} days old, forcing retrain")
            models_exist = False

        if models_exist:
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

        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]

        pred_return, direction, confidence, uncertainty = self._predict_with_fallback(
            ensemble, last_sequence, pair
        )
        if pred_return is None:
            return None

        if not np.isfinite(pred_return) or not np.isfinite(uncertainty):
            self.logger.error(f"{pair}: NaN/inf in prediction output (return={pred_return}, uncertainty={uncertainty})")
            return None

        if abs(pred_return) > MAX_PREDICTION_RETURN:
            self.logger.warning(f"{pair}: predicted return {pred_return:.6f} exceeds sanity bound of {MAX_PREDICTION_RETURN}, rejecting")
            return None

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

    def _predict_with_fallback(self, ensemble, X, pair):
        try:
            return ensemble.predict_direction_confidence(X)
        except Exception as e:
            self.logger.warning(f"{pair}: ensemble prediction failed ({e}), trying LSTM-only")

        try:
            lstm_mean, lstm_std = ensemble.lstm_trainer.predict_with_uncertainty(X)
            pred_return = float(lstm_mean[0])
            direction = "UP" if pred_return > 0 else "DOWN"
            return pred_return, direction, 0.30, float(lstm_std[0])
        except Exception as e:
            self.logger.warning(f"{pair}: LSTM-only prediction failed ({e}), trying GBM-only")

        try:
            gbm_pred = ensemble.gbm_model.predict(X)
            pred_return = float(gbm_pred[0])
            direction = "UP" if pred_return > 0 else "DOWN"
            return pred_return, direction, 0.20, 0.01
        except Exception as e:
            self.logger.error(f"{pair}: all prediction methods failed ({e})")
            return None, None, None, None

    @staticmethod
    def _model_is_stale(path: str) -> bool:
        try:
            age_seconds = time.time() - os.path.getmtime(path)
            return age_seconds > MODEL_STALENESS_DAYS * 86400
        except OSError:
            return True

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
