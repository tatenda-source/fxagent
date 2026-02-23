import os

import numpy as np

from agents.base_agent import BaseAgent
from models.lstm_model import ForexLSTM, LSTMTrainer
from models.model_utils import prepare_sequences, inverse_scale_close
from data.storage import Storage
from config import MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE


class PredictionAgent(BaseAgent):
    """Agent 3: Trains/loads LSTM models and predicts next close price per pair."""

    def __init__(self):
        super().__init__(name="PredictionAgent")
        self.storage = Storage()
        os.makedirs(MODEL_DIR, exist_ok=True)

    def run(self, input_data: dict) -> dict:
        analyzed_data = input_data["analyzed_data"]
        predictions = {}

        for pair, df in analyzed_data.items():
            try:
                pred = self._predict_pair(pair, df)
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

        model_path = os.path.join(MODEL_DIR, f"{pair.replace('=', '_')}_lstm.pt")

        if os.path.exists(model_path):
            trainer.load(model_path)
            self.logger.info(f"Loaded existing model for {pair}")
        else:
            self.logger.info(f"Training new model for {pair}...")
            losses = trainer.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
            trainer.save(model_path)
            self.logger.info(f"Model saved. Final loss: {losses[-1]:.6f}")

        # Predict using the last available sequence
        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]
        pred_scaled = trainer.predict(last_sequence)[0]

        predicted_price = inverse_scale_close(pred_scaled, scaler, num_features)
        current_price = float(df["Close"].iloc[-1])

        direction = "UP" if predicted_price > current_price else "DOWN"
        change_pct = abs(predicted_price - current_price) / current_price
        confidence = min(0.95, change_pct * 20)

        self.logger.info(
            f"{pair}: current={current_price:.5f}, predicted={predicted_price:.5f}, "
            f"direction={direction}, confidence={confidence:.2f}"
        )

        self.storage.save_prediction({
            "pair": pair,
            "predicted_price": predicted_price,
            "prediction_horizon": "1d",
            "model_version": "lstm_v1",
        })

        return {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "direction": direction,
            "confidence": confidence,
            "change_pct": change_pct,
        }
