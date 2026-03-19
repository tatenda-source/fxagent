import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_ATTENTION_HEADS,
    GRADIENT_CLIP_MAX_NORM, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    VALIDATION_SPLIT, MC_DROPOUT_PASSES,
)


class DirectionalMSELoss(nn.Module):
    """MSE loss with a directional penalty.

    Penalizes predictions that get the sign (direction) wrong more heavily
    than those that get magnitude wrong. For log-return targets, the sign
    directly maps to UP/DOWN — the thing we actually care about."""

    def __init__(self, direction_weight: float = 0.3):
        super().__init__()
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        # Direction penalty: 1 when signs disagree, 0 when they agree
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        direction_wrong = (pred_sign != target_sign).float()

        # Weight the penalty by how far off the magnitude is
        direction_penalty = (direction_wrong * (pred - target).abs()).mean()

        return mse + self.direction_weight * direction_penalty


class TemporalAttention(nn.Module):
    """Multi-head temporal attention over LSTM outputs.

    Learns which timesteps are most informative, replacing the naive
    'take last hidden state' approach. O(n) per head."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.score_layers = nn.ModuleList([
            nn.Linear(self.head_dim, 1) for _ in range(num_heads)
        ])

    def forward(self, lstm_output):
        batch, seq_len, hidden = lstm_output.shape
        multi_head = lstm_output.view(batch, seq_len, self.num_heads, self.head_dim)

        contexts = []
        for h in range(self.num_heads):
            head_data = multi_head[:, :, h, :]
            scores = self.score_layers[h](head_data).squeeze(-1)
            weights = F.softmax(scores, dim=-1)
            context_h = torch.bmm(weights.unsqueeze(1), head_data).squeeze(1)
            contexts.append(context_h)

        return torch.cat(contexts, dim=-1)


class ForexLSTM(nn.Module):
    """Multi-layer LSTM with temporal attention for forex return prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_attention_heads: int = LSTM_ATTENTION_HEADS,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size, num_attention_heads)
        self.mc_dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.mc_dropout(context)
        return self.fc(context)


class LSTMTrainer:
    """Handles training with early stopping, gradient clipping, LR scheduling,
    directional loss, and MC dropout inference for uncertainty estimation."""

    def __init__(self, model: ForexLSTM, lr: float = 0.001, direction_weight: float = 0.3):
        self.model = model
        self.criterion = DirectionalMSELoss(direction_weight=direction_weight)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR,
            min_lr=1e-6,
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state_dict = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> dict:
        """Train with validation, early stopping, gradient clipping, and LR scheduling.

        Returns dict with train_losses, val_losses, stopped_epoch, direction_accuracy."""
        # Auto-split validation if not provided
        if X_val is None:
            val_size = int(len(X_train) * VALIDATION_SPLIT)
            if val_size > 0:
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = None
        if X_val is not None and len(X_val) > 0:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)),
                batch_size=batch_size, shuffle=False,
            )

        train_losses, val_losses = [], []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state_dict = None
        stopped_epoch = epochs
        best_dir_acc = 0.0

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM,
                )
                self.optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_pred, all_true = [], []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        pred = self.model(X_batch)
                        val_loss += self.criterion(pred, y_batch).item()
                        all_pred.append(pred.numpy().flatten())
                        all_true.append(y_batch.numpy().flatten())
                avg_val = val_loss / len(val_loader)
                val_losses.append(avg_val)

                # Track directional accuracy
                all_pred = np.concatenate(all_pred)
                all_true = np.concatenate(all_true)
                dir_acc = np.mean(np.sign(all_pred) == np.sign(all_true))

                self.scheduler.step(avg_val)

                # Early stopping
                if avg_val < self.best_val_loss - EARLY_STOPPING_MIN_DELTA:
                    self.best_val_loss = avg_val
                    self.patience_counter = 0
                    best_dir_acc = dir_acc
                    self.best_state_dict = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                        stopped_epoch = epoch + 1
                        break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "stopped_epoch": stopped_epoch,
            "direction_accuracy": best_dir_acc,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standard inference."""
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy().flatten()

    def predict_with_uncertainty(self, X: np.ndarray, n_passes: int = MC_DROPOUT_PASSES) -> tuple:
        """MC Dropout inference for uncertainty estimation.

        Returns (mean_prediction, std_uncertainty) as numpy arrays."""
        self.model.train()  # Keep dropout active
        tensor = torch.FloatTensor(X)
        predictions = []
        for _ in range(n_passes):
            with torch.no_grad():
                predictions.append(self.model(tensor).numpy().flatten())

        predictions = np.array(predictions)
        self.model.eval()
        return predictions.mean(axis=0), predictions.std(axis=0)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
