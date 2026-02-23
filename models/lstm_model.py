import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT


class ForexLSTM(nn.Module):
    """Multi-layer LSTM for forex price prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class LSTMTrainer:
    """Handles training, inference, saving, and loading of ForexLSTM."""

    def __init__(self, model: ForexLSTM, lr: float = 0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32) -> list:
        """Train the LSTM. Returns list of epoch losses."""
        self.model.train()
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Returns numpy array of predictions."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X)
            return self.model(tensor).numpy().flatten()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
