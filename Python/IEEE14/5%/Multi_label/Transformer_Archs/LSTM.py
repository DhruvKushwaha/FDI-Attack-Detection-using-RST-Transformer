import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels,          # number of features per timestep
        out_features,         # number of classes
        hidden_size=64,
        lstm_layers=2,
        bidirectional=False,
        dropout=0.1,
        reduce="last",        # "last" | "mean" | "max"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduce = reduce
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), out_features),
        )

    def _to_BLC(self, x):
        # Accept (B, L, C) or (B, C, L). LSTM expects (B, L, C)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        # If second dim equals feature size, it's (B, C, L) -> transpose
        if x.shape[1] == self.in_channels and x.shape[2] != self.in_channels:
            x = x.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        return x

    def forward(self, x):
        x = self._to_BLC(x)     # (B, L, C)
        x, _ = self.lstm(x)     # (B, L, H[*2])
        if self.reduce == "last":
            x = x[:, -1]
        elif self.reduce == "mean":
            x = x.mean(dim=1)
        elif self.reduce == "max":
            x, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")
        return self.head(x)     # (B, out_features)
