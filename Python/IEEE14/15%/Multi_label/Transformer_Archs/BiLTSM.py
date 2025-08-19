import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_labels: int, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,           # features per timestep
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                # expects (B, T, C)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected (B,T,C) or (B,C,T), got {x.shape}")
        x = x.float()
        in_size = self.lstm.input_size
        if x.size(-1) == in_size:           # (B,T,C)
            x_btc = x
        elif x.size(1) == in_size:          # (B,C,T) -> (B,T,C)
            x_btc = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Input {x.shape} doesn't match input_size={in_size}")
        out, (h_n, _) = self.lstm(x_btc)    # h_n: (2*num_layers, B, H)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        return self.fc(h)                   # (B, num_labels)
