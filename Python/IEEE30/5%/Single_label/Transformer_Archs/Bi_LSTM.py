import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Accept (B, T, C) or (B, C, T); ensure we send (B, T, C) to LSTM
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, C) or (B, C, T), got {x.shape}")

        in_size = self.lstm.input_size  # 86 for your case
        if x.size(-1) == in_size:
            # already (B, T, C) -> OK
            x_btc = x
        elif x.size(1) == in_size:
            # (B, C, T) -> (B, T, C)
            x_btc = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Input {x.shape} doesn't match BiLSTM input_size={in_size}")

        out, _ = self.lstm(x_btc)            # (B, T, 2H)
        last = out[:, -1, :]                 # (B, 2H)
        return self.fc(last)                 # (B, output_size)
