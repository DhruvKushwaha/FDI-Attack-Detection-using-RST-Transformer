import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Args:
            input_size (int): Number of features per time step in the input.
            hidden_size (int): Number of hidden units in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Dimension of the model output.
        """
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define a bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Enable bidirectionality
        )
        
        # Because the LSTM is bidirectional, the effective hidden size becomes hidden_size*2.
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            out (Tensor): Output predictions of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states for both directions (total layers = num_layers*2)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate the LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))
        
        # For a many-to-one model, we use the output of the last time step
        last_time_step = out[:, -1, :]  # shape: (batch_size, hidden_size*2)
        
        # Pass through the fully connected layer to get final predictions
        out = self.fc(last_time_step)  # shape: (batch_size, output_size)
        return out