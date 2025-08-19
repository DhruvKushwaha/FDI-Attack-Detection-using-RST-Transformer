import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_size: int,
        lstm_layers: int,
        output_size: int,  # now should be 2 for two output features
        seq_length: int
    ):
        """
        Args:
            in_channels: Number of input channels (e.g., number of features per time step).
            out_channels: Number of filters in the convolutional layer.
            kernel_size: Size of the 1D convolution filter.
            hidden_size: Number of hidden units in the LSTM.
            lstm_layers: Number of LSTM layers (stacked LSTMs).
            output_size: Dimension of the model output (set to 2 for two features).
            seq_length: The length of the input sequence.
        """
        super(CNNLSTM, self).__init__()
        
        # 1D Convolution: input shape (batch, in_channels, seq_length)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        
        # Optional: add a pooling layer if you want to reduce dimensionality
        # self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM: expects input shape (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Fully connected layer mapping hidden_size to output_size (here, 2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.seq_length = seq_length

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, seq_length)
        Returns:
            Tensor of shape (batch_size, output_size) where output_size is 2.
        """
        # Convolutional layer
        x = self.conv1(x)  # (batch, out_channels, L_out)
        x = F.relu(x)
        
        # Optional: apply pooling
        # x = self.pool(x)
        
        # Permute to match LSTM input dimensions: (batch, seq, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layer: outputs shape (batch, seq, hidden_size)
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Use the last time step output for prediction
        last_output = outputs[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layer to obtain final output
        out = self.fc(last_output)  # (batch, output_size) -> (batch, 2)
        return out