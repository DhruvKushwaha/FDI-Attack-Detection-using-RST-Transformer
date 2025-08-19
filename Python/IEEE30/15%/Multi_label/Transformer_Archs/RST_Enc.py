import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reduced_sparsemax(z, alpha=1.0, dim=-1):
    """
    Computes the sparsemax function with scaling parameter alpha.
    
    The function solves
        p = argmin_p || alpha * p - z ||^2   subject to p >= 0 and sum(p) = 1,
    which is equivalent to projecting z/alpha onto the probability simplex.
    
    Args:
        z (torch.Tensor): Input tensor.
        alpha (float): Scaling parameter. Typically alpha > 1.
        dim (int): Dimension along which to apply the sparsemax.

    Returns:
        torch.Tensor: Output tensor after applying the sparsemax function.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    # Scale the input
    z_scaled = z / alpha

    # Sort z_scaled in descending order along the given dimension.
    z_sorted, _ = torch.sort(z_scaled, descending=True, dim=dim)

    # Compute the cumulative sum of the sorted z.
    z_cumsum = z_sorted.cumsum(dim)

    # Create a range tensor [1, 2, ..., d] with the same device and type as z.
    d = z.size(dim)
    # To broadcast properly, we create a shape that has ones in all dims except `dim`.
    range_shape = [1] * z.dim()
    range_shape[dim] = d
    range_tensor = torch.arange(1, d + 1, device=z.device, dtype=z.dtype).view(*range_shape)

    # Determine the support: find indices where
    #   z_sorted - (z_cumsum - 1) / range_tensor > 0
    support = z_sorted - (z_cumsum - 1) / range_tensor > 0

    # Compute k, the number of entries in the support.
    # (Keep the dimension for proper broadcasting later.)
    k = support.sum(dim=dim, keepdim=True)

    # Gather the cumulative sum at the index (k - 1) along the sorting dimension.
    # We subtract 1 because Python uses 0-based indexing.
    # Clamp k to be at least 1 to avoid issues (though there should always be at least one active entry).
    k_safe = torch.clamp(k, min=1)
    # Prepare indices for gathering.
    # We need to convert k_safe to long and subtract one.
    indices = (k_safe - 1).long()
    # Gather the cumulative sum at these indices.
    tau_sum = z_cumsum.gather(dim, indices)

    # Compute tau (note: k must be cast to the same type as z).
    tau = (tau_sum - 1) / k_safe.to(z.dtype)

    # Subtract tau and threshold at zero.
    output = torch.clamp(z_scaled - tau, min=0)
    
    return output


class SparsemaxAttention(nn.Module):
    """
    Multi-head attention layer that uses reduced_sparsemax instead of softmax.

    Assumes inputs with shape (batch, seq_len, embed_dim) (i.e. batch_first=True).
    """
    def __init__(self, d_model, n_heads, alpha):
        super(SparsemaxAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.alpha = alpha
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for reduced_sparsemax multi-head attention.

        Args:
            query (Tensor): (batch, seq_len, embed_dim)
            key (Tensor): (batch, seq_len, embed_dim)
            value (Tensor): (batch, seq_len, embed_dim)
            mask (Tensor, optional): Mask to apply on attention scores.
            
        Returns:
            output (Tensor): (batch, seq_len, embed_dim)
            attn (Tensor): Attention weights.
        """
                
        # Get the batch size
        batch_size = query.size(1)
        
        # Linear transformation of Q, K, and V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape Q, K, and V to have the same shape as the number of heads
        Q = Q.view(Q.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(K.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(V.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        
             
        scores = torch.matmul(Q, K.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.head_dim
                                , dtype=torch.float32, device=Q.device))
        
        # This is where the mask is applied
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Get the attention weights
        attn_output_weights = reduced_sparsemax(scores, self.alpha ,dim=-1)
        attn_output = torch.matmul(attn_output_weights, V)
        
        # print(f"Attention output shape: {attn_output.shape}")
        
        # Get the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output).transpose(0, 1)
    
class SparsemaxEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, alpha ,dim_feedforward, dropout=0.1):
        super(SparsemaxEncoderLayer, self).__init__()
        self.self_attn = SparsemaxAttention(d_model, n_heads, alpha=alpha)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_output = self.self_attn(src, src, src, mask=None) #self attention
        src = src + self.dropout(attn_output) #add and norm
        src = self.norm1(src) #norm
        ff_output = self.feed_forward(src) #feed forward
        src = src + self.dropout(ff_output) #add and norm
        src = self.norm2(src) #norm
        return src
    
class SparsemaxEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward, alpha, dropout=0.1):
        super(SparsemaxEncoder, self).__init__()
        self.layers = nn.ModuleList([SparsemaxEncoderLayer(d_model, n_heads, alpha, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeseriesTransformer(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, d_model, nhead, num_encoder_layers, dim_feedforward, alpha):
        super(TimeseriesTransformer, self).__init__()
        self.src_embed = nn.Linear(input_feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = SparsemaxEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, alpha)
        self.output_layer = nn.Linear(d_model, output_feature_size)
        self.alpha = alpha

    def forward(self, src):
        src = self.src_embed(src) # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src) # (seq_len, batch_size, d_model)
        src = self.encoder(src) # (seq_len, batch_size, d_model)
        src = src.mean(dim=0) # (batch_size, d_model)
        output = self.output_layer(src) # (seq_len, batch_size, output_feature_size)
        #output = F.softmax(output, dim=-1) #Sparsemax change
        return output