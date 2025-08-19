import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SoftmaxAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        #self.sparsemax = Sparsemax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # print(f"Query shape: {query.shape}")
        # print(f"Key shape: {key.shape}")
        # print(f"Value shape: {value.shape}")
        
        # Get the batch size
        batch_size = query.size(1)
        
        # Linear transformation of Q, K, and V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # print(f"Q shape after linear: {Q.shape}")
        # print(f"K shape after linear: {K.shape}")
        # print(f"V shape after linear: {V.shape}")
        
        # Reshape Q, K, and V to have the same shape as the number of heads
        Q = Q.view(Q.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(K.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(V.size(0), batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        
        # print(f"Q shape after reshape: {Q.shape}")
        # print(f"K shape after reshape: {K.shape}")
        # print(f"V shape after reshape: {V.shape}")

        # Adjust K and V if their sequence length doesn't match Q
        # if K.size(2) != Q.size(2):
        #     K = K[:, :, -Q.size(2):]
        #     V = V[:, :, -Q.size(2):]
        
        scores = torch.matmul(Q, K.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.head_dim
                                , dtype=torch.float32, device=Q.device))
        
        # This is where the mask is applied
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Get the attention weights
        attn_output_weights = F.softmax(scores, dim=1)
        attn_output = torch.matmul(attn_output_weights, V)
        
        # print(f"Attention output shape: {attn_output.shape}")
        
        # Get the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output).transpose(0, 1)
    
class SoftmaxEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(SoftmaxEncoderLayer, self).__init__()
        self.self_attn = SoftmaxAttention(d_model, n_heads)
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
    
class SoftmaxEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1):
        super(SoftmaxEncoder, self).__init__()
        self.layers = nn.ModuleList([SoftmaxEncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)])

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
    def __init__(self, input_feature_size, output_feature_size, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TimeseriesTransformer, self).__init__()
        self.src_embed = nn.Linear(input_feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = SoftmaxEncoder(d_model, nhead, num_encoder_layers, dim_feedforward)
        self.output_layer = nn.Linear(d_model, output_feature_size)

    def forward(self, src):
        src = self.src_embed(src) # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src) # (seq_len, batch_size, d_model)
        src = self.encoder(src) # (seq_len, batch_size, d_model)
        src = src.mean(dim=0) # (batch_size, d_model)
        output = self.output_layer(src) # (seq_len, batch_size, output_feature_size)
        #output = F.softmax(output, dim=-1)
        return output