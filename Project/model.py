import torch
import torch.nn as nn
import torch.optim as optim
import math


class MultiHeadAttention(nn.Module):
    """
    Multihead self-attention
    """

    def __init__(self, num_heads, embedding_size):
        """_summary_
        """
        super(MultiHeadAttention, self).__init__()
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"

        # initialize values
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        # Dimension of each head's key, query, and value
        self.d_k = embedding_size // num_heads

        # Linear layers for input embeddings, make sure Y = W_Q*X, Y has the same size with X, which means the size of W_Q, W_K, W_V is (embedding_size, embedding_size), the size of Q,K,V is (embedding_size, sequence_length)
        self.W_q = nn.Linear(embedding_size, embedding_size)  # Query
        self.W_k = nn.Linear(embedding_size, embedding_size)  # Key
        self.W_v = nn.Linear(embedding_size, embedding_size)  # Value
        self.W_o = nn.Linear(embedding_size, embedding_size)  # post attention

    def normalized_dot_product_attention(self, Q, K, V, mask=None):
        """_summary_

        Args:
            Q (torch tensor): Shape = (B, H, N, C)
            K (torch tensor): Shape = (B, H, N, C)
            V (torch tensor): Shape = (B, H, N, C)
            mask (torch tensor, optional): mask for decoder multi head attention layer. Defaults to None.
            Where,
            B - Batch size
            H - Number of heads
            N - Sequence length
            C - Embedding size
        """
        # Calculate attention scores
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # applying mask on attention, we use large negative values -1e9 instead of 0, because after softmax function, the result will be close to 0.
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        attn_output = torch.matmul(attn_probs, V)
        return attn_output

    def split_heads(self, x):
        """ reshapes the input tensor x to have multiple heads for multi-head attention. 

        Args:
            x (_type_): _description_
        """
        batch_size, seq_length, embedding_size = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_size)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.normalized_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size, max_seq_length):

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, embedding_size)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embedding_size, 2).float() * -(math.log(10000.0) / embedding_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """forward pass to generate positional embeddings

        Args:
            x - (torch tensor) embedded data. Shape = (B, N, C) 

        Returns:
            x - (torch tensor) positional embedded data. Shape = (B, N, C) 
        """
        return x + self.pe[:, :x.size(1)]

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_size, feedforward_size):

        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_size, feedforward_size)
        self.fc2 = nn.Linear(feedforward_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = PositionWiseFeedForward(embedding_size, feedforward_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x