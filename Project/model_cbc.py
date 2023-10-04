import torch
import torch.nn as nn
import torch.optim as optim
import math


class MultiHeadAttention(nn.Module):
    """
    Multihead self-attention
    """

    def __init__(self, embedding_size, num_heads):
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
        self.feed_forward = PositionWiseFeedForward(
            embedding_size, feedforward_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, embedding_size, num_heads, feedforward_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_size, num_heads)
        self.cross_attn = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = PositionWiseFeedForward(
            embedding_size, feedforward_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        attn_output = self.self_attn(x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(
            x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(attn_output))
        feedforward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feedforward_output))
        return x


class Transformer(nn.Module):

    """class to create the complete transformer architecture
    """

    def __init__(self, encoder_input_size, decoder_input_size, embedding_size, num_heads, num_layers, feedforward_size, dropout=0.1, max_seq_length=100):
        """class initializer

        Args:
            encoder_input_size (int): dimension of the encoder input, shape (B, 8, encoder_input_size)
            decoder_input_size (int): dimension of the decoder input, shape (B, 12, decoder_input_size)
            embedding_size (int): embedding size, e.g 512, change one-hot encoding to word embedding.
            num_heads (int): number of heads in multi-head self-attention
            num_layers (int): the number of encoder blocks and decoder blocks
            feedforward_size (int): size of linear layer in feedforward network (same in Encoder & Decoder)
            dropout (float): 0-1, dropout percentage, default value = 0.1
            max_seq_length (int): maximum length of sequence in Encoder and Decoder (used in PositionalEncoding)
        """
        super(Transformer, self).__init__()

        #
        self.encoder_embedding = nn.Embedding(
            encoder_input_size, embedding_size)
        self.decoder_embedding = nn.Embedding(
            decoder_input_size, embedding_size)
        self.positional_encoding = PositionalEncoding(
            embedding_size, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embedding_size, num_heads, feedforward_size, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embedding_size, num_heads, feedforward_size, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(embedding_size, decoder_input_size)
        self.dropout = nn.Dropout(dropout)
        # self.output_gen = nn.Linear()

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
