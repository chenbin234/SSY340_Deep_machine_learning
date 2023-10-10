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
            mask = mask.unsqueeze(1)
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


class Embeddings(nn.Module):
    """
    class to generate the embeddings for encoder and decoder input data
    """

    def __init__(self, input_size, embedding_size):
        """
        class initializer

        INPUT:
        input_size - (int) size of the input data
        embedding_size - (int) size of the embedding
        """
        super(Embeddings, self).__init__()

        # caching values
        self.embedding_size = embedding_size

        # creating liner layer for embedding input data
        self.linear_embd = nn.Linear(input_size, embedding_size)

        # creating object for positional encoding
        # self.pos_encoding = PositionalEncoding(embedding_size)

    def forward(self, x):
        """
        forward pass to generate input embeddings

        INPUT:
        x - (torch tensor) input data. Shape = (B, N, input_dimension)

        OUTPUT:
        x - (torch tensor) embedded data. Shape = (B, N, C)
        """

        # creating embeddings for input data
        # Shape = (B, N, C)
        x = self.linear_embd(x.float()) * math.sqrt(self.embedding_size)
        # incorporating positional embeddings
        # x = self.pos_encoding.forward(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size, max_seq_length=100):
        """_summary_

        Args:
            embedding_size (int): embedding size, e.g 512
            max_seq_length (int, optional): maximum length of sequence in Encoder and Decoder. Defaults to 100.
        """

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
        attn_output = self.self_attn.forward(x, x, x, mask)
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

        # print('start Decoder layer masked multihead attention')
        attn_output = self.self_attn.forward(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # print('start Decoder layer cross multihead attention')
        # attn_output = self.cross_attn.forward(x, encoder_output, encoder_output, source_mask)
        attn_output = self.cross_attn.forward(
            x, encoder_output, encoder_output, mask=None)

        x = self.norm2(x + self.dropout(attn_output))
        feedforward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feedforward_output))
        return x


class Transformer(nn.Module):

    """class to create the complete transformer architecture
    """

    def __init__(self, encoder_input_size, decoder_input_size, embedding_size, num_heads, num_layers, feedforward_size, dropout=0.1):
        """class initializer

        Args:
            encoder_input_size (int): dimension of the encoder input, shape (B, 8, encoder_input_size)
            decoder_input_size (int): dimension of the decoder input, shape (B, 12, decoder_input_size)
            embedding_size (int): embedding size, e.g 512, change one-hot encoding to word embedding.
            num_heads (int): number of heads in multi-head self-attention
            num_layers (int): the number of encoder blocks and decoder blocks
            feedforward_size (int): size of linear layer in feedforward network (same in Encoder & Decoder)
            dropout (float): 0-1, dropout percentage, default value = 0.1
        """
        super(Transformer, self).__init__()

        # Encoder input embeddings (word embeddings)
        # input size (B, 8, encoder_input_size), output size (B, 8, embedding_size)
        self.encoder_embedding = Embeddings(encoder_input_size, embedding_size)

        # Decoder input embeddings (word embeddings)
        # input size (B, 12, decoder_input_size), output size (B, 12, embedding_size)
        self.decoder_embedding = Embeddings(decoder_input_size, embedding_size)

        # Positional Encoding for Encoder & Decoder
        self.positional_encoding = PositionalEncoding(embedding_size)

        # stack num_layers' Encoder (or Decoder) blocks
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embedding_size, num_heads, feedforward_size, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embedding_size, num_heads, feedforward_size, dropout) for _ in range(num_layers)])

        # Linear layer after the Decoder for model ouput
        # output size (B, 12, decoder_input_size)
        self.fc = nn.Linear(embedding_size, decoder_input_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """generate mask 

        Args:
            src (torch tensor): _description_
            tgt (torch tensor): _description_

        Returns:
            _type_: _description_
        """
        # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # # tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # tgt_mask = (tgt != 0).unsqueeze(1)
        # seq_length = tgt.size(1)
        # nopeak_mask = (
        #     1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        # tgt_mask = tgt_mask & nopeak_mask

        seq_length0 = src.shape[1]
        batch_size = tgt.shape[0]
        seq_length1 = tgt.shape[1]

        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length1, seq_length1), diagonal=1)).bool()
        tgt_mask = nopeak_mask.repeat(batch_size, 1, 1)

        src_mask = torch.ones(batch_size, seq_length0, seq_length0)

        # dec_source_mask = torch.ones((enc_input.shape[0], 1, enc_input.shape[1])).to(device)
        # dec_target_mask = utils.subsequent_mask(dec_input.shape[1]).repeat(dec_input.shape[0], 1, 1).to(device)

        return src_mask, tgt_mask

    def forward(self, src, tgt):

        # genarate mask for encoder and Decoder
        # print('genarate mask')
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # print('start src_word_embeddings')
        src_word_embeddings = self.encoder_embedding.forward(src)

        # print('start src positional_encoding')
        src_embedded = self.positional_encoding.forward(src_word_embeddings)

        # print('start tgt_word_embeddings')
        tgt_word_embeddings = self.decoder_embedding.forward(tgt)

        # print('start tgt positional_encoding')
        tgt_embedded = self.positional_encoding.forward(tgt_word_embeddings)

        # src_embedded = self.dropout(
        #     self.positional_encoding(self.encoder_embedding(src)))
        # tgt_embedded = self.dropout(
        #     self.positional_encoding(self.decoder_embedding(tgt)))

        # print('start Encoder')
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # print('start Decoder')
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # output size (B, 12, decoder_input_size)
        output = self.fc(dec_output)
        return output
