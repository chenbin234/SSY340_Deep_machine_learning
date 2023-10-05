import model_cbc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import dataloader
from torch.utils.data import DataLoader


transformer = model_cbc.Transformer(encoder_input_size=2, decoder_input_size=2,
                                    embedding_size=512, num_heads=8, num_layers=6, feedforward_size=2048)
# transformer = model.TFModel(encoder_input_size=2, decoder_input_size=2, embedding_size=512, num_heads=8, num_layers=6, feedforward_size=2048)
# transformer = model.TFModel(encoder_ip_size=2, decoder_ip_size=2,
#                             model_op_size=2, emb_size=512, num_heads=8, ff_hidden_size=2048, n=6)


def subsequent_mask(size):
    """
    Function to compute the mask used in attention layer of decoder

    INPUT:
    size - (int) horizon size

    OUTPUT:
    mask - (torch tensor) boolean array to mask out the data in decoder
    """

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(mask) == 0

    return mask


# Generate random sample data
enc_input = torch.randint(1, 10, (64, 7, 2))  # (batch_size, seq_length)
dec_input = torch.randint(1, 10, (64, 7, 2))  # (batch_size, seq_length)


dec_source_mask = torch.ones(
    (enc_input.shape[0], 1, enc_input.shape[1]))
dec_target_mask = subsequent_mask(
    dec_input.shape[1]).repeat(dec_input.shape[0], 1, 1)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(2):
    optimizer.zero_grad()
    # output = transformer(src_data, tgt_data[:, :-1])
    output = transformer.forward(enc_input, dec_input)

    print('fine')
    loss = criterion(output.contiguous().view(-1, 896).float(),
                     dec_input.contiguous().view(-1, 896).float())
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
