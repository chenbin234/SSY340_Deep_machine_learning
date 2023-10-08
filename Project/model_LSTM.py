import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LSTMCell(nn.Module):
    """implement functionity for a LSTM cell 
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size  # The size of each input X_t
        self.hidden_size = hidden_size  # The size of each hidden neuron

        # W_hh * h_t_minus_1 + W_xh * x_t
        # The reason we times 4 is that we have 4 gates
        self.W_hh = nn.Linear(hidden_size, hidden_size * 4)
        self.W_xh = nn.Linear(input_size, hidden_size * 4)

        # reset the parameter
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x_t, h_t_c_t=None):
        """compute hidden state h_t and cell state c_t, 
        based on input x_t, and previous states h_{t-1} and c_{t-1}

        Args:
            x_t (torch tensor): input at time t, shape (batch_size, input_size)
            h_t_c_t (set of torch tensor): (h_{t-1}, c_{t-1}), each have the shape (batch_size, hidden_size)

        Returns:
            (h_t, c_t): each have the shape (batch_size, hidden_size)
        """

        # Initiate hidden state h_t and cell state c_t for the first step
        if h_t_c_t is None:
            # zeros = Variable(x_t.new_zeros(x_t.size(0), self.hidden_size))
            # h_t_c_t = (zeros, zeros)
            zeros = torch.zeros(x_t.size(0), self.hidden_size, dtype=x_t.dtype)
            h_t_c_t = (zeros, zeros)

        h_t_1, c_t_1 = h_t_c_t

        # gates = W_hh * h_t_1 + W_xh * x_t, gates has the shape (batch_size, 4*hidden_size)
        gates = self.W_hh(h_t_1) + self.W_xh(x_t)

        # Get 4 gates, each gate has the shape (batch_size, hidden_size)
        input_gate, forget_gate, gate_gate, output_gate = gates.chunk(4, 1)

        i = torch.sigmoid(input_gate)  # input gate
        f = torch.sigmoid(forget_gate)  # forget gate
        o = torch.sigmoid(output_gate)  # output gate
        g = torch.tanh(gate_gate)  # gate gate

        # according to LSTM architecture
        c_t = f * c_t_1 + i * g # cell state 
        h_t = o * torch.tanh(c_t) # hidden state

        return (h_t, c_t)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(LSTMBlock(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMBlock(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers,
                              input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers,
                              input.size(0), self.hidden_size))
        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out
