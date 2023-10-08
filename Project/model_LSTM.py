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
        c_t = f * c_t_1 + i * g  # cell state, shape (batch_size, hidden_size)
        # hidden state, shape (batch_size, hidden_size)
        h_t = o * torch.tanh(c_t)

        return (h_t, c_t)


class LSTM(nn.Module):
    """implement a complete LSTM model
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """initiate value

        Args:
            input_size (int): The size of each input X_t
            hidden_size (int): The size of each hidden neuron
            num_layers (int): Number of recurrent layers
            output_size (int): The size of each output
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # add the first layer, the input size is input_size
        self.LSTM_first_layer = nn.ModuleList([
            LSTMCell(self.input_size, self.hidden_size)])

        # add the left other layers, Note the input size is hidden_size
        self.LSTM_whole = self.LSTM_first_layer.extend(
            [LSTMCell(self.hidden_size, self.hidden_size) for _ in range(1, num_layers)])

        # add a linear layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, X, hx=None):
        """_summary_

        Args:
            X (torch tensor): shape (batch_size, seq_len, input_size)
            hx (torch tensor, optional): _description_. Defaults to None.

        Returns:
            output: Output of shape (batch_size, output_size)
        """

        # initiate the value for h_0 and c_0, both shape is (num_layer, batch_size, hidden_size)
        batch_size = X.size(0)
        if hx is None:
            h_0 = Variable(torch.zeros(self.num_layers,
                                       batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers,
                                       batch_size, self.hidden_size))

        outs = []

        # For each layer, we initiate a set (h_0, c_0) and store it in 'h_t_c_t' list
        # h_t_c_t is then used to update (h_t, c_t) in each layer
        h_t_c_t = list()
        for layer in range(self.num_layers):
            h_t_c_t.append((h_0[layer, :, :], c_0[layer, :, :]))

        seq_len = X.size(1)
        for seq in range(seq_len):
            for layer in range(self.num_layers):
                # For the first layer, input is X[:, seq, :] with the shape (batch_size, input_size)
                # the seq'th input was feed into the network
                if layer == 0:
                    # h_t_c_t_layer is actually (h_1, c_1)
                    h_t_c_t_layer = self.LSTM_whole[layer](
                        X[:, seq, :], (h_t_c_t[layer][0], h_t_c_t[layer][1]))
                # For the other layers, input is the output h_t from previous layer
                else:
                    h_t_c_t_layer = self.LSTM_whole[layer](
                        h_t_c_t[layer - 1][0], (h_t_c_t[layer][0], h_t_c_t[layer][1]))

                # update (h_t, c_t) for different layer in h_t_c_t
                h_t_c_t[layer] = h_t_c_t_layer

            # For each input X[:, seq, :], after go through all the layers, the final h_t is then store in outs
            # outs is a list which has seq_len element, each element is the final h_t of the corresponding x_t
            # each element has the shape (batch_size, hidden_size)
            outs.append(h_t_c_t_layer[0])

        # we pick the last h_t
        out = outs[-1].squeeze()

        output = self.fc(out)

        return output
