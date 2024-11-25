import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size, seq_size):
        super(StateCritic, self).__init__()

        self.sequence_size = seq_size
        self.static_encoder = Encoder(
            int(static_size / self.sequence_size), hidden_size
        )
        self.dynamic_encoder = Encoder(
            int(dynamic_size / self.sequence_size), hidden_size
        )

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

    def _build_inputs(self, batch, bs, t=None):
        ts = slice(None) if t is None else slice(t, t + 1)
        output = batch["obs"]
        return batch["obs"][:, ts]

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        return input_shape
