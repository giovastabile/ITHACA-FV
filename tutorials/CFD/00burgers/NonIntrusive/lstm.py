import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class ReducedCoeffsTimeSeries(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=4,
                 hidden_dim=200,
                 n_layers=2,
                 use_cuda=True):
        super(ReducedCoeffsTimeSeries, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim,
                                  hidden_dim,
                                  n_layers,
                                  batch_first=True)

        self.encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ELU(), nn.Linear(hidden_dim//2, output_dim))

        if use_cuda:
            self.cuda()

    def forward(self, x):
        z = self.lstm(x)[0]
        x_out = self.encoder(z)
        return x_out


