import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, input_size=784, encoding_dim=32, n_layers=3):
        super().__init__()
        layers =  [nn.Sequential(nn.Linear(input_size, encoding_dim*2**(n_layers-1)), nn.ReLU())]
        layers += [nn.Sequential(nn.Linear(encoding_dim*2**(i), encoding_dim*2**(i-1)), nn.ReLU()) for i in range(n_layers-1,0,-1)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, output_size=784, encoding_dim=32, n_layers=3):
        super().__init__()
        layers =  [nn.Sequential(nn.Linear(encoding_dim*2**(i), encoding_dim*2**(i+1)), nn.ReLU()) for i in range(n_layers-1)]
        layers += [nn.Sequential(nn.Linear(encoding_dim*2**(n_layers-1), output_size), nn.ReLU())]
        self.layers = nn.Sequential(*layers, nn.Sigmoid())
    def forward(self, x):
        return self.layers(x)

class AE(nn.Module):
    def __init__(self, input_size=784, encoding_dim=32, n_layers=3, use_cuda=True):
        super().__init__()
        self.encoder = EncoderBlock(input_size, encoding_dim, n_layers)
        self.decoder = DecoderBlock(input_size, encoding_dim, n_layers)
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self.decoder.cuda()
    def forward(self, x):
        enc_x = self.encoder(x)          # encode
        x = self.decoder(enc_x)          # decode
        return x#, enc_x # also return encoding


def plot_snapshot(snap, idx_train, idx_coord=0):
    m = snap.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    z = snap[idx_train, idx_coord, x, y]
    plt.figure(figsize=(7, 6))
    pl = plt.contourf(x, y, z)
    v1 = np.linspace(0, np.max(z), 15)
    cb = plt.colorbar(pl,fraction=0.046, pad=0.04, ticks=v1)
    cb.ax.tick_params(labelsize='large')
    cb.ax.set_yticklabels(["{:2.1f}".format(i) for i in v1])

def plot_compare(snap, snap_reconstruct, n_train, idx_coord=0, n_samples=5):
    x, y = np.meshgrid(np.arange(150), np.arange(150))
    index_list = np.random.randint(0, n_train, n_samples)
    z = [snap[n, idx_coord, x, y] for n in index_list]
    z_reconstruct = [snap_reconstruct[n, idx_coord, x, y] for n in index_list]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*4, 20))
    fig.suptitle("comparison of snapshots and reconstructed snapshots")
    for i, image in enumerate(z):
        axes[0, i].contourf(x, y, image)
    for i, image in enumerate(z_reconstruct):
        axes[1, i].contourf(x, y, image)
    plt.show()