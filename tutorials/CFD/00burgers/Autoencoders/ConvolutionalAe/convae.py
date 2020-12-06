import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from clock import *

# class AE(nn.Module):
#     # by default our latent space is 50-dimensional
#     # and we use 400 hidden units
#     def __init__(self, hidden_dim=400, use_cuda=True):
#         super(AE, self).__init__()
#         # create the encoder and decoder networks
#         self.encoder = Encoder(hidden_dim)
#         self.decoder = Decoder(hidden_dim)

#         if use_cuda:
#             # calling cuda() here will put all the parameters of
#             # the encoder and decoder networks into gpu memory
#             self.cuda()
#             # self.encoder.cuda()
#             # self.decoder.cuda()

#     def forward(self, x):
#         z = self.encoder.forward(x)
#         x_out = self.decoder.forward(z)
#         return x_out

# class Encoder(nn.Module):
#     def __init__(self, hidden_dim, kernel_size=10):
#         super(Encoder, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Sequential(nn.Linear(18*18*128, hidden_dim), nn.Sigmoid())

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out

# class Decoder(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Decoder, self).__init__()

#         self.fc = nn.Linear(hidden_dim, 18*18*128)
#         self.layer1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'))
#         self.layer2 = nn.Sequential(
#             nn.ConvTranspose2d(64, 3, kernel_size=16, stride=2, padding=2),
#             nn.BatchNorm2d(3))#,
#         #     nn.Upsample(scale_factor=2, mode='bilinear'))

#     def forward(self, z):
#         out = self.fc(z)
#         out = out.reshape(-1, 128, 18, 18)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         return out


class AE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, hidden_dim=400, use_cuda=True, domain_size=60, scale=1):
        super(AE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim, domain_size)
        self.decoder = Decoder(hidden_dim, domain_size, self.encoder.hl, scale)
        self.scale = scale

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self.decoder.cuda()

    def forward(self, x):
        z = self.encoder.forward(x)
        x_out = self.decoder.forward(z)
        return x_out


class Encoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(Encoder, self).__init__()
        self.ds = domain_size
        self.hl = int(self.eval_size())

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.fc = nn.Sequential(nn.Linear(32 * (self.hl)**2, hidden_dim), nn.Tanh())

    @clock
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def eval_size(self):
        convlayer = lambda x: np.floor((x + 2 * 2 - 5) / 1 + 1)
        poolayer = lambda x: np.floor((x - 2) / 2 + 1)
        return poolayer(convlayer(poolayer(convlayer(self.ds))))


class ShallowDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(ShallowDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2), nn.BatchNorm1d(32 * (self.hl)**2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(3), nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Linear(32 * (self.hl)**2, 3 * (self.ds)**2), nn.BatchNorm1d(3 * (self.ds)**2), nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2)), nn.BatchNorm1d(32 * (self.hl)**2)), nn.ReLU())

    def forward(self, z):
        out = self.layer1(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 3)
        return out* self.scale

class Decoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(Decoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2))#, nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(3), nn.ReLU())

    @clock
    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer1(out)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 3)
        return out* self.scale


def plot_snapshot(snap, idx_train, idx_coord=0):
    m = snap.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    z = snap[idx_train, idx_coord, x, y]
    plt.figure(figsize=(7, 6))
    pl = plt.contourf(x, y, z)
    v1 = np.linspace(0, 0.5, 15)
    plt.clim(0., 0.5)
    cb = plt.colorbar(pl, fraction=0.046, pad=0.04, ticks=v1)
    # cb.ax.tick_params(labelsize='large')
    # cb.ax.set_yticklabels(["{:2.5f}".format(i) for i in v1])


def plot_compare(snap, snap_reconstruct, n_train, idx_coord=0, n_samples=5):
    domain_size = snap.shape[2]
    x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
    index_list = np.random.randint(0, n_train, n_samples)
    z = [snap[n, idx_coord, x, y] for n in index_list]
    z_reconstruct = [snap_reconstruct[n, idx_coord, x, y] for n in index_list]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 4, 20))
    fig.suptitle("comparison of snapshots and reconstructed snapshots")
    for i, image in enumerate(z):
        axes[0, i].contourf(x, y, image)
    for i, image in enumerate(z_reconstruct):
        axes[1, i].contourf(x, y, image)
    plt.show()