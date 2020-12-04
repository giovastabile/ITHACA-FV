import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    def __init__(self, hidden_dim=400, use_cuda=True):
        super(AE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(hidden_dim)

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
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(37*37*32, hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(hidden_dim, 37*37*32)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU())

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32, 37, 37)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

def plot_snapshot(snap, idx_train, idx_coord=0):
    m = snap.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    z = snap[idx_train, idx_coord, x, y]
    plt.contourf(x, y, z)
    plt.colorbar()

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