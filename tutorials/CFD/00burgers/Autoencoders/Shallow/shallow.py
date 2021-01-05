import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim, output_dim, scale, hidden_dim_enc=32, hidden_dim_dec=3, latent_dim=4, domain_size=60, use_cuda=True):
        super().__init__()
        self.encoder = DeepEncoder(latent_dim, domain_size)
        self.decoder = DecoderBlock(output_dim, hidden_dim_enc, latent_dim, scale)
        if use_cuda:
            self.cuda()

    def forward(self, x):
        enc_x = self.encoder(x)
        x = self.decoder(enc_x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=6000, latent_dim=4):
        super().__init__()

        self.layer1 =  nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, output_dim=784, hidden_dim=33000, latent_dim=4, scale=(-1, 1)):
        super().__init__()
        self.scale = scale

        self.layer1 =  nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ELU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out*(self.scale[1]-self.scale[0])+self.scale[0]

class DeepEncoder(nn.Module):
    def __init__(self, latent_dim, domain_size):
        super(DeepEncoder, self).__init__()
        self.ds = domain_size
        self.hl = int(self.eval_size())
        # print(self.hl)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=1), nn.BatchNorm2d(8), nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ELU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ELU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ELU())
        self.fc = nn.Sequential(nn.Linear(64 * (self.hl)**2, latent_dim))

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # print(out.size())
        return out

    def eval_size(self):
        convlayer = lambda x: np.floor((x  - 5 + 2) / 2 + 1)
        lastconvlayer = lambda x: np.floor((x  - 4 + 2) / 2 + 1)
        # print(convlayer(self.ds))
        # print(convlayer(convlayer(self.ds)))
        # print(convlayer(convlayer(convlayer(self.ds))))
        return lastconvlayer(convlayer(convlayer(convlayer(self.ds))))


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