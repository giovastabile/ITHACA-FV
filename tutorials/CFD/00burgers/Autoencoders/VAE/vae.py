import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

from clock import *
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 29, 29]             408
       BatchNorm2d-2            [-1, 8, 29, 29]              16
               ELU-3            [-1, 8, 29, 29]               0
            Conv2d-4           [-1, 16, 14, 14]           3,216
       BatchNorm2d-5           [-1, 16, 14, 14]              32
               ELU-6           [-1, 16, 14, 14]               0
            Conv2d-7             [-1, 32, 6, 6]          12,832
       BatchNorm2d-8             [-1, 32, 6, 6]              64
               ELU-9             [-1, 32, 6, 6]               0
           Conv2d-10             [-1, 64, 3, 3]          32,832
      BatchNorm2d-11             [-1, 64, 3, 3]             128
              ELU-12             [-1, 64, 3, 3]               0
           Linear-13                    [-1, 4]           2,308
           Linear-14                  [-1, 576]           2,880
  ConvTranspose2d-15             [-1, 32, 7, 7]          51,232
      BatchNorm2d-16             [-1, 32, 7, 7]              64
              ELU-17             [-1, 32, 7, 7]               0
  ConvTranspose2d-18           [-1, 16, 15, 15]          12,816
      BatchNorm2d-19           [-1, 16, 15, 15]              32
              ELU-20           [-1, 16, 15, 15]               0
  ConvTranspose2d-21            [-1, 8, 29, 29]           3,208
      BatchNorm2d-22            [-1, 8, 29, 29]              16
              ELU-23            [-1, 8, 29, 29]               0
  ConvTranspose2d-24            [-1, 2, 60, 60]             578
      BatchNorm2d-25            [-1, 2, 60, 60]               4
================================================================
Total params: 122,666
Trainable params: 122,666
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 0.65
Params size (MB): 0.47
Estimated Total Size (MB): 1.15
----------------------------------------------------------------
"""
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 60, 60]           1,216
       BatchNorm2d-2           [-1, 16, 60, 60]              32
              ReLU-3           [-1, 16, 60, 60]               0
         MaxPool2d-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 32, 30, 30]          12,832
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
         MaxPool2d-8           [-1, 32, 15, 15]               0
            Linear-9                    [-1, 4]          28,804
             Tanh-10                    [-1, 4]               0
           Linear-11                 [-1, 7200]          36,000
  ConvTranspose2d-12           [-1, 16, 30, 30]           8,208
      BatchNorm2d-13           [-1, 16, 30, 30]              32
             ReLU-14           [-1, 16, 30, 30]               0
  ConvTranspose2d-15            [-1, 3, 60, 60]           1,731
      BatchNorm2d-16            [-1, 3, 60, 60]               6
             ReLU-17            [-1, 3, 60, 60]               0
================================================================
Total params: 88,925
Trainable params: 88,925
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 2.77
Params size (MB): 0.34
Estimated Total Size (MB): 3.15
----------------------------------------------------------------
"""
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 60, 60]             816
       BatchNorm2d-2           [-1, 16, 60, 60]              32
              ReLU-3           [-1, 16, 60, 60]               0
         MaxPool2d-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 32, 30, 30]          12,832
       BatchNorm2d-6           [-1, 32, 30, 30]              64
              ReLU-7           [-1, 32, 30, 30]               0
         MaxPool2d-8           [-1, 32, 15, 15]               0
            Linear-9                    [-1, 4]          28,804
          Sigmoid-10                    [-1, 4]               0
           Linear-11                 [-1, 7200]          36,000
  ConvTranspose2d-12           [-1, 16, 30, 30]           8,208
      BatchNorm2d-13           [-1, 16, 30, 30]              32
             ReLU-14           [-1, 16, 30, 30]               0
  ConvTranspose2d-15            [-1, 2, 60, 60]           1,154
      BatchNorm2d-16            [-1, 2, 60, 60]               4
             ReLU-17            [-1, 2, 60, 60]               0
================================================================
Total params: 87,946
Trainable params: 87,946
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 2.69
Params size (MB): 0.34
Estimated Total Size (MB): 3.05
----------------------------------------------------------------
"""
# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self, hidden_dim=400, use_cuda=True, domain_size=60, scale=(-1, 1)):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = DeepEncoder(hidden_dim, domain_size)
        self.decoder = DeepDecoder(hidden_dim, domain_size, self.encoder.hl, scale)
        self.scale = scale
        self.ds = domain_size

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self.decoder.cuda()
        self.use_cuda = use_cuda
        self.hidden_dim = hidden_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img, loc_scale = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Normal(loc_img, loc_scale).to_event(1), obs=x.reshape(-1, 2*self.ds**2))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img, loc_scale = self.decoder(z)
        return loc_img, loc_scale

class Encoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(Encoder, self).__init__()
        self.ds = domain_size
        self.hl = int(self.eval_size())

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.fc = nn.Sequential(nn.Linear(32 * (self.hl)**2, hidden_dim), nn.Sigmoid())

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

class DeepEncoder(nn.Module):
    def __init__(self, hidden_dim, domain_size):
        super(DeepEncoder, self).__init__()
        self.ds = domain_size
        self.hl = int(self.eval_size())
        # print(self.hl)
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=1), nn.BatchNorm2d(8), Swish())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16), Swish())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32), Swish())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), Swish())
        self.fc = nn.Sequential(nn.Linear(64 * (self.hl)**2, hidden_dim), Swish())
        self.sigma_net = nn.Linear(2 * self.ds**2, hidden_dim)

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
        return out, self.sigma_net(x.reshape(-1, 2*self.ds**2))

    def eval_size(self):
        convlayer = lambda x: np.floor((x  - 5 + 2) / 2 + 1)
        lastconvlayer = lambda x: np.floor((x  - 4 + 2) / 2 + 1)
        # print(convlayer(self.ds))
        # print(convlayer(convlayer(self.ds)))
        # print(convlayer(convlayer(convlayer(self.ds))))
        return lastconvlayer(convlayer(convlayer(convlayer(self.ds))))

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
            nn.ConvTranspose2d(16, 2, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(2), nn.ReLU())

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer1(out)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 2)
        return out*(self.scale[1]-self.scale[0])+self.scale[0]

class DeepDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(DeepDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64 * (self.hl)**2), Swish())
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1), nn.BatchNorm2d(32), Swish())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=5, stride=2, padding=1), nn.BatchNorm2d(16), Swish())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(8), Swish())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(8, 2, kernel_size=6, stride=2, padding=1), nn.BatchNorm2d(2))
        self.sigma_net = nn.Linear(hidden_dim, 2 * (self.ds)**2 )

    def forward(self, z):
        out = self.fc(z)
        out = out.reshape(-1, 64, self.hl, self.hl)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out).reshape(-1, 2 * self.ds**2)
        return out*(self.scale[1]-self.scale[0])+self.scale[0], self.sigma_net(z)


class ShallowDecoder(nn.Module):
    def __init__(self, hidden_dim, domain_size, hidden_length, scale):
        super(ShallowDecoder, self).__init__()
        self.ds = domain_size
        self.hl = hidden_length
        self.scale = scale

        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2), nn.BatchNorm1d(32 * (self.hl)**2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(2), nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Linear(32 * (self.hl)**2, 3 * (self.ds)**2), nn.BatchNorm1d(3 * (self.ds)**2), nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Linear(hidden_dim, 32 * (self.hl)**2)), nn.BatchNorm1d(32 * (self.hl)**2)), nn.ReLU())

    def forward(self, z):
        out = self.layer1(z)
        out = out.reshape(-1, 32, self.hl, self.hl)
        out = self.layer2(out).reshape(-1, self.ds * self.ds * 2)
        return out*(self.scale[1]-self.scale[0])+self.scale[0]

def plot_snapshot(frame, idx, idx_coord=1):
    m = frame.shape[2]
    x, y = np.meshgrid(np.arange(m), np.arange(m))
    z = frame[idx, idx_coord, x, y]
    plt.figure(figsize=(7, 6))
    pl = plt.contourf(x, y, z)
    # v1 = np.linspace(0, 0.5, 15)
    # plt.clim(0., 0.1)
    cb = plt.colorbar(pl, fraction=0.046, pad=0.04)#, ticks=v1)
    plt.show()
    # cb.ax.tick_params(labelsize='large')
    # cb.ax.set_yticklabels(["{:2.5f}".format(i) for i in v1])

def plot_two(snap, snap_reconstruct, idx, epoch, idx_coord=0, title='bu'):
    domain_size = snap.shape[2]
    x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
    z = [snap[n, idx_coord, x, y] for n in idx]
    z_reconstruct = [snap_reconstruct[n, idx_coord, x, y] for n in idx]

    fig, axes = plt.subplots(2, len(z), figsize=(len(z)* 4, 20))
    fig.suptitle(title)
    if len(z)>2:
        for i, image in enumerate(z):
            im = axes[0, i].contourf(x, y, image)
            fig.colorbar(im, ax=axes[0, i])
        for i, image in enumerate(z_reconstruct):
            im_ = axes[1, i].contourf(x, y, image)
            fig.colorbar(im_, ax=axes[1, i])
    elif len(z)==1:
        for i, image in enumerate(z):
            im = axes[0].contourf(x, y, image)
            fig.colorbar(im, ax=axes[0])
        for i, image in enumerate(z_reconstruct):
            im_ = axes[1].contourf(x, y, image)
            fig.colorbar(im_, ax=axes[1])
    plt.savefig('./data/'+ title + "_" + str(epoch) + '.png')
    plt.close()


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

class Swish(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
