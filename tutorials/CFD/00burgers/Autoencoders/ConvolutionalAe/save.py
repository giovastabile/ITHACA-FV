import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
import argparse
from convae import *
import clock
import time

DOMAIN_SIZE = 60
DIM = 2  # number of components of velocity field
WM_PROJECT = "../../"
NUM_EPOCHS = 1000
HIDDEN_DIM = 4
BATCH_SIZE = 40
LEARNING_RATE = 1e-4
ORDER = 4
checkpoint_dir = "./checkpoint/"
model_dir = "./model/"

# Device configuration
device = torch.device('cuda')
print("device is: ", device)

# snapshots have to be clipped before
snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
assert np.min(snap_vec) >= 0., "Snapshots should be clipped"

# specify how many samples should be used for training and validation
n_total = snap_vec.shape[1]
n_train = n_total-n_total//6
print("Dimension of validation set: ", n_total-n_train)

# scale the snapshots
nor = Normalize(snap_vec, center_fl=True)
snap_framed = nor.framesnap(snap_vec)
snap_scaled = nor.scale(snap_framed)
snaps_torch = torch.from_numpy(snap_scaled)
print("snapshots shape", snap_scaled.shape)
print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))

# start model
model = AE(
    HIDDEN_DIM,
    scale=(nor.min_sn, nor.max_sn),
    #mean=nor.mean(device),
    domain_size=DOMAIN_SIZE,
    use_cuda=True).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ckp_path = "./checkpoint/checkpoint.pt"
ckp_path = "./model/best_model.pt"
model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
sm = torch.jit.trace(model, torch.rand(1, 2, 60, 60).to(device))
sm.save('model_gpu_' + str(HIDDEN_DIM) + '.pt')

# Save the initial value of the latent variable
initial = model.encoder(snaps_torch[:1, :, :, :].to(
    device, dtype=torch.float)).detach().to(torch.device('cpu'), dtype=torch.float)
# torch.jit.save(initial, 'initial.pt')
print("initial latent variable shape : ", initial)
np.save("latent_initial_" + str(HIDDEN_DIM) + ".npy", initial.numpy())

# Save decoder
model.decoder.to(device)
example = torch.rand(1, HIDDEN_DIM)
sm = torch.jit.trace(model.decoder, example.to(device) )
sm.save('decoder_gpu_' + str(HIDDEN_DIM) + '.pt')
print("saved")

decoder = torch.jit.load("decoder_gpu_4.pt")
output = decoder(initial.to(device, dtype=torch.float)).to(torch.device('cpu'))
plot_snapshot(output.detach().cpu().numpy().reshape(1, 2, 60, 60), 0, idx_coord=1)

rec = np.load("../../nonIntrusiveCoeffConvAe.npy")
print(rec.shape, rec)
output = decoder(torch.from_numpy(rec).to(device, dtype=torch.float)).to(torch.device('cpu'))
plot_snapshot(output.detach().cpu().numpy().reshape(-1, 2, 60, 60), 500, idx_coord=1)