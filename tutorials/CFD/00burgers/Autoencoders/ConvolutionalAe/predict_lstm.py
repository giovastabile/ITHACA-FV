import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
from lstm import *
from convae import *

WM_PROJECT = "../../"
HIDDEN_DIM = 4
domain_size = 60
DIM = 2
device = torch.device('cuda:0')
print("device is: ", device)

# reshape as (train_samples, channel, y, x)
snapshots = np.load(WM_PROJECT+"npSnapshots.npy")
print("snapshots shape: ", snapshots.shape)

n_train = snapshots.shape[1]
snapshots = snapshots.T
snapshots_numpy = snapshots.reshape((n_train, 3, domain_size, domain_size))[:, :DIM,:, :]
sn_max = np.max(snapshots_numpy)
sn_min = np.min(snapshots_numpy)
snapshots = torch.from_numpy((snapshots_numpy-sn_min)/(sn_max-sn_min)).to(device, dtype=torch.float)
print("snapshots shape: ", snapshots.size())

# load autoencoder
model = AE(HIDDEN_DIM, scale=(sn_min, sn_max), domain_size=domain_size, use_cuda=True).to(device)
model.load_state_dict(torch.load("./model.ckpt"))
model.eval()

##################################### TEST LSTM

train_params = np.load(WM_PROJECT+"parTrain.npy").reshape(-1)
n_train_params = train_params.shape[0]

# loading training outputs
output_pre = np.load(WM_PROJECT+'ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
n_time_samples_times_n_param = output_pre.shape[0]
n_time_samples = n_time_samples_times_n_param//n_train_params
print("number of time samples: ", n_time_samples)

time_samples = output_pre[:n_time_samples, 0]
output = output_pre[:, 1:]
print("outputs shape: ", output.shape)
print("time samples: ", time_samples.shape)

# load test set
test_params = np.load(WM_PROJECT+"parTest.npy").reshape(-1)
print("test parameters shape: ", test_params.shape, test_params)

# prepare test set, first column time, second column parameters
x_test_0 = np.kron(np.ones(test_params.shape[0]).reshape(-1, 1), time_samples.reshape(-1, 1))
print("times column shape: ", x_test_0.shape)

x_test_1 = np.kron(test_params.reshape(-1, 1), np.ones((time_samples.shape[0], 1)))
print("parameters column shape: ", x_test_1.shape)

x_test = np.hstack((x_test_0, x_test_1))
print("test dataset shape: ", x_test.shape)

# load lstm
lstm_model = ReducedCoeffsTimeSeries().to(device)
lstm_model.load_state_dict(torch.load("./lstm.ckpt"))
lstm_model.eval()

# get predictions with GPR
predictions = lstm_model( torch.from_numpy(x_test.reshape(1, n_time_samples, 2)).to(device, dtype=torch.float)).squeeze()
print("predictions shape: ", predictions.shape)

predicted_snapshots = model.decoder.forward(predictions).cpu().detach().numpy()
print("predicted snapshots shape: ", predicted_snapshots.shape)
np.save(WM_PROJECT+"snapshotsReconstructedConvAe.npy",predicted_snapshots)


#################################### PROJECTION ERROR
snap_true = np.load(WM_PROJECT+"npTrueSnapshots.npy")
print("true test snapshots shape: ", snap_true.shape)
n_test = snap_true.shape[1]
snap_true = snap_true.T
snap_true_numpy = snap_true.reshape(n_test, 3, domain_size, domain_size)[:, :2, :, :]
snap_true_numpy -= sn_min
snap_true_numpy /= sn_max-sn_min
snap_true = torch.from_numpy(snap_true_numpy).to(device, dtype=torch.float)
print("snapshots shape: ", snap_true.size())

# reconstruct snapshots
snap_true_rec = model.forward(snap_true).cpu().detach().numpy()
snap_true_rec = snap_true_rec.reshape(-1, 2, domain_size, domain_size)
print("non linear reduction training coeffs: ", snap_true_rec.shape)
plot_compare(snap_true_numpy, snap_true_rec, n_test)
np.save(WM_PROJECT+"snapshotsConvAeTrueProjection.npy",snap_true_rec)

