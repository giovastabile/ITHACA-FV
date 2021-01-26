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
DOMAIN_SIZE = 60
DIM = 2
device = torch.device('cuda:0')
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

# load autoencoder
model = AE(
        HIDDEN_DIM,
        scale=(nor.min_sn, nor.max_sn),
        #mean=nor.mean(device),
        domain_size=DOMAIN_SIZE,
        use_cuda=True).to(device)

modello = torch.load("./model_4.ckpt")
model.load_state_dict(modello['state_dict'])
# model.load_state_dict(torch.load("./model_"+str(HIDDEN_DIM)+".ckpt"))
# model.eval()

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
lstm_model.load_state_dict(torch.load('lstm_'+str(HIDDEN_DIM)+'.ckpt'))
lstm_model.eval()

# get predictions with lstm
predictions = lstm_model( torch.from_numpy(x_test.reshape(1, n_time_samples, 2)).to(device, dtype=torch.float)).squeeze()
print("predictions shape: ", predictions.shape)
# predictions = np.hstack((x_test_0, predictions.detach().cpu().numpy().squeeze()))
print("predictions and timings shape: ", predictions.shape)
np.save(WM_PROJECT+"nonIntrusiveCoeffConvAe.npy", predictions.to(device, dtype=torch.float).detach().cpu().numpy())

predicted_snapshots = model.decoder.forward(predictions.to(device)).cpu().detach().numpy()
print("predicted snapshots shape: ", predicted_snapshots.shape)
np.save(WM_PROJECT+"snapshotsReconstructedConvAe.npy",predicted_snapshots)

predicted_snapshots_openfoam = np.concatenate((predicted_snapshots.reshape(time_samples.shape[0], 2, DOMAIN_SIZE, DOMAIN_SIZE), np.zeros((time_samples.shape[0], 1, DOMAIN_SIZE, DOMAIN_SIZE))), axis=1).reshape(time_samples.shape[0], -1).T
print("predicted snapshots openfoam shape: ", predicted_snapshots_openfoam.shape)
np.save(WM_PROJECT+"snapshotsReconstructedConvAeOF.npy",predicted_snapshots_openfoam)


#################################### PROJECTION ERROR
snap_vec_true = np.load(WM_PROJECT+"npTrueSnapshots.npy")
assert np.min(snap_vec_true) >= 0., "Snapshots should be clipped"

# scale the snapshots
snap_framed_true = nor.framesnap(snap_vec_true)
snap_scaled_true = nor.scale(snap_framed_true)
snap_torch_true = torch.from_numpy(snap_scaled_true)
n_test = snap_framed_true.shape[0]
test_norm = np.linalg.norm(snap_vec_true, axis=0, keepdims=False)
test_max_norm = np.max(snap_vec_true, axis=0, keepdims=False)
print("test snapshots shape: ", snap_scaled_true.shape)
print("max and min L2 norm", np.max(test_norm), np.min(test_norm))
print("snapshots shape", snap_scaled_true.shape)
print("Min max after scaling: ", np.min(snap_scaled_true), np.max(snap_scaled_true))

# reconstruct snapshots
snap_true_rec = model(snap_torch_true.to(device, dtype=torch.float)).cpu().detach().numpy()
snap_true_rec = nor.frame2d(snap_true_rec)
print("non linear reduction training coeffs: ", snap_true_rec.shape)
plot_compare(snap_framed_true, snap_true_rec, n_test)
np.save(WM_PROJECT+"snapshotsConvAeTrueProjection.npy",snap_true_rec)

err = np.abs(snap_true_rec - nor.rescale(snap_scaled_true))
error_proj = np.linalg.norm(nor.vectorize2d(err), axis=1)
error_proj = error_proj / test_norm
error_mean = np.mean(error_proj)
error_max = np.max(error_proj)
error_min = np.min(error_proj)
print("projection erorrs: mean, max, min: ", error_mean, error_max, error_min)

