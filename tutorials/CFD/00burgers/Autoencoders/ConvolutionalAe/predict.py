import numpy as np
import matplotlib.pyplot as plt
import GPy
import torch
import torch.nn as nn
from convae import *

WM_PROJECT = "../../"
HIDDEN_DIM = 4
domain_size = 60
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("device is: ", device)

# reshape as (train_samples, channel, y, x)
snapshots = np.load(WM_PROJECT+"npSnapshots.npy")
print("snapshots shape: ", snapshots.shape)

n_train = snapshots.shape[1]
snapshots = snapshots.T
snapshots_numpy = snapshots.reshape((n_train, 3, domain_size, domain_size))
sn_max = np.max(np.abs(snapshots_numpy))
snapshots = torch.from_numpy(snapshots_numpy/sn_max).to(device, dtype=torch.float)
print("snapshots shape: ", snapshots.size())

# load autoencoder
model = AE(HIDDEN_DIM, scale=sn_max, domain_size=domain_size, use_cuda=True).to(device)
model.load_state_dict(torch.load("./model.ckpt"))
model.eval()

# reconstruct snapshots
snap_rec = model.forward(snapshots).cpu().detach().numpy()

print("non linear reduction training coeffs: ", snap_rec.shape)
plot_compare(snapshots_numpy, snap_rec.reshape(-1, 3, domain_size, domain_size), n_train)

# evaluate hidden variables
nl_red_coeff = model.encoder.forward(snapshots)
print("non linear reduction training coeffs: ", nl_red_coeff.size())
nl_red_coeff = nl_red_coeff.cpu().detach().numpy()


##################################### TRAIN GPR
# load training inputs
array=[]
with open(WM_PROJECT+"ITHACAoutput/Offline/Training/mu_samples_mat.txt") as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
x = np.array(array_)
print("inputs shape: ", x.shape)

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

# perform GP regression
kern = GPy.kern.RBF(input_dim=2, ARD=True, lengthscale=0.4)
gp = GPy.models.GPRegression(x, nl_red_coeff)
gp.optimize_restarts(5)

##################################### TEST GPR
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

# get predictions with GPR
predictions = gp.predict(x_test)[0]
print("predictions shape: ", predictions.shape)

predicted_snapshots = model.decoder.forward(torch.from_numpy(predictions).to(device, dtype=torch.float)).cpu().detach().numpy().reshape(-1, n_train)
print("predicted snapshots shape: ", predicted_snapshots.shape)
np.save(WM_PROJECT+"snapshotsReconstructedConvAe.npy",predicted_snapshots)


#################################### PROJECTION ERROR
snap_true = np.load(WM_PROJECT+"npTrueSnapshots.npy")
print("true test snapshots shape: ", snap_true.shape)
n_test = snap_true.shape[1]
snap_true = snap_true.T
snap_true_numpy = snap_true.reshape((n_train, 3, domain_size, domain_size))
snap_true = torch.from_numpy(snap_true_numpy).to(device, dtype=torch.float)
print("snapshots shape: ", snap_true.size())

# reconstruct snapshots
snap_true_rec = model.forward(snap_true).cpu().detach().numpy()
snap_true_rec = snap_true_rec.reshape(-1, 3, domain_size, domain_size)
print("non linear reduction training coeffs: ", snap_true_rec.shape)
plot_compare(snap_true_numpy, snap_true_rec, n_train)
np.save(WM_PROJECT+"snapshotsConvAeTrueProjection.npy",snap_true_rec)