import numpy as np
import matplotlib.pyplot as plt
import GPy
import torch
import torch.nn as nn
from convae import *

WM_PROJECT = "../../"
HIDDEN_DIM = 4
domain_size = 60
DIM = 2
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("device is: ", device)

# reshape as (train_samples, channel, y, x)
snapshots = np.load(WM_PROJECT+"npSnapshots.npy")
print("snapshots shape: ", snapshots.shape)

# reshape snapshots
n_train = snapshots.shape[1]
snapshots = snapshots.T
snapshots_numpy = snapshots.reshape((n_train, 3, domain_size, domain_size))[:, :DIM,:, :]
print("max, min ", np.max(snapshots_numpy), np.min(snapshots_numpy))

# scale snapshots
sn_max = np.max(snapshots_numpy)
sn_min = np.min(snapshots_numpy)
snapshots_numpy = (snapshots_numpy-sn_min)/(sn_max-sn_min)
print("max, min ", np.max(snapshots_numpy), np.min(snapshots_numpy))
snapshots = torch.from_numpy(snapshots_numpy).to(device, dtype=torch.float)
print("snapshots shape: ", snapshots.size())

# load autoencoder
model = AE(HIDDEN_DIM, scale=(sn_min, sn_max), domain_size=domain_size, use_cuda=True).to(device)
model.load_state_dict(torch.load("./model.ckpt"))
model.eval()

# show filtersopen
# filters = model.encoder.layer1[0].weight.detach().cpu().numpy()
# print(filters.shape)
# for i in range(8):
#     for j in range(2):
#         plt.imshow(filters[i, j, :, :])
#         plt.show()

# show initial
inputs = torch.from_numpy(np.load("latent_initial.npy")).to(device, dtype=torch.float)
output = model.decoder.forward(inputs)
print("shapeoutput", output.shape)
# plot_snapshot(output.detach().cpu().numpy().reshape(1, 2, 60, 60), 0, idx_coord=1)

# reconstruct snapshots
snap_rec = model.forward(snapshots).cpu().detach().numpy()
print("non linear reduction training coeffs: ", snap_rec.shape)
# plot_compare(snapshots_numpy, snap_rec.reshape(-1, DIM, domain_size, domain_size), n_train)

# evaluate hidden variables
nl_red_coeff = model.encoder.forward(snapshots)
print("non linear reduction training coeffs: ", nl_red_coeff.size())
nl_red_coeff = nl_red_coeff.cpu().detach().numpy()

# test max error
err = snap_rec.reshape(-1, DIM, domain_size, domain_size)-snapshots_numpy*(sn_max-sn_min)+sn_min
plot_snapshot(err, 11000, idx_coord=1)
print(snapshots_numpy.shape, snap_rec.shape)
err_max = np.max(np.abs(err.reshape(n_train, -1)), axis=1)
norm_max = np.max(np.abs(snapshots_numpy.reshape(n_train, -1)), axis=1)
error_max = err_max/norm_max
print("error max: ", err_max, norm_max, error_max, np.max(error_max), np.min(error_max))

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
kern = GPy.kern.RBF(input_dim=2, ARD=True, lengthscale=0.04)
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
snap_true_numpy = snap_true.reshape(n_train, 3, domain_size, domain_size)[:, :2, :, :]
snap_true_numpy -= sn_min
snap_true_numpy /= sn_max-sn_min
snap_true = torch.from_numpy(snap_true_numpy).to(device, dtype=torch.float)
print("snapshots shape: ", snap_true.size())

# reconstruct snapshots
snap_true_rec = model.forward(snap_true).cpu().detach().numpy()
snap_true_rec = snap_true_rec.reshape(-1, 2, domain_size, domain_size)
print("non linear reduction training coeffs: ", snap_true_rec.shape)
plot_compare(snap_true_numpy, snap_true_rec, n_train)
np.save(WM_PROJECT+"snapshotsConvAeTrueProjection.npy",snap_true_rec)