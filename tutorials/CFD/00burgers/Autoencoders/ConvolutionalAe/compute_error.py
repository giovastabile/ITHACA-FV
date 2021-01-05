import numpy as np
from convae import *
import torch
print(torch.__version__)
WM_PROJECT = "../../"

domain_size = 60

# load reconstructed snapshots and reshape as (train_samples, channel, x, y)
snapshots_predicted = np.load(WM_PROJECT+"snapshotsReconstructedConvAe.npy")
n_test = snapshots_predicted.shape[0]
err_pred = snapshots_predicted.reshape(n_test, 2, domain_size, domain_size)
print("snapshots shape: ", err_pred.shape)
# plot_snapshot(err_pred, idx=1000)

# load true test snapshots
snapshots_true = np.load(WM_PROJECT+"npTrueSnapshots.npy")
err_snap = snapshots_true.T
err_snap = err_snap.reshape(n_test, 3, domain_size, domain_size)[:, :2, :, :]
print("snapshots shape: ", snapshots_true.shape, snapshots_predicted.shape, err_pred.shape)
plot_snapshot(err_snap, idx=1000)
plt.show()

# evaluate error snapshots on the whole domain and plot them
err = np.abs(err_pred-err_snap)
# x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
# z = err[50, 0, x, y] # change first index to change snapshot
# plt.contourf(x, y, z)
# plt.colorbar()
# plt.title("Test error on non-intrusive CAE")
# plt.show()

# evaluate L2 relative error and save it
error = np.linalg.norm(err.reshape(n_test, -1), axis=1)
norm = np.linalg.norm(snapshots_true, axis=0)
error = error/norm
print("error shape", error.shape)
np.save("errL2UconvAeNonIntrusive.npy", error)

# evaluate max relative error and save it
err_max = np.max(np.abs(err.reshape(n_test, -1)), axis=1)
norm_max = np.max(np.abs(snapshots_true), axis=0)
error_max = err_max/norm_max
print("error max: ", np.max(error_max), np.min(error_max))

################################################## COMPUTE PROJECTION ERROR

err_snap_proj_true = np.load(WM_PROJECT+"snapshotsConvAeTrueProjection.npy")
err_snap_proj_true = err_snap_proj_true.reshape((n_test, 2, domain_size, domain_size))
plot_snapshot(err_snap_proj_true, 1000)

err = np.abs(err_snap_proj_true-err_snap)
error_proj = np.linalg.norm(err.reshape(n_test, -1), axis=1)
norm = np.linalg.norm(snapshots_true, axis=0)
error_proj = error_proj/norm

x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
z = error_proj[1000, 0, x, y]
plt.contourf(x, y, z)
plt.colorbar()
plt.title("Projection error")
plt.show()

print("projection error shape", error_proj.shape)
np.save("errL2UconvAeProjection.npy", error_proj)