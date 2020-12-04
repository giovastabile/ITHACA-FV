import numpy as np
from convae import *
import torch
print(torch.__version__)
WM_PROJECT = "../../"


# reshape as (train_samples, channel, y, x)
snapshots_predicted = np.load(WM_PROJECT+"snapshotsReconstructedConvAe.npy")
n_test = snapshots_predicted.shape[1]
err_pred = snapshots_predicted.reshape((n_test, 3, 150, 150))
plot_snapshot(err_pred, 500)
plt.show()

# load true test snapshots
snapshots_true = np.load(WM_PROJECT+"npTrueSnapshots.npy")
err_snap = snapshots_true.T
err_snap = err_snap.reshape((n_test, 3, 150, 150))
print("snapshots shape: ", snapshots_true.shape, snapshots_predicted.shape, err_pred.shape)

# evaluate error snapshots on the whole domain and plot them
err = np.abs(err_pred-err_snap)
x, y = np.meshgrid(np.arange(150), np.arange(150))
z = err[50, 0, x, y] # change first index to change snapshot
plt.contourf(x, y, z)
plt.colorbar()
plt.title("Test error on non-intrusive CAE")
plt.show()

# evaluate relative error and save it
error = np.linalg.norm(err.reshape(n_test, -1), axis=1)
norm = np.linalg.norm(snapshots_true, axis=0)
error = error/norm
print("error shape", error.shape)
np.save("errL2UconvAeNonIntrusive.npy", error)

################################################## COMPUTE PROJECTION ERROR

err_snap_proj_true = np.load(WM_PROJECT+"snapshotsConvAeTrueProjection.npy")
err_snap_proj_true = err_snap_proj_true.reshape((n_test, 3, 150, 150))
err = np.abs(err_snap_proj_true-err_snap)
x, y = np.meshgrid(np.arange(150), np.arange(150))
z = err[50, 0, x, y]
plt.contourf(x, y, z)
plt.colorbar()
plt.title("Projection error")
plt.show()

error_proj = np.linalg.norm(err.reshape(n_test, -1), axis=1)
norm = np.linalg.norm(snapshots_true, axis=0)
error_proj = error_proj/norm
print("projection error shape", error_proj.shape)
np.save("errL2UconvAeProjection.npy", error_proj)