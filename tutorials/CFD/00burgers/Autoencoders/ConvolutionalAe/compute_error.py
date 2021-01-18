import numpy as np
from convae import *
import torch
print(torch.__version__)
WM_PROJECT = "../../"

domain_size = 60

# load reconstructed snapshots and reshape as (train_samples, channel, x, y)
snapshots_predicted = np.load(WM_PROJECT+"snapshotsReconstructedConvAe.npy")
n_test = snapshots_predicted.shape[0]
snap_pred_frame = snapshots_predicted.reshape(n_test, 2, domain_size, domain_size)
print("predicted snapshots shape: ", snap_pred_frame.shape, n_test)
# plot_snapshot(snap_pred_frame, idx=1000)

# load true test snapshots
snapshots_true = np.load(WM_PROJECT+"npTrueSnapshots.npy").T
snap_true_frame = snapshots_true.reshape(n_test, 3, domain_size, domain_size)[:, :2, :, :]
snap_true_vec = snap_true_frame.reshape(n_test, -1)
print("true snapshots shape: ", snapshots_true.shape, snap_true_frame.shape, snap_true_vec.shape)
# plot_snapshot(snap_true_frame, idx=1000)

# evaluate error snapshots on the whole domain and plot them
err_rec = np.abs(snap_pred_frame-snap_true_frame)
print("absolute error shape: ", err_rec.shape)
x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
z = err_rec[50, 0, x, y] # change first index to change snapshot
plt.contourf(x, y, z)
plt.colorbar()
plt.title("Test error on non-intrusive CAE")
plt.show()
err_rec_vec = err_rec.reshape(n_test, -1)
print("absolute error vectorized shape: ", err_rec_vec.shape)


# evaluate L2 relative error and save it
error = np.linalg.norm(err_rec_vec, axis=1)
norm = np.linalg.norm(snap_true_vec, axis=1)
print("min max not normalized error L2", np.max(error), np.min(error))

error = error/norm
print("max and min L2 norm", np.max(norm), np.min(norm))
print("relative error shape", error.shape)
np.save("errL2UconvAeNonIntrusive.npy", error)

# evaluate max relative error and save it
err_max = np.max(err_rec_vec, axis=1)
norm_max = np.max(np.abs(snap_true_vec), axis=1)
error_max = err_max/norm_max
print("error max: ", np.max(err_max), np.min(err_max))
print("norm error max: ", np.max(norm_max), np.min(norm_max))
print("relative error max : ", np.max(error_max), np.min(error_max))

################################################## COMPUTE PROJECTION ERROR

err_snap_proj_true = np.load(WM_PROJECT+"snapshotsConvAeTrueProjection.npy")
snap_proj_frame = err_snap_proj_true.reshape(n_test, 2, domain_size, domain_size)
# plot_snapshot(snap_proj_frame, 1000)

err_abs = np.abs(snap_proj_frame-snap_true_frame)
err_vec = err_abs.reshape(n_test, -1)
error_proj = np.linalg.norm(err_vec, axis=1)
norm = np.linalg.norm(snap_true_vec, axis=1)
error_proj_rel = error_proj/norm
print("relative projection error shape", error_proj.shape)
print("relative projection error", np.log10(np.max(error_proj_rel)), np.log10(np.min(error_proj_rel)))
np.save("errL2UconvAeProjection.npy", error_proj_rel)

x, y = np.meshgrid(np.arange(domain_size), np.arange(domain_size))
z = err_abs[1000, 0, x, y]
plt.contourf(x, y, z)
plt.colorbar()
plt.title("Projection error")
plt.show()