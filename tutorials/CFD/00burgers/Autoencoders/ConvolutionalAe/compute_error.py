import numpy as np

WM_PROJECT = "../../"

# reshape as (train_samples, channel, y, x)
snapshots_predicted = np.load(WM_PROJECT+"snapshotsReconstructedConvAe.npy").T
snapshots_true = np.load(WM_PROJECT+"npTrueSnapshots.npy").T
print("snapshots shape: ", snapshots_true.shape, snapshots_predicted.shape)

error = np.linalg.norm(snapshots_predicted-snapshots_true, axis=1)
norm = np.linalg.norm(snapshots_true, axis=1)
error = error/norm
print("error shape", error.shape)
np.save("errL2UconvAeNonIntrusive.npy", error)