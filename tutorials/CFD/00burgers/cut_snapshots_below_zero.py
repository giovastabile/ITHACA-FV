import numpy as np
domain_size = 60
snapshots = np.load("./npSnapshots.npy")
print(snapshots.shape, np.min(snapshots), np.max(snapshots))
snapshots = snapshots.clip(min=0)
print(snapshots.shape, np.min(snapshots), np.max(snapshots))
np.save("./npSnapshots.npy", snapshots)

snapshots = np.load("./npTrueSnapshots.npy")
print(snapshots.shape, np.min(snapshots), np.max(snapshots))
snapshots = snapshots.clip(min=0)
print(snapshots.shape, np.min(snapshots), np.max(snapshots))
np.save("./npTrueSnapshots.npy", snapshots)