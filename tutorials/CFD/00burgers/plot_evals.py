import numpy as np
import matplotlib.pyplot as plt

snapshots = np.load("npSnapshots.npy").reshape(3, 10000, -1)
print(snapshots.shape)

array=[]
with open('ITHACAoutput/POD/Eigenvalues_U') as f:
    for i, line in enumerate(f):
        if i>1:
            array.append(*line.split(','))

array_ = [float(item) for item in array]
print(array_)

eigenvals = np.array(array_)
print(eigenvals.shape)
plt.plot(range(eigenvals.shape[0])[:200], np.log(eigenvals)[:200])
plt.show()