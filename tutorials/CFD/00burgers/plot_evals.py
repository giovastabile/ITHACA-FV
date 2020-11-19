import numpy as np
import matplotlib.pyplot as plt

snapshots = np.load("npSnapshots.npy").reshape(3, 10000, -1)
print(snapshots.shape)

array=[]
with open('ITHACAoutput/Offline/mu_samples_mat.txt') as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
params = np.array(array_)
print(params.shape)

array=[]
with open('ITHACAoutput/POD/Eigenvalues_U') as f:
    for i, line in enumerate(f):
        if i>1:
            array.append(*line.split(','))

array_ = [float(item) for item in array]
eigenvals = np.array(array_)

plt.plot(range(eigenvals.shape[0])[:200], eigenvals[:200])
plt.xlabel("eigenvalue number")
plt.ylabel("log10 scaled eigenvalues")
plt.yscale("log")
plt.yticks(np.array([10**-20, 10**-15, 10**-10, 10**-5, 1, 10**5, 10**10]))
plt.grid(True)
plt.show()