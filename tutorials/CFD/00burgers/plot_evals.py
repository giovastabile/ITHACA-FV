import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

array=[]
with open('ITHACAoutput/Offline/Training/mu_samples_mat.txt') as f:
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

plt.semilogy(range(eigenvals.shape[0])[:200], eigenvals[:200])
plt.xlabel("eigenvalue number")
plt.ylabel("log10 scaled eigenvalues")
plt.yscale("log")
plt.title("POD eigenvalues decay")
# plt.yticks(np.array([10**-20, 10**-15, 10**-10, 10**-5, 1, 10**5, 10**10]))
plt.grid(True, which="both")
plt.ylim([1e-15,1e+1])
plt.show()