import numpy as np
import matplotlib.pyplot as plt

array=[]
with open('ITHACAoutput/POD/CumEigenvalues_U') as f:
    for i, line in enumerate(f):
        if i>1:
            array.append(*line.split(','))

array_ = [float(item) for item in array]
eigenvals = np.array(array_)
plt.plot(range(eigenvals.shape[0])[:200], eigenvals[:200])
plt.xlabel("eigenvalue number")
plt.ylabel("log10 scaled eigenvalues")
plt.yticks(np.arange(0.7, 1, 0.05))
plt.xticks(np.arange(0, 200, 5))
plt.grid(True)
plt.show()