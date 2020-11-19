import numpy as np
import matplotlib.pyplot as plt

error_L2_intrusive = np.load("./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy")
error_L2_nonintrusive= np.load("./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy")


plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_intrusive, label="intrusive")
plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_nonintrusive, label="non-intrusive")
plt.legend()
plt.title("L2 error")
plt.show()