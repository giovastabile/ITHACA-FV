import numpy as np
import matplotlib.pyplot as plt

error_L2_intrusive = np.load("./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy")
error_L2_nonintrusive= np.load("./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy")
error_L2_nonintrusive_covae = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeNonIntrusive.npy")


plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_intrusive, label="intrusive")
plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_nonintrusive, label="non-intrusive")
plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_nonintrusive_covae, label="non-intrusive convolutional AE")
plt.legend()
plt.grid()
plt.title("L2 error")
plt.show()