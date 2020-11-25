import numpy as np
import matplotlib.pyplot as plt

# ROM intrusive errors
error_L2_intrusive = np.load("./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy")

# non intrusive errors
error_L2_nonintrusive= np.load("./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy")
error_L2_nonintrusive_covae = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeNonIntrusive.npy")

# projection errors
error_L2_ROM_projection = np.load("./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy")
error_L2_CAE_projection = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeProjection.npy")


plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_intrusive, label="intrusive",  linewidth=4)
plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_nonintrusive, label="non-intrusive",  linewidth=4)
plt.plot(np.arange(error_L2_intrusive.shape[0]), error_L2_nonintrusive_covae, label="non-intrusive convolutional AE",  linewidth=4)
plt.plot(np.arange(error_L2_CAE_projection.shape[0]), error_L2_CAE_projection, label="projection error CAE",  linewidth=2)
plt.plot(np.arange(error_L2_CAE_projection.shape[0]), error_L2_ROM_projection, label="projection error ROM",  linewidth=2)

plt.legend()
plt.grid()
plt.title("L2 error")
plt.show()