import numpy as np
import matplotlib.pyplot as plt

# ROM intrusive errors
error_L2_intrusive = np.load("./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy")
error_L2_intrusive_NMLSPGCentral = np.load("./ITHACAoutput/ErrorsL2/errL2UNMLSPGCentral.npy")
error_L2_intrusive_NMLSPGFor = np.load("./ITHACAoutput/ErrorsL2/errL2UNMLSPGFor.npy")
error_L2_intrusive_NMLSPGTrue = np.load("./ITHACAoutput/ErrorsL2/errL2UNMLSPGTrue.npy")

# non intrusive errors
error_L2_nonintrusive= np.load("./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy")
error_L2_nonintrusive_covae = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeNonIntrusive.npy")

# projection errors
error_L2_ROM_projection = np.load("./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy")
error_L2_CAE_projection = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeProjection.npy")


plt.plot(np.arange(error_L2_intrusive.shape[0])[1:], np.log10(error_L2_intrusive)[1:], label="intrusive",  linewidth=4)
plt.plot(np.arange(error_L2_intrusive.shape[0])[1:], np.log10(error_L2_nonintrusive)[1:], label="non-intrusive",  linewidth=4)
plt.plot(np.arange(error_L2_intrusive.shape[0])[1:], np.log10(error_L2_nonintrusive_covae)[1:], label="non-intrusive convolutional AE",  linewidth=4)
plt.plot(np.arange(error_L2_CAE_projection.shape[0])[1:], np.log10(error_L2_CAE_projection)[1:], label="projection error CAE",  linewidth=2)
plt.plot(np.arange(error_L2_CAE_projection.shape[0])[1:], np.log10(error_L2_ROM_projection)[1:], label="projection error ROM",  linewidth=2)
plt.plot(np.arange(error_L2_CAE_projection.shape[0])[1:], np.log10(error_L2_intrusive_NMLSPGCentral)[1:], label="NM-LSPG-Central",  linewidth=2)
plt.plot(np.arange(error_L2_CAE_projection.shape[0]), np.log10(error_L2_intrusive_NMLSPGFor), label="NM-LSPG-Forward",  linewidth=2)
plt.plot(np.arange(error_L2_CAE_projection.shape[0]), np.log10(error_L2_intrusive_NMLSPGTrue), label="NM-LSPG-TrueJacobian",  linewidth=2)

plt.legend()
plt.grid()
plt.title("L2 relative error")
plt.show()