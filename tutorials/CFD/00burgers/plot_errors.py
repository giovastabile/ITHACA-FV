import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# ROM intrusive errors
error_L2_intrusive = np.load("./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy")
error_L2_intrusive_NMLSPGCentral = np.load("./ITHACAoutput/ErrorsL2/errL2UNMLSPGCentral.npy")
error_L2_intrusive_NMLSPGTrue = np.load("./ITHACAoutput/ErrorsL2/errL2UNMLSPGTrue.npy")
error_consistency = np.load("./ITHACAoutput/ErrorsL2/errConsistency.npy")
# non intrusive errors
error_L2_nonintrusive= np.load("./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy")
error_L2_nonintrusive_covae = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeNonIntrusive.npy")

# projection errors
error_L2_ROM_projection = np.load("./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy")
error_L2_CAE_projection = np.load("./Autoencoders/ConvolutionalAe/errL2UconvAeProjection.npy")


plt.semilogy(np.arange(error_L2_intrusive.shape[0])[1:]/1000, error_L2_intrusive[1:], label="intrusive",  linewidth=4)
plt.semilogy(np.arange(error_L2_intrusive.shape[0])[1:]/1000, error_L2_nonintrusive[1:], label="non-intrusive",  linewidth=4)
plt.semilogy(np.arange(error_L2_intrusive.shape[0])[1:]/1000, error_L2_nonintrusive_covae[1:], label="non-intrusive convolutional AE",  linewidth=4)
plt.semilogy(np.arange(error_L2_CAE_projection.shape[0])[1:]/1000, error_L2_CAE_projection[1:], label="projection error CAE",  linewidth=2)
plt.semilogy(np.arange(error_L2_CAE_projection.shape[0])[1:]/1000, error_L2_ROM_projection[1:], label="projection error ROM",  linewidth=2)
plt.semilogy(np.arange(error_L2_CAE_projection.shape[0])[1:]/1000, error_L2_intrusive_NMLSPGCentral[1:], label="NM-LSPG-Central",  linewidth=2)

plt.semilogy(np.arange(error_L2_CAE_projection.shape[0])[1:]/1000, error_L2_intrusive_NMLSPGTrue[1:], label="NM-LSPG-TrueJacobian",  linewidth=2)
plt.semilogy(np.arange(error_consistency.shape[0])[1:]/1000, error_consistency[1:], label="non-intrus-cae-lstm consistency",  linewidth=2)

plt.legend()
plt.ylim([1e-4,1e-0])
plt.grid(True, which="both")
plt.xlabel("time instants [s]")
plt.ylabel(" log10 relative L2 error")
plt.title("Error of test sample for reduced Burgers' PDE\n reduced dimension is 4")
plt.show()