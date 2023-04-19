import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import sys
sys.path.insert(0, "./")

#plt.style.use('classic')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

time = np.loadtxt("./ITHACAoutput/true/trueTimeVec_mat.txt")
probe_true = np.loadtxt("./ITHACAoutput/true/probe_true_mat.txt")
probe_rec = np.loadtxt("./ITHACAoutput/reconstruction/probe_rec_mat.txt")
gTrue_probe = np.loadtxt("./ITHACAoutput/reconstruction/gTrue_probe_mat.txt")
gRec_probe = np.loadtxt("./ITHACAoutput/reconstruction/gRec_probe_mat.txt")
state_min = np.loadtxt("./ITHACAoutput/reconstruction/probeState_minConf_mat.txt")
state_max = np.loadtxt("./ITHACAoutput/reconstruction/probeState_maxConf_mat.txt")
reconstructedBC = np.loadtxt("./ITHACAoutput/reconstruction/parameterMean_mat.txt")
param_min = np.loadtxt("./ITHACAoutput/reconstruction/parameter_minConf_mat.txt")
param_max = np.loadtxt("./ITHACAoutput/reconstruction/parameter_maxConf_mat.txt")
trueBC = np.loadtxt("./ITHACAoutput/true/trueBC_mat.txt")

print state_min.size
print state_max.size
print probe_rec.size
#minConfidence = np.loadtxt("./ITHACAoutput/reconstuction/probe_minConfidence_mat.txt")
#maxConfidence = np.loadtxt("./ITHACAoutput/reconstuction/probe_MaxConfidence_mat.txt")


fig = plt.figure(1,figsize=(8,6))
plt.plot(time, probe_true,"b--", linewidth = 2, label="trueT")

plt.fill_between(time, state_min, state_max, color='b', alpha=.1)
plt.plot(time,probe_rec, linewidth = 2, color='b', label="T rec" )
plt.grid()
#plt.legend()
plt.xlabel('Time [s]', fontsize=25)

fig = plt.figure(2,figsize=(8,6))
#plt.plot(time, trueBC,"k--", linewidth = 2, label="trueBC")


for i in range(reconstructedBC.shape[0]):
    plt.plot(time, reconstructedBC[i,:], label=i)
#plt.fill_between(time, param_min, param_max, color='k', alpha=.1)

#plt.fill_between(time, minConfidence, maxConfidence, color='b', alpha=.1)
#plt.plot(time,reconstructedBC, linewidth = 2, color='b', label="T rec" )

#plt.legend()
plt.xlabel('Time [s]', fontsize=25)
plt.grid()
plt.legend()


fig = plt.figure(3,figsize=(8,6))
plt.plot(time, gRec_probe,"k--", linewidth = 2, label="gRec")

plt.plot(time, gTrue_probe, linewidth = 2, color='b', label="gTrue" )
plt.grid()
plt.legend()
plt.xlabel('Time [s]', fontsize=25)

plt.show()
