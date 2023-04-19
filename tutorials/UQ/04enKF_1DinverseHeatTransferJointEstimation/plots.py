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

time = np.loadtxt("./ITHACAoutput/direct/trueTimeVec_mat.txt")
probe_true = np.loadtxt("./ITHACAoutput/direct/probe_true_mat.txt")
probe_rec = np.loadtxt("./ITHACAoutput/reconstruction/probe_rec_mat.txt")
state_min = np.loadtxt("./ITHACAoutput/reconstruction/probeState_minConf_mat.txt")
state_max = np.loadtxt("./ITHACAoutput/reconstruction/probeState_maxConf_mat.txt")
reconstructedBC = np.loadtxt("./ITHACAoutput/reconstruction/parameterMean_mat.txt")
param_min = np.loadtxt("./ITHACAoutput/reconstruction/parameter_minConf_mat.txt")
param_max = np.loadtxt("./ITHACAoutput/reconstruction/parameter_maxConf_mat.txt")
trueBC = np.loadtxt("./ITHACAoutput/direct/trueBC_mat.txt")

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
plt.legend()
plt.xlabel('Time [s]', fontsize=25)

fig = plt.figure(2,figsize=(8,6))
plt.plot(time, trueBC,"k--", linewidth = 2, label="trueBC")

plt.fill_between(time, param_min, param_max, color='k', alpha=.1)
plt.plot(time,reconstructedBC, linewidth = 2, color='k', label="recBC" )

#plt.fill_between(time, minConfidence, maxConfidence, color='b', alpha=.1)
#plt.plot(time,reconstructedBC, linewidth = 2, color='b', label="T rec" )

#plt.legend()
plt.xlabel('Time [s]', fontsize=25)
plt.grid()
plt.legend()


plt.show()
