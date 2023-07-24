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

time = np.loadtxt("./ITHACAoutput/true/trueTimeVec_mat.txt")                                  # Loading time vector [100 equal to the number of time steps]

#⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬ Plotting reconstructed mean temperature and true temperature at a specific probe(1, 0.02, 0.6) over time.
probe_true = np.loadtxt("./ITHACAoutput/true/probe_true_mat.txt")                             # Reading true temprature[100 equal to the number of time steps] file at the probe  
probe_rec =  np.loadtxt("./ITHACAoutput/reconstruction/probe_rec_mat.txt")                    # Reading reconstructed mean[100 equal to the number of time steps] temprature file at the probe 
state_min =  np.loadtxt("./ITHACAoutput/reconstruction/probeState_minConf_mat.txt")           # Reading the 5th percentile value of state ensemble at different time steps 
state_max =  np.loadtxt("./ITHACAoutput/reconstruction/probeState_maxConf_mat.txt")           # Reading the 95th  percentile value of state ensemble at different time steps 

print(state_min.size) # It must be equal to the number of time steps 
print(state_max.size) # It must be equal to the number of time steps 
print(probe_rec.size) # It must be equal to the number of time steps 

fig = plt.figure(1,figsize=(8,6))
plt.plot(time, probe_true,"b", linewidth = 2, label="Ttrue Probe")
plt.plot(time, probe_rec, linewidth=2, color='k', linestyle='--', label='Trec Probe')



# Using plt.fill_between to visualize uncertainty or confidence intervals in the state values by filling the area between state_min and  state_max curves representing the 5th and 95th percentile values of the state ensemble at different time steps, respectively.
plt.fill_between(time, state_min, state_max, color='b', alpha=.1, label='Confidence Interval') # Fill the area between state_min and state_max with a specified color and transparency.


# Set labels and title
plt.grid()
plt.legend()
plt.xlabel('Time [s]', fontsize=15)
plt.ylabel('Temperature[K] at a probe', fontsize=15)
plt.title('True and Reconstructed Mean Temperature at a Probe(1.7, 0.02, 0.24) over time.')
plt.show()
#⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫ Plotting reconstructed mean temperature and true temperature at a specific probe(1, 0.02, 0.6) over time.

#⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬ Plotting reconstructed mean heat flux and true heat flux at a specific probe (1, 0, 0.06) on the hotside boundary condition.
gTrue_probe = np.loadtxt("./ITHACAoutput/reconstruction/gTrue_probe_mat.txt")               # Reading the true heat flux file at the probe
gRec_probe = np.loadtxt("./ITHACAoutput/reconstruction/gRec_probe_mat.txt")                 # Reading the reconstructed mean heat flux at the probe


fig = plt.figure(3,figsize=(8,6))
plt.plot(time, gTrue_probe,      linewidth = 2, color='b', label="gTrue Probe" )
plt.plot(time, gRec_probe,"k--", linewidth = 2, label="gRec Probe")
plt.grid()
plt.legend()
plt.xlabel('Time [s]', fontsize=15)
#plt.ylabel('Heat Flux[w/m^2]', fontsize=25)
plt.ylabel(r'Heat Flux [$\mathrm{W/m^2}$] at a probe', fontsize=15)
plt.title('True and Reconstructed Mean Heat Flux at a Probe(1.7, 0.0, 0.24) over time.')
plt.show()
#⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫ Plotting reconstructed mean heat flux and true heat flux at a specific probe (1, 0, 0.06) on the hotside boundary condition.


#⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬⏬ Plotting reconstructed mean heat flux and true heat flux at the hotSide BC over time.

reconstructedBC  = np.loadtxt("./ITHACAoutput/reconstruction/parameterMean_mat.txt")                       # This matrix[100(mean weight),100(timesteps)] stores the reconstructed mean weight at each face (100) at each time steps (100)
heatFluxSpaceRBF = np.loadtxt("./ITHACAoutput/projection/HeatFluxSpaceRBF/heat_flux_space_basis_mat.txt")  # This matrix[100,400(faces)] stores heatFluxSpaceBasis data

# By using RBF formula with the help of two above matrix, we create a matrix named out containing the reconstructed heat flux for each face at each time step.
n, m = heatFluxSpaceRBF.shape
n1, m1 = reconstructedBC.shape

out = np.zeros((m1, m))

for i in range(m1):
    for j in range(m):
        out[i, j] = np.sum(reconstructedBC[:, i] * heatFluxSpaceRBF[:, j])            # This matrix out [100(timesteps), 400(faces)]
# out is the reconstructed heat flux matrix
out = np.transpose(out)                                                               # After transposing, out [400(faces), 100(timesteps)]
column_sums2 = out.sum(axis=0)  # Sum of each column of the matrix out. Therefore, each element of the resulted vector represents the total recunstructed heat flux at the hotSide BC at each time step

# trueBC = np.loadtxt("./ITHACAoutput/true/trueBC_mat.txt")                           # Why this vector is empty?
gTrue = np.loadtxt("./ITHACAoutput/projection/TrueHeatFlux/HeatFluxTrue_mat.txt")     # This matrix[, ]
gTrue = np.transpose(gTrue)                                                           # After transposing, gTrue [400(faces), 101(timesteps)]
column_sums1 = gTrue.sum(axis=0)  # Integral or sum of each column of the matrix gTrue. Therefore, each element of the resulted vector represents the true heat flux at the hotSide BC at each time step


fig = plt.figure(2,figsize=(8,6))
plt.plot(time, column_sums1[1:],"b", linewidth = 2, label="True Heat Flux")
plt.plot(time, column_sums2, "k--" , linewidth = 2, label="Reconstructed Heat Flux")

plt.grid()
plt.legend()
plt.xlabel('Time [s]', fontsize=15)
plt.ylabel(r'Heat Flux [$\mathrm{W/m^2}$] at hotSide BC', fontsize=15)
plt.title('True and Reconstructed Mean Heat Flux at the hotSide.')
plt.show()


param_min = np.loadtxt("./ITHACAoutput/reconstruction/parameter_minConf_mat.txt")     # This matrix[100(5th of each ensemble weight),100(timesteps)] stores the 5th  percentile value of the parameter(weight) ensemble at different time steps
param_max = np.loadtxt("./ITHACAoutput/reconstruction/parameter_maxConf_mat.txt")     # This matrix[100(95th of each ensemble weight),100(timesteps)] stores the 95th percentile value of the parameter(weight) ensemble at different time steps
param_min_vector = param_min.sum(axis=0)  # Sum each column of the param_min matrix. Therefore, each element of the resulted vector represents the total 5th  percentile value of the parameter(weight)ensemble at the hotSide BC at each time step
param_max_vector = param_max.sum(axis=0)  # Sum each column of the param_min matrix. Therefore, each element of the resulted vector represents the total 95th  percentile value of the parameter(weight) ensemble at the hotSide BC at each time step

out1 = np.zeros((m1, m))
out2 = np.zeros((m1, m))
for i in range(m1):
    for j in range(m):
        out1[i, j] = np.sum(param_min[:, i] * heatFluxSpaceRBF[:, j])                 # This matrix[100(timesteps), 400(heat flux min confidence)]
        out2[i, j] = np.sum(param_max[:, i] * heatFluxSpaceRBF[:, j])                 # This matrix[100(timesteps), 400(heat flux min confidence)]
out1 = np.transpose(out1)                                                             # After transposing, out1 [400(heat flux min confidence), 100(timesteps)]
out2 = np.transpose(out2)                                                             # After transposing, out2 [400(heat flux min confidence), 100(timesteps)]
column_sums3 = out1.sum(axis=0) # Sum of each column of the matrix out1. Therefore, each element of the resulted vector represents the total heat flux min confidence (5th)  at the hotSide BC at each time step
column_sums4 = out2.sum(axis=0) # Sum of each column of the matrix out2. Therefore, each element of the resulted vector represents the total heat flux max confidence (95th) at the hotSide BC at each time step


fig = plt.figure(4,figsize=(8,6))
plt.plot(time, column_sums1[1:],"b", linewidth = 2, label="True Heat Flux")
plt.plot(time, column_sums2, "k--" , linewidth = 2, label="Reconstructed Heat Flux")
# Using plt.fill_between to visualize uncertainty or confidence intervals in the heatflux values by filling the area between heat flux min confidence and  heat flux max confidence curves representing the 5th and 95th percentile values of the heat flux parameter ensemble at different time steps, respectively.
plt.fill_between(time, column_sums3, column_sums4 , color='b', alpha=.1, label='Confidence Interval')

#minConfidence = np.loadtxt("./ITHACAoutput/reconstuction/probe_minConfidence_mat.txt")   # not available
#maxConfidence = np.loadtxt("./ITHACAoutput/reconstuction/probe_MaxConfidence_mat.txt")   # not available

#for i in range(reconstructedBC.shape[0]):                                                # not required
    #plt.plot(time, reconstructedBC[i,:])

    
    
#plt.fill_between(time, minConfidence, maxConfidence, color='b', alpha=.1)                # not available

# Using plt.fill_between to visualize uncertainty or confidence intervals in the parameter values by filling the area between param_min and  param_max curves representing the 5th and 95th percentile values of the parameter ensemble at different time steps, respectively.
#plt.fill_between(time, param_min_vector, param_max_vector, color='b', alpha=.1, label='Confidence Interval') # Fill the area between param_min and param_max with a specified color and transparency.


plt.grid()
plt.legend()
plt.xlabel('Time [s]', fontsize=15)
plt.ylabel(r'Heat Flux [$\mathrm{W/m^2}$] at hotSide BC', fontsize=15)
plt.title('True and Reconstructed Mean Heat Flux at the hotSide.')
plt.show()
#⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫⏫ Plotting reconstructed mean heat flux and true heat flux at the hotSide BC over time.

