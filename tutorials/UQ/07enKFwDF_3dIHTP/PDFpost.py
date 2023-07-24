import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import sys
import os
from scipy.stats import norm 

#plt.style.use('classic')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


NumEns = 120       # Number of seeds (samples).                      To see this value please see the Nsamples in main c file .
NumWeight = 100    # Number of time-dependent basis function weight. To see this value please see the sizeOfParameter in the main c file.
Ntime = 100        # Number of time steps                            Please see the controlDIic file in system folder. endTime/deltaT


TimeInterval = 98 # Must be an even number between 0 and 98, because observation is available every two time stes.

path1 = "/u/k/kbakhsha/ITHACA-FV-KF/tutorials/UQ/15enKFwDF_3dIHTP/ITHACAoutput/projection/HFWposterior"
path = os.path.join(path1 + str(TimeInterval), "heatFlux_weightsPosterior_mat.txt")

# Read the file using numpy
HFWposterior = np.loadtxt(path)

#plotting the Pdf
Weight = 9 # 0 <= Weight < NumWeight, each row represents the number of ensemble for basis function weights 
wForPlot=HFWposterior[Weight,:]

# Compute the probability distribution
values,counts = np.unique(wForPlot,return_counts=True)#The argument return_counts=True instructs the function to return both the unique values(values) in the array (wForPlot) and the number of times each value appears (counts). 
probability=counts/len(wForPlot)

# Compute the mean and standard deviation
mean=np.mean(wForPlot)
std_dev=np.std(wForPlot)

# Generate value for the bell curve
x=np.linspace(np.min(wForPlot),np.max(wForPlot),100)
y=norm.pdf(x,mean,std_dev) # Probability density function

# Plot the bell Curve and PDF
plt.plot(x,y,'r-',label='Bell Curve for the Weigh number {}'.format(Weight+1))
plt.bar(values,probability)

# Set labels and title
plt.xlabel('Values')
plt.ylabel('Probability')
plt.title('Probability Distributin and Bell Curve at time Interval {} '.format(TimeInterval))
plt.legend()

# Show the plot
plt.show()
