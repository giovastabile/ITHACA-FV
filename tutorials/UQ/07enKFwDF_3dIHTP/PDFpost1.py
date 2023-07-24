import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import sys
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal

# Set matplotlib parameters
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
############################ Constant

NumEns = 120        # Number of seeds (samples).                      To see this value please see the Nsamples variable in main c file .  All the ensemble-based methods, in general, tend to converge as we increase the number of samples. Therefore, the number of samples should be as high as possible (the more sample we have usually the more accurate)
NumWeight = 100     # Number of time-dependent basis function weight. To see this value please see the sizeOfParameter variable in the main c file.
Ntime = 100         # Number of time steps                            Please see the controlDIic file in system folder. endTime/deltaT
deltaT = 0.2                                                         # Please see the controlDict file in system folder
observationInterval = 2                                              # Please see the measurementsDict in constant folder
############################ Constant
path1 = "/u/k/kbakhsha/ITHACA-FV-KF/tutorials/UQ/15enKFwDF_3dIHTP/ITHACAoutput/projection/HFWposterior"

mean_values = []     # List to store mean values of specified weight for each time step
std_dev_values = []  # List to store standard deviation values for each time step


############################ Iterate over folders (stored at each time step) and plot posterior probability distributions and bell curve for the weight number (specified with value of a variable named 'Weight' inside the loop)
for i in range(0, Ntime, observationInterval):  # Must be an even number between 0 and 98 because observation is available every two time steps.
    folder_number = i if i <= 98 else 98 # this line ensures that the folder number is capped at 98 to prevent any errors when constructing the folder path.
    folder_path = os.path.join(path1 + str(folder_number), "heatFlux_weightsPosterior_mat.txt")
    
    # Read the file using numpy
    HFWposterior = np.loadtxt(folder_path)
    
    # Plotting the PDF
    Weight = 79  # 0 <= Weight < NumWeight, each row represents the number of ensemble for basis function weights 
    wForPlot = HFWposterior[Weight, :]
    
    # Compute the probability distribution
    values, counts = np.unique(wForPlot, return_counts=True)#The argument return_counts=True instructs the function to return both the unique values(values) in the array (wForPlot) and the number of times each value appears (counts). 
    probability = counts / len(wForPlot)
    
    # Compute the mean and standard deviation
    mean = np.mean(wForPlot)
    std_dev = np.std(wForPlot)
    
    # Store the mean and standard deviation values
    mean_values.append(mean)
    std_dev_values.append(std_dev)
    
    # Generate values for the bell curve
    x = np.linspace(np.min(wForPlot), np.max(wForPlot), 100)
    y = norm.pdf(x, mean, std_dev)  # Probability density function 
    
    # Plot the bell curve and PDF
    #plt.plot(x, y, label='Bell Curve for Weight number {}'.format(Weight + 1))
    plt.plot(x, y)
    #plt.bar(values, probability, alpha=0.5) # The value of alpha ranges between 0 and 1. alpha=0.5 sets the transparency of the bars to 50%, allowing some of the underlying plot or data to be visible through the bars. This can help with visual clarity when there are overlapping bars or when you want to emphasize the background plot.
############################ Iterate over folders and plot posterior probability distributions and bell curve for the weight number (specified with Weight value inside the loop)

############################ Read the prior mean and covariance file using numpy and Plot the prior destribution of the specified weight

path2 = "/u/k/kbakhsha/ITHACA-FV-KF/tutorials/UQ/15enKFwDF_3dIHTP/ITHACAoutput/projection/PriorMeanCovariance"
folder_path2 = os.path.join(path2, "prior_weights_Cov_mat.txt")
folder_path3 = os.path.join(path2, "prior_weights_Mean_mat.txt")

prior_weights_Cov = np.loadtxt(folder_path2)
prior_weights_Mean = np.loadtxt(folder_path3)

# Define the mean and corresponding covariance value of the prior specified weight
mean1 = prior_weights_Mean[Weight]
variance1 = prior_weights_Cov[Weight, Weight]

print("The prior mean value for the ensemble weight number {} is {}.".format(Weight+1, mean1)) 
print("The prior variance value for the ensemble weight number {} is {}.".format(Weight+1, variance1))
print("The prior satandard deviation value for the ensemble weight number {} is {}.".format(Weight+1, np.sqrt(variance1)))


# Define the univariate normal distribution for the prior specified weight
univariate_normal = multivariate_normal(mean=mean1, cov=variance1)

# Create an array of x values
x1 = np.linspace(mean1 - 3*np.sqrt(variance1), mean1 + 3*np.sqrt(variance1), 100)

# Compute the probability density at each x value
pdf1 = univariate_normal.pdf(x1)

# Plot the PDF
label_text1 = 'Prior PDF for the Weigh Number {}'.format(Weight+1)
plt.plot(x1, pdf1, color='red', linestyle='--', marker='*', label=label_text1)
plt.legend()
############################ Read the prior mean and covariance file using numpy and Plot the prior destribution of the specofied weight    

############################# Set labels and title

plt.xlabel('Values')
plt.ylabel('Probability')
plt.title('Posterior PDF and Bell Curve for the Weight Number {} Over Observation Time'.format(Weight+1))
#plt.legend()

# Show the plot
plt.show()
############################ Set labels and title

############################ Plot the mean and standard deviation values over observation time.
# Create a separate figure for mean values
plt.figure()
time = np.array(range(0, Ntime, observationInterval)) * deltaT + deltaT
label_text2 = 'Prior Mean Value for Ensemble Weight Number {}'.format(Weight+1)
plt.plot(0, mean1, marker='*', color='red', markersize=10, label=label_text2)

# Add text annotation for the value of mean1
value_format = '{:.2f}'  # Formats the value with 2 decimal places
text2 = value_format.format(mean1)
plt.text(0, mean1, text2, ha='left', va='bottom')


plt.plot(time, mean_values)

# Calculating and plotting the mean of mean_values - Mean of the Posterior Mean Variation for the weight number
mean_value = sum(mean_values) / len(mean_values)
plt.axhline(mean_value, color='red', linestyle='--', label='Mean of Posterior Mean Variation')
print("Mean of the Posterior Mean Variation for the weight number {} is {}.".format(Weight+1, mean_value)) 




plt.xlabel('Observation Time (sec)')
plt.ylabel('Mean')
plt.title('Posterior Mean Variation for Ensemble Weight Number {}'.format(Weight+1))
plt.legend()
plt.grid()

# Create a separate figure for standard deviation values
plt.figure()
label_text3 = 'Prior Standard Deviation Value for Ensemble Weight Number {}'.format(Weight+1)
plt.plot(0, np.sqrt(variance1), marker='*', color='red', markersize=10, label=label_text3)
# Add text annotation for the value of standard deviation
value_format = '{:.2f}'  # Formats the value with 2 decimal places
text3 = value_format.format(np.sqrt(variance1))
plt.text(0, np.sqrt(variance1), text3, ha='left', va='bottom')

plt.plot(time, std_dev_values)

# Calculating and plotting the mean of std_dev_values - Mean of the Fluctuated Posterior Standard Deviation Variation
mean_value_SD = sum(std_dev_values) / len(std_dev_values)
plt.axhline(mean_value_SD, color='red', linestyle='--', label='Mean of the Fluctuated Posterior Standard Deviation Variation')
print("Mean of the Fluctuated Posterior Standard Deviation Variation for the weight number {} is {}.".format(Weight+1, mean_value_SD)) 

plt.xlabel('Observation Time (sec)')
plt.ylabel('Standard Deviation')
plt.title('Posterior Standard Deviation Variation for Ensemble Weight Number {}'.format(Weight+1))
plt.legend()
plt.grid()

# Show the plots
plt.show()
############################ Plot the mean and standard deviation values
