import numpy as np
import matplotlib.pyplot as plt
import GPy
import torch.nn as nn

# load training inputs
array=[]
with open('ITHACAoutput/Offline/Training/mu_samples_mat.txt') as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
x = np.array(array_)
print("inputs shape: ", x.shape)

train_params = np.load("parTrain.npy").reshape(-1)
n_train_params = train_params.shape[0]

# loading training outputs
output_pre = np.load('ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
n_time_samples_times_n_param = output_pre.shape[0]
n_time_samples = n_time_samples_times_n_param//n_train_params
print("number of time samples: ", n_time_samples)

time_samples = output_pre[:n_time_samples, 0]
output = output_pre[:, 1:]
print("outputs shape: ", output.shape)
print("time samples: ", time_samples)

# perform GP regression
kern = GPy.kern.RBF(input_dim=2, ARD=True, lengthscale=0.05)
gp = GPy.models.GPRegression(x[:, :2], output[:, :1])
gp.optimize_restarts(1)
gp.plot()
plt.show()
# load test set
test_params = np.load("parTest.npy").reshape(-1)
print("test parameters shape: ", test_params.shape, test_params)

# prepare test set, first coordinate time, second coordinate parameters
x_test_0 = np.kron(np.ones(test_params.shape[0]).reshape(-1, 1),  time_samples.reshape(-1, 1))
print("times column shape: ", x_test_0.shape)

x_test_1 = np.kron(test_params.reshape(-1, 1), np.ones((time_samples.shape[0], 1)))
print("parameters column shape: ", x_test_1.shape)

x_test = np.hstack((x_test_0, x_test_1))
print("test dataset shape: ", x_test.shape)

# get predictions with GPR
predictions = gp.predict(x_test)[0]
print("predictions shape: ", predictions.shape)
predictions = np.hstack((x_test_0, predictions))
print("predictions and timings shape: ", predictions.shape)
np.save("nonIntrusiveCoeff.npy", predictions)
tmp = np.load("nonIntrusiveCoeff.npy")

