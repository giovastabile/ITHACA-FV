import numpy as np
import matplotlib.pyplot as plt
import GPy

# load training inputs
array=[]
with open('ITHACAoutput/Offline/Training/mu_samples_mat.txt') as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
x = np.array(array_)
print("inputs shape: ", x.shape)

# loading training outputs
output_pre = np.load('ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
time_samples = output_pre[:, 0]
output = output_pre[:, 1:]
print("outputs shape: ", output.shape)
print("time samples: ", time_samples)

# perform GP regression
kern = GPy.kern.RBF(input_dim=2, ARD=True)
gp = GPy.models.GPRegression(x, output)
gp.optimize_restarts(15)

# load test set
test_params = np.load("parTest.npy").reshape(-1)
print("test parameters shape: ", test_params.shape)

# prepare test set, first coordinate time, second coordinate parameters
x_test_0 = np.repeat(time_samples.reshape(-1, 1), test_params.shape[0], axis=0)
x_test = np.hstack((x_test_0, np.kron(test_params.reshape(-1, 1), np.ones((time_samples.shape[0], 1)))))
print("test dataset shape: ", x_test.shape)

# get predictions with GPR
predictions = gp.predict(x_test)[0]
print("predictions shape: ", predictions.shape)
predictions = np.hstack((x_test_0, predictions))
print("predictions and timings shape: ", predictions.shape)
np.save("nonIntrusiveCoeff.npy", predictions)
tmp = np.load("nonIntrusiveCoeff.npy")

