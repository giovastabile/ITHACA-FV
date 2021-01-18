import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
from lstm import *
# load training inputs
array=[]
with open('../ITHACAoutput/Offline/Training/mu_samples_mat.txt') as f:
    for i, line in enumerate(f):
        array.append([*line.split()])

array_ = [[float(elem) for elem in item] for item in array]
x = np.array(array_)
print("inputs shape: ", x.shape)

# get the number of training params apart from time
train_params = np.load("../parTrain.npy").reshape(-1)
n_train_params = train_params.shape[0]

# loading training outputs
output_pre = np.load('../ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
n_time_samples_times_n_param = output_pre.shape[0]
n_time_samples = n_time_samples_times_n_param//n_train_params
print("number of time samples: ", n_time_samples)

time_samples = output_pre[:n_time_samples, 0]
output = output_pre[:, 1:]
print("outputs shape: ", output.shape)
print("time samples: ", time_samples)

# Device configuration
device = torch.device('cuda' if False else 'cpu')
print("device is: ", device)

# load lstm
input_dim = x.shape[1]
hidden_dim = output.shape[1]
n_layers = 1
model = model = ReducedCoeffsTimeSeries().to(device)
model.load_state_dict(torch.load("./model.ckpt"))
model.eval()

# load test set
test_params = np.load("../parTest.npy").reshape(-1)
print("test parameters shape: ", test_params.shape, test_params)

# prepare test set, first coordinate time, second coordinate parameters
x_test_0 = np.kron(np.ones(test_params.shape[0]).reshape(-1, 1),  time_samples.reshape(-1, 1))
print("times column shape: ", x_test_0.shape)

x_test_1 = np.kron(test_params.reshape(-1, 1), np.ones((time_samples.shape[0], 1)))
print("parameters column shape: ", x_test_1.shape)

x_test = np.hstack((x_test_0, x_test_1))
print("test dataset shape: ", x_test.shape)

# get predictions with lstm
x_test = x_test.reshape(test_params.shape[0], x_test.shape[0], x_test.shape[1])
print(x_test.shape)
predictions = model(torch.from_numpy(x_test).to(device, dtype=torch.float)).reshape(-1, output.shape[1])
print("predictions shape: ", predictions.shape)
predictions = np.hstack((x_test_0, predictions.detach().cpu().numpy().squeeze()))
print("predictions and timings shape: ", predictions.shape)
np.save("../nonIntrusiveCoeff.npy", predictions)

