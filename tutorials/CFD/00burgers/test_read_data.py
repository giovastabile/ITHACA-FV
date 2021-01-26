import numpy as np

test = np.load("parTest.npy")
print(test.shape)
print(test)

jac = np.load("parTrain.npy")
print(jac.shape)
print(jac)