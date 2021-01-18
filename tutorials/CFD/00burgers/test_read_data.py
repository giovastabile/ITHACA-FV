import numpy as np

test = np.load("x.npy")
print(test.shape)
print(test)

jac = np.load("jacobian.npy")
print(jac.shape)
print(jac)