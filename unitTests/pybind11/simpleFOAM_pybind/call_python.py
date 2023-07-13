# %%
from simpleFOAM_pybind import simpleFOAM_pybind
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

resU = []
resP = []
# Instantiate OF object
a = simpleFOAM_pybind(["."])
## Define Input Parameters
# pars = np.linspace(0.01,0.5,50)

pars = [0.3333333333,0.2,0.1428571429,0.1111111111,0.0909090909,0.0769230769,0.0666666667,0.0588235294,0.0526315789,0.0476190476,0.0434782609,0.04,0.037037037,0.0344827586,0.0322580645,0.0303030303,0.0285714286,0.027027027,0.0256410256,0.0243902439,0.023255814,0.0222222222,0.0212765957,0.0204081633,0.0196078431,0.0188679245,0.0181818182,0.0175438596,0.0169491525,0.0163934426,0.0158730159,0.0153846154,0.0149253731,0.0144927536,0.014084507,0.0136986301,0.0133333333,0.012987013,0.0126582278,0.012345679,0.0120481928,0.0117647059,0.0114942529,0.0112359551,0.010989011,0.0107526882,0.0105263158,0.0103092784,0.0101010101]

print(pars)

# %%
tol = 1e-6
k = 1
i = 1
for vis in pars:
    print(pars)
    a.changeViscosity(vis)
    while tol<a.getRes():
        a.solveOneStep()
        i=i+1
    a.exportU(str(k),"./ITHACAoutput/Offline/","U")
    a.exportP(str(k),"./ITHACAoutput/Offline/","p")
    a.restart()
    k = k+1
# %%
plt.semilogy(resU)
plt.semilogy(resP)

print(np.linalg.norm(a.getResidual()[0:27000]))
print(resU[-1])
print(resP[-1])

# %%
a.restart()
for i in range(1000):
    a.solveOneStep()


u = a.getU()
p = a.getP()
u2 = u.copy()
resU = np.zeros([u.size])
resP = np.zeros([p.size])
phi = a.getPhi()
u2[3] = 1.5
p[0] = 1.5
phi[0] = 1.5
a.setU(u2)
# a.printU()


def residual(x):
    u_set = x[0:int(len(x)/4*3)]
    p_set = x[int(len(x)/4*3)::]
    a.setU(u_set)
    a.setP(p_set)
    res = a.getResidual()
    print(np.linalg.norm(res))
    return res


# up = np.ma.concatenate((u,p),axis=0)
# up = up.flatten()
# up[0] = 1
# residual(up)
# up[0] = 2
# residual(up)
# print(up.shape)
# from scipy.optimize import least_squares
# least_squares(residual, up, method="lm")


# %%
# a.printU()
# a.printP()
# a.printPhi()
u[0] = 1.5
res = a.getResidual()
print(np.linalg.norm(res))
u[0] = 3
res = a.getResidual()
print(np.linalg.norm(res))


# %%
