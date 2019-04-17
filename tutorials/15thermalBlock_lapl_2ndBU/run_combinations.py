import numpy as np
import os
import files 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools

modes_T = [1,2,3,4,5,10,15,20,25,30,35,40]
modes_DEIM = [1,2,3,4,5,10,15,20,25,30,35,40]

a = list(itertools.product(modes_T, modes_DEIM))
for k in a:
     files.sed_variable("N_modes_T","./system/ITHACAdict",str(k[0]))
     files.sed_variable("N_modes_DEIM_A","./system/ITHACAdict",str(k[1]))
     files.sed_variable("N_modes_DEIM_B","./system/ITHACAdict",str(k[1]))
     os.system("rm -r ITHACAoutput")
     os.system("10thermalBlock_lapl_2nd")