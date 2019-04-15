import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,99,num=100)

"""
Error_L2 = open('Errors/Error_L2_mat.txt').read()
Error_L2 = [item.split() for item in Error_L2.split('\n')[:-1]]
Error_Fr = open('Errors/Error_Fr_mat.txt').read()
Error_Fr = [item.split() for item in Error_Fr.split('\n')[:-1]]

Error_L2_wm = open('Errors/Error_L2_wm_mat.txt').read()
Error_L2_wm = [item.split() for item in Error_L2_wm.split('\n')[:-1]]
Error_Fr_wm = open('Errors/Error_Fr_wm_mat.txt').read()
Error_Fr_wm = [item.split() for item in Error_Fr_wm.split('\n')[:-1]]

plt.semilogy(x,Error_L2, 'r', label = r'L$_2$')
plt.semilogy(x,Error_L2_wm, 'k', label = r'L$_2$ weighted')
plt.semilogy(x,Error_Fr, 'b--', label = r'Frobenius')
plt.semilogy(x,Error_Fr_wm, 'g--', label = r'Frobenius weighted')
plt.xlim(1,100)

plt.legend(loc = 'center')
plt.savefig("Errors/Errors comparison.png")
plt.show()

Error_L2_new = open('Errors/Error_L2_new_mat.txt').read()
Error_L2_new = [item.split() for item in Error_L2_new.split('\n')[:-1]]
Error_Fr_new = open('Errors/Error_Fr_new_mat.txt').read()
Error_Fr_new = [item.split() for item in Error_Fr_new.split('\n')[:-1]]

Error_L2_wm_new = open('Errors/Error_L2_wm_new_mat.txt').read()
Error_L2_wm_new = [item.split() for item in Error_L2_wm_new.split('\n')[:-1]]
Error_Fr_wm_new = open('Errors/Error_Fr_wm_new_mat.txt').read()
Error_Fr_wm_new = [item.split() for item in Error_Fr_wm_new.split('\n')[:-1]]

plt.semilogy(x,Error_L2_new, 'r', label = r'L$_2$')
plt.semilogy(x,Error_L2_wm_new, 'k', label = r'L$_2$ weighted')
plt.semilogy(x,Error_Fr_new, 'b--', label = r'Frobenius')
plt.semilogy(x,Error_Fr_wm_new, 'g--', label = r'Frobenius weighted')
plt.xlim(1,100)

plt.legend(loc = 'center')
plt.savefig("Errors/Errors comparison new.png")
plt.show()

Error_L2_50 = open('Errors/Error_L2_50_mat.txt').read()
Error_L2_50 = [item.split() for item in Error_L2_50.split('\n')[:-1]]
Error_Fr_50 = open('Errors/Error_Fr_50_mat.txt').read()
Error_Fr_50 = [item.split() for item in Error_Fr_50.split('\n')[:-1]]

Error_L2_wm_50 = open('Errors/Error_L2_wm_50_mat.txt').read()
Error_L2_wm_50 = [item.split() for item in Error_L2_wm_50.split('\n')[:-1]]
Error_Fr_wm_50 = open('Errors/Error_Fr_wm_50_mat.txt').read()
Error_Fr_wm_50 = [item.split() for item in Error_Fr_wm_50.split('\n')[:-1]]

plt.semilogy(x,Error_L2_50, 'r', label = r'L$_2$')
plt.semilogy(x,Error_L2_wm_50, 'k', label = r'L$_2$ weighted')
plt.semilogy(x,Error_Fr_50, 'b--', label = r'Frobenius')
plt.semilogy(x,Error_Fr_wm_50, 'g--', label = r'Frobenius weighted')
plt.xlim(1,100)

plt.legend(loc = 'center')
plt.savefig("Errors/Errors comparison 50.png")
plt.show()
"""

Error_L2 = open('Errors/ComparedErrors/L2_errors_mat.txt').read()
Error_L2 = [item.split() for item in Error_L2.split('\n')[:-1]]
Error_Fr = open('Errors/ComparedErrors/Frobenius_errors_mat.txt').read()
Error_Fr = [item.split() for item in Error_Fr.split('\n')[:-1]]

Error_L2_wm = open('Errors/ComparedErrors/L2_wm_errors_mat.txt').read()
Error_L2_wm = [item.split() for item in Error_L2_wm.split('\n')[:-1]]
Error_Fr_wm = open('Errors/ComparedErrors/Frobenius_wm_errors_mat.txt').read()
Error_Fr_wm = [item.split() for item in Error_Fr_wm.split('\n')[:-1]]

plt.semilogy(x,Error_L2, 'r', label = r'L$_2$')
plt.semilogy(x,Error_L2_wm, 'k', label = r'L$_2$ weighted')
plt.semilogy(x,Error_Fr, 'b--', label = r'Frobenius')
plt.semilogy(x,Error_Fr_wm, 'g--', label = r'Frobenius weighted')
plt.xlim(1,100)

plt.legend(loc = 'center')
plt.savefig("Errors/ComparedErrors/Errors comparison.png")
plt.show()

















