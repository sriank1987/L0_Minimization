import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import fft
import cv2
import scipy

np.random.seed(seed=0)
start = -100
stop = 100
n=stop-start

# Path to input and output folders
ip_path = '/home/asrivast/L0Minimization/inputs'
op_path = '/home/asrivast/L0Minimization/outputs'

# Generate a 1D random signal
ip_1d = np.arange(start,stop)
"""op_1d = np.zeros_like(ip_1d, dtype=np.float32)
op_1d[-45:-25] = 1
op_1d[-10:10] = 1
op_1d[25:45] = 1
op_1d = op_1d + np.random.rand(n)/10.0"""
op_1d = np.heaviside(ip_1d, 0.5) + np.random.rand(n)/10.0

# Plot input and save the figure
fig_1 = plt.figure()
plt.plot(ip_1d, op_1d)
plt.show()
#fig.savefig(ip_path+'/1_1d.png')

# L0 Minimization algo Parameters

lambdu = 0.02
beta_0 = 2*lambdu
beta_max = 1E5
kappa = 2

#initialisations
s_op_1d = op_1d
beta = beta_0
i=0

fx = np.zeros((n), dtype=np.float32)
fx[0] = 1
fx[-1] = -1

# Algo
while (beta<beta_max):
    i=i+1
    grad_0 = s_op_1d - np.roll(s_op_1d,-1)
    #print(grad_0)
    grad_1 = np.square(grad_0)
    #print(grad_1)
    h = np.where(grad_1>lambdu/beta, grad_0, 0.0)
    print(h)
    print(np.count_nonzero(h), end=" ")
    fft_dx = fft.fft(fx)
    #print(fft_dx)
    fft_dx_c = np.conjugate(fft_dx)
    #print(fft_dx_c)
    num = fft.fft(op_1d)+beta*(fft_dx_c*fft.fft(h))
    #print(num)
    den = 1+beta*(fft_dx_c*fft_dx)
    #print(den)
    s_op_1d = np.real(fft.ifft(num/den))
    #print(s_op_1d)
    beta = kappa*beta

# Plot output and save the figure
fig_2 = plt.figure()
plt.plot(ip_1d, s_op_1d)
plt.title(f"Lambda :{lambdu}")
plt.show()
#fig.savefig(op_path+'/1_1d.png')

#print('Number of iterations: ',i)