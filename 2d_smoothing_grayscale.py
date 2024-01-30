import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import fft
import cv2
import scipy

# Path to input and output folders

ip_path = 'Enter the input path here'
op_path = 'Enter the output path here'

# Load original image

orig_img = cv2.imread(ip_path+'/img1.png')
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
orig_img = np.asarray(orig_img, dtype=np.float32)/255

# L0 Minimization algo Parameters

lambdu = 0.02
beta_0 = 2*lambdu
beta_max = 1E5
kappa = 2

#initialisations

h, w = np.shape(orig_img)
fft_orig_img = np.full_like(orig_img, 0.0, dtype=np.complex64)
hp = np.zeros((h,w), dtype=np.float32)
vp = np.zeros((h,w), dtype=np.float32)

# Show original image
fig_1 = plt.figure()
plt.imshow(orig_img, cmap="gray")
plt.show()

smth_img = orig_img
beta = beta_0
i=0

fx = np.zeros((h,w), dtype=np.float32)
fx[0,0] = 1
fx[0,-1] = -1

fy = np.zeros((h,w), dtype=np.float32)
fy[0,0] = 1
fy[-1,0] = -1

fft_dx = fft.fft2(fx)
fft_dx_c = np.conjugate(fft_dx)

fft_dy = fft.fft2(fy)
fft_dy_c = np.conjugate(fft_dy)

# Algo

while (beta<beta_max):
    i=i+1

    grad_x = smth_img - np.roll(smth_img,-1, axis=1)
    grad_y = smth_img - np.roll(smth_img,-1, axis=0)

    grad_square = np.square(grad_x) + np.square(grad_y)

    hp = np.where(grad_square>lambdu/beta, grad_x, 0.0)
    vp = np.where(grad_square>lambdu/beta, grad_y, 0.0)

    num = fft.fft2(orig_img) + beta*(fft_dx_c*fft.fft2(hp) + fft_dy_c*fft.fft2(vp))
    den = 1+beta*(fft_dx_c*fft_dx + fft_dy_c*fft_dy)
    smth_img = np.real(fft.ifft2(num/den))
    beta = kappa*beta

# Show smoothed image

fig_2 = plt.figure()
plt.imshow(smth_img, cmap="gray")
plt.title(f"Smoothed Image ; lambda : {lambdu}")
plt.show()
#cv2.imwrite(op_path+f"/img1_lambda_{lambdu}.png", smth_img)

print('Number of iterations: ',i)
