# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
----------------  TOOLS    -----------------
--------------------------------------------

@author: sjuillar
"""

import numpy as np
from scipy import ndimage

def ffts(A,norm="ortho"):
    """fft shifted twice"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A),norm=norm))


def iffts(A):
    """ifft shifted twice"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(A),norm= "ortho"))


def np_fftconv(A,B):
    """Fast convolution"""
    return np.real(iffts(ffts(A)*ffts(B)))


def grad_matrix(w,l,factor):
    M = np.zeros([w,l])
    x0 = w//2
    y0 = l//2
    for x in range(0, w):
           for y in range(0, l):
               M[x,y] = factor * ( pow(x0-x,2) + pow(y0-y,2) )
    return M

def normalize2(A):
    return A/(abs(A).max()-abs(A).min())

def circle(w,l,r):
    """ Create a zeros matrix [w*l] with a circle of ones of raduis r at the centre"""
    M = np.zeros([w,l])
    for x in range(0, w):
           for y in range(0, l):
               if  pow(x-w/2,2) + pow(y-l/2,2) <= pow(r,2):
                   M[x,y] = 1
    return M

def padding(A,ech):
    Apad = np.zeros(np.array(A.shape) * ech)*0j
    Apad[ A.shape[0]*(ech-1)//2 : A.shape[0]*(ech+1)//2, A.shape[1]*(ech-1)//2 : A.shape[1]*(ech+1)//2 ] = A
    return Apad

def depadding(A,ech):
    Acut = A[ A.shape[0]*(ech-1)//(2*ech) : A.shape[0]*(ech+1)//(2*ech), A.shape[1]*(ech-1)//(2*ech) : A.shape[1]*(ech+1)//(2*ech) ]
    return Acut

def gradient_xy(A,mode="sobel"):
    """ 2D spacial graident """
    
    if mode=="sobel" :
        sx = ndimage.sobel(A,axis=0,mode='constant')
        sy = ndimage.sobel(A,axis=1,mode='constant')
        grad=np.hypot(sx,sy)
        
    elif mode=="np" :
        [sx, sy]  = np.gradient(A)
        grad=np.hypot(sx,sy)
    else :
        grad = ndimage.gaussian_gradient_magnitude(A,sigma=0.2)
        
    return grad

def gradient2_xy(A):
    """ 2D spacial graident """
    return ndimage.laplace(A)



# %% Compare fft/mft 

# import matplotlib.pyplot as plt
# import tools as tls
# from param import P,wphi,lphi,ech,w,l
# from Asterix.propagation_functions import mft

# # Carte de phase
# phi = np.zeros((wphi,lphi))

# # Calcul des images 
# i_fft  = pow( abs( tls.ffts(tls.padding(P*np.exp(1j*phi),ech)) ) ,2)
# i_mft =  pow( abs( w * mft(P*np.exp(1j*phi),wphi,w,wphi) ) ,2)

# plt.figure(1)
# plt.subplot(1,3,1),plt.imshow(i_fft,cmap='jet'),plt.title("H avec FFT"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(i_mft,cmap='jet'),plt.title("H avec MFT"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(i_mft-i_fft,cmap='jet'),plt.title("Difference"),plt.colorbar()



