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

def circle(w,l,r):
    """ Create a zeros matrix [w*l] with a circle of ones of raduis r at the centre"""
    M = np.zeros([w,l])
    for x in range(0, w):
           for y in range(0, l):
               if  pow(x-w/2,2) + pow(y-l/2,2) <= pow(r,2):
                   M[x,y] = 1
    return M

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

def padding(A,ech):
    Apad = np.zeros(np.array(A.shape) * ech)
    Apad[ A.shape[0]*(ech-1)//2 : A.shape[0]*(ech+1)//2, A.shape[1]*(ech-1)//2 : A.shape[1]*(ech+1)//2 ] = A
    return Apad

def depadding(A,ech):
    Acut = A[ A.shape[0]*(ech-1)//(2*ech) : A.shape[0]*(ech+1)//(2*ech), A.shape[1]*(ech-1)//(2*ech) : A.shape[1]*(ech+1)//(2*ech) ]
    return Acut

def gradient_xy(A):
    """ 2D spacial graident """
    [sx,sy]   = np.gradient(A)
    grad = np.hypot(sx,sy)
    return grad

def gradient2_xy(A):
    """ 2D spacial graident """
    return ndimage.laplace(A,mode='constant')

def defoc(phi_foc,defoc_factor):
    [w,l] = phi_foc.shape
    map_deofc = grad_matrix(w,l,defoc_factor) 
    phi_defoc = phi_foc + map_deofc
    return phi_defoc
