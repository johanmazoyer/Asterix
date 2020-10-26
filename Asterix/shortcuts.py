import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as spsignal
import scipy.optimize as opt
import scipy.ndimage as nd
import skimage.transform
from astropy.io import fits
import cv2
from zipfile import ZipFile



#Raccourcis FFT
fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift
ishift=np.fft.ifftshift

#Raccourcis généraux
abs=np.abs
im=np.imag
real=np.real
mean=np.mean
dot=np.dot
amax=np.amax

#Raccourcis conversions angles
dtor    = np.pi / 180.0 # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor


