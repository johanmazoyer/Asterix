#Version 26/12/2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as spsignal
import scipy.optimize as opt
import scipy.ndimage as nd
import skimage.transform
from astropy.io import fits
import glob
import os
#import cv2


#General configuration
main_dir = r"C:\Users\LESIA-BAT\Desktop\Labview_IDL_routines\Package_EFC_THD_v2\\"
Labview_dir= r"C:\Users\LESIA-BAT\Desktop\Labview_IDL_routines\Labview_routines\ITHD_v40\DATA\EFC\\"

otherbasis=False#If true, uses the basis saved by init_SCC in IDL

isz   = 400  # image size (in pixels)
wavelength = 783.25e-9 # wavelength for image (in meters)
pdiam = 8.23e-3 # pupil diameter (in meters)
lyotdiam  = 7.9e-3  # lyot diameter (in meters)
lyot=lyotdiam/pdiam
set_ldp_manually = True #Calculate lambda/D (in pixel) from the size of lyotdiam or set it manually below
ld_p=7.80*lyot #lambda/D (in pixel), taken into account if set_ldp_manually equal True
#ld_p=8.13*lyot
coronagraph ='knife' #Can be fqpm or knife
new=150     #Taille réechantillonage
dimimages=150  #Taille DH (découpage des images)


#PW parameters
amplitudePW = 34
#posprobes = [309,495,659]
posprobes = [466,498]
posprobes = [466,465]
#posprobes = [466,498,530]
cut = 1e6
cut = 5e4
cut=5e5 #Knife

#EFC parameters
# The lowest the more actuators are considered
# ratio of energy of the influence function inside the pupil wrt to energy of the influence function
MinimumSurfaceRatioInThePupil = 0.1


choosepix = [8,35,-35,35]
Nbmodes =300

choosepix = [12,29,-29,29]
Nbmodes =320

choosepix = [12,29,0,29]
Nbmodes =150


choosepix = [-35,35,-35,35] #Full DH minus 2.5 pixels on each direction
Nbmodes =500


choosepix = [11,29,-29,29]
Nbmodes =390

choosepix = [-37,37,-37,37] #Full DH minus 0.5 pixel on each side
Nbmodes =300

choosepix = [8,32,-32,32]
Nbmodes = 480

choosepix = [8,27,-32,32]
Nbmodes = 450

choosepix = [5,30,-35,35]
Nbmodes =510#510

choosepix = [8,30,-30,30]
Nbmodes =460#510

choosepix = [3,37,-37,37]
Nbmodes =500#510

choosepix = [4,20,-8,8]
Nbmodes =130#510

choosepix = [5,20,-20,20]
Nbmodes =250#510

choosepix = [-25,25,5,25]
Nbmodes =290#510

#choosepix = [-35,35,4,35]
#Nbmodes =450#510


print(str(150/2+np.array(np.fft.fftshift(choosepix))))

amplitudeEFC = 17
