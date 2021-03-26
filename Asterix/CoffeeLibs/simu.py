# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
------------  Simu optical system  ---------
--------------------------------------------

@author: sjuillar
"""

import numpy as np
import CoffeeLibs.tools as tls
from CoffeeLibs.param import P,wphi,lphi,ech,w,l,r
from Asterix.propagation_functions import mft

def defoc(phi_foc,defoc_factor):
    [w,l] = phi_foc.shape
    map_deofc = tls.grad_matrix(w,l,defoc_factor) 
    phi_defoc = phi_foc + map_deofc
    return phi_defoc

def H(phi):
    """ Compute sys rep """
    return pow( abs( w * mft(P*np.exp(1j*phi),wphi,w,wphi) ) ,2)


# %% Teste MFT echnatillonnage

# =============================================================================
# import matplotlib.pyplot as plt
# import timeit as tme
# 
# # Calcul des images
# phi = np.zeros((wphi,lphi))
# P = tls.circle(wphi,lphi,r)
# i_mft1 =  pow( abs( w * mft(P*np.exp(1j*phi),wphi,w*ech,wphi) ) ,2)
# 
# phi = tls.padding(phi,ech)
# P = tls.circle(wphi*ech,lphi*ech,r)
# i_mft2 =  pow( abs( w * mft(P*np.exp(1j*phi),wphi,w*ech,wphi) ) ,2)
# 
# # Gradients
# plt.figure(1)
# plt.subplot(1,3,1),plt.imshow(np.log10(i_mft1),cmap='jet'),plt.title("H avec padding"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(np.log10(i_mft2),cmap='jet'),plt.title("H sans padding"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(np.log10(i_mft2)-np.log10(i_mft1),cmap='jet'),plt.title("Difference"),plt.colorbar()
# 
# =============================================================================

# %% Teste MFT / FFT

# =============================================================================
# import matplotlib.pyplot as plt
# import timeit as tme
# 
# setup = '''import numpy as np 
# import tools as tls
# from param import P,wphi,lphi,ech,w,l
# from Asterix.propagation_functions import mft'''
# 
# # Carte de phase
# phi = np.zeros((wphi,lphi))
# 
# # Echantillonnage
# phi = tls.padding(phi,ech)
# 
# # Calcul des images 
# i_fft  = pow( abs( w * tls.ffts(P*np.exp(1j*phi),wphi,w,w) ) ,2)
# i_mft =  pow( abs( w * mft(P*np.exp(1j*phi),wphi,w,w) ) ,2)
# 
# 
# timz_mft = tme.timeit('phi = np.zeros((wphi,lphi))\nphi = tls.padding(phi,ech)\npow( abs( w**2 * mft(P*np.exp(1j*phi),wphi,w,1) ) ,2)',setup=setup,number=100)
# timz_fft = tme.timeit('phi = np.zeros((wphi,lphi))\nphi = tls.padding(phi,ech)\npow( abs( tls.ffts(P*np.exp(1j*phi)) ) ,2)',setup=setup,number=100)
# print("For 10 iterations :\n \ttime fft = " +str(round(timz_fft,4))+"s\n\ttime mft = "+str(round(timz_mft,4))+"s")
# 
# # Gradients
# plt.figure(1)
# plt.subplot(1,3,1),plt.imshow(i_fft,cmap='jet'),plt.title("H avec FFT"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(i_mft,cmap='jet'),plt.title("H avec MFT"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(i_mft-i_fft,cmap='jet'),plt.title("Difference"),plt.colorbar()
# 
# =============================================================================

