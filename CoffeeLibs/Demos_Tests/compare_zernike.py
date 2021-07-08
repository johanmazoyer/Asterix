# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:46:08 2021

@author: sjuillar
"""

import matplotlib.pyplot as plt
from CoffeeLibs.pzernike import pmap, zernike
from CoffeeLibs.files_manager import get_fits_as_imgs
from CoffeeLibs.tools import circle
from numpy import pi,sqrt

# %%

name      = "zernike_a4_1.00rad"
idl_Ro    = "rho"
idl_Theta = "phi"
N    = 32


idl_zern  = get_fits_as_imgs("save",name)[:,:,0]
idl_Ro    = get_fits_as_imgs("save",idl_Ro)[:,:,0]
idl_Theta = get_fits_as_imgs("save",idl_Theta)[:,:,0]

pup = circle(N,N,16)
[Ro, Theta] = pmap(N,N,16)
py_zern  = pup*zernike(Ro, Theta, 4)


# plt.figure("Comparaison Ro/Theta")
# plt.subplot(2,3,1),plt.imshow(idl_Ro,cmap="jet"),plt.colorbar(),plt.title("RO IDL")
# plt.subplot(2,3,2),plt.imshow(Ro,cmap="jet"),plt.colorbar(),plt.title("RO Python")
# plt.subplot(2,3,3),plt.imshow(abs(idl_Ro/Ro),cmap="jet"),plt.colorbar(),plt.title("Différence")
# plt.subplot(2,3,4),plt.imshow(idl_Theta,cmap="jet"),plt.colorbar(),plt.title("Theta IDL")
# plt.subplot(2,3,5),plt.imshow(Theta,cmap="jet"),plt.colorbar(),plt.title("Theta Python")
# plt.subplot(2,3,6),plt.imshow(abs(idl_Theta-Theta),cmap="jet"),plt.colorbar(),plt.title("Différence")

plt.figure("Comparaison Zernik")
plt.subplot(1,3,1),plt.imshow(idl_zern,cmap="jet"),plt.colorbar(),plt.title("zernike IDL")
plt.subplot(1,3,2),plt.imshow(py_zern,cmap="jet"),plt.colorbar(),plt.title("zernike Python")
plt.subplot(1,3,3),plt.imshow(abs(idl_zern-py_zern),cmap="jet"),plt.colorbar(),plt.title("Différence")