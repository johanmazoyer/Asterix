# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench
from CoffeeLibs.criteres import DJ_up,diff_grad_J,DJ_up_v1
from CoffeeLibs.pzernike import pmap, zernike
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt

# %% Initialisations

# Chargement des parametres de la simulation
path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Initialize test bed:
tbed = custom_bench(config["modelconfig"],'.')
tbed.grad = "np"
tbed.lap  = "lap"


# %% Treatment

N = tbed.dimScience//tbed.ech

# Images to estimate
[Ro,Theta] = pmap(N,N)

phi_foc = zernike(Ro,Theta,2) # Astig + defoc
EF_foc  = np.exp(1j*phi_foc)
i_foc   = tbed.psf(EF_foc)


# Images to estimate
[Ro,Theta] =  pmap(N,N)
PHI0       =  zernike(Ro,Theta,5) # Astig + defoc

J = pow( abs( i_foc - tbed.psf(np.exp(1j*PHI0)) ) , 2 )

# tbad.rcorno = 0

grad_analytic1  = DJ_up_v1(PHI0,0*PHI0,i_foc,tbed) # espi = 0
tbed.epsi = 1
grad_analytic2  = DJ_up_v1(PHI0,0*PHI0,i_foc,tbed)
grad_analytic3  = DJ_up(PHI0,0*PHI0,i_foc,tbed)

grad_diff       = diff_grad_J(PHI0,0*PHI0,i_foc,tbed)



# %%  Plots


plt.figure(1)
plt.subplot(2,4,1),plt.imshow(grad_analytic1,cmap='jet'),plt.title("Garident Analytique bpaul (epsi=0)"),plt.colorbar()
plt.subplot(2,4,2),plt.imshow(grad_analytic2,cmap='jet'),plt.title("Garident Analytique bpaul (epsi=1)"),plt.colorbar()
plt.subplot(2,4,3),plt.imshow(grad_analytic3,cmap='jet'),plt.title("Garident Analytique Olivier"),plt.colorbar()
plt.subplot(2,4,4),plt.imshow(grad_diff,cmap='jet'),plt.title("Gradient par différences"),plt.colorbar()

plt.subplot(2,2,3),plt.imshow(PHI0,cmap='jet'),plt.title("Point courant"),plt.colorbar()
plt.subplot(2,2,4),plt.imshow(abs(J),cmap='jet'),plt.title("Critere au point courant"),plt.colorbar()


