# -*- coding: utf-8 -*-

import os
from CoffeeLibs.criteres import *
from CoffeeLibs.coffee import custom_bench, data_simulator
from Asterix.Optical_System_functions import coronagraph

import numpy as np

from CoffeeLibs.files_manager import get_ini

import matplotlib.pyplot as plt
import pickle

# %% Initialisation

config = get_ini('my_param_file.ini')

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
# tbed      = custom_bench(config,'.')
tbed      = coronagraph(config['modelconfig'],config['Coronaconfig'])

sim       = data_simulator(tbed,var,div_factors)

# %% Generation de données 

## -- Coeff du zernike  
coeff = 10000/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
# sim.gen_zernike_phi_do([0,0,1])   # On génere le phi do

w = tbed.dim_overpad_pupil
W = tbed.dimScience


# %% Calcule d'une image - 

point = sim.get_EF_div(0)
    
L = genere_L(tbed)

img_normal = sim.get_img_div(0) # On cree les images
img_L      = pow(abs(np.dot(L,point.reshape(w*w,)).reshape(W,W)),2)

# %%  Plots


plt.figure("Formation Image")
plt.suptitle("Essai calcul matriciel de l'image")
plt.subplot(1,3,1),plt.imshow(img_normal,cmap='jet'),plt.title("Image genere par tbed.todetecor"),plt.colorbar()
plt.subplot(1,3,2),plt.imshow(img_L,cmap='jet'),plt.title("Image par L*psi"),plt.colorbar()
plt.subplot(1,3,3),plt.imshow(img_normal-img_L,cmap='jet'),plt.title("Erreur"),plt.colorbar()
plt.show()
