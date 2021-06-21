# -*- coding: utf-8 -*-
import os
from CoffeeLibs.criteres import genere_L,diff_grad_EF,gamma_terme
from CoffeeLibs.coffee import custom_bench, data_simulator
import numpy as np

from CoffeeLibs.files_manager import get_ini

import matplotlib.pyplot as plt
import pickle

# %% Initialisation

config = get_ini('my_param_file.ini',"..\..\Param_configspec.ini")

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
tbed      = custom_bench(config,'.')
sim       = data_simulator(tbed,var,div_factors)

# %% Generation de données 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,1]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])   # On génere le phi do

w = tbed.dimScience//tbed.ech
W = tbed.dimScience


# %% Calcule d'une image - 

point = sim.get_EF(0)

    
L = genere_L(tbed)

img_normal = tbed.todetector_Intensity(point) # On cree les images
img_L      = pow(abs(np.dot(L,point.reshape(w*w,)).reshape(W,W)),2)

# %% Calcule d'une image - 

i_reel = img_L     # On defini i reel
sim.gen_zernike_phi_foc(1/np.arange(1,3))  # On change le point ou calcule le gradient : EF =/= EF_reel


point = 0j*sim.get_EF(0)

# Gradient du critere a partir de L et dL au point EF
Lpoint   = pow(abs(np.dot(L,point.reshape(w*w,)).reshape(W,W)),2)
gamma  = gamma_terme(L,sim,i_reel)


dJ_matriciel  = (np.dot(np.transpose(L),gamma)).reshape(w,w)
diff_grad_h   = diff_grad_EF(sim.get_phi_foc(),tbed,i_reel)

# %%  Plots


plt.figure("Formation Image")
plt.suptitle("Essai calcul matriciel de l'image")
plt.subplot(2,3,1),plt.imshow(img_normal,cmap='jet'),plt.title("Image genere par tbed.todetecor"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(img_L,cmap='jet'),plt.title("Image par L*psi"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow(img_normal-img_L,cmap='jet'),plt.title("Erreur"),plt.colorbar()
plt.show()

plt.subplot(2,1,2),plt.imshow(np.real(dJ_matriciel),cmap='jet'),plt.title("dJ/dEF maticiel, valeur absolu"),plt.colorbar()
plt.show()