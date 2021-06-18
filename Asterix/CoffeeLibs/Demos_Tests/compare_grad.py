# -*- coding: utf-8 -*-
import os
from CoffeeLibs.criteres import DJmv_up,diff_grad_J_up,diff_grad_J_down,DJmv_down, estime_fluxfond
from CoffeeLibs.coffee import custom_bench, data_simulator
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt
import pickle

# %% Initialisation

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
tbed      = custom_bench(config,'.')
sim       = data_simulator(tbed,var,div_factors)

# %% Generation de données 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])   # On génere le phi do

img = sim.todetector_Intensity(0) # On cree les images

w = tbed.dimScience//tbed.ech


#[flux,fond] = estime_fluxfond(sim,imgs)

# %% Calcule des gradients - 
# On chage l'object sim, ca devient une autre simulation quelquonque, les info i_foc et i_div disparaissent
# Maitenant on va s'en servir comme la simulation a tester avec les données généré précédement


point = np.zeros((w,w))
# Point ou on calcule le gradients
sim.set_phi_do(point)
sim.set_phi_foc(point)


# sim.known_var['flux'] = flux
# sim.known_var['fond'] = fond

grad_analytic_up   = 0
grad_diff_up       = 0
grad_analytic_down = 0
grad_diff_down     = 0
    
    
# grad_diff_down       += diff_grad_J_down(sim.get_phi_do(),0,sim,img)

# grad_analytic_down += DJmv_down(0,img,sim)

grad_analytic_up   += DJmv_up(0,img,sim)

grad_diff_up       += diff_grad_J_up(sim.get_phi_foc(),0,sim,img)
    

# %%  Plots


plt.figure(1)
plt.subplot(2,3,1),plt.imshow(grad_analytic_up,cmap='jet'),plt.title("Garident Analytique UP"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(grad_diff_up,cmap='jet'),plt.title("Garident difference UP"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow((grad_diff_up-grad_analytic_up),cmap='jet'),plt.title("Erreur UP"),plt.colorbar()

# plt.subplot(2,3,4),plt.imshow(grad_analytic_down,cmap='jet'),plt.title("Garident Analytique DOWN"),plt.colorbar()
# plt.subplot(2,3,5),plt.imshow(grad_diff_down,cmap='jet'),plt.title("Garident difference DOWN"),plt.colorbar()
# plt.subplot(2,3,6),plt.imshow((grad_diff_down-grad_analytic_down),cmap='jet'),plt.title("Erreur DOWN"),plt.colorbar()


plt.show()

# with open('save/grad_d', 'wb') as handle:
#     pickle.dump([grad_analytic_down,grad_diff_down], handle)

