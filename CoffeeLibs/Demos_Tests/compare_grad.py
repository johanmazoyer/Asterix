# -*- coding: utf-8 -*-
import os
from CoffeeLibs.criteres import *
from CoffeeLibs.coffee import custom_bench, data_simulator
from Asterix.Optical_System_functions import coronagraph

import numpy as np
import numpy.matlib as mnp

from CoffeeLibs.files_manager import get_ini

import matplotlib.pyplot as plt
import pickle

# %% Initialisation

config = get_ini('my_param_file.ini')

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0,1,-1,0.5,-0.5]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
# tbed      = custom_bench(config,'.')
tbed      = coronagraph(config['modelconfig'],config['Coronaconfig'])

sim       = data_simulator(tbed,var,div_factors)

# %% Generation de données 

coeff = 100/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
# sim.gen_zernike_phi_do([0,0,1])   # On génere le phi do

img = sim.todetector_Intensity(0) # On cree les images

w = tbed.dim_overpad_pupil
W = tbed.dimScience


#[flux,fond] = estime_fluxfond(sim,imgs)

# %% Calcule des gradients - 
# On chage l'object sim, ca devient une autre simulation quelquonque, les info i_foc et i_div disparaissent
# Maitenant on va s'en servir comme la simulation a tester avec les données généré précédement


point = np.zeros((w,w))
# Point ou on calcule le gradients
# sim.set_phi_do(point)
# sim.set_phi_foc(point)
sim.gen_zernike_phi_foc(1/np.arange(1,65))  # On change le point ou calcule le gradient : EF =/= EF_reel


# sim.known_var['flux'] = flux
# sim.known_var['fond'] = fond

grad_analytic_up   = 0
grad_diff_up       = 0
grad_analytic_down = 0
grad_diff_down     = 0
    
    
# grad_diff_down       += diff_grad_J_down(sim.get_phi_do(),0,sim,img)

# grad_analytic_down += DJmv_down(0,img,sim)

div_id = 0

grad_analytic_up   += np.zeros((w,w))#-DJmv_up(div_id,img,sim)

grad_diff_up       += diff_grad_J_up(sim.get_phi_foc(),div_id,sim,img)

# Grad AUTO
L             = genere_L(tbed)
gamma         = gamma_terme(L,sim,img,div_id)
dJ_matriciel  = (np.dot(np.conj(np.transpose(L)),gamma)).reshape(w,w)*np.conj(sim.get_EF_div(div_id))
dJ_matriciel  = -4*np.imag(dJ_matriciel)

# %%  Plots


plt.figure(1)
plt.subplot(2,3,1),plt.imshow(grad_analytic_up,cmap='jet'),plt.title("Garident Analytique UP"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(grad_diff_up,cmap='jet'),plt.title("Garident difference UP"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow((grad_diff_up-grad_analytic_up),cmap='jet'),plt.title("Erreur UP"),plt.colorbar()


plt.subplot(2,3,4),plt.imshow(dJ_matriciel,cmap='jet'),plt.title("Garident Matriciel"),plt.colorbar()
plt.subplot(2,3,5),plt.imshow(grad_diff_up,cmap='jet'),plt.title("Garident difference UP"),plt.colorbar()
plt.subplot(2,3,6),plt.imshow((dJ_matriciel-grad_diff_up),cmap='jet'),plt.title("Erreur grad matriciel"),plt.colorbar()


plt.show()

# with open('save/grad_d', 'wb') as handle:
#     pickle.dump([grad_analytic_down,grad_diff_down], handle)

