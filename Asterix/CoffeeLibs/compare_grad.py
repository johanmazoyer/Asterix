# -*- coding: utf-8 -*-
import os
from CoffeeLibs.criteres import DJmv_up,diff_grad_J_up,diff_grad_J_down,DJmv_down, estime_fluxfond
from CoffeeLibs.coffee import custom_bench, data_simulator
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt

# %% Initialisation

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0,1]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed.dimScience//tbed.ech,var,div_factors)

# %% Generation de données 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])   # On génere le phi do

imgs = sim.gen_div_imgs(tbed) # On cree les images

w = tbed.dimScience//tbed.ech
point = np.zeros((w,w))


# %% Calcule des gradients - 
# On chage l'object sim, ca devient une autre simulation quelquonque, les info i_foc et i_div disparaissent
# Maitenant on va s'en servir comme la simulation a tester avec les données généré précédement

# Point ou on calcule le gradients
sim.set_phi_do(point)
sim.set_phi_foc(point)

[flux,fond] = estime_fluxfond(sim,tbed,imgs)
# sim.known_var['flux'] = flux
# sim.known_var['fond'] = fond

grad_analytic_up   = 0
grad_diff_up       = 0
grad_analytic_down = 0
grad_diff_down     = 0
    
for div_id in range(len(div_factors)):
    img = imgs[:,:,div_id]
    
    grad_analytic_down += DJmv_down(div_id,img,sim,tbed)
    grad_diff_down     += diff_grad_J_down(sim.get_phi_do(),div_id,sim,img,tbed)
    
    grad_analytic_up   += DJmv_up(div_id,img,sim,tbed)
    grad_diff_up       += diff_grad_J_up(sim.get_phi_foc(),div_id,sim,img,tbed)
    



# %%  Plots


plt.figure(1)
plt.subplot(2,3,1),plt.imshow(grad_analytic_up,cmap='jet'),plt.title("Garident Analytique UP"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(grad_diff_up,cmap='jet'),plt.title("Garident difference UP"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow((grad_diff_up-grad_analytic_up),cmap='jet'),plt.title("Erreur UP"),plt.colorbar()

plt.subplot(2,3,4),plt.imshow(grad_analytic_down,cmap='jet'),plt.title("Garident Analytique DOWN"),plt.colorbar()
plt.subplot(2,3,5),plt.imshow(grad_diff_down,cmap='jet'),plt.title("Garident difference DOWN"),plt.colorbar()
plt.subplot(2,3,6),plt.imshow((grad_diff_down-grad_analytic_down),cmap='jet'),plt.title("Erreur DOWN"),plt.colorbar()


text = "estimate flux = "  + str(flux) +"\n"
text += "estimate fond = " + str(fond)
plt.suptitle(text)

text  =  "estimate flux = " + "{:.2f}".format(flux) + " --> error = +/-" + "{:.2f}".format(100*(abs(sim.get_flux()-flux))/sim.get_flux()) + "%\n"
text += "estimate fond = "  + "{:.2f}".format(fond) + " --> error = "    + "{:.2e}".format(abs(sim.get_fond()-fond))
plt.suptitle(text)

print(text)