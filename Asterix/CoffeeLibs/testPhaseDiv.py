# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
from CoffeeLibs.pzernike import pmap, zernike, pzernike
import numpy as np
from Asterix.propagation_functions import mft

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import pickle

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
estimator = Estimator(tbed,**config["Estimationconfig"])

# %% Traitement 

coeff = 1/np.arange(1,6)          # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])
imgs = sim.gen_div_imgs(tbed,RSB) # On cree les images

#Estimation
known_var = {'flux':1, 'fond':0}
# known_var = var
e_sim = estimator.estimate(imgs,div_factors,known_var)


# %% Save / Load

# with open('./save/est_4q', 'rb') as f:
#     e_sim = pickle.load(f)

# with open('./save/est_4q', 'wb') as f:
#     pickle.dump(e_sim, f)  

# %%  Plots


plt.figure(3)

cropEF = sim.get_phi_foc()*tbed.pup
cropEFd = sim.get_phi_do()*tbed.pup_d

plt.subplot(2,3,1),plt.imshow(cropEF,cmap='jet'),plt.title("Phi_up Attendu"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(e_sim.get_phi_foc(),cmap='jet'),plt.title("Estimation"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow(cropEF-e_sim.get_phi_foc(),cmap='jet'),plt.title("Erreur"),plt.colorbar()

plt.subplot(2,3,4),plt.imshow(cropEFd,cmap='jet'),plt.title("Phi_do Attendu"),plt.colorbar()
plt.subplot(2,3,5),plt.imshow(e_sim.get_phi_do(),cmap='jet'),plt.title("Estimation"),plt.colorbar()
plt.subplot(2,3,6),plt.imshow(cropEFd-e_sim.get_phi_do(),cmap='jet'),plt.title("Erreur"),plt.colorbar()


text =  "estimate flux = " + "{:.2f}".format(e_sim.get_flux()) + " --> error = +/-" + "{:.2f}".format(100*(abs(sim.get_flux()-e_sim.get_flux()))/sim.get_flux()) + "%\n"
text += "estimate fond = " + "{:.2f}".format(e_sim.get_fond()) + " --> error = "    + "{:.2f}".format(abs(sim.get_fond()-e_sim.get_fond()))
plt.suptitle(text)


# %%  Introspection 

view_list  = tbed.introspect(sim.get_EF(),sim.get_EF_do())
title_list = ["Entrence EF", "Upsteam pupil", "MFT", "Corno", "MFT", "Downstream EF + pupil", "MFT - detecteur"]

def update(val):
    plt.subplot(1,1,1),plt.imshow(abs(view_list[val]),cmap='jet'),plt.suptitle(title_list[val]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[val])**2))),plt.subplots_adjust(bottom=0.25)
    

plt.figure("Introscpetion")
plt.subplot(1,1,1),plt.imshow(abs(view_list[0]),cmap='jet'),plt.suptitle(title_list[0]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[0])**2))),plt.subplots_adjust(bottom=0.25)

slide = Slider(plt.axes([0.25,0.1,0.65,0.03]),"view ",0,len(view_list)-1,valinit=0,valstep=1)
slide.on_changed(update)
