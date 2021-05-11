# -*- coding: utf-8 -*-
"""
Created on Tue May  11 10:55:09 2021

--------------------------------------------
------------  Evals for COFFEE  ------------
------------    under SNR       ------------
--------------------------------------------

@author: sjuillar
"""

import os
from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
from Asterix.propagation_functions import mft
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle

# %% Initialisation

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [-1,1,0,2,-2]  # List of div factor's images diversity
RSB         = 30000

# %%  Initalisation of objetcs
tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed,var,div_factors)
estimator = Estimator(**config["Estimationconfig"])

coeff = 1/np.arange(1,6)          # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])

known_var = {'flux':1, 'fond':0}

# %% Variables

Neval   = 10    # Nombre de point a evaluer
start   = -200   # En Rapport signal a bruit
stop    = 300

nb_iter = 1    # Moyenne sur nb_iter

plist   = range(start,stop,stop//Neval) 

N = sim.N

# %% Test loop

error_list = []
time_list  = []
fig_list   = []

for RSB in plist:
        
    elapsed = 0
    error   = 0
    
    for ii in range(nb_iter):
       
        # Add some perturbation
        imgs = sim.gen_div_imgs(RSB) # On cree les images

        # Estimation      
        
        e_sim = estimator.estimate(imgs,tbed,div_factors,known_var)
        error    += sum(sum(abs(sim.get_phi_foc()*tbed.pup - e_sim.get_phi_foc())))/(N**2)
    
    
    # Update data 
    time_list.append(estimator.toc/nb_iter)   
    error_list.append(error/nb_iter)
    fig_list.append(e_sim.get_phi_foc())
    
    print("Error : " + "%.5f" % error)
    print("Minimize took : " + str(elapsed/60) + " mins\n")
    print(" ^ Simu for RSB :"+ str(RSB) +"\n")
    print("----------------------------------------")

# Saves

with open('./save/eval1', 'wb') as f:
    pickle.dump(fig_list, f)
    
    
# %%  Plots

# plt.figure(1)
# plt.subplot(1,2,1),plt.plot(plist,time_list),plt.title("Time")
# plt.subplot(1,2,2),plt.plot(plist,error_list),plt.title("RMS")


with open('./save/eval1', 'rb') as f:
    fig_list = pickle.load(f)

def update(val):
    img = (val - start)//(stop//Neval)
    plt.subplot(1,2,1),plt.imshow(fig_list[img],cmap='jet'),plt.title("For RSB = " + str(val))
    plt.subplot(1,2,2),plt.imshow(abs(mft(np.exp(1j*fig_list[img]),N,N*tbed.ech,N))**2,cmap='jet'),plt.title("abs(FT(phi_est))**2)")

plt.figure(2)
plt.subplot(1,2,1),plt.imshow(fig_list[0],cmap='jet'),plt.title("For RSB = " + str(start))
plt.subplot(1,2,2),plt.imshow(abs(mft(np.exp(1j*fig_list[0]),N,N*tbed.ech,N))**2,cmap='jet'),plt.title("abs(FT(phi_est))**2)"),plt.colorbar()
slide = Slider(plt.axes([0.25,0.1,0.65,0.03]),"RSB",start,stop-(stop//Neval),valinit=start,valstep=stop//Neval)

slide.on_changed(update)
