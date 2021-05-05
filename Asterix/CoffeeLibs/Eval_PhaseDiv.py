# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator
from CoffeeLibs.pzernike import pmap, zernike
from sklearn.preprocessing import normalize
from Asterix.propagation_functions import mft
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import pickle

# %% Initialisations

# Chargement des parametres de la simulation
path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Initialize test bed:
tbed = custom_bench(config["modelconfig"],'.')


# %% Variables

Neval   = 10    # Nombre de point a evaluer
start   = -200   # En Rapport signal a bruit
stop    = 300

nb_iter = 1    # Moyenne sur nb_iter

plist   = range(start,stop,stop//Neval) 


# %% Initalisation

N = tbed.dimScience//tbed.ech
estimator = Estimator(tbed,**config["Estimationconfig"])

# Images to estimate
[Ro,Theta] = pmap(N,N)
phi_foc    = normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6))
phi_div    = phi_foc + zernike(Ro,Theta,4)

EF_foc = np.exp(1j*phi_foc)
EF_div = np.exp(1j*phi_div)

i_foc_0 = tbed.psf(entrance_EF=EF_foc)
i_div_0 = tbed.psf(entrance_EF=EF_div)


# %% Test loop

error_list = []
time_list  = []
fig_list   = []

for RSB in plist:
        
    elapsed = 0
    error   = 0
    
    for ii in range(nb_iter):
       
        # Add some perturbation
        varb   = 10**(-RSB/20)
        i_foc  = i_foc_0 + np.random.normal(0, varb, i_foc_0.shape)
        i_div  = i_div_0 + np.random.normal(0, varb, i_div_0.shape)
        
        # Estimation      
        t         = time.time() 
        phi_est,_ = estimator.estimate(i_foc, i_div)
        elapsed  += time.time() - t
        error    += sum(sum(abs(phi_foc*tbed.pup - phi_est)))/(N**2)
    
    
    # Update data 
    time_list.append(elapsed/nb_iter)   
    error_list.append(error/nb_iter)
    fig_list.append(phi_est)
    
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
