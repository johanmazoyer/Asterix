# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator
from CoffeeLibs.pzernike import pmap, zernike
from sklearn.preprocessing import normalize
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt
import time


# %% Initialisations

# Chargement des parametres de la simulation
Asterixroot = os.path.dirname(os.path.realpath(__file__))

parameter_file = Asterixroot + os.path.sep+ 'my_param_file.ini'
configspec_file = Asterixroot + os.path.sep + "..\Param_configspec.ini"



config = ConfigObj(parameter_file,
                   configspec=configspec_file,
                   default_encoding="utf8")

vtor = Validator()
checks = config.validate(vtor, copy=True)

modelconfig = config["modelconfig"]

model_dir = os.path.join(Asterixroot, "Model") + os.path.sep
Model_local_dir = os.path.join(config["Data_dir"], "Model_local") + os.path.sep

# Initialize test bed:
tbed = custom_bench(modelconfig,model_dir=model_dir)



# %% Treatment

N = tbed.dimScience//tbed.ech
estimator = Estimator(tbed,**config["Estimationconfig"])
estimator.hypp = 70

# Images to estimate
[Ro,Theta] = pmap(N,N)

varb  = 0.1 # Variance du bruit

lap_methodes  = ["grad","lap"]
grad_methodes = ["sobel","magn","np"]

nbl = len(lap_methodes)
nbg = len(grad_methodes)


res_matrix = 0

#Init fig
phi_foc =  normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6)) # Astig + defoc
fig = plt.imshow(phi_foc)

for lap in lap_methodes:
    
    error_list = []
    time_list  = []
    
    for grad in grad_methodes :
           
        tbed.grad = grad
        tbed.lap  = lap
        
        phi_foc =  normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6)) # Astig + defoc
        phi_div = phi_foc + zernike(Ro,Theta,4)
        
        
        EF_foc = np.exp(1j*phi_foc)
        EF_div = np.exp(1j*phi_div)
        
        
        i_foc = tbed.psf(entrance_EF=EF_foc)
        
        i_div = tbed.psf(entrance_EF=EF_div)
        
        
        # %% BBGC
        
        # Add some perturbation
        i_foc  = i_foc + np.random.normal(0, varb, i_foc.shape)
        i_div  = i_div + np.random.normal(0, varb, i_div.shape)
        
        
        
        # %% Estimation
        
        
        
        t = time.time() 
        phi_est,_ = estimator.estimate(i_foc, i_div)
        elapsed = time.time() - t
        error = sum(sum(pow(phi_foc - phi_est,2)))/(N**2)
        
        time_list.append(elapsed)   
        error_list.append(error)
        
        # fig.set_data(phi_est)
        # plt.title("Simu for  lap :"+ str(lap) +", grad :" + grad)
        # plt.show()
        # plt.pause(0.0001)
        print("Error : " + "%.5f" % error)
        print("Minimize took : " + str(elapsed/60) + " mins\n")
        print(" ^ Simu for  lap :"+ str(lap) +", grad :" + grad +"\n")
        print("----------------------------------------")
        
    if not isinstance(res_matrix,int) :    
        res_matrix = np.concatenate((res_matrix,np.array(error_list).reshape(nbg,1)),axis=1)
    else :
        res_matrix = np.array(error_list).reshape(nbg,1)
        
        
# %%  Plots

lap_methodes = np.array(lap_methodes)
grad_methodes = np.array(grad_methodes)

fig, ax = plt.subplots(1,1)
img = ax.imshow(res_matrix,cmap='jet',extent=[0,nbl,0,nbg], aspect='auto')


ax.set_xticks(np.arange(nbl))
ax.set_yticks(np.arange(nbg))

ax.set_xticklabels(lap_methodes)
ax.set_yticklabels(grad_methodes[::-1])

fig.colorbar(img)