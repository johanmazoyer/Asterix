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
tbed.grad = "np"
tbed.lap  = "lap"

# %% Treatment

N = tbed.dimScience//tbed.ech
estimator = Estimator(tbed,**config["Estimationconfig"])

# Images to estimate
[Ro,Theta] = pmap(N,N)

# varb  = 0 # Variance du bruit
Neval      = 10 # Nombre de point a evaluer
plist      = [0,5,10,15,20,25,30,35,40,50] # Rapport signal a bruit


hlist   = [50,200,1e3,1e100]
nbh     = len(hlist)

Hypvar  = 0

maxiter = 10


#Init fig
# phi_foc =  normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6)) # Astig + defoc
# fig = plt.imshow(phi_foc)

for RSB in plist:
    
    error_list = []
    time_list  = []
    
    for hypp in hlist: 
        
        elapsed = 0
        error   = 0
        
        for ii in range(maxiter):
           
            # if hypp == 100 :
            #     hypp = varb
            
            phi_foc =  normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6)) # Astig + defoc
            phi_div = phi_foc + zernike(Ro,Theta,4)
            
            
            EF_foc = np.exp(1j*phi_foc)
            EF_div = np.exp(1j*phi_div)
            
            
            i_foc = tbed.psf(entrance_EF=EF_foc)
            
            i_div = tbed.psf(entrance_EF=EF_div)
            
            
            # %% BBGC
            
            # Add some perturbation
            varb = 10**(-RSB/20)
            i_foc  = i_foc + np.random.normal(0, varb, i_foc.shape)
            i_div  = i_div + np.random.normal(0, varb, i_div.shape)
            
            
            
            # %% Estimation
            
            
            
            t = time.time() 
            estimator.hypp = hypp
            phi_est = estimator.estimate(i_foc, i_div)
            elapsed += time.time() - t
            error   += sum(sum(pow(tbed.EF_through(phi_foc) - phi_est,2)))/(N**2)
        
        time_list.append(elapsed/maxiter)   
        error_list.append(error/maxiter)
        
        # fig.set_data(phi_est)
        # plt.title("Simu for  hyper :"+str(hypp)+", varb :"+"%.2f" % varb)
        # plt.show()
        # plt.pause(0.0001)
        print("Error : " + "%.5f" % error)
        print("Minimize took : " + str(elapsed/60) + " mins\n")
        print(" ^ Simu for  hyper :"+str(hypp)+", varb :"+"%.2f" % varb+"\n")
        print("----------------------------------------")
        
    if not isinstance(Hypvar,int) :    
        Hypvar = np.concatenate((Hypvar,np.array(error_list).reshape(nbh,1)),axis=1)
    else :
        Hypvar = np.array(error_list).reshape(nbh,1)
        
        
# %%  Plots

hlist = np.array(hlist)
plist = np.array(plist)

plt.figure(3)
fig, ax = plt.subplots(1,1)
img = ax.imshow(Hypvar,cmap='jet',extent=[plist.min(),plist.max(),nbh,0], aspect='auto')


ax.set_yticks(np.arange(nbh)[::-1])
ax.set_xticks(plist[::-1])

hliststr = ["{:.0e}".format(x) for x in hlist[::-1]]
ax.set_yticklabels(hliststr)
ax.set_xticklabels(plist[::-1])
ax.set_ylabel('hyper-parametre'),ax.set_xlabel('RSB')

fig.colorbar(img)


# %% Plot alternatifs

plt.figure(1)
plt.title("Variation de l'erreur pour différentes valeurs d'hyper-paramètre")
img = plt.plot(plist[::-1],np.transpose(np.fliplr(Hypvar)))
hliststr = ["{:.0e}".format(x) for x in hlist]
plt.legend(hliststr)
plt.ylabel("Erreur RMS"),plt.xlabel('RSB')