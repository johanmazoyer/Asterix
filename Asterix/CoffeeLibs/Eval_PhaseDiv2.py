# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator
from CoffeeLibs.pzernike import pmap, zernike, pzernike
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

# Images to estimate
[Ro,Theta] = pmap(N,N)

# varb  = 0 # Variance du bruit
Neval      = 10 # Nombre de point a evaluer
plist      = np.linspace(0, 100, Neval) # Rapport signal a bruit


glist   = [1,1e-1,1e-2,1e-3]
nbh     = len(glist)
Hypvar  = 0
Hypvar2 = 0

maxiter = 15

#Init
coeff = 1/np.arange(1,6)
coeff[0:3] = [0,0,0]
            
phi_foc =  pzernike(Ro,Theta,coeff) # Astig + defoc
phi_div =  phi_foc + zernike(Ro,Theta,4)

EF_foc = np.exp(1j*phi_foc)
EF_div = np.exp(1j*phi_div)

i_foc_0 = tbed.psf(EF_foc)

i_div_0 = tbed.psf(EF_div)

fig = plt.imshow(phi_foc*tbed.pup)
plt.show()
plt.pause(1)

for RSB in plist:
    
    error_list = []
    time_list  = []
    
    for gtol in glist: 
        
        elapsed = 0
        error   = 0
        
        for ii in range(maxiter):
        
            # if hypp == 100 :
            #     hypp = varb
            
            
            # Add some perturbation
            varb = 10**(-RSB/20)
            i_foc  = i_foc_0 + np.random.normal(0, varb, i_foc_0.shape)
            i_div  = i_div_0 + np.random.normal(0, varb, i_div_0.shape)
            
            
            
            # %% Estimation
            
            t = time.time() 
            estimator.gtol = gtol
            phi_est,_ = estimator.estimate(i_foc, i_div)
            elapsed += time.time() - t
            error   += sum(sum(pow(phi_foc*tbed.pup - phi_est*tbed.pup,2)))/(N**2)
        
        time_list.append(elapsed/maxiter)   
        error_list.append(error/maxiter)
        
        fig.set_data(phi_est)
        plt.title("Current")
        plt.show()
        plt.pause(0.01)
        print("Error : " + "%.5f" % error)
        print("Minimize took : " + str(elapsed/60) + " mins\n")
        print(" ^ Simu for  gtol :"+str(gtol)+", varb :"+"%.2f" % varb+"\n")
        print("----------------------------------------")
        
    if not isinstance(Hypvar,int) :    
        Hypvar = np.concatenate((Hypvar,np.array(error_list).reshape(nbh,1)),axis=1)
    else :
        Hypvar = np.array(error_list).reshape(nbh,1)
        
    if not isinstance(Hypvar2,int) :    
        Hypvar2 = np.concatenate((Hypvar2,np.array(time_list).reshape(nbh,1)),axis=1)
    else :
        Hypvar2 = np.array(time_list).reshape(nbh,1)
        
        
# %%  Plots

glist = np.array(glist)
plist = np.array(plist)

plt.figure(3)

fig, ax = plt.subplots(1,1)
img = ax.imshow(Hypvar,cmap='jet',extent=[plist.min(),plist.max(),nbh,0], aspect='auto')


ax.set_yticks(np.arange(nbh)[::-1])
ax.set_xticks(plist[::-1])

ax.set_yticklabels(glist[::-1])
ax.set_xticklabels(plist[::-1].round(2))

ax.set_ylabel('gtol'),ax.set_xlabel('variance du bruit')

fig.colorbar(img)


plt.title("Erreur RMS")

plt.figure(4)
fig, ax = plt.subplots(1,1)
img = ax.imshow(Hypvar2,cmap='jet',extent=[plist.min(),plist.max(),nbh,0], aspect='auto')


ax.set_yticks(np.arange(nbh)[::-1])
ax.set_xticks(plist[::-1])

ax.set_yticklabels(glist[::-1])
ax.set_xticklabels(plist[::-1].round(2))

ax.set_ylabel('gtol'),ax.set_xlabel('variance du bruit')
plt.title("Temps d'execution (en s)")

fig.colorbar(img)

# %% Plot alternatifs

plt.figure(1)
plt.title("Variation du temps d'execution pour différentes valeurs de gtol")
img = plt.plot(plist[::-1],np.transpose(np.fliplr(Hypvar2)))
gliststr = ["{:.0e}".format(x) for x in glist]
plt.legend(gliststr)
plt.ylabel("Temps d'execution en seconde"),plt.xlabel('RSB')

plt.figure(2)
plt.title("Variation de l'erreur pour différentes valeurs de gtol")
img = plt.plot(plist[::-1],np.transpose(np.fliplr(Hypvar))*10e1)
gliststr = ["{:.0e}".format(x) for x in glist]
plt.legend(gliststr)
plt.ylabel("Erreur RMS "+"{:.0e}".format(10e1)),plt.xlabel('RSB')
