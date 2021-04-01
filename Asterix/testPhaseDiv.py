# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator
from CoffeeLibs.pzernike import pmap, zernike
from sklearn.preprocessing import normalize
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt

# %% Initialisations

# Chargement des parametres de la simulation
Asterixroot = os.path.dirname(os.path.realpath(__file__))

parameter_file = Asterixroot + os.path.sep+ 'Example_param_file.ini'
configspec_file = Asterixroot + os.path.sep + "Param_configspec.ini"



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

N = tbed.dim_overpad_pupil

# Images to estimate
[Ro,Theta] = pmap(N,N)
phi_foc =  normalize(zernike(Ro,Theta,4) + zernike(Ro,Theta,6)) # Astig + defoc
phi_div = phi_foc + zernike(Ro,Theta,4)


EF_foc = np.exp(1j*phi_foc)
EF_div = np.exp(1j*phi_div)

plt.subplot(1,2,1),plt.imshow(abs(EF_foc),cmap='jet'),plt.title("Evaluation H avec Valeur Attendu"),plt.colorbar()
plt.subplot(1,2,2),plt.imshow(abs(tbed.EF_through(EF_foc)),cmap='jet'),plt.title("Evaluation H avec Valeur Attendu"),plt.colorbar()


# %% BBGC

i_foc = tbed.psf(entrance_EF=EF_foc)

i_div = tbed.psf(entrance_EF=EF_div)


# %% BBGC

# Add some perturbation

# varb  = 0 # Variance du bruit
# i_foc  = i_foc + np.random.normal(0, varb, i_foc.shape)
# i_div  = i_div + np.random.normal(0, varb, i_div.shape)



# %% Estimation
estimator = Estimator(tbed,**config["coffeeEstimator"])

phi_est = estimator.estimate(i_foc, i_div)

i_est = tbed.psf(entrance_EF=np.exp(1j*phi_est))


# %%  Plots 



plt.figure(3)
plt.subplot(2,2,1),plt.imshow(phi_est,cmap='jet'),plt.title("Estimation"),plt.colorbar()
plt.subplot(2,2,2),plt.imshow(phi_foc,cmap='jet'),plt.title("Valeur Attendu"),plt.colorbar()

plt.subplot(2,2,3),plt.imshow(i_est,cmap='jet'),plt.title("Evaluation H avec estimation"),plt.colorbar()
plt.subplot(2,2,4),plt.imshow(i_foc,cmap='jet'),plt.title("Evaluation H avec Valeur Attendu"),plt.colorbar()


