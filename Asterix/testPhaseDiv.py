# -*- coding: utf-8 -*-
import os
import Asterix.InstrumentSimu_functions as instr
from CoffeeLibs.coffee import Estimator
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

# Initialize thd:
thd2 = instr.THD2_testbed(modelconfig,
                          config["DMconfig"],
                          config["Coronaconfig"],
                          save_fits=True,
                          model_dir=model_dir,
                          Model_local_dir=Model_local_dir)



# %% Traitement

N = thd2.dim_overpad_pupil

# Images a bases for the estimator 
EF_foc = np.zeros((N,N)) #Trouver ou est le prametre dans config file
EF_div = np.zeros((N,N))

i_foc = thd2.EF_through(entrance_EF=EF_foc,
                        wavelength=None,
                        DM1phase=0.,
                        DM3phase=0.,
                        noFPM=False,
                        save_all_planes_to_fits=False,
                        dir_save_fits=None)

i_div = thd2.EF_through(entrance_EF=EF_div,
                        wavelength=None,
                        DM1phase=0.,
                        DM3phase=0.,
                        noFPM=False,
                        save_all_planes_to_fits=False,
                        dir_save_fits=None)


# %% Ajut de bruit

# Add some perturbation
varb  = 1e-5 # Variance du bruit
i_foc  = i_foc + np.random.normal(0, varb, i_foc.shape)
i_div  = i_div + np.random.normal(0, varb, i_div.shape)



# %% Estimation
estimator = Estimator(thd2,**config["coffeeEstimator"])

EF_est = estimator.estimate(i_foc, i_div)

i_est = thd2.EF_through(entrance_EF=EF_est,
                        wavelength=None,
                        DM1phase=0.,
                        DM3phase=0.,
                        noFPM=False,
                        save_all_planes_to_fits=False,
                        dir_save_fits=None)


# %%  Plots 



plt.figure(3)
plt.subplot(2,2,1),plt.imshow(EF_est,cmap='jet'),plt.title("Estimation"),plt.colorbar()
plt.subplot(2,2,2),plt.imshow(EF_foc,cmap='jet'),plt.title("Valeur Attendu"),plt.colorbar()

plt.subplot(2,2,3),plt.imshow(abs(i_foc),cmap='jet'),plt.title("Evaluation H avec estimation"),plt.colorbar()
plt.subplot(2,2,4),plt.imshow(abs(i_est),cmap='jet'),plt.title("Evaluation H avec Valeur Attendu"),plt.colorbar()


