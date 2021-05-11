# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
import numpy as np

from configobj import ConfigObj
from validate import Validator

# %% Chargement des parametres

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file peut-etre ?
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [-1,1,0,2,-2]  # List of div factor's images diversity
RSB         = 30000

# Coeff pour généré les de zernikes de phi_foc
coeff = 1/np.arange(1,6)          
coeff[0:3] = [0,0,0]

# %% Initalisation of objetcs and generation of images

tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed,var,div_factors)
estimator = Estimator(**config["Estimationconfig"])

# On set phi_foc et phi _do 
sim.gen_zernike_phi_foc(coeff)
sim.gen_zernike_phi_do([0,0,1])

imgs = sim.gen_div_imgs(RSB) # On cree les images bruité


# %% Traitement 


known_var = {'flux':1, 'fond':0} # Variables défini comme connu

e_sim = estimator.estimate(imgs,tbed,div_factors,known_var) # Estimation


# %% Save / Load and plots 

from CoffeeLibs.tools import tempalte_plot
import pickle

# with open('./save/9999evals', 'rb') as f:
#     e_sim = pickle.load(f)

# with open('./save/9999evals', 'wb') as f:
#     pickle.dump(e_sim, f)  

tempalte_plot(sim,e_sim)                       # Result of minimiz

tbed.introspect(sim.get_EF(),sim.get_EF_do())  # Introdpection

