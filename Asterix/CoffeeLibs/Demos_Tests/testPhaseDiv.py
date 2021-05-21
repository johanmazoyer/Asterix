# -*- coding: utf-8 -*-
import os
from configobj import ConfigObj
from validate import Validator

from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator

import CoffeeLibs.tools as tls
import numpy as np


# %% Chargement des parametres

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file peut-etre ?
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [1,0]  # List of div factor's images diversity
RSB         = 50000000

# Coeff pour généré les de zernikes de phi_foc
coeff = 1/np.arange(1,6)          
coeff[0:3] = [0,0,0]

# %% Traitement

# Initialisation
tbed      = custom_bench(config,'.')
sim       = data_simulator(tbed,var,div_factors)
estimator = Estimator(**config["Estimationconfig"])

# Definition of phis 
sim.gen_zernike_phi_foc(coeff)
sim.gen_zernike_phi_do([0,0,1])

estimator.var_phi = np.var(tls.gradient_xy(sim.get_phi_foc()))

imgs = sim.gen_div_imgs(RSB) # Compute images 

# Estimation
#known_var = {'downstream_EF':sim.get_EF_do()} # Variables défini comme connu
known_var = {'flux':1, 'fond':0}
e_sim = estimator.estimate(imgs,tbed,div_factors,known_var) # Estimation


# %% Save / Load and plots 

from CoffeeLibs.tools import tempalte_plot

tempalte_plot(sim,e_sim,estimator,disp=True) # Result of minimiz

# tbed.introspect(sim.get_EF(),sim.get_EF_do())  # Introdpection

