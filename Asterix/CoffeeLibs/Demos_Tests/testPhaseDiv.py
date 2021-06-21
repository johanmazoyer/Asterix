# -*- coding: utf-8 -*-
import os
from configobj import ConfigObj
from validate import Validator

from CoffeeLibs.coffee import coffee_estimator,data_simulator,custom_bench
from Asterix.estimator import Estimator
import numpy as np

# %% Chargement des parametres

name = "mySim"

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file peut-etre ?
var   = {'downstream_EF':1, 'flux':2e2, 'fond':1e3}
div_factors = [0,12.2]  # List of div factor's images diversity
RSB         = 10

# Coeff pour généré les de zernikes de phi_foc
coeff = 1/np.arange(1,6)          
coeff[0:3] = [0,0,0]

# %% Traitement

## Initialisation of objects ##
tbed      = custom_bench(config,'.')
sim       = data_simulator(tbed,var,div_factors)

# Constructor by CoffeeLibs
# estimator = coffee_estimator(**config["Estimationconfig"])

# Constructor by asterix
estimator = Estimator(config["Estimationconfig"],tbed)

## Generation of images to test estimator
sim.gen_zernike_phi_foc(coeff)
# sim.gen_zernike_phi_do([0,0,0,1/30])

imgs = sim.gen_div_imgs(RSB) # Compute images 
estimator.var_phi      = 1 / np.var(sim.get_phi_foc())

## Estimation ##

known_var = {'downstream_EF':1}  # Variables défini comme connu

# From Asterix
e_sim = estimator.estimate(tbed,imgs=imgs,div_factors=div_factors,known_var=known_var,result_type="simulator")
# From CoffeeLibs
# e_sim = estimator.estimate(tbed,imgs,div_factors,known_var) # Estimation

# %% Save / Load and plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2
# import matplotlib.pyplot as plt

## Result of minimiz ##
# tempalte_plot(sim,e_sim,estimator,name=name,disp=True)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator.coffee,name=name,disp=True)

## Introdpection ##
# tbed.introspect(sim.get_EF(),sim.get_EF_do())