# -*- coding: utf-8 -*-
import os
from configobj import ConfigObj
from validate import Validator

from CoffeeLibs.coffee import coffee_estimator,data_simulator,custom_bench
from CoffeeLibs.files_manager import get_ini

from Asterix.estimator import Estimator
from Asterix.Optical_System_functions import coronagraph
import numpy as np

# %% Initialisation

config = get_ini('my_param_file.ini',"..\..\Param_configspec.ini")
tbed      = custom_bench(config,'.')
corono    = coronagraph(config['modelconfig'],config['Coronaconfig'])

# tbed = corono

name = "mySim_auto"

config["Estimationconfig"]["auto"]  = True
config["Estimationconfig"]["cplx "] = False

## -- Constructor by CoffeeLibs
estimator = coffee_estimator(**config["Estimationconfig"])
estimator.simGif = name
## -- Constructor by asterix
# config["Estimationconfig"]["estimation"] = "coffee"
# estimator = Estimator(config["Estimationconfig"],tbed)


# %% Simulation de données

# Paramètres
var   = {'downstream_EF':1, 'flux':[1,1], 'fond':[0,0]}
div_factors = [0,1]  # List of div factor's images diversity
RSB         = 500000000

# Coeff du zernike  
coeff = 1/np.arange(1,6)         
coeff[0:3] = [0,0,0]


## -- Generation des images avec data_simulator

sim = data_simulator(tbed,var,div_factors)
sim.gen_zernike_phi_foc(coeff)
# sim.gen_zernike_phi_do([0,0,0,1/30])

imgs = sim.gen_div_imgs(RSB) # Compute images 


# %% Traitement

known_var = {'downstream_EF':1, 'flux':[1,1], 'fond':[0,0]}  # Variables défini comme connu
estimator.var_phi      = 0 / np.var(sim.get_phi_foc())

# From Asterix
# e_sim = estimator.estimate(tbed,
#                            imgs=imgs,
#                            div_factors=div_factors,
#                            known_var=known_var,
#                            result_type="simulator")

# From CoffeeLibs
e_sim = estimator.estimate(tbed,imgs,div_factors,known_var) # Estimation

# %% Save / Load and plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2,tempalte_plotauto
# import matplotlib.pyplot as plt

## Result of minimiz ##

if isinstance(estimator,Estimator) : estimator = estimator.coffee

# tempalte_plotauto(sim,e_sim,estimator,name=name,disp=True)  
tempalte_plot(sim,e_sim,estimator,name=name,disp=True)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator,name=name,disp=True)

## Introdpection ##
# tbed.introspect(sim.get_EF(),sim.get_EF_do())