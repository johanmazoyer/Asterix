# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:22:28 2021

@author: sjuillar
"""

from CoffeeLibs.coffee import custom_bench, coffee_estimator

import matplotlib.pyplot as plt

from CoffeeLibs.tools import get_pup_size, get_science_sampling, detector_shifts2div_factors
from CoffeeLibs.files_manager import get_ini,get_fits_as_imgs

# %% Varaibles

div_dist = [0,12]  # List of div factor's EN DECALAGE DU DETECTEUR mm
Ld  = 640          # longeur d'onde
F_D = 61.5         # Nombre d'ouverture
name = "NoCornoTests"

# Info to get your fits
path_root  = "imgs"                        # folder
prefix     = "PSF_640nm_pup823_noLyot_P2"  # Prefix of your fits Or Name if you have only one (no need to put ".fits") 
exts       = ["","_p12mm"]                 # List of extention of your fits

# %% Chargement des parametres

config = get_ini('my_param_file.ini',"..\..\Param_configspec.ini")
imgs = get_fits_as_imgs(path_root,prefix,exts)

# Update config from our datas
# config["modelconfig"]["Science_sampling"] = get_science_sampling(imgs[:,:,0])
# config["modelconfig"]["diam_pup_in_pix"]  = get_pup_size(imgs[:,:,0])
config["modelconfig"]["Science_sampling"] = 6
config["modelconfig"]["diam_pup_in_pix"]  = 23
config['modelconfig']['dimScience']       = imgs.shape[0]

div_factors = detector_shifts2div_factors(div_dist,Ld,F_D)


# %% Traitement

## Initialisation of objects ##
tbed      = custom_bench(config,'.')
estimator = coffee_estimator(**config["Estimationconfig"])
estimator.bound      = None
estimator.var_phi    = 0
estimator.simGif     = name

## Estimation ##

known_var = {'downstream_EF':1}  # Variables défini comme connu

e_sim = estimator.estimate(imgs,tbed,div_factors,known_var) # Estimation


# %% Save / Load and plots 

from CoffeeLibs.tools import tempalte_plot2

tempalte_plot2(imgs,e_sim,estimator,name=name,disp=True,save=False)


