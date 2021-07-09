# -*- coding: utf-8 -*-

from CoffeeLibs.coffee import coffee_estimator,data_simulator,custom_bench
from CoffeeLibs.files_manager import get_ini

from Asterix.estimator import Estimator
from Asterix.Optical_System_functions import coronagraph
import numpy as np

# %% Initialisation

config = get_ini('my_param_file.ini')

tbed    = custom_bench(config,'.')
# tbed      = coronagraph(config['modelconfig'],config['Coronaconfig'])
name = "mySim_cplx_auto"

config["Estimationconfig"]["auto"] = True
config["Estimationconfig"]["cplx"] = False

## -- Constructor by CoffeeLibs
estimator = coffee_estimator(**config["Estimationconfig"])
estimator.simGif = name

## -- Constructor by asterix
# config["Estimationconfig"]["estimation"] = "coffee"
# estimator = Estimator(config["Estimationconfig"],tbed)


# %% Simulation de données

## -- Paramètres
fall = 1
var   = {'downstream_EF':1, 'flux':[fall,fall], 'fond':[0,0]}
div_factors = [0,0.12]  # List of div factor's images diversity
RSB         = None

## -- Coeff du zernike  
coeff = 1/np.arange(1,6)         
coeff[0:3] = [0,0,0]


## -- Generation des images avec data_simulator

sim = data_simulator(tbed,var,div_factors)

# Phi_up 
# sim.gen_zernike_phi_foc(coeff)

# Phi Complex
phi_r = sim.gen_zernike_phi(coeff)
phi_i =  sim.gen_zernike_phi([0,0,0.1])
sim.set_phi_foc(phi_r+1j*phi_i)

# Phi_do
# sim.gen_zernike_phi_do([0,0,0,1/4])

imgs = sim.gen_div_imgs(RSB) # Compute images 


# %% Traitement

known_var = {'downstream_EF':1, 'flux':[fall,fall], 'fond':[0,0]}  # Variables défini comme connu
estimator.var_phi      = 0 / np.var(sim.get_phi_foc())

## -- From Asterix
# e_sim = estimator.estimate(tbed,
#                            imgs=imgs,
#                            div_factors=div_factors,
#                            known_var=known_var,
#                            result_type="simulator")

## -- From CoffeeLibs
e_sim = estimator.estimate(tbed,imgs,div_factors,known_var) # Estimation

# %% Plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2,tempalte_plotauto

## Result of minimiz ##

if isinstance(estimator,Estimator) : estimator = estimator.coffee

tempalte_plotauto(sim,e_sim,estimator,name=name,disp=True)  
tempalte_plot(sim,e_sim,estimator,name=name,disp=True)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator,name=name,disp=True)

## Introdpection ##
# tbed.introspect(sim.get_EF(),sim.get_EF_do())


## Test zbiaz ##

import numpy as np
tbed.zbiais = True
one = np.ones((sim.N,sim.N))
tbed.introspect(one,one)

# %% Save as fits 

# =============================================================================

# 
# from astropy.io import fits
# hdu = fits.PrimaryHDU(sim.get_phi_foc())
# hdul = fits.HDUList([hdu])
# hdul.writeto('phase_tilt.fits')
# hdu = fits.PrimaryHDU(imgs[:,:,0])
# hdul = fits.HDUList([hdu])
# hdul.writeto('image4q_tilt.fits')

# =============================================================================
