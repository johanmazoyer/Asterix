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
name = "mySim_cplx"

config["Estimationconfig"]["auto"]  = False
config["Estimationconfig"]["cplx"]  = False
config["Estimationconfig"]["myope"] = False

## -- Constructor by asterix
config["Estimationconfig"]["estimation"] = "coffee"
estimator = Estimator(config["Estimationconfig"],tbed)


# %% Simulation de données

## -- Paramètres
fall = 1
var   = {'downstream_EF':1, 'flux':[fall,fall], 'fond':[0,0]}
div_factors = [0,1/100]  # List of div factor's images diversity
RSB         = None

## -- Coeff du zernike  
coeff = 0.01/np.arange(1,6)         
coeff[0:3] = [0,0,0]
coeff = [0,0,0,1/100]

## -- Generation des images avec data_simulator

sim = data_simulator(tbed,var,div_factors)

# Phi_up 
sim.gen_zernike_phi_foc(coeff)

# Phi Complex
# phi_r = sim.gen_zernike_phi(coeff)
# phi_i =  sim.gen_zernike_phi([0,0,0.1])
# sim.set_phi_foc(phi_r+1j*phi_i)

# Phi_do
# sim.gen_zernike_phi_do([0,0,0,1/4])

imgs = sim.gen_div_imgs(RSB) # Compute images 


# %% Traitement

known_var = {'downstream_EF':1, 'flux':[fall,fall], 'fond':[0,0]}  # Variables défini comme connu
estimator.var_phi      = 0 / np.var(sim.get_phi_foc())

## -- From Asterix
e_sim = estimator.estimate(tbed,
                            imgs=imgs,
                            div_factors=div_factors,
                            known_var=known_var,
                            result_type="simulator")

# %% Estimation Plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2,tempalte_plotauto
if isinstance(estimator,Estimator) : estimator = estimator.coffee

# tempalte_plotauto(sim,e_sim,estimator,name=name,disp=True)  
tempalte_plot(sim,e_sim,estimator,name=name,disp=True)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator,name=name,disp=True)


# %% Custom plots ##

# import matplotlib.pyplot as plt
# if isinstance(estimator,Estimator) : estimator = estimator.coffee

# plt.ion()
# plt.figure("Err div")

# div_id  = 1
# plt.subplot(1,3,1),plt.imshow(e_sim.div_err[:,:,div_id], cmap='jet'),plt.title("Err div 0"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(sim.div_map[:,:,div_id], cmap='jet'),plt.title("True div 0"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(e_sim.div_map[:,:,div_id] - e_sim.div_err[:,:,0], cmap='jet'),plt.title("Diff true div / apriori + err"),plt.colorbar()

# %% Introdpection ##

# tbed.introspect(sim.get_EF(),sim.get_EF_do())

# %% Save as fits 

# from astropy.io import fits

# hdu = fits.PrimaryHDU(sim.get_phi_foc())
# hdul = fits.HDUList([hdu])
# hdul.writeto('phase_tilt.fits')

# hdu = fits.PrimaryHDU(imgs[:,:,0])
# hdul = fits.HDUList([hdu])
# hdul.writeto('image4q_defocnoz2.fits')
