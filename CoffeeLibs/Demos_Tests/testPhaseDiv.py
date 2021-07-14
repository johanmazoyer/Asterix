# -*- coding: utf-8 -*-

from CoffeeLibs.coffee import coffee_estimator,data_simulator,custom_bench
from CoffeeLibs.files_manager import get_ini

from Asterix.estimator import Estimator
from Asterix.Optical_System_functions import coronagraph
import numpy as np


# %% Initialisation

config = get_ini('my_param_file.ini')

# tbed    = custom_bench(config,'.')
tbed      = coronagraph(config['modelconfig'],config['Coronaconfig'])

config["Estimationconfig"]["auto"]  = True
config["Estimationconfig"]["cplx"]  = False
config["Estimationconfig"]["myope"] = False

name = "mySim_cplx"

## -- Constructor by CoffeeLibs
estimator = coffee_estimator(**config["Estimationconfig"])
estimator.simGif = name


# %% Simulation de données

## -- Paramètres
fu = 1
var   = {'downstream_EF':1, 'flux':[fu,fu], 'fond':[0,0]}
div_factors = [0,0.1]  # List of div factor's images diversity
RSB         = None

## -- Coeff du zernike  
coeff = 0.1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

## -- Generation des images avec data_simulator

sim = data_simulator(tbed,var,div_factors)

# ~ Phi_up reel 
sim.gen_zernike_phi_foc(coeff)

# ~ Phi Complex
# phi_r = sim.gen_zernike_phi(coeff)
# phi_i =  sim.gen_zernike_phi([0,0,0.1])
# sim.set_phi_foc(phi_r+1j*phi_i)

# ~ Phi_do 
# sim.gen_zernike_phi_do([0,0,0,1/4])

imgs = sim.gen_div_imgs(RSB) # Compute images 


# %% Traitement

# Variables défini comme connu
known_var = {'downstream_EF':1, 'flux':[fu,fu], 'fond':[0,0]}  
estimator.var_phi      = 1 / np.var(sim.get_phi_foc())

e_sim = estimator.estimate(tbed,imgs,div_factors,known_var) # Estimation

# %% Estimation Plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2,tempalte_plotauto
if isinstance(estimator,Estimator) : estimator = estimator.coffee

tempalte_plotauto(sim,e_sim,estimator,name=name,disp=True)  
tempalte_plot(sim,e_sim,estimator,name=name,disp=True)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator,name=name,disp=True)


# %% Custom plots ##

import matplotlib.pyplot as plt
plt.ion()

# plt.figure("Err div")

# div_id  = 1
# plt.subplot(1,3,1),plt.imshow(e_sim.div_err[:,:,div_id], cmap='jet'),plt.title("Err div 0"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(sim.div_map[:,:,div_id], cmap='jet'),plt.title("True div 0"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(e_sim.div_map[:,:,div_id] - e_sim.div_err[:,:,0], cmap='jet'),plt.title("Diff true div / apriori + err"),plt.colorbar()

plt.figure("Imgs")

plt.suptitle("Img non-corono Johan (votex charge=0)")
plt.subplot(1,2,1),plt.imshow(imgs[:,:,0], cmap='jet'),plt.title("Img 0"),plt.colorbar()
plt.subplot(1,2,2),plt.imshow(imgs[:,:,1], cmap='jet'),plt.title("Img 1"),plt.colorbar()
img_normal = sim.get_img_div(0) # On cree les images

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
