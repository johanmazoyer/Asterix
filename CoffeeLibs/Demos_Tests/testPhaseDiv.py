
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

config["Estimationconfig"]["auto"]  = True
config["Estimationconfig"]["cplx"]  = False
config["Estimationconfig"]["myope"] = False

name = "test"
gif  = True

# %% Simulation de données

## -- Paramètres
fu = 1
# Conseil : Plus flux elevé, plus on pourra être précis (explication -> cf. coffee.py )

var   = {'flux':fu, 'fond':0}
div_factors = [0,0.0000001]
sim = data_simulator(tbed,var,div_factors) # Init la simulation

## -- Coeff du zernike  
coeff = 0.1/np.arange(1,7) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]

# ~ Phi_up reel 
sim.gen_zernike_phi_foc(coeff)

# ~ Phi Complex
if config["Estimationconfig"]["cplx"] == True :
    phi_r = sim.gen_zernike_phi(coeff)
    phi_i =  sim.gen_zernike_phi([0,0,0.0001])
    sim.set_phi_foc(phi_r+1j*phi_i)

# ~ Phi_do 
# sim.gen_zernike_phi_do([0,0.2])

RSB  = None
imgs = sim.gen_div_imgs(RSB) # Compute images 

# %% Traitement

## -- Constructor by CoffeeLibs
estimator = coffee_estimator(**config["Estimationconfig"])
estimator.bound = None
if gif : estimator.simGif = name

# Variables défini comme connu
known_var = {'downstream_EF':1, 'flux':fu,'fond':0}
estimator.var_phi      = 0 / np.var(sim.get_phi_foc())
div_factors_knwon = np.array(div_factors) # Add error on div_factor ?

e_sim = estimator.estimate(tbed,imgs,div_factors_knwon,known_var) # Estimation

# %% Estimation Plots 

from CoffeeLibs.tools import tempalte_plot,tempalte_plot2,tempalte_plotauto
saveall = True

e_sim.set_phi_foc(e_sim.get_phi_foc()) # Spot a piston you need to remove ? 

tempalte_plotauto(sim,e_sim,estimator,name=name,disp=True,save=saveall)  
tempalte_plot(sim,e_sim,estimator,name=name,disp=True,save=saveall)     
tempalte_plot2(sim.gen_div_imgs(),e_sim,estimator,name=name,disp=True,save=saveall)


# %% Custom plots ##

import matplotlib.pyplot as plt
plt.ion()

# plt.figure("Err div")
# for div_id in range(sim.nb_div):
#     plt.subplot(sim.nb_div,3,1+3*div_id),plt.imshow(e_sim.get_div_est(div_id), cmap='jet'),plt.title("Estimated diversity "+str(div_id)),plt.colorbar()
#     plt.subplot(sim.nb_div,3,2+3*div_id),plt.imshow(sim.div_map[:,:,div_id], cmap='jet'),plt.title("True divversity  "+str(div_id)),plt.colorbar()
#     plt.subplot(sim.nb_div,3,3+3*div_id),plt.imshow(e_sim.get_div_est(div_id) - sim.div_map[:,:,div_id], cmap='jet'),plt.title("Difference"),plt.colorbar()


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
