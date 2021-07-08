# -*- coding: utf-8 -*-

from CoffeeLibs.coffee import coffee_estimator,data_simulator,custom_bench
from CoffeeLibs.files_manager import get_ini

from Asterix.estimator import Estimator
from Asterix.Optical_System_functions import coronagraph
import numpy as np

# %% Initialisation

config = get_ini('my_param_file.ini')

tbed_me      = custom_bench(config,'.')
tbed_jo      = coronagraph(config['modelconfig'],config['Coronaconfig'])

# Paramètres
fall = 1
var   = {'downstream_EF':1, 'flux':[fall,fall], 'fond':[0,0]}
div_factors = [0,1/10]  # List of div factor's images diversity
RSB         = None

# %% Simulation de données


# Coeff du zernike  
# coeff = 1/np.arange(1,6)         
# coeff[0:3] = [0,0,0]
coeff = [0,0,0,1/100]
# coeff = [0,0,0,1/100]


sim_me = data_simulator(tbed_me,var,div_factors)
sim_jo = data_simulator(tbed_jo,var,div_factors)

sim_me.gen_zernike_phi_foc(coeff)
sim_jo.gen_zernike_phi_foc(coeff)


imgs_me = sim_me.gen_div_imgs(RSB) # Compute images 
imgs_jo = sim_jo.gen_div_imgs(RSB) # Compute images 


import matplotlib.pyplot as plt
plt.ion()

plt.subplot(2,3,1),plt.imshow(imgs_me[:,:,0],cmap='jet'),plt.colorbar(),plt.title("me focalisé")
plt.subplot(2,3,2),plt.imshow(imgs_jo[:,:,0],cmap='jet'),plt.colorbar(),plt.title("jo focalisé")
plt.subplot(2,3,3),plt.imshow(imgs_me[:,:,0] - imgs_jo[:,:,0],cmap='jet'),plt.colorbar(),plt.title("diff focalisé")

plt.subplot(2,3,4),plt.imshow(imgs_me[:,:,1],cmap='jet'),plt.colorbar(),plt.title("me diversité")
plt.subplot(2,3,5),plt.imshow(imgs_jo[:,:,1],cmap='jet'),plt.colorbar(),plt.title("jo diversité")
plt.subplot(2,3,6),plt.imshow(imgs_me[:,:,1] - imgs_jo[:,:,1],cmap='jet'),plt.colorbar(),plt.title("diff diversité")

