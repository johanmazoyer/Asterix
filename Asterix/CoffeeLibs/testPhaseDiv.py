# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator
from CoffeeLibs.pzernike import pmap, zernike, pzernike
from sklearn.preprocessing import normalize
import numpy as np
from CoffeeLibs.tools import depadding
from Asterix.propagation_functions import mft


from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt

# # %% Initialisations

# # Chargement des parametres de la simulation
# path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
# config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
# config.validate(Validator(), copy=True)

# # Initialize test bed:
# tbed = custom_bench(config["modelconfig"],'.')
# tbed.grad = "np"
# tbed.lap  = "lap"


# # %% Treatment

# N = tbed.dimScience//tbed.ech

# Images to estimate
[Ro,Theta] = pmap(N,N)

coeff = 1/np.arange(1,6)
coeff[0:3] = [0,0,0]


phi_foc =  normalize(pzernike(Ro,Theta,coeff)) # Astig + defoc
phi_div =  phi_foc + zernike(Ro,Theta,4)

EF_foc = np.exp(1j*phi_foc)
EF_div = np.exp(1j*phi_div)


# i_foc = tbed.psf(EF_foc)

# i_div = tbed.psf(EF_div)

# # %% BBGC

# # Add some perturbation
# varb   = 0 # Variance du bruit
# i_foc  = i_foc + np.random.normal(0, varb, i_foc.shape)
# i_div  = i_div + np.random.normal(0, varb, i_div.shape)


# # %% Estimation

# estimator = Estimator(tbed,**config["Estimationconfig"])
# estimator.disp = True


# est_u,est_d = estimator.estimate(i_foc, i_div)


# EF_est = np.exp(1j*est_u)
# i_est  = tbed.psf(EF_est)
    

# # %%  Plots

# plt.figure(3)
# plt.subplot(3,2,1),plt.imshow(phi_foc,cmap='jet'),plt.title("Valeur Attendu"),plt.colorbar()
# plt.subplot(3,2,2),plt.imshow(est_u,cmap='jet'),plt.title("Estimation"),plt.colorbar()

# plt.subplot(3,2,3),plt.imshow(np.abs(tbed.EF_through(EF_foc)),cmap='jet'),plt.title("abs EF au detecteur"),plt.colorbar()
# plt.subplot(3,2,4),plt.imshow(np.abs(tbed.EF_through(EF_est)),cmap='jet'),plt.title("abs EF au detecteur"),plt.colorbar()


# plt.subplot(3,2,5),plt.imshow(tbed.psf(phi_foc),cmap='jet'),plt.title("PSF"),plt.colorbar()
# plt.subplot(3,2,6),plt.imshow(tbed.psf(est_u),cmap='jet'),plt.title("PSF"),plt.colorbar()


# # plt.subplot(2,3,4),plt.imshow(abb_est,cmap='jet'),plt.title("Estimation"),plt.colorbar()
# # plt.subplot(2,3,5),plt.imshow(phi_abb,cmap='jet'),plt.title("Valeur Attendu"),plt.colorbar()
# # plt.subplot(2,3,6),plt.imshow(abs( tbed.EF_through(phi_abb) - tbed.EF_through(abb_est) ),cmap='jet'),plt.title("Evaluation H avec estimation"),plt.colorbar()

# %%  Test

N = 16
ech = 2
dim_img = N * ech
dim_pup = N

nbres = (N,2*N)

EF_foc = np.ones((N,N))
EF_foc[0,0] = 0

EF_1 = np.ones((N,N))

print(sum(sum(abs(EF_foc)**2)))

factor = 1 / sum(sum( abs(mft(EF_1 ,dim_pup,dim_img,nbres,inverse=False)) ))

i = factor * mft(EF_foc ,dim_pup,dim_img,nbres,inverse=False)

print(sum(sum(abs(i))))


plt.imshow(abs(i))