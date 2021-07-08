# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:52:25 2021

@author: sjuillar
"""
# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, data_simulator
import numpy as np

from configobj import ConfigObj
from validate import Validator

import matplotlib.pyplot as plt

# %% Initialisation

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
tbed      = custom_bench(config,'.')
sim       = data_simulator(tbed,var,div_factors)

# %% Expériences MFT 

from Asterix.propagation_functions import mft
from CoffeeLibs.tools import depadding

N  = tbed.dimScience
Ne = tbed.dimScience//tbed.ech
test = tbed.pup * np.exp(1j*0)

mft1   = mft(test,Ne,N,Ne)
mft12  = mft(test,Ne,Ne,Ne)

mft2 = mft(mft1,N,N,N)
mft3 = mft(mft1,N,Ne,Ne,inverse=True)
mft4 = mft(mft1,Ne,N,Ne)
mft5 = mft(mft12,Ne,Ne,Ne)


plt.figure(2)
plt.subplot(2,4,1),plt.imshow(abs(mft1),cmap='jet'),plt.title("E = "+str(np.sum(abs(mft1)**2))[:5]),plt.colorbar()
plt.subplot(2,4,2),plt.imshow(np.log(abs(mft1)**2),cmap='jet'),plt.title("mft ech"),plt.colorbar()
plt.subplot(2,4,3),plt.imshow(abs(mft12),cmap='jet'),plt.title("mft pas ech : E = "+str(np.sum(abs(mft12)**2))[:5]),plt.colorbar()
plt.subplot(2,4,4),plt.imshow(np.log(abs(mft12)**2),cmap='jet'),plt.title("mft pas ech"),plt.colorbar()

plt.subplot(2,4,5),plt.imshow(abs(mft2),cmap='jet'),plt.title("N,N,N : E = "+str(np.sum(abs(mft2)**2))[:5]),plt.colorbar()
plt.subplot(2,4,6),plt.imshow(abs(mft3),cmap='jet'),plt.title("N,Ne,Ne : E = "+str(np.sum(abs(mft3)**2))[:5]+"\n Ce que j'utilisait maitenant"),plt.colorbar()
plt.subplot(2,4,7),plt.imshow(abs(mft4),cmap='jet'),plt.title("Ne,N,Ne  :E = "+str(np.sum(abs(mft4)**2))[:5]+"\n Ce que j'utilisait avant (avec depadding)"),plt.colorbar()
plt.subplot(2,4,8),plt.imshow(abs(mft5),cmap='jet'),plt.title("Pas ech : E = "+str(np.sum(abs(mft5)**2))[:5]),plt.colorbar()

print(np.sum(abs(mft5 - test)))

print(np.sum(abs(mft5)))
print(np.sum(abs(test)))

print(np.sum(abs(mft5 - test)))
print(np.sum(abs(mft3 - test)))
print(np.sum(abs(depadding(mft4,2) - test)))

plt.imshow(abs(mft3 - test))
