# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:13:39 2021

@author: sjuillar
"""

import unittest
# -*- coding: utf-8 -*-
import os
from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
from CoffeeLibs.pzernike import pmap, zernike, pzernike
import numpy as np
from Asterix.propagation_functions import mft

from configobj import ConfigObj
from validate import Validator

import xlsxwriter as wrt

# %% Initialisation

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
known_var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0,1]  # List of div factor's images diversity
RSB         = 30000

# Initalisation of objetcs
tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed.dimScience//tbed.ech,known_var,div_factors)
estimator = Estimator(tbed,**config["Estimationconfig"])

threshold = 1e-2  # Error max 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]
sim.set_phi_do(1)

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
imgs = sim.gen_div_imgs(tbed)     # On cree les images
cropEF = sim.get_phi_foc()*tbed.pup

# %% Things for write result in a ~nice~ spreadsheet

book  = wrt.Workbook('./save/Tests.xlsx')

Good        = book.add_format({'font_color': 'green'})
NotGood     = book.add_format({'bold': True, 'font_color': 'red'})
labels_font = book.add_format({'bold': True, 'font_color': 'blue'})

def fontchoice(assertRes):
    if (assertRes): return Good
    else          : return NotGood

def init_worksheet(name,labels):
    global sheet
    sheet = book.add_worksheet(name) 
    global col
    col = 0
    row = 0
    for lab in labels : 
        sheet.write(col,row,lab,labels_font)
        row += 1
    col += 1

def new_row(datas,error):
    global col
    row = 0
    for data in datas :
        sheet.write(col,row,data,fontchoice(error))
        row += 1
    col += 1

# %% unit Test

class TestCoffee(unittest.TestCase):
    
    def test_known_var(self):
        
        labels = ["Name","error","errror_on_phi_do","error_on_flux","error_on_fond"]
        init_worksheet("test_known_var",labels)

        #Estimation All fixed
        known_var   = {'downstream_EF':1, 'flux':1, 'fond':0}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        new_row(["All fixed",error,None,None,None], (error<threshold))
        self.assertLess(error,threshold)
        
        known_var   = {'downstream_EF':1}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_flux = sim.get_flux()-e_sim.get_flux()
        error_fond = sim.get_fond()-e_sim.get_fond()
        new_row(["downstream_EF fixed",error,None,error_flux,error_fond], (error<threshold))
        self.assertLess(error,threshold)
        
        known_var   = {'flux':1, 'fond':0}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_on_phi_do = sum(sum(sim.get_phi_do()-e_sim.get_phi_do()))/cropEF.size
        new_row(["Flux/fond fixed",error,error_on_phi_do,None,None], (error<threshold))
        self.assertLess(error,threshold)
        
        known_var   = {'downstream_EF':1,'flux':1}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_fond = sim.get_fond()-e_sim.get_fond()
        new_row(["Flux fixed",error,None,None,error_fond], (error<threshold))
        self.assertLess(error,threshold)
        
        known_var   = {'downstream_EF':1,'fond':0}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_flux = sim.get_flux()-e_sim.get_flux()
        new_row(["Fond fixed",error,None,error_flux,None], (error<threshold))
        self.assertLess(error,threshold)
        
        known_var   = {}
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_on_phi_do = sum(sum(sim.get_phi_do()-e_sim.get_phi_do()))/cropEF.size
        error_flux = sim.get_flux()-e_sim.get_flux()
        error_fond = sim.get_fond()-e_sim.get_fond()
        new_row(["Estime All",error,error_on_phi_do,error_flux,error_fond], (error<threshold))
        self.assertLess(error,threshold)
        

    def test_diff_corono(self):
        
        labels = ["Corno","error"]
        init_worksheet("test_diff_corono",labels)

        known_var = {'downstream_EF':1, 'flux':1, 'fond':0}
        #Estimation All fixed
        
        tbed.set_corono("lyot") 
        imgs  = sim.gen_div_imgs(tbed)     # On cree les images
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        estimator.tbed = tbed
        error = sum(sum(sim.get_phi_foc()*tbed.pup-e_sim.get_phi_foc()))/cropEF.size
        new_row(["lyot",error], (error<threshold))
        self.assertLess(error,threshold)
        
        tbed.set_corono("R&R") 
        imgs  = sim.gen_div_imgs(tbed)     # On cree les images
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        estimator.tbed = tbed
        error = sum(sum(sim.get_phi_foc()*tbed.pup-e_sim.get_phi_foc()))/cropEF.size
        new_row(["R&R",error], (error<threshold))
        self.assertLess(error,threshold)
        
        tbed.set_corono("4q") 
        imgs  = sim.gen_div_imgs(tbed)     # On cree les images
        estimator.tbed = tbed
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(sim.get_phi_foc()*tbed.pup-e_sim.get_phi_foc()))/cropEF.size
        new_row(["4q",error], (error<threshold))
        self.assertLess(error,threshold)
        
        tbed.set_corono("4q")
        tbed.zbiais = True
        imgs  = sim.gen_div_imgs(tbed)     # On cree les images
        estimator.tbed = tbed
        e_sim = estimator.estimate(imgs,div_factors,known_var)
        error = sum(sum(sim.get_phi_foc()*tbed.pup-e_sim.get_phi_foc()))/cropEF.size
        new_row(["4q zbiais",error], (error<threshold))
        self.assertLess(error,threshold)
        
# %% Main


unittest.main()
book.close()