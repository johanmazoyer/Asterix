# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:13:39 2021

@author: sjuillar
"""

from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
import numpy as np

import os
from configobj import ConfigObj
from validate import Validator

import xlsxwriter as wrt
import unittest

# %% Chargement des parametres

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
known_var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0,1]  # List of div factor's images diversity
RSB         = 30000

# %% Initalisation of objetcs

tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed,known_var,div_factors)
estimator = Estimator(tbed,**config["Estimationconfig"])

threshold = 1e-2  # Error max 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]
sim.set_phi_do(1)

sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])   # On phi do

imgs = sim.gen_div_imgs()         # On cree les images
cropEF = sim.get_phi_foc()*tbed.pup

# %% Things to write result in a ~nice~ spreadsheet

book  = wrt.Workbook('./save/UnitTests.xlsx')
sheet = book.add_worksheet("setting")
sheet.write_comment(0,0,str(config))

labels = ["Name","error","errror_on_phi_do","error_on_flux","error_on_fond","time"]

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

def do_a_test(name):
        e_sim = estimator.estimate(imgs,tbed,div_factors,known_var)
        error = sum(sum(cropEF-e_sim.get_phi_foc()))/cropEF.size
        error_flux = sim.get_flux()-e_sim.get_flux()
        error_fond = sim.get_fond()-e_sim.get_fond()
        error_on_phi_do = sum(sum(sim.get_phi_do()-e_sim.get_phi_do()))/cropEF.size
        new_row([name,error,error_on_phi_do,error_flux,error_fond,estimator.toc], (error<threshold))
        return error

# %% unit Test

class TestCoffee(unittest.TestCase):
    
    def test_known_var(self):
        
        init_worksheet("test_known_var",labels)

        #Estimation All fixed
        known_var   = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        error = do_a_test("All fixed")
        self.assertLess(error,threshold)
        
        known_var   = {'downstream_EF':sim.get_EF_do()}
        error = do_a_test("downstream_EF fixed")
        self.assertLess(error,threshold)
        
        known_var   = {'flux':1, 'fond':0}
        error = do_a_test("Flux/fond fixed")
        self.assertLess(error,threshold)

        
        known_var   = {'flux':1}
        error = do_a_test("Flux fixed")
        self.assertLess(error,threshold)
        
        known_var   = {'downstream_EF':sim.get_EF_do(),'fond':0}
        error = do_a_test("Fond and downstream_EF fixed")
        self.assertLess(error,threshold)
        
        known_var   = {}
        error = do_a_test("Estime All")
        self.assertLess(error,threshold)

    def test_diff_corono(self):
        
        init_worksheet("test_diff_corono",labels)

        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        #Estimation All fixed
        
        tbed.set_corono("lyot")
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("lyot")
        self.assertLess(error,threshold)
        
        tbed.set_corono("R&R") 
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("lyot")
        self.assertLess(error,threshold)
        
        tbed.set_corono("4q") 
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("4q")
        self.assertLess(error,threshold)
        
        tbed.set_corono("4q zbiaiz")
        tbed.zbiais = True
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("4q zbiaiz")
        self.assertLess(error,threshold)
        
# %% Main

unittest.main()
book.close()