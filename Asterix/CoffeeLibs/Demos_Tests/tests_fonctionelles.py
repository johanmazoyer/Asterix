# -*- coding: utf-8 -*-
"""
Created on Thu May  11 11:13:39 2021

@author: sjuillar
"""

from CoffeeLibs.coffee import custom_bench, Estimator, data_simulator
from CoffeeLibs.tools import tempalte_plot

import numpy as np

import os
from configobj import ConfigObj
from validate import Validator

import xlsxwriter as wrt
import unittest

# %% Chargement des parametres

# Chargement des parametres de la simulation
path   = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
config = ConfigObj(path + 'my_param_file.ini', configspec=path + "..\..\Param_configspec.ini")
config.validate(Validator(), copy=True)

# Paramètres qu'il faudra ranger dans ini file..
known_var   = {'downstream_EF':1, 'flux':1, 'fond':0}
div_factors = [0,1]  # List of div factor's images diversity
RSB         = 30000

# %% Initalisation of objetcs

tbed      = custom_bench(config["modelconfig"],'.')
sim       = data_simulator(tbed,known_var,div_factors)
estimator = Estimator(**config["Estimationconfig"])

threshold = 3  # Error max 

coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]


sim.gen_zernike_phi_foc(coeff)    # On génere le phi focalisé
sim.gen_zernike_phi_do([0,0,1])   # On phi do

imgs = sim.gen_div_imgs()         # On cree les images

cropEF   = sim.get_phi_foc()*tbed.pup
cropEFd  = sim.get_phi_do()*tbed.pup_d
pup_size = np.sum(tbed.pup)

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
    
        e_sim = estimator.estimate(imgs,tbed,div_factors,known_var) # Estimation
        error = 100*abs((cropEF-e_sim.get_phi_foc())/cropEF)
        error[np.isnan(error) | np.isinf(error)] = 0
        error_flux = sim.get_flux()-e_sim.get_flux()
        error_fond = sim.get_fond()-e_sim.get_fond()
        error_do = 100*abs((cropEFd-(e_sim.get_phi_do()*tbed.pup_d))/cropEFd)
        error_do[np.isnan(error_do) | np.isinf(error_do)] = 0
        new_row([name,np.sum(error)/pup_size,np.sum(error_do)/pup_size,error_flux,error_fond,estimator.toc], ((np.sum(error)/error.size)<threshold))
        
        tempalte_plot(sim,e_sim,estimator,name="fig/"+name,disp=False,save=True)

        return np.sum(error)/pup_size

# %% unit Test

class TestCoffee(unittest.TestCase):
    
    def test_iter_until_conv(self):
        
        init_worksheet("test_iter_until_convergence",labels)
        
        global known_var
        global estimator
        estimator = Estimator(**config["Estimationconfig"])

        known_var   = {'flux':1, 'fond':0}
        
        estimator.setting['options']['maxiter'] = 10
        error = do_a_test("updo 10 iters")
        
        estimator.setting['options']['maxiter'] = 100
        error = do_a_test("updo 100 iters")
        
        estimator.setting['options']['maxiter'] = 500
        error = do_a_test("updo 500 iters")

        estimator.setting['options']['maxiter'] = 1000
        error = do_a_test("updo 1k iters")
        
        estimator.setting['options']['maxiter'] = 10
        error = do_a_test("up 10 iters")
        
        estimator.setting['options']['maxiter'] = 100
        error = do_a_test("up 100 iters")
        
        estimator.setting['options']['maxiter'] = 500
        error = do_a_test("up 500 iters")

        estimator.setting['options']['maxiter'] = 1000
        error = do_a_test("up 1k iters")
        
    
    def test_known_var(self):
        
        init_worksheet("test_known_var",labels)
        global known_var
        global estimator
        estimator = Estimator(**config["Estimationconfig"])

        
        #Estimation All fixed
        known_var   = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        error = do_a_test("All fixed")
        
        known_var   = {'downstream_EF':sim.get_EF_do()}
        error = do_a_test("downstream_EF fixed")
        
        known_var   = {'flux':1, 'fond':0}
        error = do_a_test("Flux_fond fixed")

        
        known_var   = {'flux':1}
        error = do_a_test("Flux fixed")
        
        known_var   = {'downstream_EF':sim.get_EF_do(),'fond':0}
        error = do_a_test("Fond and downstream_EF fixed")
        
        known_var   = {}
        error = do_a_test("Estime All")

    def test_diff_corono(self):
        
        init_worksheet("test_diff_corono",labels)

        global known_var
        global sim
        global tbed
        global estimator
        global imgs
        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        estimator = Estimator(**config["Estimationconfig"])

        #Estimation All fixed
        
        tbed.set_corono("lyot")
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("lyot")
        
        tbed.set_corono("R&R") 
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("lyot")
        
        tbed.set_corono("4q") 
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("4q")
        
        tbed.set_corono("4q zbiaiz")
        tbed.zbiais = True
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        error = do_a_test("4q zbiaiz")
        
    def test_RSB_updo(self):
        
        init_worksheet("RSB_updo",labels)
        
        global known_var
        global estimator
        global imgs
        known_var = {'flux':1, 'fond':0}
        estimator = Estimator(**config["Estimationconfig"])
        estimator.hypp = 1
        
        imgs  = sim.gen_div_imgs(RSB=1)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=10)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=50)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=100)
        error = do_a_test("RSB 1")
        
    def test_RSB_up(self):
        
        init_worksheet("RSB_up",labels)
        
        global known_var
        global estimator
        global imgs
        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        estimator = Estimator(**config["Estimationconfig"])
        estimator.hypp = 1
        
        imgs  = sim.gen_div_imgs(RSB=1)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=10)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=50)
        error = do_a_test("RSB 1")
        
        imgs  = sim.gen_div_imgs(RSB=100)
        error = do_a_test("RSB 1")
        
    def test_regul(self):
        
        init_worksheet("regul",labels)
        
        global known_var
        global estimator
        
        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':1, 'fond':0}
        estimator = Estimator(**config["Estimationconfig"])
        
        estimator.hypp = 10
        error = do_a_test("regul 10")
        
        estimator.hypp = 1
        error = do_a_test("regul 1")
        
        estimator.hypp = 1e-1
        error = do_a_test("regul 1e-1")
        
        estimator.hypp = 1e-2
        error = do_a_test("regul 1e-2")
        
        estimator.hypp = 1e-3
        error = do_a_test("regul 1e-3")
        
        estimator.hypp = 1e-100
        error = do_a_test("regul 1e-100")
        
        
        
        
# %% Main

unittest.main()
book.close()