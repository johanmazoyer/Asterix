# -*- coding: utf-8 -*-
"""
Created on Thu May  11 11:13:39 2021

@author: sjuillar


----------------   Test fonctionelle PyCoffee  -----------------


Section 1  -> parametres de base

Section 2  -> Fonction pour écrire les logs/spreadsheet etc

Section 3  -> Définition des test avec unit test

Section 4  -> Main ou on lance les tests

----
Fait les tests. Ecris un fichier exels avec les résultats d'erreur
Fait aussi les figures de phase dans un fichier. (template_plot)

Beware of how you reset estimator sim and tbed ! 

"""

from CoffeeLibs.coffee import custom_bench, coffee_estimator, data_simulator
from CoffeeLibs.tools import tempalte_plot
from CoffeeLibs.files_manager import get_ini


import numpy as np

import os
from configobj import ConfigObj
from validate import Validator

import xlsxwriter as wrt
from datetime import datetime
import unittest


# %% ################ SECTION 1 #################


prefix = datetime.now().strftime("%d_%m_%H_%M_")# Prefix pour différentier les series. Prfix date

threshold = 1 # Error max on recontruction in percent (precsion 2decimals)

config = get_ini('my_param_file.ini')

# Paramètres globaux 
fu = 1e1
known_var   = {'downstream_EF':1, 'flux':fu, 'fond':0}
div_factors = [0,0.0000001]
RSB         = None

def reset_param():
    global fu,known_var,known_var,div_factors,RSB
    fu = 1e1
    known_var   = {'downstream_EF':1, 'flux':fu, 'fond':0}
    div_factors = [0,0.0000001]
    RSB         = None

def reset_tbed():
    global tbed
    tbed      = custom_bench(config,'.')
    
def reset_sim():
    global imgs,sim    
    sim = data_simulator(tbed,known_var,div_factors)
    coeff = 0.1/np.arange(1,7)
    coeff[0:3] = [0,0,0]
    
    sim.gen_zernike_phi_foc(coeff)
    
    imgs = sim.gen_div_imgs(RSB) 


reset_tbed()
reset_sim()

# %% ################ SECTION 2 #################
# Things to write result in a ~nice~ spreadsheet


log = './save/'+prefix+'test_error_log.txt'
book  = wrt.Workbook('./save/'+prefix+'UnitTests.xlsx')
sheet = book.add_worksheet("setting")
sheet.write_comment(0,0,str(config))

labels = ["Name","%imgs","%phi_up","%phi_do","diff flux","diff fond","time (s)"]

Good        = book.add_format({'font_color': 'green'})
NotGood     = book.add_format({'bold': True, 'font_color': 'red','text_wrap': True})
labels_font = book.add_format({'bold': True, 'font_color': 'blue','text_wrap': True})

def write_log(name,e):
    f = open(log, "a")
    f.write(name +" : \n" + str(e) + '\n')
    f.close()

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
        path = prefix + "_" + "unit_test/"
        
        estimator.simGif = None
        try :  e_sim = estimator.estimate(tbed,imgs,div_factors,known_var) # Estimation
        except Exception as e : 
            write_log(name,e)
            return False
        
        # Erreur percentage 2decs precision
        error      = 100 * np.sum(abs( sim.get_phi_foc() - e_sim.get_phi_foc() )) / np.sum(abs(sim.get_phi_foc()))
        
        error_flux = sim.get_flux(0)- e_sim.get_flux(0)
        error_fond = sim.get_fond(0)- e_sim.get_fond(0)
        
        if np.sum(sim.get_phi_do()) == 0 : error_do = 0
        else : error_do   = 100 * np.sum(abs( sim.get_phi_do()  - e_sim.get_phi_do())) /  np.sum(abs(sim.get_phi_do()))
        
     # Compute error of reconstruction
        e_imgs = e_sim.gen_div_imgs(RSB)
        error_img = []
        for div in range(e_sim.nb_div) :
            error_img.append( 100 * ( np.sum(abs(e_imgs[:,:,div] - imgs[:,:,div])) ) / np.sum(abs(imgs[:,:,div])) )
        error_img = np.mean(error_img)
            
        new_row([name,error_img,error,error_do,error_flux,error_fond,estimator.toc], (error_img<threshold))
        tempalte_plot(sim,e_sim,estimator,name=name,disp=False,save=True,path=path)

        return True
    
def reset_estimator():
    global estimator
    estimator = coffee_estimator(**config["Estimationconfig"])
    estimator.minimiz_options["disp"] = False
    estimator.var_phi =  0 / np.var(sim.get_phi_foc())
    estimator.bound = None


# %% ################ SECTION 3 #################
# unit Test

class TestCoffee(unittest.TestCase):
    
    def test_known_var(self):
        
        init_worksheet("test_known_var",labels)
        global known_var
        global estimator
        global sim
        global imgs
        reset_tbed()
        reset_estimator()
        reset_sim()
        
        # _________
        known_var   = {'downstream_EF':sim.get_EF_do(), 'flux':fu, 'fond':0}
        self.assertTrue(do_a_test("All fixed"))
        
        # _________
        known_var   = {'downstream_EF':sim.get_EF_do(), 'flux':fu}
        self.assertTrue(do_a_test("Estime fond"))
        
        # _________
        known_var   = {'downstream_EF':sim.get_EF_do(),'fond':0}
        self.assertTrue(do_a_test("Estime flux"))
        
        # _________
        sim.gen_zernike_phi_do([0,0,0.5])
        imgs  = sim.gen_div_imgs()
        known_var   = {'flux':fu, 'fond':0}
        self.assertTrue(do_a_test("Estime downstream EF"))
        
        # _________
        known_var   = {}
        self.assertTrue(do_a_test("Estime All"))

    def test_diff_corono(self):
        
        init_worksheet("test_diff_corono",labels)

        global known_var
        global sim
        global tbed
        global estimator
        global imgs
        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':fu, 'fond':0}
        reset_tbed()
        reset_estimator()
        reset_sim()

        #Estimation All fixed
        
        tbed.set_corono("lyot")
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        self.assertTrue(do_a_test("lyot"))
        
        tbed.set_corono("R&R") 
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        self.assertTrue(do_a_test("R&R"))
        
        tbed.set_corono("4q") 
        tbed.zbiais = False
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        self.assertTrue(do_a_test("4q no zbiaiz"))
        
        tbed.set_corono("4q")
        tbed.zbiais = True
        sim.tbed = tbed
        global div_factors
        div_factors = [0,0.1]
        coeff = 1/np.arange(1,7)
        coeff[0:3] = [0,0,0]
        reset_sim()
        sim.gen_zernike_phi_foc(coeff)
        imgs  = sim.gen_div_imgs()
        self.assertTrue(do_a_test("4q"))
        reset_param()
        
    def test_regul(self):
        
        init_worksheet("regul",labels)
        
        global known_var
        global estimator
        reset_tbed()
        reset_estimator()
        reset_sim()
        known_var = {'downstream_EF':sim.get_EF_do(), 'flux':fu, 'fond':0}
        
        # _________
        estimator.var_phi =  0
        self.assertTrue(do_a_test("No regularisation"))
        
        # _________
        estimator.var_phi =  1 / np.var(sim.get_phi_foc())
        self.assertTrue(do_a_test("Perfect regularisation"))
        
        # _________
        estimator.var_phi =  0.5 / np.var(sim.get_phi_foc())
        self.assertTrue(do_a_test("Regularisation halves"))
        
        # _________
        estimator.var_phi =  2 / np.var(sim.get_phi_foc())
        self.assertTrue(do_a_test("Regularisation double"))
        
    def test_mode(self):
        
        init_worksheet("modes",labels)
        
        global known_var
        global estimator
        global tbed
        global sim
        global imgs
        global normA
        known_var = {'downstream_EF':1, 'flux':1, 'fond':0}
        reset_tbed()
        reset_estimator()
        reset_sim()

        tbed.rcorno = 0
        tbed.set_corono("lyot")
        sim.tbed = tbed
        imgs  = sim.gen_div_imgs()
        
        # _________
        estimator.myope = True
        # TODO changer la diversité
        self.assertTrue(do_a_test("Myope"))
        estimator.myope = False

        # _________
        estimator.set_auto(True)
        self.assertTrue(do_a_test("Auto"))
        estimator.set_auto(False)

        # _________
        estimator.cplx = True
        known_var = {'downstream_EF':1, 'flux':1e3, 'fond':0}
        reset_sim()
        tbed.set_corono("4q") 
        sim.tbed = tbed

        coeff = 1/np.arange(1,7)
        coeff[0:3] = [0,0,0]
        
        phi_r = sim.gen_zernike_phi(coeff)
        phi_i =  sim.gen_zernike_phi([0,0.1,0.1])
        sim.set_phi_foc(phi_r+1j*phi_i)
        
        imgs = sim.gen_div_imgs(RSB) 

        self.assertTrue(do_a_test("Complex"))
        

        
        
        
        
# %% ################ SECTION 4 - Main ##########

unittest.main()
book.close()