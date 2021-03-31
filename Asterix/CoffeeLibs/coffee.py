# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
------------  COFFEE Classes   -------------
----------   estim and correct  ------------
--------------------------------------------

@author: sjuillar
"""

import numpy as np

import CoffeeLibs.tools as tls
import CoffeeLibs.simu as sim
import CoffeeLibs.criteres as cacl
from scipy.optimize import minimize
from Asterix.InstrumentSimu_functions import Optical_System, pupil
from Asterix.propagation_functions import mft
from CoffeeLibs.param import *  # Doit être enlever !!


class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """
    
    def __init__(self, thd2, gtol=1e-5, maxiter=5000, eps=1e-10):
        self.thd2 = thd2    # Initialize thd:
        # Estimator parameters
        self.gtol = gtol
        self.maxiter = maxiter
        self.eps = eps
        
    
    def estimate(self,i_foc,i_div):
        
        # Random pour l'instant parce que le modèle l'oblige
        N = self.thd2.dim_overpad_pupil
        EF_ini   = np.zeros((N,N))
        
        res  = minimize(cacl.V_map_J,
                        EF_ini.reshape(N*N,),
                        args=(self.thd2,i_foc,i_div),
                        method='BFGS',
                        jac=cacl.V_grad_J,
                        options={'disp': True,'gtol':self.gtol,'eps':self.eps,'maxiter':self.maxiter}) 
            
        EF_est  = res.get('x').reshape(N,N)
        return EF_est
    
    
class custom_bench(Optical_System):
    
    def __init__(self, modelconfig, model_dir=''):
        
        super().__init__(modelconfig)
        self.entrancepupil = pupil(modelconfig,
                                   prad=self.prad,
                                   model_dir=model_dir,
                                   filename=modelconfig["filename_instr_pup"])

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   save_all_planes_to_fits=False,
                   dir_save_fits=None):
        
        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        EF_afterentrancepup = self.entrancepupil.EF_through(
            entrance_EF=entrance_EF,
            wavelength=wavelength,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_fits=dir_save_fits)

        return EF_afterentrancepup
    
    def psf(self,
            entrance_EF=1.,
            wavelength=None,
            save_all_planes_to_fits=False,
            dir_save_fits=None):
        
        EF_out = self.EF_through(entrance_EF=entrance_EF,
                                wavelength=wavelength)

        dim_pup = self.dim_overpad_pupil
        dim_img = EF_out.shape[0]
        
        return pow( abs( mft(EF_out,dim_pup,dim_img,dim_pup) ) ,2)