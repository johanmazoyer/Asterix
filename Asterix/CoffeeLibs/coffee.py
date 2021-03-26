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

from CoffeeLibs.param import *  # Doit être enlever !!


class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """
    
    def __init__(self, thd2, gtol=1e-5, maxiter=1000, eps=1e-10):
        self.thd2 = thd2    # Initialize thd:
        # Estimator parameters
        self.gtol = gtol
        self.maxiter = maxiter
        self.eps = eps
        
    
    def estimate(self,i_foc,i_div):
        
        # Random pour l'instant parce que le modèle l'oblige
        N = self.thd2.dim_overpad_pupil
        EF_ini = np.random.normal(0.5, 0.5, [N,N])
        # PHI0   = np.zeros((wphi,lphi))
        
        res  = minimize(cacl.V_map_J,
                        EF_ini.reshape(N*N,),
                        args=(self.thd2,i_foc,i_div),
                        method='BFGS',
                        jac=cacl.V_grad_J,
                        options={'disp': True,'gtol':self.gtol,'eps':self.eps,'maxiter':self.maxiter}) 
            
        EF_est  = res.get('x').reshape(N,N)
        return EF_est