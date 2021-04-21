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
import CoffeeLibs.criteres as cacl
from scipy.optimize import minimize
from Asterix.Optical_System_functions import Optical_System, pupil
from Asterix.propagation_functions import mft



class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """

    def __init__(self, tbed, gtol=1e-1, maxiter=5000, eps=1e-10,disp=False,hypp=1,lap="lap",grad="grad",**kwarg):
        self.tbed = tbed    # Initialize thd:
        # Estimator parameters
        self.gtol = gtol
        self.maxiter = maxiter
        self.eps  = eps
        self.disp = disp
        self.hypp = hypp
        self.lap = lap
        self.grad = grad


    def estimate(self,i_foc,i_div):

        N = self.tbed.dimScience//self.tbed.ech
        EF_ini   = np.zeros((2*N*N,))
        
        res  = minimize(cacl.V_map_J,
                        EF_ini,
                        args=(self.tbed,i_foc,i_div,self.hypp),
                        method='BFGS',
                        jac=cacl.V_grad_J,
                        options={'disp': self.disp,'gtol':self.gtol,'eps':self.eps,'maxiter':self.maxiter})

        EFs        = res.get('x')
        EF_est     = EFs[:N*N].reshape(N,N)
        EF_abb_est = EFs[N*N:].reshape(N,N)
        
        self.complete_res = res
        
        return EF_est,EF_abb_est


class custom_bench(Optical_System):

    def __init__(self, modelconfig, model_dir=''):

        super().__init__(modelconfig)
        self.diam_pup_in_pix = modelconfig["diam_pup_in_pix"]
        self.entrancepupil = pupil(modelconfig, prad=self.prad)
        # self.measure_normalization()
        
        # Sand stuff
        self.abberation = 1
        
        self.rcorno = 5
        self.ech    = 2
        
        
        self.pup    = tls.circle(self.dimScience//self.ech,self.dimScience//self.ech,self.prad)
        self.pup    = np.ones((self.dimScience//self.ech,self.dimScience//self.ech))
        
        if(self.rcorno) : self.corno  = abs(tls.circle(self.dimScience,self.dimScience,self.rcorno)-1)
        else :      self.corno = np.ones((self.dimScience,self.dimScience)) 
        
        self.epsi   = 0
        

    def EF_through(self,
                   entrance_EF=1.,
                   downstream_EF=1.,
                   wavelength=None,
                   save_all_planes_to_fits=False,
                   dir_save_fits=None):

        # call the Optical_System super function to check and format the variable entrance_EF
        # entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        # if wavelength == None:
        #     wavelength = self.wavelength_0

        # EF_afterentrancepup = self.entrancepupil.EF_through(
        #     entrance_EF=entrance_EF,
        #     wavelength=wavelength,
        #     save_all_planes_to_fits=save_all_planes_to_fits,
        #     dir_save_fits=dir_save_fits)

        #Sand EF_through
        dim_img = self.dimScience
        dim_pup = dim_img//self.ech
        
        EF_afterentrancepup = entrance_EF*self.pup
        EF_aftercorno       = tls.depadding( mft( self.corno   * mft(EF_afterentrancepup,dim_pup,dim_img,dim_pup,inverse=False) ,dim_pup,dim_img,dim_pup), self.ech)
        EF_out              = tls.depadding( mft(EF_aftercorno * downstream_EF * self.pup ,dim_pup,dim_img,dim_pup), self.ech)

        return EF_out

    def psf(self,
            entrance_EF=1.,
            aberration_EF=1.,
            wavelength=None,
            save_all_planes_to_fits=False,
            dir_save_fits=None):

        EF_out = self.EF_through(entrance_EF,aberration_EF)

        dim_img = self.dimScience
        dim_pup = dim_img//self.ech
        

        return pow( abs( EF_out ) ,2)
