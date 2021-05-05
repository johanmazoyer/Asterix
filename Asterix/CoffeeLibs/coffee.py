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
from CoffeeLibs.pzernike import pmap, zernike, pzernike
from scipy.optimize import minimize
from Asterix.Optical_System_functions import Optical_System, pupil
from Asterix.propagation_functions import mft


# %% DATA SIMULATOR

class data_simulator():

    def __init__(self, size,known_var = "default", div_factors = [0,1],phi_foc=None):
        
        # Store simulation Variables
        self.div_factors = div_factors
        self.known_var = known_var
        self.phi_foc = phi_foc
        
        # Things we don't want to recompute
        self.N = size
        [Ro,Theta] = pmap(size,size)
        self.defoc   = zernike(Ro,Theta,4)
        
        # To keep track on what I am suppose to know
        # If default, we know everything
        self.remeber_known_var_as_bool()
        
        
        # Set default everywhere it is needed
        if known_var == "default" : self.set_default_value()
        elif "downstream_EF" in known_var.keys() : 
            self.set_phi_do(np.imag(np.log(known_var["downstream_EF"])))
        if phi_foc is None : self.phi_foc = zernike(Ro,Theta,1)
        
    # Generators
    def gen_zernike_phi_foc(self,coeff):
        self.phi_foc = self.gen_zernike_phi(coeff)

    def gen_zernike_phi_do(self,coeff):
        self.set_phi_do(self.gen_zernike_phi(coeff))
    
    def gen_zernike_phi(self,coeff):
        [Ro,Theta] = pmap(self.N,self.N)
        return pzernike(Ro,Theta,coeff)
    
    def gen_div_phi(self):
        
        n = self.N
        phis = np.zeros((n,n,len(self.div_factors)))
        
        # List of matricial a*defocs
        defoc_list = np.array(self.div_factors).reshape(1,1,len(self.div_factors)) * self.defoc.reshape(n,n,1)
        
        # List of phi_up = phi_foc + a*defoc
        phis = (phis + self.phi_foc.reshape(n,n,1)) + defoc_list
        
        return phis
    
    def gen_div_imgs(self,tbed,RSB=None):
        return self.phi2img(tbed,self.gen_div_phi(),RSB)
    
    # Setters
    def set_phi_foc(self,phi_foc):
        self.phi_foc = phi_foc
        
    def set_phi_do(self,phi_do):
        """Set phi downstream and EF downstream.
            input : matrix NxN or int """
        if isinstance(phi_do, int) |  isinstance(phi_do, float) : self.phi_do = phi_do*np.ones((self.N,self.N))
        else : self.phi_do = phi_do
        self.known_var['downstream_EF'] = np.exp(1j*self.phi_do)
        
    def set_flux(self,flux):
        self.known_var['flux'] = flux
        
    def set_fond(self,fond):
        self.known_var['fond'] = fond
    
    # Getters
    def get_EF(self,defoc=0):
        return np.exp(1j*self.phi_foc)
    
    def get_EF_div(self,div_id):
        return np.exp(1j*self.get_phi_div(div_id))
    
    def get_img(self,tbed,defoc=0):
        return tbed.psf(self.get_EF(defoc),**self.known_var)
    
    def get_img_div(self,tbed,div_id,ff=True):
        if ff : return tbed.psf(self.get_EF_div(div_id),**self.known_var)
        else  : return tbed.psf(self.get_EF_div(div_id),self.get_EF_do())
    
    def get_phi_foc(self):
        return self.phi_foc
    
    def get_phi_div(self,div_id):
        return self.phi_foc + ( self.div_factors[div_id] * self.defoc )
    
    def get_EF_do(self):
        return self.known_var['downstream_EF']
    
    def get_phi_do(self):
        return self.phi_do
        
    def get_flux(self):
        return self.known_var['flux']
        
    def get_fond(self):
        return self.known_var['fond']
    
    # Checkers
    def phi_do_is_known(self):
        return self.known_var_bool['downstream_EF']
        
    def flux_is_known(self):
        return self.known_var_bool['flux']
        
    def fond_is_known(self):
        return self.known_var_bool['fond']
    
    # Tools
    def phi2img(self,tbed,phis,RSB=None):
        
        EFs  = np.exp(1j*phis)
        imgs = tbed.psf(EFs,**self.known_var)
        
        if RSB is not None :
            varb = 10**(-RSB/20)
            imgs = imgs + np.random.normal(0, varb, imgs.shape)
            
        return imgs
    
    def remeber_known_var_as_bool(self):
        known_var_bool = {'downstream_EF':False,'flux':False,'fond':False}
        keys = (self.known_var).keys()
        if 'downstream_EF' in keys: known_var_bool['downstream_EF'] = True
        if 'flux'          in keys: known_var_bool['flux'] = True
        if 'fond'          in keys: known_var_bool['fond'] = True
        self.known_var_bool = known_var_bool
        
    def print_know_war(self):
        
        msg = "Estimator will find phi_foc, "
        if not self.phi_do_is_known() : msg += "phi_do,"       
        if not self.flux_is_known()   : msg += "flux,"
        if not self.fond_is_known()   : msg += "fond"
        
        msg += "\n With "
        if self.phi_do_is_known() : msg += "known phi_do\n"       
        if self.flux_is_known()   : msg += "flux = " + str(self.get_flux()) +"\n"
        if self.fond_is_known()   : msg += "fond = " + str(self.get_fond()) +"\n"
        msg += "and diversity : " + str(self.div_factors)
        
        print(msg)
        
    def set_default_value(self):
        
        # In case known_var was an str flag
        if not isinstance(self.known_var, dict) : self.known_var=dict()
        keys = (self.known_var).keys()
        
        if 'downstream_EF' not in keys: self.set_phi_do(np.zeros((self.N,self.N)))
        else : self.phi_do = np.imag(np.log(self.get_EF_do()))
        if 'flux'          not in keys: self.set_flux(1)
        if 'fond'          not in keys: self.set_fond(0)

        
    # Fonctions for optimize minimizer intergration   
    def opti_unpack(self,pack):
        """Unconcatene variables from minimizer """
        n = self.N
        self.set_phi_foc(pack[:n*n].reshape(n,n)) # We always estime phi_foc
        indx = n*n
        
        if not self.phi_do_is_known() : 
            self.set_phi_do(pack[indx:indx+n*n].reshape(n,n))
            indx+=n*n
        
        return self.get_phi_foc(),self.get_phi_do(),self.get_flux(),self.get_fond()

    def opti_pack(self,phi_foc=None):
        """Concatene variables to fit optimize minimiza syntax"""
        
        self.set_default_value()
        
        if phi_foc is None : phi_foc = self.phi_foc
        [w,l] = phi_foc.shape
        
        pack = phi_foc.reshape(w*l,)
        
        # If the var is unknown, then pack to pass to minimizer
        if not self.phi_do_is_known() : pack = np.concatenate((pack, self.get_phi_do().reshape(w*l,)), axis=None)         

    
        return pack
    
    def opti_update(self,pack,tbed,imgs):
        """Uptade sim with varaibles from minimizer optimize"""
        n = self.N
        
        self.phi_foc = pack[:n*n].reshape(n,n) # We always estime phi_foc
        
        indx = n*n
        if not self.phi_do_is_known() : 
            self.set_phi_do(pack[indx:indx+n*n].reshape(n,n))
            indx+=n*n
        
        # Matrice Inversion
        if (not self.flux_is_known()) | (not self.fond_is_known()) : flux,fond = cacl.estime_fluxfond(self,tbed,imgs)
            
        if not self.flux_is_known() : self.set_flux(flux)
        if not self.fond_is_known() : self.set_fond(fond)


    def EF_through(self,div_id,tbed,ff=True):
        return tbed.EF_through(self.get_EF_div(div_id),self.get_EF_do())

    def psf(self,div_id,tbed):
        return abs(self.EF_through(div_id,tbed))**2


# %% ESTIMATOR
    
class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """

    def __init__(self, tbed, gtol=1e-1, maxiter=5000, eps=1e-10,disp=False,hypp=1,lap="lap",grad="np",**kwarg):
        self.tbed = tbed    # Initialize thd:
        # Estimator parameters
        self.gtol = gtol
        self.maxiter = maxiter
        self.eps  = eps
        self.disp = disp
        self.hypp = hypp
        self.lap = lap
        self.grad = grad


    def estimate(self,imgs,div_factors,known_var=dict()):

        N = self.tbed.dimScience//self.tbed.ech

        # Create our simulation
        sim      = data_simulator(N,known_var,div_factors)
        ini_pack = sim.opti_pack()
        
        if self.disp : sim.print_know_war()
        
        res  = minimize(cacl.V_map_J,
                        ini_pack,
                        args=(self.tbed,sim,imgs,self.hypp),
                        method='BFGS',
                        jac=cacl.V_grad_J,
                        options={'disp': self.disp,'gtol':self.gtol,'eps':self.eps,'maxiter':self.maxiter})

        
        sim.opti_unpack(res.get('x'))
        self.complete_res = res
        
        return sim


# %% CUSTOM BENCH

class custom_bench(Optical_System):

    def __init__(self, modelconfig, model_dir=''):

        super().__init__(modelconfig)
        self.diam_pup_in_pix = modelconfig["diam_pup_in_pix"]
        self.entrancepupil = pupil(modelconfig, prad=self.prad)
        # self.measure_normalization()
        
        # A mettre en parametres 
        self.rcorno = 5
        self.ech    = 2
        self.zbiais = False
        self.epsi   = 0

        # Definitions
        wech = self.dimScience//self.ech
        w    = self.dimScience
        
        self.pup      = tls.circle(wech,wech,self.prad)
        self.pup_d    = tls.circle(wech,wech,self.prad)

        if  (modelconfig["filename_instr_pup"]=="4q") : self.corno = tls.daminer(w,w)
        elif(modelconfig["filename_instr_pup"]=="R&R"): self.corno = 2*tls.circle(w,w,self.rcorno)-1
        else                                          : self.corno = abs(tls.circle(w,w,self.rcorno)-1)
        
    def EF_through(self,entrance_EF=1.,downstream_EF=1.,zbiais = False):

        # If multiple EF
        if isinstance(entrance_EF, np.ndarray) : 
            if (entrance_EF.ndim == 3) : return self.EF_through_loop(entrance_EF,downstream_EF,zbiais)        
        
        dim_img = self.dimScience
        dim_pup = dim_img//self.ech

        EF_afterentrancepup = entrance_EF*self.pup
        EF_aftercorno       = mft( self.corno * mft(EF_afterentrancepup,dim_pup,dim_img,dim_pup,inverse=False) ,dim_pup,dim_img,dim_pup)
        if(zbiais)          : EF_aftercorno = EF_aftercorno - self.z_biais()
        EF_out              = mft( tls.padding( downstream_EF * self.pup_d, self.ech) * EF_aftercorno ,dim_pup,dim_img,dim_pup)
        
        return EF_out

    def EF_through_loop(self,EFs=1.,downstream_EF=1.,zbiais = False):

        shape = (self.dimScience,self.dimScience,1)
        nb_img = EFs.shape[2]
        res    = self.EF_through(EFs[:,:,0],downstream_EF,zbiais).reshape(shape)
        

        for ii in range(1,nb_img):
            out = self.EF_through(EFs[:,:,ii],downstream_EF,zbiais).reshape(shape)
            res = np.append(res,out,axis=2)
        
        return res
        
    def z_biais(self):
        N = self.dimScience
        return mft( self.corno * mft(self.pup,N//self.ech,N,N//self.ech,inverse=False) ,N//self.ech,N,N//self.ech)

    def psf(self,entrance_EF=1.,downstream_EF=1.,flux=1,fond=0):

        EF_out = self.EF_through(entrance_EF,downstream_EF)

        dim_img = self.dimScience
        dim_pup = dim_img//self.ech
        

        return flux*pow( abs( EF_out ) ,2) + fond

    def set_corono(self,cortype):
        w    = self.dimScience
        if  (cortype=="4q") : self.corno = tls.daminer(w,w)
        elif(cortype=="R&R"): self.corno = 2*tls.circle(w,w,self.rcorno)-1
        else                : self.corno = abs(tls.circle(w,w,self.rcorno)-1)
  

    def introspect(self,
            entrance_EF=1.,
            downstream_EF=1.):
    
        view_list = []
        N = self.dimScience//self.ech

        # entrance_EF = entrance_EF * dim_pup
        view_list.append(entrance_EF)
        
        view_list.append(view_list[-1]*self.pup)
        
        view_list.append(mft(view_list[-1],N,N*self.ech,N))
        
        view_list.append(self.corno*view_list[-1])
        
        view_list.append(mft(view_list[-1],N,N*self.ech,N))
        
        if(self.zbiais): view_list.append(view_list[-1] - self.EF_through(iszbiais=True))
        
        view_list.append(tls.padding( downstream_EF * self.pup_d, self.ech)*view_list[-1] )
        
        view_list.append(mft(view_list[-1],N,N*self.ech,N))
        
        
        return view_list