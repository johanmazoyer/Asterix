# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
------------  COFFEE Classes   -------------
--------------------------------------------

@author: sjuillar

Class data simuator : 
    Create a simulation
    
Class Estimator : 
    Estim a simulation from images
    
Class Custom Test Bench :
    Optical systeme

"""

import numpy as np

import CoffeeLibs.tools as tls
import CoffeeLibs.criteres as cacl
from CoffeeLibs.pzernike import pmap, zernike, pzernike
from scipy.optimize import minimize
from Asterix.Optical_System_functions import Optical_System, pupil
from Asterix.propagation_functions import mft
import time as t

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# %% DATA SIMULATOR

class data_simulator():

    def __init__(self,tbed, known_var = "default", div_factors = [0,1], phi_foc=None):
        
        # Things we don't want to recompute
        self.N = tbed.dimScience//tbed.ech
        [Ro,Theta] = pmap(self.N,self.N)
        self.defoc   = zernike(Ro,Theta,4)
        
        # Store simulation Variables
        self.tbed = tbed
        self.phi_foc = phi_foc
        self.set_div_map(div_factors)
        self.known_var = known_var
        
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
        phis = np.zeros((n,n,self.nb_div))
        
        # List of phi_up = phi_foc + a*defoc
        phis = (phis + self.phi_foc.reshape(n,n,1)) + self.div_map
        
        return phis
    
    def gen_div_imgs(self,RSB=None):
        return self.phi2img(self.gen_div_phi(),RSB)
    
    # Setters
    def set_phi_foc(self,phi_foc):
        self.phi_foc = phi_foc
        
    def set_phi_do(self,phi_do):
        """Set phi downstream and EF downstream.
            input : matrix NxN or int """
        if isinstance(phi_do, int) |  isinstance(phi_do, float) : self.phi_do = phi_do*np.ones((self.N,self.N))
        else : self.phi_do = phi_do
        self.known_var['downstream_EF'] = np.exp(1j*self.phi_do)
       
    def set_div_map(self,div_factors):
        n = self.N
        self.set_nb_div(div_factors)
        self.div_map = np.zeros((n,n,self.nb_div))
        tpm = 0
        for fact in div_factors:
            if isinstance(fact, int) |  isinstance(fact, float) : self.div_map[:,:,tpm] = fact*self.defoc
            else : self.div_map[:,:,tpm] = fact
            tpm +=1
       
    def set_nb_div(self,div_factors):
        self.nb_div = len(div_factors)
        
       
    def set_flux(self,flux):
        self.known_var['flux'] = flux
        
    def set_fond(self,fond):
        self.known_var['fond'] = fond
    
    # Getters
    def get_EF(self,defoc=0):
        return np.exp(1j*self.phi_foc)
    
    def get_EF_div(self,div_id):
        return np.exp(1j*self.get_phi_div(div_id))
    
    def get_img(self,defoc=0):
        return self.tbed.psf(self.get_EF(defoc),**self.known_var)
    
    def get_img_div(self,div_id,ff=True):
        if ff : return self.tbed.psf(self.get_EF_div(div_id),**self.known_var)
        else  : return self.tbed.psf(self.get_EF_div(div_id),self.get_EF_do())
    
    def get_phi_foc(self):
        return self.phi_foc
    
    def get_phi_div(self,div_id):
        return self.phi_foc + self.div_map[:,:,div_id]
    
    def get_EF_do(self):
        return self.known_var['downstream_EF']
    
    def get_phi_do(self):
        return self.phi_do
        
    def get_flux(self):
        return self.known_var['flux']
        
    def get_fond(self):
        return self.known_var['fond']
    
    def get_know_var_bool(self):
        return self.known_var_bool
    
    # Checkers
    def phi_foc_is_known(self):
        return self.known_var_bool['phi_foc']
    
    def phi_do_is_known(self):
        return self.known_var_bool['downstream_EF']
        
    def flux_is_known(self):
        return self.known_var_bool['flux']
        
    def fond_is_known(self):
        return self.known_var_bool['fond']
    
    # Tools
    def phi2img(self,phis,RSB=None):
        
        EFs  = np.exp(1j*phis)
        imgs = self.tbed.psf(EFs,**self.known_var)
        
        if RSB is not None :
            varb = 10**(-RSB/20)
            imgs = imgs + np.random.normal(0, varb, imgs.shape)
            
        return imgs
    
    def remeber_known_var_as_bool(self):
        known_var_bool = {'downstream_EF':False,'flux':False,'fond':False,'phi_foc':True}
        keys = (self.known_var).keys()
        
        if  self.phi_foc   is None: known_var_bool['phi_foc'] = False
        if 'downstream_EF' in keys: known_var_bool['downstream_EF'] = True
        if 'flux'          in keys: known_var_bool['flux'] = True
        if 'fond'          in keys: known_var_bool['fond'] = True
        
        self.known_var_bool = known_var_bool
        
    def print_know_war(self):
        
        msg = "Estimator will find "
        if not self.phi_foc_is_known() : msg += "phi_foc,"
        if not self.phi_do_is_known()  : msg += "phi_do,"       
        if not self.flux_is_known()    : msg += "flux,"
        if not self.fond_is_known()    : msg += "fond"
        
        msg += "\nWith " + str(self.nb_div) + " diversities"
        
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
        
        indx = 0
        if not self.phi_foc_is_known() : 
            self.set_phi_foc(pack[:n*n].reshape(n,n)) # We always estime phi_foc
            indx += n*n
        
        if not self.phi_do_is_known() : self.set_phi_do(pack[indx:indx+n*n].reshape(n,n))
        
        return self.get_phi_foc(),self.get_phi_do(),self.get_flux(),self.get_fond()

    def opti_pack(self,phi_foc=None):
        """Concatene variables to fit optimize minimiza syntax"""
        
        self.set_default_value()
        n = self.N
        pack = []
        
        if not self.phi_foc_is_known() : pack = np.concatenate((pack, self.get_phi_foc().reshape(n**2,)), axis=None)
        if not self.phi_do_is_known()  : pack = np.concatenate((pack, self.get_phi_do().reshape(n**2,)), axis=None)         

        return pack
    
    def opti_update(self,pack,imgs):
        """Uptade sim with varaibles from minimizer optimize"""
        n = self.N        
        indx = 0

        if not self.phi_foc_is_known() :
            self.phi_foc = pack[:n*n].reshape(n,n)
            indx += n*n
        
        if not self.phi_do_is_known() : self.set_phi_do(pack[indx:indx+n*n].reshape(n,n))
        
        # Matrice Inversion
        if (not self.flux_is_known()) | (not self.fond_is_known()) : flux,fond = cacl.estime_fluxfond(self,imgs)
            
        if not self.flux_is_known() : self.set_flux(flux)
        if not self.fond_is_known() : self.set_fond(fond)


    def EF_through(self,div_id,ff=True):
        return self.tbed.EF_through(self.get_EF_div(div_id),self.get_EF_do())

    def psf(self,div_id):
        return self.tbed.psf(self.get_EF_div(div_id),**self.known_var)


# %% ESTIMATOR
    
class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """

    def __init__(self,hypp=1 ,method = 'BfGS', gtol=1e-1 , maxiter=10000, eps=1e-10,disp=False,**kwarg):
        
        self.hypp = hypp
        self.disp = disp
        # Minimize parameters
        options = {'disp': disp,'gtol':gtol,'eps':eps,'maxiter':maxiter}
        self.setting = {'options' : options, 'method': method }

    def estimate(self,imgs,tbed,div_factors,known_var=dict(),phi_foc=None):
        
        # Create our simulation
        sim      = data_simulator(tbed,known_var,div_factors,phi_foc)
        ini_pack = sim.opti_pack()
        if self.disp : sim.print_know_war()
        
        tic  = t.time() 
        res  = minimize(cacl.V_map_J,
                        ini_pack,
                        args=(sim,imgs,self.hypp),
                        jac=cacl.V_grad_J,
                        **self.setting)
        toc = t.time() - tic
       
        mins = int(toc//60)
        sec  = 100*(toc-60*mins)
        if self.disp : print("Minimize took : " + str(mins) + "m" + str(sec)[:3])
        sim.opti_unpack(res.get('x'))
        
        # If you need more info after
        self.complete_res = res
        self.toc          = toc
        
        return sim


# %% CUSTOM BENCH

class custom_bench(Optical_System):

    def __init__(self, modelconfig, model_dir=''):

        super().__init__(modelconfig)
        self.diam_pup_in_pix = modelconfig["diam_pup_in_pix"]
        self.entrancepupil = pupil(modelconfig, prad=self.prad)
        # self.measure_normalization()
        
        # A mettre en parametres 
        self.rcorno = 3
        self.ech    = 2
        self.zbiais = True
        self.epsi   = 0

        # Definitions
        w = self.dimScience//self.ech
        
        self.pup      = tls.circle(w,w,self.prad)
        self.pup_d    = tls.circle(w,w,self.prad)

        if  (modelconfig["filename_instr_pup"]=="4q") : self.corno = tls.daminer(w,w)
        elif(modelconfig["filename_instr_pup"]=="R&R"): self.corno = 2*tls.circle(w,w,self.rcorno)-1
        else                                          : self.corno = abs(tls.circle(w,w,self.rcorno)-1)
        
        if (modelconfig["filename_instr_pup"]!="4q")  : self.zbiais = False
        
    def EF_through(self,entrance_EF=1.,downstream_EF=1.,zbiais = False):

        # If multiple EF
        if isinstance(entrance_EF, np.ndarray) : 
            if (entrance_EF.ndim == 3) : return self.EF_through_loop(entrance_EF,downstream_EF,zbiais)        
        
        dim_img = self.dimScience
        dim_pup = dim_img//self.ech

        EF_afterentrancepup = entrance_EF*self.pup
        EF_aftercorno       = mft( self.corno * mft(EF_afterentrancepup,dim_pup,dim_pup,dim_pup,inverse=True) ,dim_pup,dim_pup,dim_pup)
        if(zbiais)          : EF_aftercorno = EF_aftercorno - self.z_biais()
        EF_out              = mft( downstream_EF * self.pup_d * EF_aftercorno ,dim_pup,dim_img,dim_pup,inverse=True)
        
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

        return flux*pow( abs( EF_out ) ,2) + fond
    
    
    def set_corono(self,cortype):
        w    = self.dimScience//self.ech
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
        
        view_list.append(mft(view_list[-1],N,N,N))
        
        view_list.append(self.corno*view_list[-1])
        
        view_list.append(mft(view_list[-1],N,N,N))
        
        if(self.zbiais): view_list.append(view_list[-1] - self.EF_through(iszbiais=True))
        
        view_list.append( downstream_EF * self.pup_d *view_list[-1] )
        
        view_list.append(mft(view_list[-1],N,N*self.ech,N))
                

        
        self.view_list = view_list
        if(self.zbiais): self.title_list = ["Entrence EF", "Upsteam pupil", "MFT", "Corno", "MFT", "Downstream EF + pupil", "MFT - detecteur"]     
        else           : self.title_list = ["Entrence EF", "Upsteam pupil", "MFT", "Corno", "MFT", "Correction zbiais" ,"Downstream EF + pupil", "MFT - detecteur"]     

        self.into   = plt.figure("Introscpetion")
        self.intoax = self.into.add_subplot(1,1,1),plt.imshow(abs(view_list[0]),cmap='jet'),plt.suptitle(self.title_list[0]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[0])**2))),plt.subplots_adjust(bottom=0.25)
        
        plt.subplots_adjust(bottom=0.2)
        self.slide = Slider(plt.axes([0.25,0.1,0.65,0.03]),"view ",0,len(view_list)-1,valinit=0,valstep=1)
        self.slide.on_changed(self.update_introspect)
    
    def update_introspect(self,val):
        view_list = self.view_list
        self.into.add_subplot(1,1,1),plt.imshow(abs(view_list[val]),cmap='jet'),plt.suptitle(self.title_list[val]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[val])**2))),plt.subplots_adjust(bottom=0.25)
       