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
    Optical system

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
        self.__remeber_known_var_as_bool__()
        
        # Set default everywhere it is needed
        if known_var == "default" : self.set_default_value()
        if phi_foc is None : self.phi_foc = zernike(Ro,Theta,1)
     
    ################################################################
    ####  Generators  ####
    # Generate stuff using object propreties
    
    def gen_zernike_phi_foc(self,coeff):
        """ Generate a zernike polynome using coeff and assign phi_foc  """
        self.phi_foc = self.gen_zernike_phi(coeff)

    def gen_zernike_phi_do(self,coeff):
        """ Generate a zernike polynome using coeff and assign phi_do  """
        self.set_phi_do(self.gen_zernike_phi(coeff))
    
    def gen_zernike_phi(self,coeff):
        """ Generate a zernike polynome using coeff and return it  """
        [Ro,Theta] = pmap(self.N,self.N)
        return pzernike(Ro,Theta,coeff)
    
    def gen_div_phi(self):
        """ Generate list of phis with different diverity
            i.e : phi[k] = phi_foc + div_map[k]
        """
        n = self.N
        phis = np.zeros((n,n,self.nb_div))
        
        # List of phi_up = phi_foc + a*defoc
        phis = (phis + self.phi_foc.reshape(n,n,1)) + self.div_map
        
        return phis
    
    def gen_div_imgs(self,RSB=None):
        """ Generate diversity images list from object propreties
            with additif noise (ajusted to given SNR (RSB en français)) 
        """
        return self.phi2img(self.gen_div_phi(),RSB)
    
    ####################################################
    ####   Setters and getters  #####
    # To access class propreties proprely
    
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
    
    def get_EF(self,defoc=0):
        return np.exp(1j*self.phi_foc)
    
    def get_EF_div(self,div_id):
        return np.exp(1j*self.get_phi_div(div_id))
    
    def get_img(self,defoc=0):
        return self.tbed.todetector_Intensity(self.get_EF(defoc),**self.known_var)
    
    def get_img_div(self,div_id,ff=True):
        if ff : return self.tbed.todetector_Intensity(self.get_EF_div(div_id),**self.known_var)
        else  : return self.tbed.todetector_Intensity(self.get_EF_div(div_id),self.get_EF_do())
    
    def get_phi_foc(self):
        return self.phi_foc
    
    def get_phi_div(self,div_id):
        return self.phi_foc + self.div_map[:,:,div_id]
    
    def get_EF_do(self):
        return self.known_var['downstream_EF']
    
    def get_phi_do(self):
        if hasattr(self, 'phi_do') : return self.phi_do
        
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
    
    
    ############################################
    ####   Tools 1 : Optimize wrappers ####
    
    # Some useful stuff -> for optimize 
    #                   -> for display purpose
        
    def print_know_war(self):
        """ Fonction for display purposes """
        msg = "Estimator will find "
        if not self.phi_foc_is_known() : msg += "phi_foc,"
        if not self.phi_do_is_known()  : msg += "phi_do,"       
        if not self.flux_is_known()    : msg += "flux,"
        if not self.fond_is_known()    : msg += "fond"
        
        msg += "\nWith " + str(self.nb_div) + " diversities"
        
        print(msg)

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

    #######################################
    ####   Tools 2 : tbed wrappers ####
    
    # call to bench   -> to avoid redonduncy 
    #                 -> to make syntaxs clearer
    #                 -> to manage 3D matrixs 

    def phi2img(self,phis,RSB=None):
        """ Generarate diverity images from phis """
        
        EFs  = np.exp(1j*phis)
        imgs = self.__psf__(EFs)
        
        if RSB is not None :
            varb = 10**(-RSB/20)
            imgs = imgs + np.random.normal(0, varb, imgs.shape)
            
        return imgs

    def EF_through(self,div_id,ff=True):
        """ call EF_through bench"""
        return self.tbed.EF_through(self.get_EF_div(div_id),self.get_EF_do())

    def todetector(self,div_id,ff=True):
        """ call EF_through bench"""
        if isinstance(div_id, int) : return self.tbed.todetector(self.get_EF_div(div_id),self.get_EF_do())
        else                       : return self.todetector_loop(self.get_EF_div(div_id),self.get_EF_do())    

    def todetector_loop(self,EFs=1.,downstream_EF=1):
        """ Just to make a big array of all div EFs because it is cleaner to me"""
        shape  = (self.tbed.dimScience,self.tbed.dimScience,1)
        res    = self.tbed.todetector(EFs[:,:,0],downstream_EF).reshape(shape)
        
        for ii in range(1,EFs.shape[2]):
            out = self.tbed.todetector(EFs[:,:,ii],downstream_EF).reshape(shape)
            res = np.append(res,out,axis=2)
        
        return res

    def __psf__(self,EFs):
        """ to detector intensity but EFs are 3D so we loop """
        return self.get_flux() * abs( self.todetector_loop(EFs,self.get_EF_do()) ** 2 ) + self.get_fond()


    def todetector_Intensity(self,div_id):
        """ call to detector intensity bench for a specific div id image"""
        return self.tbed.todetector_Intensity(self.get_EF_div(div_id),**self.known_var)

    #######################################
    ### Tools 3 : MISC ###
    
    # Some other methode -> to make syntaxs clearer
    #                    -> to avoid redonduncy

    def __remeber_known_var_as_bool__(self):
        """ if vars wasn't set in init, there will be considere unknowns (wich mean estimator will estimate them) """
        known_var_bool = {'downstream_EF':False,'flux':False,'fond':False,'phi_foc':True}
        keys = (self.known_var).keys()
        
        if  self.phi_foc   is None: known_var_bool['phi_foc'] = False
        if 'downstream_EF' in keys: known_var_bool['downstream_EF'] = True
        if 'flux'          in keys: known_var_bool['flux'] = True
        if 'fond'          in keys: known_var_bool['fond'] = True
        
        self.known_var_bool = known_var_bool
        
    def set_default_value(self):
        """ if vars are unset, set them to default values"""
        # In case known_var was an str flag
        if not isinstance(self.known_var, dict) : self.known_var=dict()
        keys = (self.known_var).keys()
        
        if 'downstream_EF' not in keys: self.set_phi_do(np.zeros((self.N,self.N)))
        if 'flux'          not in keys: self.set_flux(1)
        if 'fond'          not in keys: self.set_fond(0)  

# %% ESTIMATOR
    
class Estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """

    def __init__(self,var_phi=0 ,method = 'BfGS', gtol=1e-1 , maxiter=10000, eps=1e-10,disp=False,**kwarg):
        
        self.var_phi = var_phi
        self.disp = disp
        # Minimize parameters
        options = {'disp': disp,'gtol':gtol,'eps':eps,'maxiter':maxiter}
        self.setting = {'options' : options, 'method': method }

    def estimate(self,imgs,tbed,div_factors,known_var=dict(),phi_foc=None):
        """ Do a COFFEE estimation
        INPUT :
            
            imgs : 3D ndarry (or list of 2D maps). -> Diversity images
            
            tbed : Optical Sytem class.            -> Model of the bench
            
            div_factor : 3D ndarry (or list of 2D maps/int)
               -> Lisr of diversity map, or int if diverity is only defoc
               
            known_var : dict with possible keys {downstream_EF, flux, fond}.
               -> If any of thoses parameters are known, assign 
                   the given key. Else do nothing.
        
        OUTPUT : 
            e_sim : data_generator objetc. Minimize will estimate the best
            e_sim propreties to mach given images. 
            (i.e : phi_foc phi_do flux fond)
        """
        
        # Create our simulation
        sim      = data_simulator(tbed,known_var,div_factors,phi_foc)
        ini_pack = sim.opti_pack()
        if self.disp : sim.print_know_war()
        sim.info  = [] # To store information during iterations
        sim.info2 = []
        
        tic  = t.time() 
        res  = minimize(cacl.V_map_J,
                        ini_pack,
                        args=(sim,imgs,self.var_phi),
                        jac=cacl.V_grad_J,
                        **self.setting)
        toc = t.time() - tic
       
        mins = int(toc//60)
        sec  = 100*(toc-60*mins)
        if self.disp : print("Minimize took : " + str(mins) + "m" + str(sec)[:3])
        
        sim.opti_unpack(res.get('x'))
        sim.info  = np.array(sim.info)   #list to array because it is better
        sim.info2 = np.array(sim.info2)
        
        # If you need more info after
        self.complete_res = res
        self.toc          = toc
        
        return sim


# %% CUSTOM BENCH

class custom_bench(Optical_System):

    def __init__(self, modelconfig, model_dir=''):

        super().__init__(modelconfig["modelconfig"])
        self.diam_pup_in_pix = modelconfig["modelconfig"]["diam_pup_in_pix"]
        #self.entrancepupil = pupil(modelconfig, prad=self.diam_pup_in_pix)
        # self.measure_normalization()
        
        # A mettre en parametres 
        self.rcorno = modelconfig["Coronaconfig"]["diam_lyot_in_m"]
        self.ech    = 2
        self.epsi   = 0

        # Definitions
        w = self.dimScience//self.ech
        
        self.pup      = tls.circle(w,w,self.diam_pup_in_pix//2)
        self.pup_d    = tls.circle(w,w,self.diam_pup_in_pix//2)

        self.set_corono(modelconfig["modelconfig"]["filename_instr_pup"])
        
    def EF_through(self,entrance_EF=1.,downstream_EF=1.):

        dim_img = self.dimScience
        dim_pup = dim_img//self.ech

        EF_afterentrancepup = entrance_EF*self.pup
        EF_aftercorno       = mft( self.corno * mft(EF_afterentrancepup,dim_pup,dim_pup,dim_pup,inverse=True) ,dim_pup,dim_pup,dim_pup)
        if(self.zbiais)          : EF_aftercorno = EF_aftercorno - self.z_biais()
        EF_out              = downstream_EF * self.pup_d * EF_aftercorno
        
        return EF_out

    def todetector(self,entrance_EF=1.,downstream_EF=1.):
        w = self.dimScience//self.ech
        return mft(self.EF_through(entrance_EF,downstream_EF),w,self.dimScience,w,inverse=True)
        
    def z_biais(self):
        N = self.dimScience//self.ech
        return mft( self.corno * mft( self.pup ,N,N,N,inverse=True) ,N,N,N)

    def todetector_Intensity(self,entrance_EF=1.,downstream_EF=1.,flux=1,fond=0):

        EF_out = self.todetector(entrance_EF,downstream_EF)

        return flux*pow( abs( EF_out ) ,2) + fond
    
    
    def set_corono(self,cortype):
        """ Change corono type """
        w    = self.dimScience//self.ech
        if  (cortype=="4q") : self.corno = tls.daminer(w,w)
        elif(cortype=="R&R"): self.corno = 2*tls.circle(w,w,self.rcorno)-1
        else                : self.corno = abs(tls.circle(w,w,self.rcorno)-1)
        
        if (cortype=="4q")  : self.zbiais = True
        else                : self.zbiais = False


    def introspect(self,
            entrance_EF=1.,
            downstream_EF=1.):
        """ Display tools to see inside the tbench """
        view_list = []
        N = self.dimScience//self.ech

        # entrance_EF = entrance_EF * dim_pup
        view_list.append(entrance_EF)
        
        view_list.append(view_list[-1]*self.pup)
        
        view_list.append(mft(view_list[-1],N,N,N))
        
        view_list.append(self.corno*view_list[-1])
        
        view_list.append(mft(view_list[-1],N,N,N))
        
        if(self.zbiais): view_list.append(view_list[-1] - self.z_biais())
        
        view_list.append( downstream_EF * self.pup_d *view_list[-1] )
        
        view_list.append(mft(view_list[-1],N,N*self.ech,N))
                

        
        self.view_list = view_list
        if(self.zbiais): self.title_list = ["Entrence EF", "Upsteam pupil", "MFT", "Corno", "MFT", "Correction zbiais" ,"Downstream EF + pupil", "MFT - detecteur"]
        else           : self.title_list = ["Entrence EF", "Upsteam pupil", "MFT", "Corno", "MFT", "Downstream EF + pupil", "MFT - detecteur"]     

        self.into   = plt.figure("Introscpetion")
        self.intoax = self.into.add_subplot(1,1,1),plt.imshow(abs(view_list[0]),cmap='jet'),plt.suptitle(self.title_list[0]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[0])**2))),plt.subplots_adjust(bottom=0.25)
        
        plt.subplots_adjust(bottom=0.2)
        self.slide = Slider(plt.axes([0.25,0.1,0.65,0.03]),"view ",0,len(view_list)-1,valinit=0,valstep=1)
        self.slide.on_changed(self.__update_introspect__)
    
    def __update_introspect__(self,val):
        """ Handler for instrosect slider """
        view_list = self.view_list
        self.into.add_subplot(1,1,1),plt.imshow(abs(view_list[val]),cmap='jet'),plt.suptitle(self.title_list[val]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[val])**2))),plt.subplots_adjust(bottom=0.25)
       