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

## Pondertion moche 
normA = 1e5 #
  # normA pour ~Normalisation Amplifié~
  # pour eviter les valeurs trop faible pour le minimiseur 
  # psq sinon images sont trop vite identiques 
  # <=> pas bcp de diff entre deux points différents du gradient
  # <=> point arret atteint prématurement
  # Et que il faut que ça marche pour corono asterix ET corono moi
  # Et que ducoup de doit changer tt la normalisation des mfts partout
  # Parce que j'ai pas mis de flux pour le gradient auto
  # Et c'est penbible de remplacer toute les iterations du coeff à la main
  # J'en oublie systematiquement 1
  # En + avec le corono Asterix ça ne marche même pas
  # et j'ai pas le temps de faire un truc plus propre
  # ! doit aussi être changé dans critère.py !
  # PS : ça casse la ponderation avec la regularisation...
  

# %% DATA SIMULATOR

class data_simulator():

    def __init__(self,tbed, known_var = "default", div_factors = [0,1], phi_foc = None, cplx = False, myope = False):
        
        # Store simulation Variables
        self.tbed      = tbed
        self.phi_foc   = phi_foc
        self.known_var = known_var
        self.cplx      = cplx
        self.myope     = myope
        
        self.N     = int(tbed.dim_overpad_pupil)
        [Ro,Theta] = pmap(self.N,self.N,tbed.prad//2)
        self.defoc = zernike(Ro,Theta,4)
        
        self.set_div_map(div_factors)

        # To keep track on what I am suppose to know
        # If default, we know everything
        self.__remeber_known_var_as_bool__()
        self.fondflux_forall()
        
        # Set default everywhere it is needed
        if known_var == "default" : self.set_default_value()
        if phi_foc is None : self.set_phi_foc(np.zeros((self.N,self.N)))
     
    ################################################################
    ####  Generators  ####
    # Generate stuff using object propreties
    
    def gen_zernike_phi_foc(self,coeff):
        """ Generate a zernike polynome using coeff and assign phi_foc  """
        self.set_phi_foc(self.gen_zernike_phi(coeff))

    def gen_zernike_phi_do(self,coeff):
        """ Generate a zernike polynome using coeff and assign phi_do  """
        self.set_phi_do(self.gen_zernike_phi(coeff))
    
    def gen_zernike_phi(self,coeff):
        """ Generate a zernike polynome using coeff and return it  """
        [Ro,Theta] = pmap(self.N,self.N,self.tbed.prad//2)
        return pzernike(Ro,Theta,coeff)
    
    def gen_div_phi(self):
        """ Generate list of phis with different diverity
            i.e : phi[k] = phi_foc + div_map[k]
        """
        n = self.N
        phis = np.zeros((n,n,self.nb_div))
        
        # List of phi_up = phi_foc + carte de diversité
        phis = (phis + self.phi_foc.reshape(n,n,1)) + self.div_map
        if self.myope : phis += self.div_err
        
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
        self.EF_foc  = np.exp(1j*self.phi_foc)
        
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
        
        # Init err fiv if mode myope
        if self.myope : self.div_err = np.zeros((n,n,self.nb_div))
       
    def set_nb_div(self,div_factors):
        self.nb_div = len(div_factors)
        
    def set_flux(self,flux,div_id=None):
        if div_id is not None : self.known_var['flux'][div_id] = flux
        self.known_var['flux'] = flux
        self.fondflux_forall()
        
    def set_fond(self,fond,div_id=None):
        if div_id is not None : self.known_var['fond'][div_id] = fond
        self.known_var['fond'] = fond
        self.fondflux_forall()
    
    def get_EF(self,defoc=0):
        return self.EF_foc
    
    def get_EF_div(self,div_id,fromEF=False):
        if fromEF : return self.EF_foc * self.get_defocEF(div_id)
        else      : return np.exp(1j*self.get_phi_div(div_id))
    
    def get_img(self,div_id=0):
        return self.todetector_Intensity(div_id)
    
    def get_img_div(self,div_id,ff=True):
        if ff : return self.todetector_Intensity(div_id)
        else  : return self.tbed.todetector_Intensity(self.get_EF_div(div_id),self.get_EF_do())
    
    def get_phi_foc(self):
        return self.phi_foc * self.tbed.pup
    
    def get_phi_div(self,div_id):
        if self.myope : 
            return self.phi_foc + self.div_map[:,:,div_id] + self.div_err[:,:,div_id] 
        else : 
            return self.phi_foc + self.div_map[:,:,div_id]
    
    def get_div_err(self,div_id):
        """return error on diversity """
        return self.div_err[:,:,div_id]

    def get_div_est(self,div_id):
        """return estimated diversity """
        return self.div_map[:,:,div_id] + self.div_err[:,:,div_id]

    def get_EF_do(self):
        return self.known_var['downstream_EF']
    
    def get_phi_do(self):
        if hasattr(self, 'phi_do') : return self.phi_do
        else : return 0 
        
    def get_flux(self,div_id=None):
        if div_id is None : return self.known_var['flux']
        else              : return self.known_var['flux'][div_id]
        
    def get_fond(self,div_id=None):
        if div_id is None : return self.known_var['fond']
        else              : return self.known_var['fond'][div_id]
        
    def get_defocEF(self,div_id):
        return np.exp(1j*self.div_map[:,:,div_id])
    
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
        
        if not self.phi_foc_is_known() and not self.cplx : pack = np.concatenate((pack, self.get_phi_foc().reshape(n**2,)), axis=None)
        if not self.phi_foc_is_known() and     self.cplx : pack = np.concatenate((pack, tls.complex2real(self.get_phi_foc())), axis=None)
        if not self.phi_do_is_known()  : pack = np.concatenate((pack, self.get_phi_do().reshape(n**2,)), axis=None)         
        if self.myope  : pack = np.concatenate((pack, np.zeros( (n**2*self.nb_div,)) ), axis=None)         

        return pack
    
    def opti_bound(self,bound = 1,phi_foc=None):
        """Concatene variables to fit optimize minimiza syntax"""
        
        self.set_default_value()
        n = self.N
        pack = []
        pup = self.tbed.pup
        
        if not self.phi_foc_is_known() and not self.cplx : pack = np.concatenate((pack, (bound*pup).reshape(n**2,)), axis=None)
        if not self.phi_foc_is_known() and     self.cplx : pack = np.concatenate((pack, tls.complex2real(bound*pup)), axis=None)
        if not self.phi_do_is_known()  : pack = np.concatenate((pack, (bound*pup).reshape(n**2,)), axis=None)         
        if self.myope  : pack = np.concatenate( (pack, np.matlib.repmat((bound*pup).reshape(n**2,),self.nb_div,1) ), axis=None)         

        return np.transpose([-pack,pack])
    
    
    def opti_update(self,pack,imgs):
        """Uptade sim with varaibles from minimizer optimize"""
        
        n = self.N        
        indx = 0

        if not self.phi_foc_is_known() :
            if self.cplx : 
                self.set_phi_foc(tls.real2complex(pack[:n*n].reshape(n,n),pack[n*n:2*n*n].reshape(n,n)))
                indx += 2*n*n
            else    : 
                self.set_phi_foc(pack[:n*n].reshape(n,n))
                indx += n*n
        
        if not self.phi_do_is_known() : 
            self.set_phi_do(pack[indx:indx+n*n].reshape(n,n))
            indx += n*n
        
        if self.myope  : 
            for div_id in range(self.nb_div):
                self.div_err[:,:,div_id] = pack[indx:indx+n*n].reshape(n,n)
                indx += n*n
        
        # Matrice Inversion
        if (not self.flux_is_known()) | (not self.fond_is_known()) : ff_list = cacl.estime_fluxfond(self,imgs)
            
        if not self.flux_is_known() : self.set_flux(ff_list[0,:])
        if not self.fond_is_known() : self.set_fond(ff_list[1,:])

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
            imgs = imgs + self.get_flux()*np.random.normal(0, varb, imgs.shape)
            
        return imgs

    def EF_through(self,div_id,ff=True):
        """ call EF_through bench"""
        return self.tbed.EF_through(self.get_EF_div(div_id),EF_aberrations_LS=self.get_EF_do())

    def todetector(self,div_id,ff=True):
        """ call EF_through bench"""
        if isinstance(div_id, int) : return normA*self.tbed.todetector(self.get_EF_div(div_id),EF_aberrations_LS=self.get_EF_do())
        else                       : return normA*self.todetector_loop(self.get_EF_div(div_id),EF_aberrations_LS=self.get_EF_do())    

    def todetector_loop(self,EFs=1.,downstream_EF=1):
        """ Just to make a big array of all div EFs because it is cleaner to me"""
        shape  = (self.tbed.dimScience,self.tbed.dimScience,1)
        res    = normA * self.tbed.todetector(EFs[:,:,0],EF_aberrations_LS=downstream_EF).reshape(shape)
        
        for ii in range(1,EFs.shape[2]):
            out = normA * self.tbed.todetector(EFs[:,:,ii],EF_aberrations_LS=downstream_EF).reshape(shape)
            res = np.append(res,out,axis=2)
        
        return res

    def __psf__(self,EFs):
        """ to detector intensity but EFs are 3D so we loop """
        return self.get_flux() * abs( self.todetector_loop(EFs,self.get_EF_do()) ** 2 ) + self.get_fond()


    def todetector_Intensity(self,div_id=0):
        """ call to detector intensity bench for a specific div id image"""
        return self.get_flux(div_id) * abs( self.todetector(div_id) )**2 + self.get_fond(div_id)

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
        
    
    def fondflux_forall(self):
        """If flux/fond is int, them defin list of flux fond equal for each diversity """
        
        if not 'flux' in self.known_var : self.known_var['flux'] = 1
        if not 'fond' in self.known_var : self.known_var['fond'] = 0
        
        fu = self.get_flux()
        fo = self.get_fond()
        if not (isinstance(fu,list) or isinstance(fu,np.ndarray)) : self.known_var['flux'] = [self.get_flux()] * self.nb_div
        if not (isinstance(fo,list) or isinstance(fo,np.ndarray)) : self.known_var['fond'] = [self.get_fond()] * self.nb_div
        
        if (len(self.get_flux()) is not self.nb_div) or ( len(self.get_fond()) is not self.nb_div) :
            raise Exception('Flux/fond list len have to match number of diversity')
        
        
    def set_default_value(self):
        """ if vars are unset, set them to default values"""
        # In case known_var was an str flag
        if not isinstance(self.known_var, dict) : self.known_var=dict()
        keys = (self.known_var).keys()
        
        if 'downstream_EF' not in keys: self.set_phi_do(np.zeros((self.N,self.N)))
        if 'flux'          not in keys: self.set_flux(1)
        if 'fond'          not in keys: self.set_fond(0)  

# %% ESTIMATOR
    
class coffee_estimator:
    """ --------------------------------------------------
    COFFEE estimator
    -------------------------------------------------- """

    def __init__(self,var_phi=0 ,method = 'L-BfGS-B',auto=False ,cplx=False,myope=False, bound=3, name=None,disp=False,H=None,**kargs):
        
        self.var_phi = var_phi
        self.disp    = disp
        self.bound   = bound
        self.simGif  = name
        self.auto    = auto
        self.cplx    = cplx
        self.myope   = myope
        
        # Minimize parameters
        self.minimiz_options = {'disp' : True} # default disp activated
        for key in kargs.keys() :
            if key in ['disp','maxcor','ftol','gtol','eps','maxfun','maxiter','iprint','maxls','finite_diff_rel_step'] :
                self.minimiz_options[key] = kargs[key]
                
        self.setting = {'options' : self.minimiz_options, 'method': method }
        # (option is mutable -> you change option it will change in settings :)
        
        # Pour le gradient automatique H = matrice de gradient
        self.set_auto(auto)
        
        
    def set_auto(self,auto):
        self.auto    = auto
        if auto : 
            self.setting["fun"] = cacl.V_map_J_auto
            self.setting["jac"] = cacl.V_map_dJ_auto
        else :
            self.setting["jac"] = cacl.V_grad_J
            self.setting["fun"]  = cacl.V_map_J    

    def estimate(self,tbed,imgs,div_factors,known_var=dict(),phi_foc=None,**kwargs):
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
            e_sim : data_simulator objetc. Minimize will estimate the best
            e_sim propreties to mach given images. 
            (i.e : phi_foc phi_do flux fond)
        """
        
        ## -- Create our simulation -- ##
        
        mode    = {'cplx':self.cplx,'myope':self.myope} 
        sim   = data_simulator(tbed,known_var,div_factors,phi_foc,**mode)
        sim   = sim_init_infos(sim)
        if not sim.flux_is_known() : sim.set_flux(np.sum(imgs[:,:,0])) # Pour eviter l'erreur LinAlgError: Singular matrix.

        ## -- Agrs : Fucntion J and DJ required arguments -- ##
        
        varb = np.median(imgs) + 1   # Ponderation varb
        # prad = diametre. so cut 2 pix on rad = 4 pix on diam
        spup  = tls.circle(sim.N, sim.N, sim.tbed.prad//2 - 2) # Small pup for regularisation border effect
        
        self.setting['args'] = sim,imgs,self.var_phi,varb,spup,self.simGif 
        if self.auto : self.setting['args'] += (cacl.genere_L(tbed),)
    
        ## -- xo and bounds initilisation
        self.setting['x0']     = sim.opti_pack()
        if self.bound : self.setting['bounds'] = sim.opti_bound(self.bound)

        
        ## --  Estimation with minimize  -- ##
        
        if self.disp : sim.print_know_war()
        
        tic  = t.time()
        res  = minimize(**self.setting)
        toc  = t.time() - tic
        
        ## -- Disp and saves -- ##
        
        if self.disp : print_time(tic,toc)
        if self.simGif : tls.iter_to_gif(self.simGif)
        
        # Store infos proprely
        sim.opti_update(res.get('x'),imgs)
        sim = sim_info2array(sim)
        self.complete_res = res
        self.toc          = toc
        
        return sim

def print_time(tic,toc):
    mins = int(toc//60)
    sec  = 100*(toc-60*mins)
    print("Minimize took : " + str(mins) + "m" + str(sec)[:3])
    
def sim_init_infos(sim):
    sim.iter  = 0
    sim.info  = [] 
    sim.info_gard = []
    sim.info_div  = []
    return sim

def sim_info2array(sim):
    # list to array because it is better
    sim.info_gard  = np.array(sim.info_gard)
    sim.info_div   = np.array(sim.info_div)
    sim.info       = np.array(sim.info)
    return sim



# %% CUSTOM BENCH

class custom_bench(Optical_System):

    def __init__(self, modelconfig, model_dir=''):

        super().__init__(modelconfig["modelconfig"])
        self.prad = modelconfig["modelconfig"]["diam_pup_in_pix"]
        #self.entrancepupil = pupil(modelconfig, prad=self.prad)
        # self.measure_normalization()
        
        # A mettre en parametres 
        self.rcorno = modelconfig["Coronaconfig"]["rad_lyot_fpm"]
        if "rlyot_in_pix" in modelconfig["Coronaconfig"] : self.rcorno = modelconfig["Coronaconfig"]["rlyot_in_pix"]

        self.Science_sampling = int(modelconfig["modelconfig"]["Science_sampling"])
        self.epsi   = 0

        # Definitions
        w = self.dimScience//self.Science_sampling
        self.dim_overpad_pupil = w   # In this bed no overpadding
        
        self.pup      = tls.circle(w,w,self.prad//2)
        self.pup_d    = tls.circle(w,w,self.prad//2)

        self.set_corono(modelconfig["modelconfig"]["filename_instr_pup"])
        
        
        self.offest = {'X_offset_input':-0.5,'Y_offset_input':-0.5,'X_offset_output':-0.5,'Y_offset_output':-0.5}

        
    def EF_through(self,entrance_EF=1.,EF_aberrations_LS=1.):

        dim_img = self.dimScience
        dim_pup = dim_img//self.Science_sampling

        EF_afterentrancepup = entrance_EF*self.pup
        EF_aftercorno       = mft( self.corno * mft(EF_afterentrancepup,dim_pup,dim_pup,dim_pup,inverse=True,**self.offest) ,dim_pup,dim_pup,dim_pup,**self.offest)
        if(self.zbiais)          : EF_aftercorno = EF_aftercorno - self.z_biais()
        EF_out              = EF_aberrations_LS * self.pup_d * EF_aftercorno
        
        return EF_out

    def todetector(self,entrance_EF=1.,EF_aberrations_LS=1.):
        w = self.dimScience//self.Science_sampling
        return mft(self.EF_through(entrance_EF,EF_aberrations_LS),w,self.dimScience,w,inverse=True,**self.offest)
        
    def z_biais(self):
        N = self.dimScience//self.Science_sampling
        return mft( self.corno * mft( self.pup ,N,N,N,inverse=True,**self.offest) ,N,N,N,**self.offest)

    def todetector_Intensity(self,entrance_EF=1.,EF_aberrations_LS=1.):

        EF_out = self.todetector(entrance_EF,EF_aberrations_LS)

        return pow( abs( EF_out ) ,2)
    
    
    def set_corono(self,cortype):
        """ Change corono type """
        w    = self.dimScience//self.Science_sampling
        if  (cortype=="4q") : self.corno = tls.daminer(w,w)
        elif(cortype=="R&R"): self.corno = 2*tls.circle(w,w,self.rcorno)-1
        else                : self.corno = abs(tls.circle(w,w,self.rcorno)-1)
        
        if (cortype=="4q")  : self.zbiais = True
        else                : self.zbiais = False


    def introspect(self,
            entrance_EF=1.,
            EF_aberrations_LS=1.):
        """ Display tools to see inside the tbench """
        view_list = []
        plt.ion()

        N = self.dim_overpad_pupil

        # entrance_EF = entrance_EF * dim_pup
        view_list.append(entrance_EF)
        
        view_list.append(view_list[-1]*self.pup)
        
        view_list.append(N * mft(view_list[-1],N,N,N,inverse=True,**self.offest))
        
        view_list.append(self.corno*view_list[-1])
        
        view_list.append(1/N * mft(view_list[-1],N,N,N,**self.offest))
        
        if(self.zbiais): view_list.append(view_list[-1] - self.z_biais())
        
        view_list.append( EF_aberrations_LS * self.pup_d * view_list[-1] )
        
        view_list.append(N*self.Science_sampling * mft(view_list[-1],N,N*self.Science_sampling,N,inverse=True,**self.offest))
                

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
        val = int(val)
        self.into.add_subplot(1,1,1),plt.imshow(abs(view_list[val]),cmap='jet'),plt.suptitle(self.title_list[val]),plt.title("Energie = %.5f" % sum(sum(abs(view_list[val])**2))),plt.subplots_adjust(bottom=0.25)
       
