# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
------------  Calcule des critère  ---------
------------    et des gradients   ---------
--------------------------------------------

@author: sjuillar
"""

import numpy as np
import CoffeeLibs.tools as tls
import CoffeeLibs.simu as simu
from copy import deepcopy
from CoffeeLibs.param import *
from sklearn.preprocessing import normalize
from Asterix.propagation_functions import mft
from Asterix.InstrumentSimu_functions import THD2_testbed

# %% #######################
""" Calcule critère  """

def meanSquare(x,y,tb):
    """Compte mean square distance between data y and input x"""
    
    Hx = tb.EF_through(entrance_EF=x,
                   wavelength=None,
                   DM1phase=0.,
                   DM3phase=0.,
                   noFPM=False,
                   save_all_planes_to_fits=False,
                   dir_save_fits=None)
    
    d = pow( abs( y - Hx ) , 2 ) # Distance mean square
    
    return np.sum(np.sum( d ))


def regule_v1(psi):
    """Compte regule terme """
    psi_gard  = tls.gradient_xy(psi)  # Spacial gardient
    var_d2psi = np.var(psi_gard)      # Var of gardient
    
    return sum(sum(normalize(psi_gard))) / var_d2psi
    
def regule(psi):
    """Compte regule terme """
    return sum(sum((psi**2))) / (2*wphi*lphi)
    
    
def map_J(phi,phi_defoc,i_foc,i_div,tb):
    """Compte critere J (as a matrix)"""
    
    Jfoc =  meanSquare(phi,i_foc,tb)
    Jdiv =  meanSquare(phi_defoc,i_div,tb)
    
    # return Jfoc + Jdiv
    return Jfoc + Jdiv + regule(phi)





# %% ################################
""" Calcule gradient du critère  """

def DJ(phi,img,tb):
    
    P = tb.entrancepupil.pup
    psi  = P*np.exp(1j*phi) 
    Hphi = tb.EF_through(entrance_EF=phi,
                   wavelength=None,
                   DM1phase=0.,
                   DM3phase=0.,
                   noFPM=False,
                   save_all_planes_to_fits=False,
                   dir_save_fits=None)
    
    diff  = Hphi - img
    
    wphi = phi.shape[0]
    w    = Hphi.shape[0]
    
    ### !!!!!!  Pas de prise en compte d'échtillonage   !!!!!!! ###
    Dj = 4*np.imag( np.conj(psi) *  (1/w) * mft( diff * w * mft(psi,wphi,w,wphi) ,wphi,w,wphi,inv=1) )

    return Dj

def Dregule_v1(psi):
    """Compte derive of regule terme """
    psi_gard  = tls.gradient_xy(psi)     # Spacial gardient
    psi_lap   = tls.gradient2_xy(psi)    # Laplacien == 2nd derive
    var_d2psi = np.var(psi_gard)         # Var of gardient
    
    dR  = psi_lap   
    return dR

def Dregule(psi):
    """Compte derive of regule terme """
    return psi/(wphi*lphi)

def grad_map_J(phi,phi_div,i_foc,i_div,tb):
    """ Compute gradient of critere J """
    # Dj foc +  Dj div 
    return DJ(phi,i_foc,tb) + DJ(phi_div,i_div,tb) + Dregule(phi)
    # return DJ(phi,i_foc) + DJ(phi_div,i_div)




# %% ######################################
""" Gradient différentielle """

def diff_grad_J(PHI0,i_foc,i_div, dphi=10e-5):
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    PHI0 = tls.padding(PHI0,ech)
    for a in range(0, w):
          for b in range(0, l):
              
              # Delta au point courant
              point = deepcopy(PHI0)
              point[a,b] = point[a,b] + dphi
              
              # dphi_list.append( - map_J(PHI0,i_foc,i_div) + map_J(point,i_foc,i_div) ) 
              dphi_list.append(-meanSquare(PHI0,i_foc) + meanSquare(point,i_foc))
              
    return tls.depadding( np.array(dphi_list).reshape(w,l) / dphi ,ech) 



# %% ############################
""" Wrappers for optimize """

def V_map_J(phi,thd2,psi_foc,psi_div):
    n = int(np.sqrt(len(phi)))
    phi    = phi.reshape(n,n)
    phidef = simu.defoc(phi,defoc_factor)
    return  map_J(phi,phidef,psi_foc,psi_div,thd2)

def V_grad_J(phi,thd2,psi_foc,psi_div):
    n = int(np.sqrt(len(phi)))
    phi    = phi.reshape(n,n)
    phidef = simu.defoc(phi,defoc_factor)
    return grad_map_J(phi,phidef,psi_foc,psi_div,thd2).reshape(n*n,)
