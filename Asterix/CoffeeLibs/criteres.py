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
from CoffeeLibs.pzernike import pmap, zernike
from sklearn.preprocessing import normalize
from Asterix.propagation_functions import mft

# %% #######################
""" Calcule critère  """

def meanSquare(x,y,tb):
    """Compte mean square distance between data y and input x"""
    
    Hx = tb.psf(entrance_EF=x)
    
    d = pow( abs( y - Hx ) , 2 ) # Distance mean square
    
    return np.sum(np.sum( d ))


def regule_v1(psi):
    """Compte regule terme """
    psi_gard  = tls.gradient_xy(psi)  # Spacial gardient
    var_d2psi = np.var(psi_gard)      # Var of gardient
    
    return sum(sum(normalize(psi_gard))) / var_d2psi
    
    
def map_J(phi,phi_defoc,i_foc,i_div,tb):
    """Compte critere J (as a matrix)"""
    
    EF_foc = np.exp(1j*phi)
    EF_div = np.exp(1j*phi_defoc)
    
    Jfoc =  meanSquare(EF_foc,i_foc,tb)
    Jdiv =  meanSquare(EF_div,i_div,tb)
    
    return Jfoc + Jdiv
    # return Jfoc + Jdiv + regule(phi)





# %% ################################
""" Calcule gradient du critère  """

def DJ(phi,img,tb):
    
    psi_in = np.exp(1j*phi)
    
    psi  = tb.EF_through(entrance_EF=psi_in)
    Hphi = tb.psf(entrance_EF=psi_in)
    
    diff  = Hphi - img
    
    w    = tb.dim_im
    wphi = tb.diam_pup_in_pix
    
    Dj = 4*np.imag( np.conj(psi) *  (1/w) * mft( diff * w * mft(psi,wphi,w,wphi) ,wphi,w,wphi,inv=1) )

    return Dj

def Dregule_v1(psi):
    """Compte derive of regule terme """
    psi_gard  = tls.gradient_xy(psi)     # Spacial gardient
    psi_lap   = tls.gradient2_xy(psi)    # Laplacien == 2nd derive
    var_d2psi = np.var(psi_gard)         # Var of gardient
    
    dR  = psi_lap   
    return dR/var_d2psi


def grad_map_J(phi,phi_div,i_foc,i_div,tb):
    """ Compute gradient of critere J """
    # Dj foc +  Dj div 
    
    # return DJ(phi,i_foc,tb) + DJ(phi_div,i_div,tb) + Dregule(phi)
    return DJ(phi,i_foc,tb) + DJ(phi_div,i_div,tb)



# %% ############################
""" Wrappers for optimize """

def V_map_J(phi,thd2,psi_foc,psi_div):
    n = int(np.sqrt(len(phi)))
    [Ro,Theta] = pmap(n,n)
    phi    = phi.reshape(n,n)
    phidef = phi + zernike(Ro,Theta,4)
    return  map_J(phi,phidef,psi_foc,psi_div,thd2)

def V_grad_J(phi,thd2,psi_foc,psi_div):
    n = int(np.sqrt(len(phi)))
    [Ro,Theta] = pmap(n,n)
    phi    = phi.reshape(n,n)
    phidef = phi + zernike(Ro,Theta,4)
    return grad_map_J(phi,phidef,psi_foc,psi_div,thd2).reshape(n*n,)
