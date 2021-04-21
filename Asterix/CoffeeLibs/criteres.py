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
from Asterix.propagation_functions import mft

# %% #######################
""" Calcule critère  """

def meanSquare(x_u,x_d,y,tb):
    """Compte mean psf square distance between data y and input x"""

    Hx = tb.psf(x_u,x_d)

    d = pow( abs( y - Hx ) , 2 ) # Distance mean square

    return np.sum(np.sum( d ))


def regule(psi,hypp,mode="np"):
    """Compte regule terme """
    psi_gard  = tls.gradient_xy(psi,mode)  # Spacial gardient    
    return sum(sum(psi_gard)) / hypp


def map_J(phi,phi_defoc,abb,i_foc,i_div,tb,hypp):
    """Compte critere J (as a matrix)"""

    EF_foc = np.exp(1j*phi)
    EF_div = np.exp(1j*phi_defoc)
    EF_abb = np.exp(1j*abb)

    Jfoc =  meanSquare(EF_foc,EF_abb,i_foc,tb)
    Jdiv =  meanSquare(EF_div,EF_abb,i_div,tb)

    return Jfoc + Jdiv + regule(phi,hypp,tb.grad)

def diff_grad_J(phi_u,phi_d,i_foc,tb, dphi=10e-5):
    
    EF_u = np.exp(1j*phi_u)
    EF_d = np.exp(1j*phi_d)
    w = tb.dimScience//tb.ech
    
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    point = np.zeros((w,w))
    for a in range(0, w):
          for b in range(0, w):
              
              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              dphi_list.append(-meanSquare(EF_u,EF_d,i_foc,tb) + meanSquare(EF_u*np.exp(1j*point),EF_d,i_foc,tb))
              point[a,b] = point[a,b] - dphi
              
    return np.array(dphi_list).reshape(w,w) / dphi

# %% ################################
""" Calcule gradient du critère  """

def DJ_nc(phi,abb,img,tb):

    psi_in = np.exp(1j*phi)

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi  = tb.EF_through(entrance_EF=psi_in)
    Hphi = tb.psf(entrance_EF=psi_in)

    diff  = Hphi - img
    
    Dj = 4*np.imag( np.conj(psi) * tls.depadding( mft( diff * mft(psi,wphi,w,wphi) ,wphi,w,wphi,inverse=1),tb.ech) )

    return Dj

def DJ_down(phi,abb,img,tb):

    psi_in = np.exp(1j*phi)

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi_u  = tb.pup * psi_in
    psi_d  = np.exp(1j*abb)
    
    psi_0  =  psi_u * psi_d
    PSI_0  =  mft(psi_0,wphi,w,wphi)
    
    psi_c  =  tb.psf(psi_in,psi_d)
    PSI_c  =  mft(psi_c,wphi,w,wphi)
    
    dj_h  = psi_c - img
    
    terme   = mft( dj_h * ( PSI_0 - tb.epsi * PSI_c ) ,wphi,w,wphi,inverse=False)
    # corno_up = tls.depadding( mft( tb.corno * (mft(psi_u,wphi,w,wphi) ) ,wphi,w,wphi,inverse=False),tb.ech)
    
    
    Dj = 2*np.imag( (np.conj(psi_0) - tb.epsi * psi_d ) * tls.depadding(terme,tb.ech) )
    
    return Dj

def DJ_up_v1(phi_u,phi_d,img,tb):

    psi_u = np.exp(1j*phi_u)

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi_u  = tb.pup * psi_u
    psi_d  = tb.pup * np.exp(1j*phi_d)
    
    psi_0  =  psi_u * np.exp(1j*(phi_d+phi_u))
    
    phi_c  =  tb.EF_through(psi_u,psi_d)
    
    dj_h  = phi_c - img
    
    terme   = mft( dj_h * ( phi_c - tb.epsi * phi_c ) ,wphi,w,wphi,inverse=False)
    c_terme = mft( tb.corno * mft( np.conj(mft(psi_d,wphi,w,wphi)) * terme ,wphi,w,wphi),wphi,w,wphi,inverse=False)
    
    
    Dj = 2*np.imag( np.conj(psi_0) * tls.depadding(terme,tb.ech) - tls.depadding( tb.epsi * np.conj( mft(psi_d,wphi,w,wphi)) * c_terme ,tb.ech) )
    
    return Dj

def DJ_up(phi_u,phi_d,img,tb):

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi_u  = tb.pup * np.exp(1j*phi_u)
    psi_d  = tb.pup * np.exp(1j*phi_d)
    
    psi_det =  tb.EF_through(psi_u,psi_d)
    h_det   =  tb.psf(psi_u,psi_d)

    diff    =  h_det - img 
    
    terme1  = tls.depadding( mft( np.conj(psi_det) * diff ,wphi,w,wphi,inverse=False),tb.ech)
    terme2  = tls.depadding( tb.corno * mft( psi_d * terme1 ,wphi,w,wphi),tb.ech)
    terme3  = tls.depadding( mft( terme2 ,wphi,w,wphi,inverse=False),tb.ech)
    
    Dj = - wphi**2 * 2 * np.imag( psi_u * terme3 )
    
    return Dj


def Dregule(psi,hypp,mode="lap"):
    """Compte derive of regule terme """
    if mode=="lap" : return tls.gradient2_xy(psi) / hypp
    else : return tls.gradient_xy(tls.gradient_xy(psi)) / hypp


def grad_map_J_up(phi_u,phi_d,i_foc,i_div,tb,hypp):
    """ Compute gradient of critere up J = Jdiv + Jfoc + regul"""
    
    n = tb.dimScience//tb.ech
    [Ro,Theta] = pmap(n,n)
    defoc = zernike(Ro,Theta,4)

    return   DJ_up(phi_u,phi_d,i_foc,tb) \
           + DJ_up(phi_u + defoc,phi_d + defoc ,i_div,tb) \
           + Dregule(phi_u,hypp,tb.lap)


def grad_map_J_down(phi,phi_div,abb,i_foc,i_div,tb,hypp):
    """ Compute gradient of critere down J = Jdiv + Jfoc + regul"""

    return DJ_down(phi,abb,i_foc,tb) + DJ_down(phi_div,abb,i_div,tb)



# %% ############################
""" Wrappers for optimize """

def V_map_J(var,tb,psi_foc,psi_div,hypp):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    n = tb.dimScience//tb.ech
    [Ro,Theta] = pmap(n,n)
    defoc  =  zernike(Ro,Theta,4)
    
    phi      = var[:n*n].reshape(n,n)
    phi_d    = var[n*n:].reshape(n,n)
    
    return  map_J(phi,phi + defoc ,phi_d,psi_foc,psi_div,tb,hypp)


def V_grad_J(var,tb,psi_foc,psi_div,hypp):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    n = tb.dimScience//tb.ech
    
    phi_u  = var[:n*n].reshape(n,n)
    phi_d  = var[n*n:].reshape(n,n)
    
    grad_u = grad_map_J_up  (phi_u,phi_d,psi_foc,psi_div,tb,hypp).reshape(n*n,)
    grad_d = 0*grad_map_J_up  (phi_u,phi_d,psi_foc,psi_div,tb,hypp).reshape(n*n,)
    
    return  np.concatenate((grad_u, grad_d), axis=0)


