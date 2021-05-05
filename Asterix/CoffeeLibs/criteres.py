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

def meanSquare(Hx,y):
    """Compte mean square distance between data y and input Hx"""
    d = pow( abs( y - Hx ) , 2 ) # Distance mean square

    return np.sum(np.sum( d ))

def regule(psi,hypp,mode="sobel"):
    """Compte regule terme """
  
    psi_grad  = tls.gradient_xy(psi,mode)  # Spacial gardient
    var       = np.var(psi_grad)
    
    return sum(sum(psi_grad)) * var**2 * hypp

def diff_grad_J_up(point,div_id,sim,i_ref,tb,dphi=1e-6):
    
    
    w = tb.dimScience//tb.ech
    
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    sim.set_phi_foc(point)
    i_point = sim.get_img_div(tb,div_id)

    for a in range(0, w):
          for b in range(0, w):
              
              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              
              sim.set_phi_foc(point)
              i_dpoint = sim.get_img_div(tb,div_id)
              dphi_list.append(-meanSquare(i_point,i_ref) + meanSquare(i_dpoint,i_ref))
              
              point[a,b] = point[a,b] - dphi
              
    return np.array(dphi_list).reshape(w,w) / dphi

def diff_grad_J_down(point,div_id,sim,i_ref,tb,dphi=1e-6):
    
    
    w = tb.dimScience//tb.ech
    
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    sim.set_phi_do(point)
    i_point = sim.get_img_div(tb,div_id)

    for a in range(0, w):
          for b in range(0, w):
              
              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              
              sim.set_phi_do(point)
              i_dpoint = sim.get_img_div(tb,div_id)
              dphi_list.append(-meanSquare(i_point,i_ref) + meanSquare(i_dpoint,i_ref))
              
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

    psi_u  = tb.pup   * psi_u
    psi_d  = tb.pup_d * np.exp(1j*phi_d)
    
    psi_0  =  psi_u * np.exp(1j*(phi_d+phi_u))
    
    phi_c  =  tb.EF_through(psi_u,psi_d)
    
    dj_h   = phi_c - img
    
    terme   = mft( dj_h * ( phi_c - tb.epsi * phi_c ) ,wphi,w,wphi,inverse=False)
    c_terme = mft( tb.corno * mft( np.conj(mft(psi_d,wphi,w,wphi)) * terme ,wphi,w,wphi),wphi,w,wphi,inverse=False)
    
    
    Dj = 2*np.imag( np.conj(psi_0) * tls.depadding(terme,tb.ech) - tls.depadding( tb.epsi * np.conj( mft(psi_d,wphi,w,wphi)) * c_terme ,tb.ech) )
    
    return Dj

def DJmv_up(div_id,img,sim,tb):

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.EF_through(div_id,tb)
    h_det   =  sim.psf(div_id,tb)

    diff    =  sim.get_flux()*(h_det - img)
    
    
    terme1  = mft( np.conj(psi_det) * diff ,wphi,w,wphi,inverse=False)
    terme2  = tb.corno * mft( tls.padding(psi_d,tb.ech) * terme1 ,wphi,w,wphi)
    terme3  = mft( terme2 ,wphi,w,wphi,inverse=False)
    
    Dj = - 4 * sim.get_flux() * np.imag( tls.padding(psi_u,tb.ech) * terme3 )
    
    return tls.depadding(Dj,tb.ech)

def DJmv_down(div_id,img,sim,tb):

    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech

    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.EF_through(div_id,tb)
    h_det   =  sim.psf(div_id,tb)

    diff    =  h_det - img
    
    
    terme1  = mft( np.conj(psi_det) * diff ,wphi,w,wphi)
    terme2  = mft( tb.corno * mft( psi_u ,wphi,w,wphi),wphi,w,wphi)
    
   # Dj = - 4 * np.imag( tls.padding(psi_d,tb.ech) * terme2 * terme1 )
    
   # return tls.depadding(Dj,tb.ech)
   
    Dj = - 4 * np.imag( psi_d * terme2 * terme1 )
    
    return Dj

def Dregule(psi,hypp,mode="lap"):
    """Compte derive of regule terme """
    if mode=="lap" :  lap =  tls.gradient2_xy(psi)
    else :            lap = tls.gradient_xy(tls.gradient_xy(psi))
    var       = np.var(tls.gradient_xy(psi))
    return lap * var**2 * hypp

# %% ######################
"""Flux fond """

def estime_fluxfond(sim,tbed,imgs):
    for div_id in range(0,len(sim.div_factors)) : 
        h    = sim.get_img_div(tbed,div_id,ff=False)
        img  = imgs[:,:,div_id]
        hsum = np.sum(h)
        mat  = np.array( [[ np.sum(h*h) , hsum ],[ hsum , img.size ]])/img.size
        vect = np.array( [ np.sum(sim.get_img_div(tbed,div_id,ff=False)*img),np.sum(img)])/img.size
        [flux,fond]   = np.linalg.solve(mat,vect)
    return flux,fond


# %% ############################
""" Wrappers for optimize """

def V_map_J(var,tb,sim,imgs,hypp):
    """ Wrapper for minimize syntax"""
    
    sim.opti_update(var,tb,imgs)
    Hx = sim.gen_div_imgs(tb)

    return  meanSquare(Hx,imgs) + regule(sim.get_phi_foc(),hypp)


def V_grad_J(var,tb,sim,imgs,hypp):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    sim.opti_update(var,tb,imgs)
    
    # Compute gradient = dj/dphi
    grad = 0
    for div_id in range(0,imgs.shape[2]):
        grad += DJmv_up(div_id,imgs[:,:,div_id],sim,tb).reshape(sim.N**2,)
    grad += Dregule(sim.get_phi_foc(),hypp).reshape(sim.N**2,) # Add Regulatrisation

    # Other varaible gradient
    if not sim.phi_do_is_known() : 
        grad_d = 0
        for div_id in range(0,imgs.shape[2]):
            grad_d += DJmv_down(div_id,imgs[:,:,div_id],sim,tb).reshape(sim.N**2,)
        grad_d += Dregule(sim.get_phi_do(),hypp).reshape(sim.N**2,) # Add Regulatrisation
        grad    = np.concatenate((grad, grad_d), axis=0)
                                                  
    return  grad


