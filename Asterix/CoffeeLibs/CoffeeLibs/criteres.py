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
import copy

# %% #######################
""" Calcule critère  """

def meanSquare(Hx,y):
    """Compte mean square distance between data y and input Hx"""
    d = pow( abs( y - Hx ) , 2 ) # Distance mean square

    return np.sum(np.sum( d ))

def regule(psi,mode="np"):
    """Compte regule terme """
  
    psi_grad  = tls.gradient_xy(psi,mode)  # Spacial gardient
    
    return np.sum(psi_grad)**2

def diff_grad_J_up(point,div_id,sim,i_ref,dphi=1e-6):
        
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    i_point = sim.get_img_div(div_id)

    for a in range(0, sim.N):
          for b in range(0, sim.N):
              
              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              
              sim.set_phi_foc(point)
              i_dpoint = sim.get_img_div(div_id)
              
              dphi_list.append(meanSquare(i_point,i_ref) - meanSquare(i_dpoint,i_ref))
              
              point[a,b] = point[a,b] - dphi
              
    return np.array(dphi_list).reshape(sim.N,sim.N) / dphi

def diff_grad_J_down(point,div_id,esim,i_ref,dphi=1e-6):
    
    sim = copy.deepcopy(esim)
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    i_point = sim.get_img_div(div_id)
    
    for a in range(0, sim.N):
          for b in range(0, sim.N):
              
              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              
              sim.set_phi_do(point)
              i_dpoint = sim.get_img_div(div_id)
              
              dphi_list.append(meanSquare(i_dpoint,i_ref) - meanSquare(i_point,i_ref))
              
 
              point[a,b] = point[a,b] - dphi
                     
    return np.array(dphi_list).reshape(sim.N,sim.N) / dphi


# %% ################################
""" Calcule gradient du critère  """

def DJmv_up(div_id,img,sim):

    tb = sim.tbed    

    w     = tb.dimScience
    wphi  = tb.dimScience//tb.ech
    alpha = sim.get_flux()
    
    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.todetector(div_id)
    h_det   =  sim.todetector_Intensity(div_id)

    diff    =  h_det - img
    
    
    terme1  = mft( np.conj(psi_det) * alpha * diff ,w,wphi,wphi,inverse=True)
    terme2  = tb.corno * mft( psi_d * terme1 ,wphi,wphi,wphi)
    terme3  = mft( terme2 ,wphi,wphi,wphi,inverse=True)
    
    Dj = - 4 * np.imag( psi_u * terme3 )
    
    return Dj

def DJmv_down(div_id,img,sim):

    tb = sim.tbed    
    
    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech
    alpha = sim.get_flux()

    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.todetector(div_id)
    h_det   =  sim.todetector_Intensity(div_id)

    diff    =  h_det - img
    
    
    terme1  = mft( np.conj(psi_det) * alpha * diff ,w,wphi,wphi,inverse=True)
    terme2  = mft( tb.corno * mft( psi_u ,wphi,wphi,wphi,inverse=True),wphi,wphi,wphi)
    
    Dj = - 4 * np.imag( psi_d * terme2 * terme1 )
    
    return Dj

def Dregule(psi,mode="lap"):
    """Compte derive of regule terme """
    if mode=="lap" :  lap =  tls.gradient2_xy(psi)
    else :            lap = tls.gradient_xy(tls.gradient_xy(psi))
    return lap

# %% ######################
"""Flux fond """

def estime_fluxfond(sim,imgs):
    h    = sim.get_img_div(0,ff=False)
    img  = imgs[:,:,0]
    hsum = np.sum(h)
    mat  = np.array( [[ np.sum(h*h) , hsum ],[ hsum , img.size ]])/img.size
    vect = np.array( [ np.sum(sim.get_img_div(0,ff=False)*img),np.sum(img)])/img.size
    [flux,fond]   = np.linalg.solve(mat,vect)
    return flux,fond


# %% ############################
""" Wrappers for optimize """

def V_map_J(var,sim,imgs,hypp):
    """ Wrapper for minimize syntax"""
    
    # sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    Hx = sim.gen_div_imgs()
    
    Jmv = meanSquare(Hx,imgs)
    R   =  hypp * regule(sim.get_phi_foc()) 
    
    sim.info2.append([Jmv, R])
    
    return  Jmv + R 


def V_grad_J(var,sim,imgs,hypp):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    # sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    
    info = []
    
    # Compute gradient = dj/dphi
    grad = []
    if not sim.phi_foc_is_known() : 
        grad_u = 0
        for div_id in range(0,imgs.shape[2]):
            grad_u_k = DJmv_up(div_id,imgs[:,:,div_id],sim).reshape(sim.N**2,)
            info.append(np.sum(grad_u_k**2))
            grad_u   += grad_u_k
        grad_u *= 1  # Ponderation
        R       =  hypp * Dregule(sim.get_phi_foc()).reshape(sim.N**2,) # Regulatrisation
        grad_u += R
        grad    = np.concatenate((grad, grad_u), axis=0)
        info.append(np.sum(R**2))


    # Other varaibles gradient
    if not sim.phi_do_is_known() : 
        grad_d = 0
        for div_id in range(0,imgs.shape[2]):
            grad_d += DJmv_down(div_id,imgs[:,:,div_id],sim).reshape(sim.N**2,)
        grad_d *= 1  # Ponderation
        grad_d += hypp * Dregule(sim.get_phi_do()).reshape(sim.N**2,) # Regulatrisation
        grad    = np.concatenate((grad, grad_d), axis=0)
                   
    sim.info.append(info)                            
    return  grad


