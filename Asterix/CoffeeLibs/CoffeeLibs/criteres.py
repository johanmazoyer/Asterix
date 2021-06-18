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
    
    return np.sum(psi_grad**2)

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

def diff_grad_R(dpoint,pup=1,mode="np",dphi=1e-6):
    
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    Rpoint = regule(dpoint,pup,mode)
    
    w = dpoint.shape[0]
    for a in range(0,w):
          for b in range(0,w):
              
              # Delta au point courant
              dpoint[a,b] = dpoint[a,b] + dphi
              
              dphi_list.append(regule(dpoint,pup,mode) - Rpoint)
              
              dpoint[a,b] = dpoint[a,b] - dphi
                     
    return np.array(dphi_list).reshape(w,w) / dphi

# %% ################################
""" Calcule gradient du critère  """

def DJmv_up(div_id,img,sim):

    tb = sim.tbed    
    offset = tb.offest

    w     = tb.dimScience
    wphi  = tb.dimScience//tb.ech
    alpha = sim.get_flux()
    
    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.todetector(div_id)
    h_det   =  sim.todetector_Intensity(div_id)

    diff    =  h_det - img
    
    
    terme1  = mft( np.conj(psi_det) * alpha * diff ,w,wphi,wphi,inverse=True,**offset)
    terme2  = tb.corno * mft( psi_d * terme1 ,wphi,wphi,wphi,**offset)
    terme3  = mft( terme2 ,wphi,wphi,wphi,inverse=True,**offset)
    
    Dj = - 4 * np.imag( psi_u * terme3 )
    
    return Dj

def DJmv_down(div_id,img,sim):

    tb = sim.tbed    
    offset = tb.offest
    
    w    = tb.dimScience
    wphi = tb.dimScience//tb.ech
    alpha = sim.get_flux()

    psi_u  = tb.pup   * np.exp(1j*sim.get_phi_div(div_id))
    psi_d  = tb.pup_d * sim.get_EF_do()
    
    psi_det =  sim.todetector(div_id)
    h_det   =  sim.todetector_Intensity(div_id)

    diff    =  h_det - img
    
    
    terme1  = mft( np.conj(psi_det) * alpha * diff ,w,wphi,wphi,inverse=True,**offset)
    terme2  = mft( tb.corno * mft( psi_u ,wphi,wphi,wphi,inverse=True,**offset),wphi,wphi,wphi,**offset)
    
    Dj = - 4 * np.imag( psi_d * terme2 * terme1 )
    
    return Dj

def Dregule(psi,mode="lap"):
    """Compte derive of regule terme """
    if mode=="lap" :  lap =  tls.gradient2_xy(psi)
    else :            lap = tls.gradient_xy(tls.gradient_xy(psi))
    return (1/2)*lap

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

def V_map_J(var,sim,imgs,hypp,simGif):
    """ Wrapper for minimize syntax"""
    
    #sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    Hx = sim.gen_div_imgs()
    
    Jmv = meanSquare(Hx,imgs)
    R   =  (1/2) * hypp * regule(sim.get_phi_foc()) 
    
    sim.info.append([Jmv, R])
    
    return  Jmv + R


def V_grad_J(var,sim,imgs,hypp,simGif):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    # sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    
    info_gard      = []
    info_div       = []
    sim.iter +=1
    
    pup  = tls.circle(sim.N, sim.N, sim.tbed.diam_pup_in_pix//2 - 1)
    
    # Compute gradient = dj/dphi
    grad = []
    if not sim.phi_foc_is_known() : 
        grad_u = 0
        for div_id in range(0,imgs.shape[2]):
            grad_u_k = DJmv_up(div_id,imgs[:,:,div_id],sim)
            info_div.append(np.sum(grad_u_k**2))
            grad_u   += grad_u_k
        grad_u *= 1  # Ponderation
        dR      =  pup * hypp * Dregule(sim.get_phi_foc()) # Regulatrisation
        grad    = np.concatenate((grad, (grad_u + dR).reshape(sim.N**2,)), axis=0)


    # Other varaibles gradient
    if not sim.phi_do_is_known() : 
        grad_d = 0
        for div_id in range(0,imgs.shape[2]):
            grad_d += DJmv_down(div_id,imgs[:,:,div_id],sim)
        grad_d *= 1  # Ponderation
        dRdo    = pup * hypp * Dregule(sim.get_phi_do()) # Regulatrisation
        grad    = np.concatenate((grad, (grad_d + dRdo).reshape(sim.N**2,)), axis=0)
                   
    if simGif and not sim.phi_do_is_known() : tls.plot_sim_entries(sim,dR,grad_u,grad_d,dRdo,name="iter"+str(sim.iter),disp=False,save=True)
    elif simGif : tls.plot_sim_entries(sim,dR,grad_u_k,name="iter"+str(sim.iter),disp=False,save=True)              
    
    sim.info_gard.append([np.sum(grad_u**2),np.sum(dR)**2])     
    sim.info_div.append(info_div)
                    
    return  grad


