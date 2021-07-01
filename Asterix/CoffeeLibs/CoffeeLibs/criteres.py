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

def diff_grad_EF(point,tbed,i_ref,dphi=1e-6):
        
    dh_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    i_point = tbed.todetector_Intensity(point)
    
    N = tbed.dimScience//tbed.ech
    for a in range(0,N):
          for b in range(0,N):

              # Delta au point courant
              point[a,b] = point[a,b] + dphi
              i_dpoint = tbed.todetector_Intensity(point)
              dh_list.append(meanSquare(i_point,i_ref) - meanSquare(i_dpoint,i_ref))
              point[a,b] = point[a,b] - dphi
              
    return np.array(dh_list).reshape(N,N) / dphi


# %% ################################
""" Calcule gradient du critère  """

def DJmv_up(div_id,img,sim):

    tb = sim.tbed    
    offset = tb.offest

    w     = tb.dimScience
    wphi  = tb.dimScience//tb.ech
    alpha = sim.get_flux(div_id)
    
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
    alpha = sim.get_flux(div_id)

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
    else :            lap =  tls.gradient_xy(tls.gradient_xy(psi))
    
    return (1/2)*lap

# %% ######################
"""Flux fond """

def estime_fluxfond(sim,imgs):
    ff_list = np.zeros((2,sim.nb_div))
    
    for ii in range(sim.nb_div):
        h    = sim.get_img_div(ii,ff=False)
        img  = imgs[:,:,ii]
        hsum = np.sum(h)
        mat  = np.array( [[ np.sum(h*h) , hsum ],[ hsum , img.size ]])/img.size
        vect = np.array( [ np.sum(sim.get_img_div(ii,ff=False)*img),np.sum(img)])/img.size
        ff_list[:,ii]   = np.linalg.solve(mat,vect)
    
    return ff_list


# %% ############################
""" Wrappers for optimize ANALYTQIUE"""

def V_map_J(var,sim,imgs,hypp,simGif):
    """ Wrapper for minimize syntax"""
    
    #sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    Hx = sim.gen_div_imgs()
    
    varb = np.median(imgs) + 1
    
    Jmv = meanSquare(Hx,imgs) / (sim.nb_div * varb)
    
    R   = (1/2) * hypp * regule(sim.get_phi_foc())
    
    sim.info.append([Jmv, R])
    
    return  Jmv + R


def V_grad_J(var,sim,imgs,hypp,simGif):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    # sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    
    varb = np.median(imgs) + 1
    
    info_div       = []
    sim.iter +=1
    
    pup  = tls.circle(sim.N, sim.N, sim.tbed.diam_pup_in_pix//2 - 1)
    
    # Compute gradient = dj/dphi
    grad = []
    if not sim.phi_foc_is_known() : 
        grad_u = 0
        for div_id in range(0,sim.nb_div):
            grad_u_k = DJmv_up(div_id,imgs[:,:,div_id],sim)
            info_div.append(np.sum(grad_u_k**2))
            grad_u   += grad_u_k
        grad_u *= 1/(sim.nb_div * varb)  # Ponderation
        dR      = pup * hypp * Dregule(sim.get_phi_foc()) # Regulatrisation
        grad    = tls.add_to_list(grad, grad_u + dR)

    # Other varaibles gradient
    if not sim.phi_do_is_known() : 
        grad_d = 0
        for div_id in range(0,sim.nb_div):
            grad_d += DJmv_down(div_id,imgs[:,:,div_id],sim)
        grad_d *= 1  # Ponderation
        dRdo    = pup * hypp * Dregule(sim.get_phi_do()) # Regulatrisation
        grad    = tls.add_to_list(grad, grad_d + dRdo)
                   
    if simGif and not sim.phi_do_is_known() : tls.plot_sim_entries(sim,dR,grad_u,grad_d,dRdo,name="iter"+str(sim.iter),disp=False,save=True)
    elif simGif : tls.plot_sim_entries(sim,dR,grad_u_k,name="iter"+str(sim.iter),disp=False,save=True)              
    
    sim.info_gard.append([np.sum(abs(grad_u)),np.sum(abs(dR))])     
    sim.info_div.append(info_div)
                    
    return  grad

# %% ############################
""" Wrappers for optimize AUTOMATIQUE """

# TODO A implémenter dans coffee_estimator

def genere_L(tbed):
    
    N = tbed.dimScience
    n = tbed.dimScience//tbed.ech
    
    L     = 0j*np.zeros((N*N,n*n)) #Cast to complex
    point = np.zeros((n*n,1))
    
    for a in range(0,n*n):
        point[a] = 1
        L[:,a]   = tbed.todetector(point.reshape(n,n)).reshape(N*N,)
        point[a] = 0
                     
    return L
    
def V_map_J_auto(var,sim,imgs,hypp,simGif,L):
    """ Wrapper for minimize syntax"""

    sim.opti_auto_update(var)
    N = sim.tbed.dimScience
    n = sim.tbed.dimScience//sim.tbed.ech

    Hx = 0j*np.zeros((N,N,sim.nb_div))
    for div_id in range(sim.nb_div):
        point       = sim.get_EF_div(div_id,True).reshape(n*n,)
        Hx[:,:,div_id]  = np.dot(L,point).reshape(N,N)
    
    Jmv = meanSquare(Hx,imgs)
        
    return  Jmv


def V_map_dJ_auto(var,sim,imgs,hypp,simGif,L):
    """ Wrapper for minimize syntax"""

    sim.opti_auto_update(var)
    n = sim.tbed.dimScience//sim.tbed.ech

    dJ_matriciel = 0
    
    for div_id in range(sim.nb_div):
        gamma         = gamma_terme(L,sim,imgs[:,:,div_id])
        dJ_matriciel += (np.dot(np.transpose(L),gamma)).reshape(n,n)
    
    if simGif : print("# TODO")
    
    return tls.add_to_list([],dJ_matriciel)


def gamma_terme(L,sim,img):
    """ Wrapper for minimize syntax"""
    # dh/dphi 

    w  = sim.tbed.dimScience//sim.tbed.ech
    W  = sim.tbed.dimScience

    LEf     = np.dot(L,sim.get_EF_div(0,True).reshape(w*w,))
    gamma = 4*img.reshape(W*W,)*LEf + 4*pow(LEf,3)

    return gamma
