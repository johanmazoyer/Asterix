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
import numpy.matlib as mnp

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
    
    N = tbed.dimScience//tbed.Science_sampling
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
    wphi  = tb.dimScience//tb.Science_sampling
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
    wphi = tb.dimScience//tb.Science_sampling
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
    
    varb = 1
    
    Jmv = meanSquare(Hx,imgs) / (sim.nb_div * varb)
    
    R   = (1/2) * hypp * regule(sim.get_phi_foc())
    
    sim.info.append([Jmv, R])
    
    return  Jmv + R


def V_grad_J(var,sim,imgs,hypp,simGif):
    """ Wrapper that resize variables to fit minimize syntax"""
    
    # sim = copy.deepcopy(esim)
    sim.opti_update(var,imgs)
    
    info_div       = []
    sim.iter +=1
    
    pup  = tls.circle(sim.N, sim.N, sim.tbed.prad//2 - 1)
    
    # Compute gradient = dj/dphi
    grad = []
    if not sim.phi_foc_is_known() : 
        
        grad_u = 0
        
        
        for div_id in range(0,sim.nb_div):
            # Calcul de Jmv pour chaque diversité
            grad_u_k = DJmv_up(div_id,imgs[:,:,div_id],sim)
            grad_u   += grad_u_k
            info_div.append(np.sum(grad_u_k**2))
            
        # Ponderation
        varb    = np.median(imgs) + 1
        grad_u *= 1 / (sim.nb_div * varb)  
        
        # Regulatrisation
        dR = pup * hypp * Dregule(sim.get_phi_foc()) 
        grad    = tls.add_to_list(grad, grad_u + dR)

    # Other varaibles gradient
    if not sim.phi_do_is_known() : 
        
        grad_d = 0
        
        # Jmv pour chaque diverstié
        for div_id in range(0,sim.nb_div):
            grad_d += DJmv_down(div_id,imgs[:,:,div_id],sim)
        
        # Ponderation
        grad_d *= 1  
        
        # Regulatrisation
        dRdo    = pup * hypp * Dregule(sim.get_phi_do()) # Regulatrisation
        grad    = tls.add_to_list(grad, grad_d + dRdo)
                  
        
    # Save and Infos
    
    if simGif and not sim.phi_do_is_known() : tls.plot_sim_entries(sim,dR,grad_u,grad_d,dRdo,name="iter"+str(sim.iter),disp=False,save=True)
    elif simGif : tls.plot_sim_entries(sim,dR,grad_u,name="iter"+str(sim.iter),disp=False,save=True)              
    
    sim.info_gard.append([np.sum(abs(grad_u)),np.sum(abs(dR))])     
    sim.info_div.append(info_div)
                    
    return  grad

# %% ############################
""" Wrappers for optimize AUTOMATIQUE """

# TODO A implémenter dans coffee_estimator

def genere_L(tbed):
    
    N = tbed.dimScience
    n = tbed.dim_overpad_pupil
    
    L     = 0j*np.zeros((N*N,n*n)) #Cast to complex
    point = np.zeros((n*n,1))
    
    for a in range(0,n*n):
        point[a] = 1
        L[:,a]   = tbed.todetector(point.reshape(n,n)).reshape(N*N,)
        point[a] = 0
                     
    return L
    
def V_map_J_auto(var,sim,imgs,hypp,simGif,L):
    """ Wrapper for minimize syntax"""

    sim.opti_update(var,imgs)
    N = sim.tbed.dimScience
    n = sim.tbed.dim_overpad_pupil

    Hx = 0j*np.zeros((N,N,sim.nb_div))
    for div_id in range(0,sim.nb_div):
        point           = sim.get_EF_div(div_id).reshape(n*n,)
        Hx[:,:,div_id]  = pow(abs(np.dot(L,point).reshape(N,N)),2)
    
    R   = (1/2) * hypp * regule(sim.get_phi_foc())
    Jmv = meanSquare(Hx,imgs)/ (sim.nb_div)
    
    sim.info.append([Jmv, R])
    
    return  Jmv + R


def V_map_dJ_auto(var,sim,imgs,hypp,simGif,L):
    """ Wrapper for minimize syntax"""

    sim.opti_update(var,imgs)
    n = sim.tbed.dim_overpad_pupil
    N = sim.tbed.dimScience

    info_div       = []
    sim.iter +=1
    dJ_matriciel = 0
    
    for div_id in range(0,sim.nb_div):
        gamma         = gamma_terme(L,sim,imgs[:,:,div_id],div_id)
        grad_u_k      = (np.dot(np.conj(np.transpose(L)),gamma)).reshape(n,n)*np.conj(sim.get_EF_div(div_id))
        info_div.append(np.sum( abs( np.imag(grad_u_k / (sim.N**2) ) ) ))
        dJ_matriciel += 4*np.imag(grad_u_k)

    # Ponderation
    dJ_matriciel *= 1 / sim.nb_div
    
    # Regulatrisation
    pup  = tls.circle(sim.N, sim.N, sim.tbed.prad//2 - 1)
    dR      = pup * hypp * Dregule(sim.get_phi_foc()) 
    
    if simGif : tls.plot_sim_entries(sim,dR,dJ_matriciel,name="iter"+str(sim.iter),disp=False,save=True)
    
    sim.info_gard.append([np.sum(abs(dJ_matriciel)),np.sum(dR)])  
    sim.info_div.append(info_div)

    return tls.add_to_list([],dJ_matriciel+dR)


def gamma_terme(L,sim,img,div_id):
    """ Wrapper for minimize syntax"""
    # dh/dphi 

    w    = sim.tbed.dim_overpad_pupil
    W    = sim.tbed.dimScience
    
    EF  = sim.get_EF_div(div_id).reshape(w*w,)
    
    psi_det =  sim.todetector(div_id)
    h_det   =  sim.todetector_Intensity(div_id)
    diff    =  h_det - img
    
    gamma = diff * psi_det

    return gamma.reshape((W*W,))