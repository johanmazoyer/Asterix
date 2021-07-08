# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:41:26 2021

@author: sjuillar
"""

from Asterix.propagation_functions import mft
import matplotlib.pyplot as plt
from CoffeeLibs.pzernike import pmap, zernike, pzernike
import CoffeeLibs.tools as tls
import numpy as np
import pickle


# %% Simple functions definition 

def todetector(entrance_EF,downstream_EF):
    
    EF_afterentrancepup = entrance_EF*pup
    EF_aftercorno       = mft( corno * mft(EF_afterentrancepup,wphi,wphi,wphi,inverse=True) ,wphi,wphi,wphi)
    EF_out              = downstream_EF * pup_d * EF_aftercorno
    return mft(EF_out,wphi,w,wphi,inverse=True) 

def todetector_Intensity(entrance_EF,downstream_EF):
    EF_out = todetector(entrance_EF,downstream_EF)
    return pow( abs( EF_out ) ,2)
    

def grad_d(point_up,point_down,img):
    
    psi_u  = pup   * np.exp(1j*point_up)
    psi_d  = pup_d * np.exp(1j*point_down)
    
    psi_det =  todetector(psi_u,psi_d)
    h_det   =  todetector_Intensity(psi_u,psi_d)
    
    diff    =  h_det - img
    
    terme1  = mft( np.conj(psi_det) * diff ,w,wphi,wphi,inverse=True)
    terme2  = mft( corno * mft( psi_u ,wphi,wphi,wphi,inverse=True),wphi,wphi,wphi)
    
    Dj = - 4 * np.imag( psi_d * terme2 * terme1 )

    return Dj

def grad_d_diff(point_up,point_down,img,dphi=1e-6):
    
    dphi_list = [] # List of gradient dphi(a,b) for all possible (a,b)
    img_point = todetector_Intensity(np.exp(1j*point_up),np.exp(1j*point_down))

    for a in range(0, wphi):
          for b in range(0, wphi):
              
              # Delta au point courant
              point_down[a,b] = point_down[a,b] + dphi
              
              img_pointdphi = todetector_Intensity(np.exp(1j*point_up),np.exp(1j*point_down))
              
              dphi_list.append(meanSquare(img_pointdphi,img) - meanSquare(img_point,img))
              
              point_down[a,b] = point_down[a,b] - dphi
              
    return np.array(dphi_list).reshape(wphi,wphi) / dphi

def meanSquare(Hx,y):
    """Compte mean square distance between data y and input Hx"""
    d = pow( abs( y - Hx ) , 2 ) # Distance mean square

    return np.sum(np.sum( d ))

# %% Parametres

# Tailles
w       = 64
ech     = 2
wphi    = w//ech
prad    = wphi//2 
rcorno  = 5

# Parametres du banc
pup      = tls.circle(wphi,wphi,prad)
pup_d    = tls.circle(wphi,wphi,prad)
corno    = abs(tls.circle(wphi,wphi,rcorno)-1)


# Generation image
[Ro,Theta]  =  pmap(wphi,wphi)
coeff = 1/np.arange(1,6) # Coeff to generate phi_foc
coeff[0:3] = [0,0,0]
phi_up_reel =  pzernike(Ro,Theta,coeff)
phi_do_reel =  pzernike(Ro,Theta,[0,0,1])

EF_up_reel    = np.exp(1j*phi_up_reel)
EF_down_reel  = np.exp(1j*phi_do_reel)

img = todetector_Intensity(EF_up_reel,EF_down_reel)

# %% Comparaison des gradients

point_up = np.zeros((wphi,wphi))
point_dw = np.zeros((wphi,wphi))

grad_analitique = grad_d(point_up,point_dw,img)
grad_diff       = grad_d_diff(point_up,point_dw,img)

plt.figure(1),plt.suptitle("TEST GRADIENTS SIMPLE")
plt.subplot(1,3,1),plt.imshow(grad_analitique,cmap='jet'),plt.title("grad analytique"),plt.colorbar()
plt.subplot(1,3,2),plt.imshow(grad_diff,cmap='jet'),plt.title("grad par difference"),plt.colorbar()
plt.subplot(1,3,3),plt.imshow(grad_diff-grad_analitique,cmap='jet'),plt.title("Erreur"),plt.colorbar()


# %% Comparaison des gradients simple avec gradient arcitecture

with open('save/grad_d', 'rb') as handle:
    [grad_analytic_down,grad_diff_down] = pickle.load(handle)
    

plt.figure(2),plt.suptitle("COMAPRAISON GRADIENTS SIMPLE / GRADIENT PYCOFFEE")
plt.subplot(2,3,1),plt.imshow(grad_analitique,cmap='jet'),plt.title("grad analytique simple"),plt.colorbar()
plt.subplot(2,3,2),plt.imshow(grad_analytic_down,cmap='jet'),plt.title("grad analytique pycoffe"),plt.colorbar()
plt.subplot(2,3,3),plt.imshow(grad_analytic_down-grad_analitique,cmap='jet'),plt.title("Erreur"),plt.colorbar()

plt.subplot(2,3,4),plt.imshow(grad_diff,cmap='jet'),plt.title("grad difference simple"),plt.colorbar()
plt.subplot(2,3,5),plt.imshow(grad_diff_down,cmap='jet'),plt.title("grad par difference pycoffe"),plt.colorbar()
plt.subplot(2,3,6),plt.imshow(grad_diff-grad_diff_down,cmap='jet'),plt.title("Erreur"),plt.colorbar()
