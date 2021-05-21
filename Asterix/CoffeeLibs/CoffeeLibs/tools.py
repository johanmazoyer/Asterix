# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:20:09 2021

--------------------------------------------
----------------  TOOLS    -----------------
--------------------------------------------

@author: sjuillar
"""

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

# %% FFT

def ffts(A,norm="ortho"):
    """fft shifted twice"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A),norm=norm))


def iffts(A):
    """ifft shifted twice"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(A),norm= "ortho"))


def np_fftconv(A,B):
    """Fast convolution"""
    return np.real(iffts(ffts(A)*ffts(B)))

# %% Matrix

def grad_matrix(w,l,factor):
    M = np.zeros([w,l])
    x0 = w//2
    y0 = l//2
    for x in range(0, w):
           for y in range(0, l):
               M[x,y] = factor * ( pow(x0-x,2) + pow(y0-y,2) )
    return M

def circle(w,l,r):
    """ Create a zeros matrix [w*l] with a circle of ones of raduis r at the centre"""
    M = np.zeros([w,l])
    for x in range(0, w):
           for y in range(0, l):
               if  pow(x-(w)//2,2) + pow(y-(l)//2,2) < pow(r,2):
                   M[x,y] = 1
    return M

def daminer(w,l):
    """ Create a zeros matrix [w*l] with a circle of ones of raduis r at the centre"""
    M = np.ones([w,l])
    M[w//2:,l//2:] = -1
    M[:w//2,:l//2] = -1
    return M

# %% Operation on matrix

def normalize2(A):
    return A/(abs(A).max()-abs(A).min())

def padding(A,ech):
    Apad = np.zeros(np.array(A.shape) * ech)*0j
    Apad[ A.shape[0]*(ech-1)//2 : A.shape[0]*(ech+1)//2, A.shape[1]*(ech-1)//2 : A.shape[1]*(ech+1)//2 ] = A
    return Apad

def depadding(A,ech):
    Acut = A[ A.shape[0]*(ech-1)//(2*ech) : A.shape[0]*(ech+1)//(2*ech), A.shape[1]*(ech-1)//(2*ech) : A.shape[1]*(ech+1)//(2*ech) ]
    return Acut

# %% Spacial derivate of matrix

def gradient_xy(A,mode="np"):
    """ 2D spacial graident """
    
    if mode=="sobel" :
        sx = ndimage.sobel(A,axis=0,mode='constant')
        sy = ndimage.sobel(A,axis=1,mode='constant')
        grad=np.hypot(sx,sy)
        
    elif mode=="np" :
        [sx, sy]  = np.gradient(A)
        grad=np.hypot(sx,sy)
    else :
        grad = ndimage.gaussian_gradient_magnitude(A,sigma=0.2)
        
    return grad

def gradient2_xy(A):
    """ 2D spacial graident """
    return ndimage.laplace(A)


# %% Template plots 

def tempalte_plot(sim,e_sim,es,name="res",disp=True,save=False):
    
    if not disp : 
        plt.ioff()
    
    ## GET DATAS 
    
    tbed = sim.tbed
    
    cropEF = sim.get_phi_foc()*tbed.pup
    cropEFd = sim.get_phi_do()*tbed.pup_d
    
    error    = abs( cropEF  - e_sim.get_phi_foc() )
    error_do = abs( cropEFd - e_sim.get_phi_do()*tbed.pup_d )

    ## PLOTS FIGURES
    
    plt.figure("Minimization Results",figsize = (8, 6))
    plt.suptitle(name)
    plt.gcf().subplots_adjust(wspace = 0.4, hspace = 0.5)

    plt.subplot(3,3,1),plt.imshow(cropEF,cmap='jet'),plt.title("Phi_up Attendu"),plt.colorbar()
    plt.subplot(3,3,2),plt.imshow(e_sim.get_phi_foc(),cmap='jet'),plt.title("Estimation"),plt.colorbar()
    
    plt.subplot(3,3,3),plt.imshow(error,cmap='jet'),plt.title("Erreur en %"),plt.colorbar()
    
    plt.subplot(3,3,4),plt.imshow(cropEFd,cmap='jet'),plt.title("Phi_do Attendu"),plt.colorbar()
    plt.subplot(3,3,5),plt.imshow(e_sim.get_phi_do(),cmap='jet'),plt.title("Estimation"),plt.colorbar()
    plt.subplot(3,3,6),plt.imshow(error_do,cmap='jet'),plt.title("Erreur"),plt.colorbar()
    
    
    ## PLOT TEXT BOX
    
    plt.subplot(3,1,3)
    mins = int(es.toc//60)
    sec  = 100*(es.toc-60*mins)
    pup_size = np.sum(tbed.pup)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.axis('off')
    
    textbox = ""
    if not e_sim.phi_foc_is_known() : textbox += "Erreur moyenne up : "   + "{:.3f}".format(np.sum(error)/pup_size) 
    if not e_sim.phi_do_is_known()  : textbox += "\nErreur moyenne do : " + "{:.3e}".format(np.sum(error_do)/pup_size)
    if not e_sim.flux_is_known()    : textbox += "\nEstimate flux = "     + "{:.2f}".format(e_sim.get_flux()) + " --> error = +/-" + "{:.3f}".format(100*(abs(sim.get_flux()-e_sim.get_flux()))/sim.get_flux()) + "%"
    if not e_sim.fond_is_known()    : textbox += "\nEstimate fond = "     + "{:.2f}".format(e_sim.get_fond()) + " --> error = "    + "{:.2e}".format(abs(sim.get_fond()-e_sim.get_fond()))

    textbox += "\n\n"+ str(es.complete_res['message']) + "\nIterations : "+ str(es.complete_res['nit'])+ "\nTime : " + str(mins) + "m" + str(sec)[:3]
    
    pond2 = 100 * sum(e_sim.info2) / sum(sum(e_sim.info2))
    pond  = 100 * sum(e_sim.info) / sum(sum(e_sim.info))
    pond   = np.round(pond,2)
    pond2  = np.round(pond2,2)
    textbox += "\n\nPonderation   :   " + "Jmv: " + str(pond2[0]) + "%" + ", R: " + str(pond2[1]) + "%\n" + "Par gradient  :   DJmv: " + str(sum(pond[:-1])) + "%" + ", DR: " + str(pond[-1]) + "%\nPar diversité :   " + str( 100 * pond[:-1] / np.sum(pond[:-1]) )
    
    plt.text(0.2,-0.3,textbox,bbox=props)      
    
    ## DISP / SAVE 
    
    if disp : plt.show()
    
    if save :
        plt.savefig("save/"+name,pad_inches=0.5)
        if not disp : 
            plt.cla()
            plt.clf()
            plt.close()

# %% Compare fft/mft 

# import matplotlib.pyplot as plt
# import tools as tls
# from param import P,wphi,lphi,ech,w,l
# from Asterix.propagation_functions import mft

# # Carte de phase
# phi = np.zeros((wphi,lphi))

# # Calcul des images 
# i_fft  = pow( abs( tls.ffts(tls.padding(P*np.exp(1j*phi),ech)) ) ,2)
# i_mft =  pow( abs( w * mft(P*np.exp(1j*phi),wphi,w,wphi) ) ,2)

# plt.figure(1)
# plt.subplot(1,3,1),plt.imshow(i_fft,cmap='jet'),plt.title("H avec FFT"),plt.colorbar()
# plt.subplot(1,3,2),plt.imshow(i_mft,cmap='jet'),plt.title("H avec MFT"),plt.colorbar()
# plt.subplot(1,3,3),plt.imshow(i_mft-i_fft,cmap='jet'),plt.title("Difference"),plt.colorbar()

