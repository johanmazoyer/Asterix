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

from PIL import Image
import glob
import os

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

def circle(w,l,r,offset=0.5):
    """ Create a zeros matrix [w*l] with a circle of ones of raduis r at the centre"""
    M = np.zeros([w,l])
    for x in range(0, w):
           for y in range(0, l):
               if  pow(x-(w/2) + offset ,2) + pow(y-(l/2) + offset,2) < pow(r,2):
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
        sx = ndimage.sobel(A,axis=0,mode='constant',cval=0.0)
        sy = ndimage.sobel(A,axis=1,mode='constant',cval=0.0)
        grad=np.hypot(sx,sy)
        
    elif mode=="np" :
        [sx, sy]  = np.gradient(A,1)
        
        if np.iscomplexobj(sx) : grad=np.hypot(abs(sx),abs(sy))
        else                   : grad=np.hypot(sx,sy)
   
    else :
        grad = ndimage.gaussian_gradient_magnitude(A,sigma=0.2)
        
    return grad

def gradient2_xy(A):
    """ 2D spacial graident """
    if np.iscomplexobj(A) : 
        R = ndimage.laplace(np.imag(A),mode='nearest')
        I = ndimage.laplace(np.real(A),mode='nearest')
        return np.sqrt(R**2 + 1j*I**2)
    else : 
        return ndimage.laplace(A,mode='nearest')


# %% Operations on nymppy array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_pup_size(img):
    
    N = img.shape[0]
    
    #Estime pup size in pix
    psf  = abs(ffts(img))
    
    psf_slice  = psf[N//2,:]
    
    plt.figure("FTO")
    plt.plot(range(1,N+1),psf_slice,'g')
    plt.show()
    
    # Click on coupure yourself.
    coupure = plt.ginput(2)
    coupure = (np.array(coupure)[:,0]).astype(int)
    
    plt.plot(coupure,psf_slice[coupure],'r*')
    
    print("pup_size = " + str((coupure[1] - coupure[0])//2))
    return  (coupure[1] - coupure[0])//2

def get_science_sampling(img):
    
    N = img.shape[0]
    psf_slice  = abs(ffts(img)[N//2,:])**2

    max_psf = max(psf_slice)
    max_id  = int(np.where(psf_slice == max_psf)[0])
    
    half_values = [find_nearest(psf_slice[:max_id],max_psf/2),find_nearest(psf_slice[max_id:],max_psf/2)]
    half_ids    = [np.where(psf_slice ==  half_values[0])[0] + 1, np.where(psf_slice ==  half_values[1])[0] + 1 ]
    
    plt.figure("Transfert")
    plt.plot(range(1,N+1),psf_slice,'g')    
    plt.yscale('log')
    plt.plot(half_ids,half_values,'r-')
    plt.show()
    k = plt.ginput(1)
    
    print("Science sampling = " + str(int(half_ids[1] - half_ids[0])))
    return int(half_ids[1] - half_ids[0])

def detector_shifts2div_factors(div_dist,Ld,F_D):
    return np.pi*np.array(div_dist)*1e-3/(8*np.sqrt(3)*Ld*1e-9*F_D**2)

def complex2real(M):
    """Matrix of complex to two matrix of real A + iB """
    return np.concatenate((np.real(M),np.imag(M)),axis=None)
    
def real2complex(A,B):
    """ Two matrix of real A + iB as dnarray(n,n,2) to complex matric C """
    return A + 1j*B

def add_to_list(A,B):
    """Concactene B in A where A is vector and B a matrix """
    if np.iscomplexobj(B) : 
        return np.concatenate((A,complex2real(B)),axis=None)
    else :
        return np.concatenate((A,B),axis=None)


# %% Template plots 

def tempalte_plot(sim,e_sim,estimator,name="res",disp=True,save=False):
    
    if not disp : 
        plt.ioff()
    
    ## GET DATAS 
    
    tbed = sim.tbed
    
    col = 2
    if np.iscomplexobj(sim.get_phi_foc()) : col = col + 1
    
    
    ## PLOTS FIGURES
    
    plt.figure("Minimization Results",figsize = (8, 6))
    plt.suptitle(name)
    plt.gcf().subplots_adjust(wspace = 0.4, hspace = 0.5)

    if not e_sim.phi_do_is_known():
        col = col+1
        error_do = abs( sim.get_phi_do() - e_sim.get_phi_do()  ) * tbed.pup_d
        plt.subplot(col,3,4),plt.imshow(sim.get_phi_do()   * tbed.pup_d,cmap='jet'),plt.title("Phi_do Attendu"),plt.colorbar()
        plt.subplot(col,3,5),plt.imshow(e_sim.get_phi_do() * tbed.pup_d,cmap='jet'),plt.title("Estimation"),plt.colorbar()
        plt.subplot(col,3,6),plt.imshow(error_do,cmap='jet'),plt.title("Erreur"),plt.colorbar()
        

    if np.iscomplexobj(sim.get_phi_foc()) : 
        error    = abs( np.imag(sim.get_phi_foc())  - np.imag(e_sim.get_phi_foc()) ) * tbed.pup
        plt.subplot(col,3,1),plt.imshow(np.imag(sim.get_phi_foc()) * tbed.pup,cmap='jet'),plt.title("Phi_up Attendu I"),plt.colorbar()
        plt.subplot(col,3,2),plt.imshow(np.imag(e_sim.get_phi_foc()) * tbed.pup,cmap='jet'),plt.title("Estimation I"),plt.colorbar()
        plt.subplot(col,3,3),plt.imshow(error,cmap='jet'),plt.title("Erreur en difference de I"),plt.colorbar()
        
    error    = abs( np.real(sim.get_phi_foc())  - np.real(e_sim.get_phi_foc()) ) * tbed.pup
    plt.subplot(col,3,1),plt.imshow(np.real(sim.get_phi_foc()) * tbed.pup,cmap='jet'),plt.title("Phi_up Attendu"),plt.colorbar()
    plt.subplot(col,3,2),plt.imshow(np.real(e_sim.get_phi_foc()) * tbed.pup,cmap='jet'),plt.title("Estimation"),plt.colorbar()
    plt.subplot(col,3,3),plt.imshow(error,cmap='jet'),plt.title("Erreur en difference"),plt.colorbar()
    

    ## PLOT TEXT BOX
    
    plt.subplot(col,1,col)
    mins = int(estimator.toc//60)
    sec  = 100*(estimator.toc-60*mins)
    pup_size = np.sum(tbed.pup)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.axis('off')
    
    textbox = ""
    if not e_sim.phi_foc_is_known() : textbox += "Erreur moyenne up : "   + "{:.3f}".format(np.sum(error)/pup_size) 
    if not e_sim.phi_do_is_known()  : textbox += "\nErreur moyenne do : " + "{:.3e}".format(np.sum(error_do)/pup_size)
    if not e_sim.flux_is_known()    : textbox += "\nEstimate flux = "     + ",  ".join(['{:.2e}'.format(x) for x in e_sim.get_flux()])
    if not e_sim.fond_is_known()    : textbox += "\nEstimate fond = "     + ",  ".join(['{:.2e}'.format(x) for x in e_sim.get_fond()])

    textbox += "\n\n"+ str(estimator.complete_res['message']) + "\nIterations : "+ str(estimator.complete_res['nit'])+ "\nTime : " + str(mins) + "m" + str(sec)[:3]
    
    pondCritere = np.round( 100 * sum(e_sim.info)      / sum(sum(e_sim.info))     ,2)
    pondGrad    = np.round( 100 * sum(e_sim.info_gard) / sum(sum(e_sim.info_gard)),2)
    pondDiv     = np.round( 100 * sum(e_sim.info_div)  / sum(sum(e_sim.info_div)) ,2)
    textbox += "\n\nPonderation   :   " + "Jmv: " + str(pondCritere[0]) + "%" + ", R: " + str(pondCritere[1]) + "%\n" + "Par gradient  :   DJmv: " + str(pondGrad[0]) + "%" + ", DR: " + str(pondGrad[1]) + "%\nPar diversité :   " + str( pondDiv )
    
    plt.text(0.2,-0.3,textbox,bbox=props)      
    
    ## DISP / SAVE 
    
    if disp : plt.show()
    
    if save :
        plt.savefig("save/"+name,pad_inches=0.5)
        if not disp : 
            plt.cla()
            plt.clf()
            plt.close()


def plot_sim_entries(sim,dRup,dJup,dJdo=None,dRdo=None,name="res",disp=True,save=False):
    
    if not disp : 
        plt.ioff()
    
    ## GET DATAS 

    tbed = sim.tbed
    
    cropEF  = sim.get_phi_foc()*tbed.pup
    cropEFd = sim.get_phi_do()*tbed.pup_d
    
    plt.figure("Simulation gif",figsize = (8, 6))
    plt.subplot(3,2,1),plt.imshow(cropEF,cmap='jet'),plt.title("Phi_up"),plt.colorbar()
    plt.subplot(3,2,2),plt.imshow(cropEFd,cmap='jet'),plt.title("Phi_do"),plt.colorbar()
    
    if dJdo is not None : plt.subplot(3,2,4),plt.imshow(dJdo,cmap='jet'),plt.title("dJdo"),plt.colorbar()
    plt.subplot(3,2,3),plt.imshow(dJup,cmap='jet'),plt.title("dJup"),plt.colorbar()

    if dRdo is not None : plt.subplot(3,2,6),plt.imshow(dRdo,cmap='jet'),plt.title("dRdo"),plt.colorbar()
    plt.subplot(3,2,5),plt.imshow(dRup,cmap='jet'),plt.title("dRup"),plt.colorbar()
        
    plt.suptitle(name)

    ## DISP / SAVE 
    
    if save :
        plt.savefig("save/iter/"+name,pad_inches=0.5)
        if not disp : 
            plt.cla()
            plt.clf()
            plt.close()


def iter_to_gif(name="sim"):
    
    images = []
    for file in sorted(glob.glob("./save/iter/*.png"), key=len):
        images.append(Image.open(file))
    images[0].save(fp="./save/gif/"+str(name)+".gif", format='GIF', append_images=images, save_all=True,duration=300, loop=0)
    
    ii=0
    for file in sorted(glob.glob("./save/iter/*.png"), key=len):
        images[ii].close()
        os.remove(file)
        ii += 1


def tempalte_plot2(imgs,e_sim,estimator,name="res",disp=True,save=False):
    
    if not disp : 
        plt.ioff()
    
 
    ## PLOTS FIGURES
    e_imgs = e_sim.gen_div_imgs()
    
    plt.figure("Comparaison Images estimé / Image réel",figsize = (8, 6))
    plt.gcf().subplots_adjust(wspace = 0.4, hspace = 0.5)

    plt.subplot(3,3,1),plt.imshow(imgs[:,:,0],cmap='jet'),plt.colorbar(),plt.title("Image 1 reel")
    plt.subplot(3,3,4),plt.imshow(imgs[:,:,1],cmap='jet'),plt.colorbar(),plt.title("Image 2 reel")
    
    plt.subplot(3,3,2),plt.imshow(e_imgs[:,:,0],cmap='jet'),plt.colorbar(),plt.title("Image 1 estimé")
    plt.subplot(3,3,5),plt.imshow(e_imgs[:,:,1],cmap='jet'),plt.colorbar(),plt.title("Image 2 estimé")

    plt.subplot(3,3,3),plt.imshow((abs(e_imgs[:,:,0] - imgs[:,:,0])),cmap='jet'),plt.colorbar(),plt.title("Erreur de recontruction Image 1")
    plt.subplot(3,3,6),plt.imshow((abs(e_imgs[:,:,1] - imgs[:,:,1])),cmap='jet'),plt.colorbar(),plt.title("Erreur de recontruction Image 2")

    ## PLOT TEXT BOX
    
    plt.subplot(3,1,3)
    mins = int(estimator.toc//60)
    sec  = 100*(estimator.toc-60*mins)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.axis('off')
    
    textbox = ""
    if not e_sim.flux_is_known()    : textbox += "\nEstimate flux = "     + ",  ".join(['{:.2e}'.format(x) for x in e_sim.get_flux()])
    if not e_sim.fond_is_known()    : textbox += "\nEstimate fond = "     + ",  ".join(['{:.2e}'.format(x) for x in e_sim.get_fond()])

    textbox += "\n\n"+ str(estimator.complete_res['message']) + "\nIterations : "+ str(estimator.complete_res['nit'])+ "\nTime : " + str(mins) + "m" + str(sec)[:3]
    
    pondCritere = np.round( 100 * sum(e_sim.info)      / sum(sum(e_sim.info))     ,2)
    pondGrad    = np.round( 100 * sum(e_sim.info_gard) / sum(sum(e_sim.info_gard)),2)
    pondDiv     = np.round( 100 * sum(e_sim.info_div)  / sum(sum(e_sim.info_div)) ,2)
    textbox += "\n\nPonderation   :   " + "Jmv: " + str(pondCritere[0]) + "%" + ", R: " + str(pondCritere[1]) + "%\n" + "Par gradient  :   DJmv: " + str(pondGrad[0]) + "%" + ", DR: " + str(pondGrad[1]) + "%\nPar diversité :   " + str( pondDiv )
    
    plt.text(0.2,-0.3,textbox,bbox=props)      
    
    ## DISP / SAVE 
    
    if disp : plt.show()
    
    if save :
        plt.savefig("save/image_"+name,pad_inches=0.5)
        if not disp : 
            plt.cla()
            plt.clf()
            plt.close()
    
    plt.figure("Profile de la phase estimé")
    phase  = e_sim.get_phi_foc()
    if np.iscomplexobj(phase) : 
        plt.subplot(1,2,1),plt.imshow(np.real(phase),cmap='jet'),plt.title("Partie Reel de la phase"),plt.colorbar()
        plt.subplot(1,2,2),plt.imshow(np.imag(phase),cmap='jet'),plt.title("Partie Imaginaire de la phase"),plt.colorbar()
    else : 
        plt.imshow(phase,cmap='jet'),plt.colorbar()

    
    ## DISP / SAVE 
    
    if disp : plt.show()
    
    if save :
        plt.savefig("save/phase_"+name,pad_inches=0.5)
        if not disp : 
            plt.cla()
            plt.clf()
            plt.close()


def tempalte_plotauto(sim,e_sim,estimator,name="res",disp=True,save=False):
    
    if not disp : 
        plt.ioff()
     
    ## GET DATAS 
    tbed = sim.tbed
    
    col = 2
    if np.iscomplexobj(sim.get_phi_foc()) : col = col + 1
    
    
    ## PLOTS FIGURES
    
    plt.figure("Minimization EF Results",figsize = (8, 6))
    plt.suptitle(name)
    plt.gcf().subplots_adjust(wspace = 0.4, hspace = 0.5)

    ephase  = e_sim.get_EF_div(0,True)
    phase   =   sim.get_EF()

    error    = abs( np.imag(phase)  - np.imag(ephase) ) * tbed.pup
    plt.subplot(col,3,1),plt.imshow(np.imag(phase) * tbed.pup,cmap='jet'),plt.title("EF I Attendu"),plt.colorbar()
    plt.subplot(col,3,2),plt.imshow(np.imag(ephase) * tbed.pup,cmap='jet'),plt.title("Estimation I"),plt.colorbar()
    plt.subplot(col,3,3),plt.imshow(error,cmap='jet'),plt.title("Erreur en difference de I"),plt.colorbar()
    
    error    = abs( np.real(phase)  - np.real(ephase) ) * tbed.pup
    plt.subplot(col,3,4),plt.imshow(np.real(phase) * tbed.pup,cmap='jet'),plt.title("EF R Attendu"),plt.colorbar()
    plt.subplot(col,3,5),plt.imshow(np.real(ephase) * tbed.pup,cmap='jet'),plt.title("Estimation"),plt.colorbar()
    plt.subplot(col,3,6),plt.imshow(error,cmap='jet'),plt.title("Erreur en difference"),plt.colorbar()  
    
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

