# -*- coding: utf-8 -*-

import numpy as np
from math import factorial

## Recurrencial zernike polynome
def zernike(Ro,Theta,j):
    """
    Compute Zernike polynomial matrix

    Parameters
    ----------
    Ro,THETA : 2D matrixs
        2D matrix of polar cooridnate (ro,theta)
    j : int
        Zernike polynome OSA's index


    Returns
    -------
    int
        DESCRIPTION.

    """
    
    if j==1 : return np.zeros(Ro.shape)
    [n,m] = osa2mn(j)
    
    # Radial Zernike polynomial
    R = 0
    for k in np.linspace(0,((n-m)//2),((n-m)//2)+1):
        R += Ro**(n-2*k) * (-1)**k * factorial(n-k) / ( factorial(k) * factorial( (n+m)//2 - k )*factorial( (n-m)//2 -k) )
    
    if m == 0 :
        poly = np.sqrt(n+1) * R 
    elif j%2:
        poly = np.sqrt(2*(n+1)) * R * np.sin(m*Theta)
    else:
        poly = np.sqrt(2*(n+1)) * R * np.cos(m*Theta)
     
    # Normalize 1
    return poly

    
def osa2mn(j):
    """
    Converte OSA unique index to azimutal/radial degre 
    Indexes that descibe Zernike Polynomes
    Inspired from COFFEE, " valid until J=10000 " \cite{coffee}
    Py version tested until 10
    
    Parameters
    ----------
    j : int
        OSA's indice.
    
    Returns
    -------
    [n,m] : [int,int]
        azimutal/radial degre
    
    """
    
    n = int(np.sqrt(8*j -7) - 1) //2
    
    if (n%2): # Evens radial orders
        m = 1 + 2*( ( j-1 - n*(n+1)//2 ) //2 )
    else :    # Odds radial orders
        m = 2 * ( ( j - n*(n+1)//2  ) //2 )

    return int(n),int(m)


def pmap(w,l,rpup):
    """
    Generate polar coordinate map of size (w,l)

    Parameters
    ----------
    w,l : int,int
        sizes of polar map

    Returns
    -------
    Ro,THETA
        the two 2D matrix of (ro,theta) coordinates

    """
    leng = w/2
    coord = np.concatenate((np.arange(-leng+0.5,0) , np.arange(0.5,leng+0.5)), axis=0)
    X,Y   = np.meshgrid(coord,coord)
    return [np.sqrt(X**2+Y**2)/rpup, np.arctan2(Y,X)] 
    

def pzernike(Ro,Theta,poly):
    res = np.zeros(np.shape(Ro))
    ordre=1
    for a in poly:
        res += a*zernike(Ro,Theta,ordre)
        ordre +=1
    
    return res

# %%  Exemple

# =============================================================================
# import matplotlib.pyplot as plt
# 
# #Size
# w,l = 50,50
# 
# [Ro,Theta] = pmap(w,l)
# 
# Z = zernike(Ro,Theta,4)
# 
# plt.figure(1)
# plt.imshow(Z)
# =============================================================================

# print(osa2mn(6))