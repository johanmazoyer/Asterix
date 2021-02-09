__author__ = 'Axel Potier'
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from random import random

import png


def quickshow(tab):
    """
    Function to quickly show an array.
    tab: array to be shown
    """


    tmp = tab
    # tmp = tmp.T
    plt.axis('off')
    plt.imshow(tmp, origin='lower', cmap='gray')
    plt.show()


def quickfits(tab, dir='', name='tmp'):
    """
    Function to quickly save in fits. 
    By default, it will save on the desktop with a random name to avoid overwriting
    
    tab: array to be saved
    dir (optionnal): directory where to save the .fits
    name (optionnal): name of the .fits. By defaut tmpsXX.fits where xx is a random number
    """

    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/') 
    if dir == '': 
        dir = desktop

    if name == 'tmp':
        name = name + str(int(random() * 100))
    fits.writeto(dir + name + '.fits', tab)


def quickpng(tab, dir='', name='tmp'):
    """
    Function to quickly save in .png. 
    By default, it will save on the desktop with a random name

    tab: array to be saved
    dir (optionnal): directory where to save the .png
    name (optionnal): name of the .png.  By defaut tmpXX.png where xx is a random number
    """
    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
    if dir == '': 
        dir = desktop
    plt.figure(figsize=(10, 10))
    tmp = tab
    # tmp = tmp.T
    plt.axis('off')
    plt.imshow(tmp, origin='lower', cmap='gray')
    if name == 'toto':
        name = name + str(int(random() * 100))
    plt.tight_layout()
    plt.savefig(dir + name + '.png', dpi=300)

# def CubeFits(docs_dir):
#     """ --------------------------------------------------
#     Load all the fits images from a directory into a cube
    
#     Parameters:
#     ----------
#     doc_dir: Input directory
    
#     Return:
#     ------
#     image_array: numpy array
#     -------------------------------------------------- """
#     image_list = []
#     for filename in sorted(glob.glob(docs_dir + "*.fits")):
#         image = fits.getdata(filename)
#         image_list.append(image)

#     image_array = np.array(image_list)
#     return image_array


# def AverageFits(docs_dir):
#     """ --------------------------------------------------
#     Load all the fits from a directory and create an averaged numpy array
    
#     Parameters:
#     ----------
#     doc_dir: Input directory
    
#     Return:
#     ------
#     imagemoyenne: numpy array
#     -------------------------------------------------- """
#     Cubeimage = CubeFits(docs_dir)
#     Sommeimage = 0
#     for i in np.arange((Cubeimage.shape[0])):
#         Sommeimage = Sommeimage + Cubeimage[i]
#     imagemoyenne = Sommeimage / Cubeimage.shape[0]
#     return imagemoyenne


# def GetFluxmetreValue(fitspath):
#     """ --------------------------------------------------
#     Extract measured flux recorded in fits header by the THD2 experiment
    
#     Parameters:
#     ----------
#     fitspath: Name of the file with extension
    
#     Return:
#     ------
#     fluxmetre: float
#     -------------------------------------------------- """
#     # openfits=fits.open(fitspath)
#     # hdu=openfits[0].header
#     # fluxmetre=hdu['FLUX_W']
#     # return fluxmetre
#     return fits.getval(fitspath, "FLUX_W")

def from_param_to_header(config):
    """ --------------------------------------------------
    Convert ConfigObj parameters to fits header type list
    
    Parameters:
    ----------
    config: config obj
    
    Return:
    ------
    header: list of parameters
    -------------------------------------------------- """
    header = fits.Header()
    for sect in config.sections:
        # print(config[str(sect)])
        for scalar in config[str(sect)].scalars:
            header[str(scalar)[:8]] = str(config[str(sect)][str(scalar)])
    return header

