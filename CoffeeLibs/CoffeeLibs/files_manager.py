# -*- coding: utf-8 -*-
"""
Created on Tue Mar  16 11:03:09 2021

--------------------------------------------
----------------  FILE MANAGER  ------------
--------------------------------------------

@author: sjuillar

Littles fonction to easily do stuff with files


"""
import os
import sys
import glob

from configobj import ConfigObj
from validate import Validator

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import Asterix as atrx


import numpy as np # For array

def get_ini(my_file):
    """ Get dict congif from ini files
    INPUT :
        
        my_file  : path to the ini file

    OUTPUT : 
        config : validated config dict 
    
    """
    
    template = atrx.__path__[0]+"\Param_configspec.ini"
    
    config = ConfigObj(my_file, configspec=template)
    config.validate(Validator(), copy=True)
    
    return config

    
def get_fits_as_imgs(path_root,prefix,exts=[""]):
    """ Get Images from fits
    INPUT :
        
        path_root : STRING Folder of your fits
        
        prefix : STRING static prefix of your targeted fits
        
        exts : list of exts of your fits.
            
        N : static size of all your fits
            
    OUTPUT : 
        imgs : table of your img N*N*nb_img
    
    """
    imgs0 = fits.getdata(path_root + os.path.sep + prefix + exts[0]+ ".fits", ext=0)
    N     = imgs0.shape[0]
    
    imgs = np.zeros((N,N,len(exts)))
    for k in range(len(exts)) :
        imgs[:,:,k] = fits.getdata(path_root + os.path.sep + prefix + exts[k]+ ".fits", ext=0)
          
    return imgs



