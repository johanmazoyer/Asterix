# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:28:44 2021

@author: sjuillar
Tools for do repetitive tasks

"""
# Files and folders managment
import glob 
import os

from copy import deepcopy 
import numpy as np

# Fits handlers
from astropy.utils.data import get_pkg_data_filename  
from astropy.io import fits

import matplotlib.pyplot as plt


# %% Basic operation that I put inside function so it is cleaner

def creat_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("Directory " , folder ,  " Created ")
    else:    
        print("Directory " , folder ,  " already exists")


def creat_header(keys):
    head = fits.Header()
    for key in keys : 
        head[key] = keys[key]
    return head

def write_fits(new_file,data,header_dict):
    
    if os.path.exists(new_file)  : 
        os.remove(new_file)
        print("Overwrite file")
        
    hdu   = fits.PrimaryHDU(data=data,header=creat_header(header_dict))
    hdul  = fits.HDUList([hdu])
    hdul.writeto(new_file)
    hdul.close()
    
# %%

"""
-----------  CROP CENTER -----------------


AIM : crop image given a center point
    
    You give -> folder 
             -> the new size you want for your fits
    
    For each images it will ask you the center
    (You click on it)
    
    It create for you -> New folder with croped fits
    (fits have same names)

!!!! I made two versions depending of how you organized your fits
 beacause i am too lazy to do a if  !!!


TIPS :  Since you can't zoom responsivly, you can put the window
where you want to zoom (varaible zoom1 and zoom2)  (OPTIONAL)

v1 -> of fits in given folder
v2 -> of fits in folders  in given folder

"""

def centrage_v2(path_root,base_folder,size,zoom1=150,zoom2=210):
    """Centrage of fits in folder in given folder"""
    creat_folder(base_folder)
    
    w = size//2
    plt.figure("Image",(10,10))
    
    for folder in glob.glob(path_root + "\*") :
        
        new_folder = folder.replace(path_root,base_folder)
        creat_folder(new_folder)
        k = 0
        for file in glob.glob(folder + "\*.fits") : 
           
            # Get data
            data   = fits.getdata(file, ext=0)
            flux   = fits.getheader(file, ext=0)['Flux_W']

            
            # Show and ask you for center and crop
            plt.imshow(data[zoom1:zoom2,zoom1:zoom2],cmap='jet'),plt.title("image" + str(k) )
            plt.show()
            k +=1 
            
            # Wait for a click
            k = plt.ginput(1)
            print(k)
            plt.cla()
            plt.clf()
            plt.close()
            
            x = int(zoom1 + k[0][0])
            y = int(zoom1 + k[0][1])
            print('Image cropped at coordinates: x = ' + str(x) + ", y = " + str(y))
            data = data[x-w:x+w,y-w:y+w]
    

            
            new_file = file.replace(path_root,base_folder)
            write_fits(new_file,data,{'Flux_W':flux})
            
            
    print("Copping done")
    plt.close()

    return 1
            

def centrage_v1(path_root,base_folder,size,zoom1=150,zoom2=210):
    """Centrage all fits in given folder"""
   
    creat_folder(base_folder)
    
    w = size//2
    plt.figure("Image",(10,10))
    img = 0
    
    for file in glob.glob(path_root + "*.fits") : 
       
        # Get data
        data   = fits.getdata(file, ext=0)
        flux   = fits.getheader(file, ext=0)['Flux_W']
        
        # Show and ask you for center and crop
        plt.imshow(data[zoom1:zoom2,zoom1:zoom2],cmap='jet'),plt.title("image" + str(img) )
        img +=1
        
        # Wait for a click
        k = plt.ginput()
        print(k)
        plt.cla()
        plt.clf()
        plt.close()
        
        x = int(zoom1 + k[0][1])
        y = int(zoom1 + k[0][0])
        print('Image cropped at coordinates: x = ' + str(x) + ", y = " + str(y))
        ndata = deepcopy(data[x-w:x+w,y-w:y+w])

        new_file = file.replace(path_root,base_folder)
        write_fits(new_file,ndata,{'Flux_W':flux})
            
            
    print("Copping done")
    plt.close()
    return 1

# %%

"""

-----------  MOY_FOLDER -----------------


AIM : crop image given a center point
    
    You give -> folder and new folder name

    It create for you -> file(s) named after it's folder
    wich is the average of all fits that were inside
    
    
!!!! I made two versions depending of how you organized your fits
 beacause i am too lazy to do a if  !!!
 
v1 -> of fits in folders in given folder
v2 -> of fits in given folder

"""

def moy_fits_v2(path_root,base_folder):
    """Moyenne all fits in folders in given folder"""
    creat_folder(base_folder)
    
    for folder in glob.glob(path_root + "\*") :

        k=0
        data = 0
        flux = 0
        
        for file in glob.glob(folder + "\*.fits") :
            flux   += fits.getheader(file, ext=0)['Flux_W']
            data   += fits.getdata(file, ext=0)
            k      += 1
        
        if k == 0: 
            print("A folder with now fits")
        else : 
            data *= 1/k
            flux *= 1/k
            new_file = folder.replace(path_root,base_folder) + ".fits"
            write_fits(new_file,data,{'Flux_W':flux})
            
            
    print("moyennage done")
    return 1
            

def moy_fits_v1(path_root,base_folder):
    """Moyenne all fits in given folder"""
    data = 0
    flux = 0
    k = 0
    creat_folder(base_folder)
    
    for file in glob.glob(path_root + "*.fits") :
        flux   += fits.getheader(file, ext=0)['Flux_W']
        data   += fits.getdata(file, ext=0)
        k += 1
        
    data *= 1/k
    flux *= 1/k
    
    new_file = file.replace(path_root,base_folder)
    
    write_fits(new_file,data,{'Flux_W':flux})
            
    print("moyennage done")
    return 1

# %%

"""

-----------  MINUS DARK -----------------

you will find out 

"""

def minus_dark(path_root,path_dark,base_folder):
   
    for file in glob.glob(path_dark + "\*.fits") :
        dark      = fits.getdata(file, ext=0)
        flux_dark = fits.getheader(file, ext=0)['Flux_W']
        
    creat_folder(base_folder)
    
    for file in glob.glob(path_root + "\*.fits") : 
        data   = fits.getdata(file, ext=0)
        flux   = fits.getheader(file, ext=0)['Flux_W']
        
        
        new_file = file.replace(path_root,base_folder)
        write_fits(new_file,data-dark,{'Flux_W':flux-flux_dark})
            
            
    print("minus dark done")
    return 1
            

# %%

"""

-----------  NORMALIZATION -----------------


AIM : moyenne by flux
    
    You give -> folder and new folder name

    It create for you -> file(s) named after it's folder
    wich is the average of all flux
    
    
!!!! I made two versions depending of how you organized your fits
 beacause i am too lazy to do a if  !!!
 
v1 -> of fits in folders in given folder
v2 -> of fits in given folder

"""

def normliz_v2(path_root,base_folder,weight=1e10):
    """Moyenne all fits in folders in given folder"""
    creat_folder(base_folder)
    
    for folder in glob.glob(path_root + "\*") :
   
        for file in glob.glob(folder + "\*.fits") :
            flux   = fits.getheader(file, ext=0)['Flux_W']
            data   = fits.getdata(file, ext=0) / (flux * weight)
        
            new_file = folder.replace(path_root,base_folder) + ".fits"
            write_fits(new_file,data,{'Flux_W':1})
            
            
    print("normalize done")
    return 1
            

def normliz_v1(path_root,base_folder,weight=1e10):
    """Moyenne all fits in given folder"""

    creat_folder(base_folder)
    
    for file in glob.glob(path_root + "*.fits") :
        flux   = fits.getheader(file, ext=0)['Flux_W']
        data   = fits.getdata(file, ext=0) / (flux * weight)
        
        new_file = file.replace(path_root,base_folder)
        write_fits(new_file,data,{'Flux_W':1})
            
    print("normalize done")
    return 1
