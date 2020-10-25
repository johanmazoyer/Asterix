#Version 29 Janvier 2020
import numpy as np
from astropy.io import fits
import glob
import os

# def LoadImageFits(docs_dir):
#     ''' --------------------------------------------------
#     Load fits file
#     TO BE REPLACED BY astropy.io.fits FUNCTION DOING JUST THAT : getdata()
#     Parameters:
#     ----------
#     docs_dir: Name of the file with extension
    
#     Return:
#     ------
#     image: numpy array
#     -------------------------------------------------- '''
#     openfits=fits.open(docs_dir)
#     image=openfits[0].data
#     return image
    
    
# def SaveFits(image,head,doc_dir2,name,replace=False):
#     ''' --------------------------------------------------
#     Save numpy array into a fits file
#     TO BE REPLACED BY astropy.io.fits FUNCTION DOING JUST THAT : writeto()

#     Parameters:
#     ----------
#     image: Array to be saved
#     head: Header to be saved in this format: ['name',param]
#     doc_dir2: Directory where the file will be saved
#     name: Name of the saved file, without extension .fits
#     replace: Boolean, If True, overwrite on the pre-existing file
    
#     Return:
#     ------
#     image: numpy array
#     -------------------------------------------------- '''
#     hdu=fits.PrimaryHDU(image)
#     hdul = fits.HDUList([hdu])
#     hdr=hdul[0].header
#     head=np.array(head)
    
#     if head.ndim>=2:
        
#         for newline in np.arange(len(head)):
#             try:
#                 hdr.set(head[newline,0][:8],str(head[newline,1]))
#             except ValueError:
#                 print('Value error when Fits Saving:' , head[newline,0])
#                 continue

#     else:
#         hdr.set(head[0][:8],str(head[1]))
            
#     hdu.writeto(doc_dir2+name+'.fits', overwrite=replace )
    
    
def CubeFits(docs_dir):
    ''' --------------------------------------------------
    Load all the fits image in a directory
    
    Parameters:
    ----------
    doc_dir: Input directory
    
    Return:
    ------
    image_array: numpy array
    -------------------------------------------------- '''
    image_list = []
    for filename in sorted(glob.glob(docs_dir+'*.fits')):
        image=fits.getdata(filename)
        image_list.append(image)
        
    image_array = np.array(image_list)
    return image_array
    

def AverageFits(docs_dir):
    ''' --------------------------------------------------
    Load all the fits in a directory to create an averaged numpy array
    
    Parameters:
    ----------
    doc_dir: Input directory
    
    Return:
    ------
    imagemoyenne: numpy array
    -------------------------------------------------- '''
    Cubeimage=CubeFits(docs_dir)
    Sommeimage=0
    for i in np.arange((Cubeimage.shape[0])):
        Sommeimage=Sommeimage+Cubeimage[i]
    imagemoyenne=Sommeimage/Cubeimage.shape[0]
    return imagemoyenne
    
    
    
def GetFluxmetreValue(fitspath):
    ''' --------------------------------------------------
    Extract the information about the fluxmeter in an image header
    
    Parameters:
    ----------
    fitspath: Name of the file with extension
    
    Return:
    ------
    fluxmetre: float
    -------------------------------------------------- '''
    # openfits=fits.open(fitspath)
    # hdu=openfits[0].header
    # fluxmetre=hdu['FLUX_W']
    # return fluxmetre
    return fits.getval(fitspath,'FLUX_W')




def from_param_to_header(config):
    ''' --------------------------------------------------
    Convert ConfigObj parameters to fits header type list
    
    Parameters:
    ----------
    config: config obj
    
    Return:
    ------
    header: list of parameters
    -------------------------------------------------- '''
    header= fits.Header()
    for sect in config.sections:
        #print(config[str(sect)])
        for scalar in config[str(sect)].scalars:
            header[str(scalar)[:8]] = str(config[str(sect)][str(scalar)])
    return header
