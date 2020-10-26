import glob
import numpy as np
from astropy.io import fits


def CubeFits(docs_dir):
    """ --------------------------------------------------
    Load all the fits image in a directory
    
    Parameters:
    ----------
    doc_dir: Input directory
    
    Return:
    ------
    image_array: numpy array
    -------------------------------------------------- """
    image_list = []
    for filename in sorted(glob.glob(docs_dir + "*.fits")):
        image = fits.getdata(filename)
        image_list.append(image)

    image_array = np.array(image_list)
    return image_array


def AverageFits(docs_dir):
    """ --------------------------------------------------
    Load all the fits in a directory to create an averaged numpy array
    
    Parameters:
    ----------
    doc_dir: Input directory
    
    Return:
    ------
    imagemoyenne: numpy array
    -------------------------------------------------- """
    Cubeimage = CubeFits(docs_dir)
    Sommeimage = 0
    for i in np.arange((Cubeimage.shape[0])):
        Sommeimage = Sommeimage + Cubeimage[i]
    imagemoyenne = Sommeimage / Cubeimage.shape[0]
    return imagemoyenne


def GetFluxmetreValue(fitspath):
    """ --------------------------------------------------
    Extract the information about the fluxmeter in an image header
    
    Parameters:
    ----------
    fitspath: Name of the file with extension
    
    Return:
    ------
    fluxmetre: float
    -------------------------------------------------- """
    # openfits=fits.open(fitspath)
    # hdu=openfits[0].header
    # fluxmetre=hdu['FLUX_W']
    # return fluxmetre
    return fits.getval(fitspath, "FLUX_W")


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
