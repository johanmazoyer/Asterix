
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime

from random import random


def quickshow(tab):
    """
    Function to quickly show an array.
    tab: array to be shown

    Johan's quick function
    """

    tmp = np.copy(tab)
    # tmp = tmp.T
    plt.axis('off')
    plt.imshow(tmp, origin='lower', cmap='gray')
    plt.show()
    plt.close()


def save_plane_in_fits(dir_save_fits, name_plane, image):

    current_time_str = datetime.datetime.today().strftime('%H_%M_%S_%f')[:-3]
    name_fits = current_time_str + '_' + name_plane

    # sometime the image can be a single float (0 for phase or 1 for EF).
    if isinstance(image, (int, float, np.float)):
        return

    if np.iscomplexobj(image):
        tofits_array = np.zeros( (2,)+ image.shape )
        tofits_array[0] = np.real(image)
        tofits_array[1] = np.imag(image)
        fits.writeto(os.path.join(dir_save_fits,
                                  name_fits + '_RE_and_IM.fits'),
                     tofits_array,
                     overwrite=True)
    else:
        fits.writeto(os.path.join(dir_save_fits, name_fits + '_RE.fits'),
                     image,
                     overwrite=True)


def quickfits(tab, dir='', name='tmp'):
    """
    Function to quickly save in fits.
    By default, it will save on the desktop with a random name to avoid overwriting.
    Not sure the default saving on Desktop works for windows OS, but it work on mac and linux

    tab: array to be saved
    dir (optionnal): directory where to save the .fits. by default the Desktop.
    name (optionnal): name of the .fits. By defaut tmp_currenttimeinms.fits
    Johan's quick function
    """

    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    if dir == '':
        dir = desktop

    if name == 'tmp':
        current_time_str = datetime.datetime.today().strftime(
            '_%H_%M_%S_%f')[:-3]
        name = name + current_time_str
    fits.writeto(os.path.join(dir, name + '.fits'), tab, overwrite=True)


def quickpng(tab, dir='', name='tmp'):
    """
    Function to quickly save in .png.
    By default, it will save on the desktop with a random name

    tab: array to be saved
    dir (optionnal): directory where to save the .png
    name (optionnal): name of the .png.  By defaut tmpXX.png where xx is a random number

    Johan's quick function
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
    plt.close()


def check_and_load_fits(directory, filename):
    """ --------------------------------------------------
    check existence of a .fits file and load it.
    TODO can probably be modified to do that
    for other format automtaically

    Parameters:
    ----------
    directory : the directory in whih to searc
    filename :

    Return:
    ------
    the data result
    raise error if the file does not exist
    -------------------------------------------------- """
    if os.path.exists(directory + filename + '.fits') == True:
        return fits.getdata(os.path.join(directory, filename + '.fits'))
    else:
        raise Exception(
            "You need to create " + filename + ".fits before loading it." +
            "Please run the initialization with 'save_fits = True' before")


def from_param_to_header(config):
    """ --------------------------------------------------
    Convert ConfigObj parameters to fits header type list

    Parameters:
    ----------
    config: config obj

    Return:
    ------
    header: list of parameters

    Author: Axel Potier
    -------------------------------------------------- """
    header = fits.Header()
    for sect in config.sections:
        # print(config[str(sect)])
        for scalar in config[str(sect)].scalars:
            header[str(scalar)[:8]] = str(config[str(sect)][str(scalar)])
    return header



def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 0)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()