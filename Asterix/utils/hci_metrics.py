# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import warnings

import numpy as np
import matplotlib
from IPython import get_ipython
if get_ipython() is None:  # this matplotlib option is just in non-notebook case
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_contrast_curves(reduced_data,
                         xcen=None,
                         ycen=None,
                         delta_raddii=3,
                         numberofpix_per_loD=None,
                         numberofmas_per_pix=None,
                         type_of_contrast='mean',
                         mask_DH=None,
                         xtitle=None,
                         ytitle=None,
                         title=None,
                         legend_labels=None,
                         xrange=None,
                         yrange=None,
                         path='',
                         filename=''):
    """
    Plot and save in pdf contrast curves from a image or a cube of images (assumed to be 
    several iterations of a loop) using concentric rings.
    You can chooose the center, the size of the rings, the type of contrast (mean or std)
    The DH is set using a binary mask (1s where the DH is, 0 elsewhere)
    The abcisse unit can be in pixel mas or in lambda/D.

    AUTHOR: J. Mazoyer
    16/03/2022
    
    Parameters
    ----------
        reduced_data: array [dim, dim] or [nb_iter, dim, dim]
            array containing the reduced data. Assume to be already in contrast unit (divided by max of PSF)
            if the array is of dimension 3, the first dimension is assumed to be the number of iter and a 
            contrast curve will be plotted for each
        
        xcen: float, default None (reduced_data.shape[0]/2 - 1/2)
            pixel, position x of the star
        
        ycen: float, default None (reduced_data.shape[1]/2 - 1/2)
            pixel, position y of the star
        
        delta_raddii: default 3
            pixel, width of the small concentric rings
        
        type_of_contrast: string default 'mean'
            can be  'mean' : mean contrast on the rings 
                    'stddev_1sig' : 1 sigma standard deviation on the rings
                    'stddev_5sig' : 5 sigma standard deviation on the rings
    
        numberofpix_per_loD: float, defaut None
            resolution of the focal plane in # of pixel per lambda/D (useful for testbed)
            If set the absciss unit will be in lambda/D 

        numberofmas_per_pix: float, defaut None
            resolution of the focal plane in # of mas per pixel  (useful for real instruments)
            If set the absciss unit will be in mas
            
            If none of these keywords are set, the absciss unit will be in pixels
            If both are set, it will raise an error
        
        mask_DH : 2d binary Array  default is all focal plane
            mask delimiting the DH
        
        path: string, default ''
            path where to save the pdf plot file
        
        filename: string, default ''
            base of the file name to save the pdf plot file
        
        legend_labels: string array of the same number of images in the first cube, default None
            Name of the legend labels,
            If None and if the array is of dimension 2, no legend
            If None and if the array is of dimension 3, we assume these are iterations

    """

    filename = filename + '_ContrastCurve_DH'

    if numberofpix_per_loD is not None and numberofmas_per_pix is None:
        # absice is in lambda over D
        absicemultiplicationfactor = delta_raddii / numberofpix_per_loD
        abscise_String_unit = '(λ/D)'
        filename += '_unitlod'

    elif numberofpix_per_loD is None and numberofmas_per_pix is not None:
        # absice is in mas
        absicemultiplicationfactor = delta_raddii * numberofmas_per_pix
        abscise_String_unit = '(mas)'
        filename += '_unitmas'

    elif numberofpix_per_loD is None and numberofmas_per_pix is None:
        # absice is in pixel
        absicemultiplicationfactor = delta_raddii
        abscise_String_unit = '(pix)'
        filename += '_unitpix'
    else:
        raise Exception("either numberofpix_per_loD or numberofmas_per_pix need to be filled, not both")

    plt.figure()

    if len(reduced_data.shape) == 2:
        # reduced_data is a single image
        contrast1dcurve = contrast_curves(reduced_data,
                                          xcen=xcen,
                                          ycen=ycen,
                                          delta_raddii=delta_raddii,
                                          type_of_contrast=type_of_contrast,
                                          mask_DH=mask_DH)
        absice = np.arange(len(contrast1dcurve)) * absicemultiplicationfactor
        iwa = np.nanmin(absice[~np.isnan(contrast1dcurve)])
        owa = np.nanmax(absice[~np.isnan(contrast1dcurve)])

        plt.plot(absice, contrast1dcurve)
    else:

        # reduced_data is a cube
        if legend_labels is None:
            legend_labels = list()
            legend_labels.append("Initial")
            for i in range(1, reduced_data.shape[0]):
                legend_labels.append(f"iter #{i}")
        else:
            if len(legend_labels) != reduced_data.shape[0]:
                raise Exception("legend_labels must be a string list of size as reduced_data.shape[0]")

        for i, frame in enumerate(reduced_data):
            contrast1dcurve = contrast_curves(frame,
                                              xcen=xcen,
                                              ycen=ycen,
                                              delta_raddii=delta_raddii,
                                              type_of_contrast=type_of_contrast,
                                              mask_DH=mask_DH)
            if i == 0:
                absice = np.arange(len(contrast1dcurve)) * absicemultiplicationfactor
                iwa = np.nanmin(absice[~np.isnan(contrast1dcurve)])
                owa = np.nanmax(absice[~np.isnan(contrast1dcurve)])

            plt.plot(absice, contrast1dcurve, label=legend_labels[i])

        plt.legend(fontsize=6, loc='upper right')

    if xrange is None:
        xrange = [0, 1.2 * owa]

    plt.xlim(xrange[0], xrange[1])

    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])

    font_size = 14

    if title is None:
        title = "Separation " + abscise_String_unit
    plt.title(title, fontsize=font_size)

    if xtitle is None:
        xtitle = "Separation " + abscise_String_unit
    plt.xlabel(xtitle, fontsize=font_size)

    if ytitle is None:
        if type_of_contrast == 'mean':
            ytitle = "Mean contrast in Dark Hole"
            filename += '_mean'
        if type_of_contrast == 'stddev_1sig':
            ytitle = "1σ std contrast in Dark Hole"
            filename += '_1sigstd'
        if type_of_contrast == 'stddev_5sig':
            ytitle = "5σ std contrast in Dark Hole"
            filename += '_5sigstd'
    plt.ylabel(ytitle, fontsize=font_size)
    plt.yscale("log")

    tick_size = 12
    plt.yticks(fontsize=tick_size)
    plt.xticks(fontsize=tick_size)

    plt.savefig(os.path.join(path, filename + ".pdf"))
    plt.close()


def contrast_curves(reduced_data,
                    xcen=None,
                    ycen=None,
                    delta_raddii=3,
                    type_of_contrast='mean',
                    mask_DH=None):
    """  
    create a contrast curve from a image using concentric rings
    You can chooose the center, the size of the rings, the type of contrast (mean or std)
    The DH is set using a binary mask (1s where the DH is, 0 elsewhere)

    AUTHOR: J. Mazoyer
    16/03/2022
    
    Parameters
    ----------
        
        reduced_data: array
            [dim dim] array containing the reduced data
        
        xcen: float, default None (reduced_data.shape[0]/2 -1/2) 
            pixel, position x of the star
        
        ycen: float, default None (reduced_data.shape[1]/2 -1/2) 
            pixel, position y of the star
        
        delta_raddii: default 3
            pixel, width of the small concentric rings
        
        type_of_contrast: string default 'mean'
            can be  'mean' : mean contrast on the rings 
                    'stddev_1sig' : 1 sigma standard deviation on the rings
                    'stddev_5sig' : 5 sigma standard deviation on the rings
        
        mask_DH : 2d binary Array  default is all focal plane
            mask delimiting the DH


    Returns
    ------

        1d array with the contrast on concentric rings measure with different metrics
        Values outside of the mask are nan

     """
    if xcen is None:
        xcen = reduced_data.shape[0] / 2 - 1 / 2

    if ycen is None:
        ycen = reduced_data.shape[0] / 2 - 1 / 2

    dim = reduced_data.shape[1]

    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    contrast_curve = []

    if mask_DH is None:
        mask_DH = np.ones((reduced_data.shape))

    mask_DH[np.where(mask_DH == 0)] = np.nan

    # chech the maximum number of ring we can fit in the image, which depends on the position of the center
    maximum_number_of_points = np.min(
        (np.floor(xcen / delta_raddii), np.floor((reduced_data.shape[0] - xcen) / delta_raddii),
         np.floor(ycen / delta_raddii), np.floor((reduced_data.shape[1] - ycen) / delta_raddii)))

    for i_ring in range(0, int(maximum_number_of_points) - 1):

        wh_rings = np.where((rho2d >= i_ring * delta_raddii) & (rho2d < (i_ring + 1) * delta_raddii))

        masked_data = reduced_data * mask_DH
        if type_of_contrast == 'mean':
            contrast_curve.append(np.nanmean(masked_data[wh_rings]))
        elif type_of_contrast == 'stddev_1sig':
            contrast_curve.append(np.nanstd(masked_data[wh_rings]))
        elif type_of_contrast == 'stddev_5sig':
            contrast_curve.append(5 * np.nanstd(masked_data[wh_rings]))
        else:
            raise Exception("This type of contrast curve does not exists: ", type_of_contrast)
    contrast_curve = np.asarray(contrast_curve)
    contrast_curve[np.where(contrast_curve == 0)] = np.nan
    return contrast_curve
