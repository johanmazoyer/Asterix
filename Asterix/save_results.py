
import os
import datetime
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

import Asterix.Optical_System_functions as OptSy
from Asterix.fits_functions import _quickfits


def plot_contrast_curves(reduced_data,
                         xcen=None,
                         ycen=None,
                         delta_raddii=3,
                         numberofpix_per_loD=None,
                         numberofmas_per_pix=None,
                         type_of_contrast='mean',
                         mask_DH=None,
                         path='',
                         filename=''):

    """  -------------------------------------------------- 
    Plot and save in pdf contrast curves from a image or a cube of image using concentring rings
    
    You can chooose the center size of the rings, the type of contrast (mean or std)
    The DH is set my putting a binary mask in the parameter
    The abciss unit can be in pixel mas or in l/D.

    AUTHOR: J. Mazoyer
    16/03/2022
    
    Parameters
    ----------
        
        reduced_data: array [dim, dim] or [nb_iter, dim, dim]
            array containing the reduced data. Assume to be already in contrast unit (divided by max of PSF)
            if the array is of dimension 3, the first dimension is assumed to be the dimensions and a 
            contrast curve will be plotted for each
        
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
    
        numberofpix_per_loD: float, defaut None
            resolution of the focal plane in # of pixel per lambda / D (useful for testbed)
            If set the absciss unit will be in λ/D 

        numberofmas_per_pix: float, defaut None
            resolution of the focal plane in # of mas per lambda / D  (useful for real instruments)
            If set the absciss unit will be in mas
            If none of these are set, the absciss unit will be in pixels
            If both are set, it will rais an error
        
        mask_DH : 2d binary Array  default is all focal plane
            mask delimiting the DH
        
        path: string, default ''
            path where to save the pdf plot file
        
        filename: string, default ''
            base of the file name to save the pdf plot file

    Returns
    ------

        No return

     -------------------------------------------------- """

    filename = filename +'_ContrastCurve_DH'

    if numberofpix_per_loD is not None and numberofmas_per_pix is None:
        # absice is in lambda over D
        absicemultiplicationfactor = delta_raddii/numberofpix_per_loD
        abscise_String_unit = '(λ/D)'
        filename += '_unitlod'

    elif numberofpix_per_loD  is None and numberofmas_per_pix is not None:
        # absice is in mas
        absicemultiplicationfactor = delta_raddii * numberofmas_per_pix
        abscise_String_unit = '(mas)'
        filename += '_unitmas'

    elif numberofpix_per_loD is None and numberofmas_per_pix  is None:
        # absice is in pixel
        absicemultiplicationfactor = delta_raddii
        abscise_String_unit = '(pix)'
        filename += '_unitpix'
    else:
        raise Exception("either numberofpix_per_loD or numberofmas_per_pix need to be filled, not both")
    

    plt.figure()

    if len(reduced_data) == 2:
        # this is a single image
        contrast1dcurve = contrast_curves(reduced_data,
                                          xcen=xcen,
                                          ycen=ycen,
                                          delta_raddii=delta_raddii,
                                          type_of_contrast=type_of_contrast,
                                          mask_DH = mask_DH)
        absice = np.arange(len(contrast1dcurve))*absicemultiplicationfactor

        plt.plot(absice, contrast1dcurve)
    else:
        # this ia cube
        for i, frame in enumerate(reduced_data):
            contrast1dcurve = contrast_curves(
                frame,
                xcen=xcen,
                ycen=ycen,
                delta_raddii=delta_raddii,
                type_of_contrast=type_of_contrast,
                mask_DH= mask_DH)
            if i == 0: 
                absice = np.arange(len(contrast1dcurve))*absicemultiplicationfactor
            
            plt.plot(absice, contrast1dcurve, label = "iter #{}".format(i))
        
        plt.legend()

    plt.xlabel("Separation " + abscise_String_unit)

    plt.title("Dark hole contrast")

    if type_of_contrast == 'mean':
        plt.ylabel("Mean contrast in Dark Hole")
        filename += '_mean'
    if type_of_contrast == 'stddev_1sig':
        plt.ylabel("1σ std contrast in Dark Hole")
        filename += '_1sigstd'
    if type_of_contrast == 'stddev_5sig':
        plt.ylabel("5σ std contrast in Dark Hole")
        filename += '_5sigstd'

    plt.yscale("log")

    plt.savefig(os.path.join(path, filename + ".pdf"))
    plt.close()


def contrast_curves(reduced_data,
                    xcen=None,
                    ycen=None,
                    delta_raddii=3,
                    type_of_contrast='mean',
                    mask_DH=None):
    """  -------------------------------------------------- 
    create a contrast curve map from a image using concentring rings
    You can chooose the center size of the rings, the type of contrast (mean or std)
    The DH is set my putting a binary mask in the parameter
    
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

     -------------------------------------------------- """
    if xcen is None:
        xcen = reduced_data.shape[0] / 2 - 1/2

    if ycen is None:
        ycen = reduced_data.shape[0] / 2 - 1/2

    dim = reduced_data.shape[1]

    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    contrast_curve = []

    if mask_DH is None:
        mask_DH = np.ones((reduced_data.shape))

    # chech the maximum number of ring we can fit in the image, which depends on the position of the center
    maximum_number_of_points = np.min(
        (np.floor(xcen / delta_raddii),
         np.floor((reduced_data.shape[0] - xcen) / delta_raddii),
         np.floor(ycen / delta_raddii),
         np.floor((reduced_data.shape[1] - ycen) / delta_raddii)))

    for i_ring in range(0, int(maximum_number_of_points) -1 ):

        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))

        masked_data = reduced_data * mask_DH
        if type_of_contrast == 'mean':
            contrast_curve.append(np.nanmean(masked_data[wh_rings]))
        elif type_of_contrast == 'stddev_1sig':
            contrast_curve.append(np.nanstd(masked_data[wh_rings]))
        elif type_of_contrast == 'stddev_5sig':
            contrast_curve.append(5 * np.nanstd(masked_data[wh_rings]))
        else:
            raise Exception("This type of contrast curve does not exists: ",
                            type_of_contrast)
    contrast_curve = np.asarray(contrast_curve)
    contrast_curve[np.where(contrast_curve == 0)] = np.nan
    return contrast_curve


def Save_loop_results(CorrectionLoopResult, config, testbed: OptSy.Testbed,
                    MaskScience,
                      result_dir):
    """ --------------------------------------------------
    Save the result from a correction loop in result_dir
    
    All fits have all parameters in the header.
    The config is also saved in a .ini file
    . No return

    AUTHOR : Johan Mazoyer
    
    Parameters
    ----------
    CorrectionLoopResult: dict 
        dictionnary containing the results from CorrectionLoop1Matrix or CorrectionLoop

    config: dict
        complete parameter dictionnary

    testbed: Optical_System
        an Optical_System object which describes your testbed
    
    mask_dh: 2d numpy array
        binary array of size [dimScience, dimScience] : dark hole mask
    
    result_dir: path
        directory where to save the results

    
    -------------------------------------------------- """

    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

    FP_Intensities = CorrectionLoopResult["FP_Intensities"]
    meancontrast = CorrectionLoopResult["MeanDHContrast"]
    voltage_DMs = CorrectionLoopResult["voltage_DMs"]
    nb_total_iter = CorrectionLoopResult["nb_total_iter"]
    EF_estim = CorrectionLoopResult["EF_estim"]

    ## SAVING...
    header = from_param_to_header(config)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_FocalPlane_Intensities" +
                 ".fits",
                 np.array(FP_Intensities),
                 header,
                 overwrite=True)

    fits.writeto(result_dir + current_time_str + "_Mean_Contrast_DH" + ".fits",
                 np.array(meancontrast),
                 header,
                 overwrite=True)

    fits.writeto(result_dir + current_time_str + "_estimationFP_RE" + ".fits",
                 np.real(np.array(EF_estim)),
                 header,
                 overwrite=True)

    fits.writeto(result_dir + current_time_str + "_estimationFP_IM" + ".fits",
                 np.imag(np.array(EF_estim)),
                 header,
                 overwrite=True)

    voltage_DMs_nparray = np.zeros((nb_total_iter, testbed.number_act))

    DM_phases = np.zeros(
        (len(testbed.name_of_DMs), nb_total_iter, testbed.dim_overpad_pupil,
         testbed.dim_overpad_pupil))

    for i in range(len(voltage_DMs)):
        allDMphases = testbed.voltage_to_phases(voltage_DMs[i])

        if isinstance(voltage_DMs[i], (int, float)):
            voltage_DMs_nparray[i, :] += float(voltage_DMs[i])
        else:
            voltage_DMs_nparray[i, :] = voltage_DMs[i]

        for j, DM_name in enumerate(testbed.name_of_DMs):
            DM_phases[j, i, :, :] = allDMphases[j]

    DMstrokes = DM_phases * testbed.wavelength_0 / (2 * np.pi * 1e-9) / 2

    indice_acum_number_act = 0
    plt.figure()

    for j, DM_name in enumerate(testbed.name_of_DMs):
        fits.writeto(result_dir + current_time_str + '_' + DM_name +
                     "_phases" + ".fits",
                     DM_phases[j],
                     header,
                     overwrite=True)

        fits.writeto(result_dir + current_time_str + '_' + DM_name +
                     "_strokes" + ".fits",
                     DMstrokes[j],
                     header,
                     overwrite=True)

        DM = vars(testbed)[DM_name]  # type: OptSy.deformable_mirror
        voltage_DMs_tosave = voltage_DMs_nparray[:, indice_acum_number_act:
                                                 indice_acum_number_act +
                                                 DM.number_act]
        indice_acum_number_act += DM.number_act

        fits.writeto(result_dir + current_time_str + '_' + DM_name +
                     "_voltages" + ".fits",
                     voltage_DMs_tosave,
                     header,
                     overwrite=True)

        plt.plot(np.std(DMstrokes[j], axis=(1, 2)), label=DM_name + " RMS")
        plt.plot(np.max(DMstrokes[j], axis=(1, 2)) -
                 np.min(DMstrokes[j], axis=(1, 2)),
                 label=DM_name + " PV")

    plt.xlabel("Number of iterations")
    plt.ylabel("DM Strokes (nm)")
    plt.legend()
    plt.savefig(
        os.path.join(result_dir, current_time_str + "_DM_Strokes" + ".pdf"))
    plt.close()
    # TODO Now FP_Intensities are save with photon noise if it's on
    # We need to do them without just to save in the results

    # if config["SIMUconfig"]["photon_noise"] == True:
    #     FP_Intensities_photonnoise = np.array(FP_Intensities) * 0.
    #     for i in range(nb_total_iter):
    #         FP_Intensities_photonnoise[i] = np.random.poisson(
    #             FP_Intensities[i] * testbed.normPupto1 *
    #             config["SIMUconfig"]["nb_photons"])

    #     fits.writeto(result_dir + current_time_str + "_NoPhoton_noise" + ".fits",
    #                  FP_Intensities_photonnoise,
    #                  header,
    #                  overwrite=True)

    config.filename = result_dir + current_time_str + "_Simulation_parameters" + ".ini"
    config.write()

    plt.figure()
    plt.plot(meancontrast)
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean contrast in Dark Hole")
    plt.savefig(
        os.path.join(result_dir,
                     current_time_str + "_Mean_Contrast_DH" + ".pdf"))
    plt.close()

    plot_contrast_curves(FP_Intensities,
                         delta_raddii=3,
                         numberofpix_per_loD=config["modelconfig"]["Science_sampling"],
                         type_of_contrast='mean',
                         mask_DH=MaskScience,
                         path=result_dir,
                         filename=current_time_str)


def from_param_to_header(config):
    """ --------------------------------------------------
    Convert ConfigObj parameters to fits header type list
    AUTHOR: Axel Potier

    Parameters
    ----------
    config: dict
        config obj

    Returns
    ------
    header: dict
        list of parameters


    
    -------------------------------------------------- """
    header = fits.Header()
    for sect in config.sections:
        # print(config[str(sect)])
        for scalar in config[str(sect)].scalars:
            header[str(scalar)[:8]] = str(config[str(sect)][str(scalar)])
    return header
