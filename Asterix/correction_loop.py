## Correction loop
import os
import datetime
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc

import Asterix.Optical_System_functions as OptSy

from Asterix.MaskDH import MaskDH
from Asterix.estimator import Estimator
from Asterix.corrector import Corrector

import Asterix.fits_functions as useful


def CorrectionLoop(testbed: OptSy.Testbed,
                   estimator: Estimator,
                   corrector: Corrector,
                   mask_dh: MaskDH,
                   SIMUconfig,
                   input_wavefront=0,
                   initial_DM_voltage=0.,
                   silence=False):
    """ --------------------------------------------------
    Run a full loop for several Matrix. at each iteration, we update the matrix and
    run CorrectionLoop1Matrix. 

    Parameters:
    ----------
        
    testbed: an Optical_System object which describes your testbed
    
    estimator: an estimator object. This contains all information about the estimation
    
    corrector: a corrector object. This contains all information about the correction
    
    mask_dh: binary array of size [dimScience, dimScience] : dark hole mask
    
    SIMUconfig: simulation parameters containing the loop parameters
    
    input_wavefront: initial wavefront at the beginning of this loop.
        Electrical Field which can be a :
            float 1. if no phase / amplitude (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic   
        
        !!CAREFUL!!: right now we do not use this wf to measure the matrix, although update_matrices
                    function allows it. Currently each matrix is measured with a flat field in 
                    entrance of the testbed (input_wavefront = 1). 
                    input_wavefront is only used in the loop once the matrix is calcultated   
                    This can be changed but be careful.             

    initial_DM_voltage. initial DM voltages at the beginning of this loop. The Matrix is measured 
                        using this initial DM voltages. Can be:
            float 0 if flat DMs (default)
            1D array of size testbed.number_act
    
    photon_noise: boolean, default False
                    If True, add photon noise.
    
    nb_photons int, optional default 1e30
                Number of photons entering the pupil
    
    silence=False: Boolean, default False
                if False, print and plot results as the loop runs

    Return:
    ------
    CorrectionLoopResult : a dictionnary containing the results of all loops

    Author: Johan Mazoyer
    -------------------------------------------------- """

    CorrectionLoopResult = dict()
    CorrectionLoopResult["nb_total_iter"] = 0
    CorrectionLoopResult["Nb_iter_per_mat"] = list()
    CorrectionLoopResult["voltage_DMs"] = list()
    CorrectionLoopResult["FP_Intensities"] = list()
    CorrectionLoopResult["EF_estim"] = list()
    CorrectionLoopResult["MeanDHContrast"] = list()

    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:

        Nbmode_corr = [int(i) for i in SIMUconfig["Nbmode_corr"]]
        Linesearch = SIMUconfig["Linesearch"]
        Linesearchmodes = [int(i) for i in SIMUconfig["Linesearchmodes"]]
        gain = SIMUconfig["gain"]
        CorrectionLoopResult["SVDmodes"] = list()

    if corrector.correction_algorithm == 'sm':
        Linesearch = False

    Nbiter_corr = [int(i) for i in SIMUconfig["Nbiter_corr"]]
    Number_matrix = SIMUconfig["Number_matrix"]
    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]

    for i in range(Number_matrix):

        if i > 0:
            # the first matrix is done during initialization
            corrector.update_matrices(testbed,
                                      estimator,
                                      initial_DM_voltage=initial_DM_voltage,
                                      input_wavefront=1.)

        Resultats_correction_loop = CorrectionLoop1Matrix(
            testbed,
            estimator,
            corrector,
            mask_dh,
            Nbiter_corr,
            CorrectionLoopResult,
            gain=gain,
            Nbmode_corr=Nbmode_corr,
            Linesearch=Linesearch,
            Linesearchmodes=Linesearchmodes,
            input_wavefront=input_wavefront,
            initial_DM_voltage=initial_DM_voltage,
            photon_noise=photon_noise,
            nb_photons=nb_photons,
            silence=silence)

        min_contrast = min(CorrectionLoopResult["MeanDHContrast"])
        min_index = CorrectionLoopResult["MeanDHContrast"].index(min_contrast)
        initial_DM_voltage = Resultats_correction_loop["voltage_DMs"][
            min_index]

        print("end Matrix ", i)
        if i != Number_matrix - 1:
            print("We will restart next matrix from contrast = ", min_contrast)

    return Resultats_correction_loop


def CorrectionLoop1Matrix(testbed: OptSy.Testbed,
                          estimator: Estimator,
                          corrector: Corrector,
                          mask_dh: MaskDH,
                          Nbiter_corr,
                          CorrectionLoopResult,
                          gain=0.1,
                          Nbmode_corr=None,
                          Linesearch=False,
                          Linesearchmodes=None,
                          Search_best_Mode=False,
                          input_wavefront=1.,
                          initial_DM_voltage=0.,
                          photon_noise=False,
                          nb_photons=1e30,
                          silence=False):
    """ --------------------------------------------------
    Run a loop for a given interraction matrix

    Parameters:
    ----------
        
    testbed: an Optical_System object which describes your testbed
    
    estimator: an estimator object. This contains all information about the estimation
    
    corrector: a corrector object. This contains all information about the correction
    
    mask_dh: binary array of size [dimScience, dimScience] : dark hole mask
    
    Nbiter_corr: int or list of int, number of iterations in the loop
    
    CorrectionLoopResult: dictionnary containing the result of the previous loop. 
                            This will be updated with the result of this loop
    
    gain:  gain of the loop in EFC mode. 
            float between 0 and 1, default 0.1
    
    Nbmode_corr: int or list of int of same size as Nbiter_corr,
                    SVD modes for each iteration
    
    Linesearch: bool, default False. If True, In this mode, the function CorrectionLoop1Matrix
                        will call itself at each iteration to find the best SVD inversion mode 
                        among Linesearchmodes. 
    
    Linesearchmodes: list of int. List of modes that are probed to find the best mode at  
                                each iteration
    
    Search_best_Mode: bool. default False. If true, the algorithms does not return the
                                loop information, just the best mode and best contrast. 
                                This mode is used in Linesearch mode
                                Be careful when using this parameter, it can create an infinite loop
    
    input_wavefront: initial wavefront at the beginning of this loop.
        Electrical Field which can be a :
            float 1. if no phase / amplitude (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic                   

    initial_DM_voltage. initial DM voltages at the beginning of this loop. Can be:
            float 0 if flat DMs (default)
            1D array of size testbed.number_act
    
    photon_noise: boolean, default False
                    If True, add photon noise.
    
    nb_photons int, optional default 1e30
                Number of photons entering the pupil
    
    silence=False: Boolean, default False
                if False, print and plot results as the loop runs

    Return:
    ------
    if Search_best_Mode == True, return [bestMode, bestContrast]
    else return CorrectionLoopResult dictionnary updated with the results from this loop

    Author: Johan Mazoyer
    -------------------------------------------------- """

    if Search_best_Mode:
        # This is to prevent an infinite loop
        Linesearch = False

    thisloop_expected_iteration_number = sum(Nbiter_corr)

    ## Number of modes that is used as a function of the iteration cardinal
    # in the EFC case
    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
        modevector = []
        for i in np.arange(len(Nbiter_corr)):
            modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    initialFP = testbed.todetector_Intensity(entrance_EF=input_wavefront,
                                             voltage_vector=initial_DM_voltage,
                                             save_all_planes_to_fits=False,
                                             dir_save_all_planes=None)

    estim_init = estimator.estimate(testbed,
                                    voltage_vector=initial_DM_voltage,
                                    entrance_EF=input_wavefront,
                                    wavelength=testbed.wavelength_0,
                                    photon_noise=photon_noise,
                                    nb_photons=nb_photons)

    initialFP_contrast = np.mean(initialFP[np.where(mask_dh != 0)])

    thisloop_voltages_DMs = list()
    thisloop_FP_Intensities = list()
    thisloop_MeanDHContrast = list()
    thisloop_EF_estim = list()

    thisloop_voltages_DMs.append(initial_DM_voltage)
    thisloop_FP_Intensities.append(initialFP)
    thisloop_EF_estim.append(estim_init)
    thisloop_MeanDHContrast.append(initialFP_contrast)

    if not silence:
        print("Initial contrast in DH: ", initialFP_contrast)
        plt.ion()
        plt.figure()

    # we start at 1 because we count the initial state as an iteration of the loop
    iteration_number = 1

    for iteration in range(thisloop_expected_iteration_number):

        if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
            mode = modevector[iteration]

            if mode > corrector.total_number_modes:
                if not Search_best_Mode:
                    print(
                        "You cannot use a cutoff mode ({:d}) larger than the total size of basis ({:d})"
                        .format(mode, corrector.Gmatrix.shape[1]))
                    print("We skip this iteration")
                continue

            if Linesearch:
                # if we are just trying to find the best mode, we just call the function itself
                # on the Linesearchmodes but without updating the results.
                # this is elegant but must be carefully done if we want to avoid infinite loop.
                bestcontrast, bestmode = CorrectionLoop1Matrix(
                    testbed,
                    estimator,
                    corrector,
                    mask_dh,
                    np.ones(len(Linesearchmodes), dtype=int),
                    dict(),
                    Linesearchmodes,
                    gain=gain,
                    Linesearch=False,
                    Search_best_Mode=True,
                    input_wavefront=input_wavefront,
                    initial_DM_voltage=thisloop_voltages_DMs[iteration],
                    silence=True)
                print("Search Best Mode: ", bestmode, " contrast: ",
                      bestcontrast)
                mode = bestmode

        if not silence:
            print("--------------------------------------------------")
            print("Iteration number: ", iteration, " SVD truncation: ", mode)

        # for now monochromatic estimation
        resultatestimation = estimator.estimate(
            testbed,
            voltage_vector=thisloop_voltages_DMs[-1],
            entrance_EF=input_wavefront,
            wavelength=testbed.wavelength_0,
            photon_noise=photon_noise,
            nb_photons=nb_photons,
            perfect_estimation=Search_best_Mode)

        solution = corrector.toDM_voltage(
            testbed,
            resultatestimation,
            mode=mode,
            gain=gain,
            ActualCurrentContrast=thisloop_MeanDHContrast[-1])

        if isinstance(solution, str) and solution == "StopTheLoop":
            # for each correction algorithm, we can break the loop by
            # the string "StopTheLoop" instead of a correction vector
            print("we stop the correction")
            break

        elif isinstance(solution, str) and solution == "RebootTheLoop":
            # for each correction algorithm, we can break the loop by
            # the string "RebootTheLoop" instead of a correction vector
            print("we go back to last best correction")
            ze_arg_of_ze_best = np.argmin(thisloop_MeanDHContrast)
            new_voltage = thisloop_voltages_DMs[ze_arg_of_ze_best]

        else:
            new_voltage = thisloop_voltages_DMs[-1] + solution

        thisloop_FP_Intensities.append(
            testbed.todetector_Intensity(entrance_EF=input_wavefront,
                                         voltage_vector=new_voltage))
        thisloop_EF_estim.append(resultatestimation)
        thisloop_MeanDHContrast.append(
            np.mean(thisloop_FP_Intensities[-1][np.where(mask_dh != 0)]))

        if Search_best_Mode == False:
            # if we are only looking for the best mode, we do not update the DM shape
            # for the next iteration
            thisloop_voltages_DMs.append(new_voltage)

        iteration_number += 1
        if not silence:
            print("Mean contrast in DH: ", thisloop_MeanDHContrast[-1])
            print("")
            plt.clf()
            plt.imshow(np.log10(thisloop_FP_Intensities[-1]), vmin=-8, vmax=-5)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.pause(0.01)

    if Search_best_Mode:
        # in Search_best_Mode mode we return the mode that gives the best contrast
        return np.amin(thisloop_MeanDHContrast[1:]), modevector[np.argmin(
            thisloop_MeanDHContrast[1:])]

    else:
        # create a dictionnary to save all results

        if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
            modevector = [
                mode for mode in modevector
                if mode <= corrector.total_number_modes
            ]
            CorrectionLoopResult["SVDmodes"].append(modevector)

        CorrectionLoopResult["nb_total_iter"] += iteration_number
        CorrectionLoopResult["Nb_iter_per_mat"].append(iteration_number)

        CorrectionLoopResult["voltage_DMs"].extend(thisloop_voltages_DMs)
        CorrectionLoopResult["FP_Intensities"].extend(thisloop_FP_Intensities)
        CorrectionLoopResult["EF_estim"].extend(thisloop_EF_estim)
        CorrectionLoopResult["MeanDHContrast"].extend(thisloop_MeanDHContrast)

        if not silence:
            plt.close()

        if Linesearch:
            print(
                "Linesearch Mode. We found the following modes to dig the contrast best:"
            )
            print(modevector)

        return CorrectionLoopResult


def Save_loop_results(CorrectionLoopResult, config, testbed, result_dir):
    """ --------------------------------------------------
    Save the result from a correction loop 

    Parameters:
    ----------
    CorrectionLoopResult: dictionnary containing the results from CorrectionLoop1Matrix or CorrectionLoop

    config: complete parameter dictionnary

    testbed: an Optical_System object which describes your testbed
    
    result_dir: directory where to save the results

    Return:
    ------


    Author: Johan Mazoyer
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

    indice_acum_number_act = 0
    for j, DM_name in enumerate(testbed.name_of_DMs):
        fits.writeto(result_dir + current_time_str + '_' + DM_name +
                     "_phases" + ".fits",
                     DM_phases[j],
                     header,
                     overwrite=True)

        DM = vars(testbed)[DM_name]
        voltage_DMs_tosave = voltage_DMs_nparray[:, indice_acum_number_act:
                                                 indice_acum_number_act +
                                                 DM.number_act]
        indice_acum_number_act += DM.number_act

        fits.writeto(result_dir + current_time_str + '_' + DM_name +
                     "_voltages" + ".fits",
                     voltage_DMs_tosave,
                     header,
                     overwrite=True)

    if config["SIMUconfig"]["photon_noise"] == True:
        FP_Intensities_photonnoise = np.array(FP_Intensities) * 0.
        for i in range(nb_total_iter):
            FP_Intensities_photonnoise[i] = np.random.poisson(
                FP_Intensities[i] * testbed.normPupto1 *
                config["SIMUconfig"]["nb_photons"])

        fits.writeto(result_dir + current_time_str + "_Photon_noise" + ".fits",
                     FP_Intensities_photonnoise,
                     header,
                     overwrite=True)

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
