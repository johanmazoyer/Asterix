## Correction loop
import os
import datetime
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc


def CorrectionLoop(testbed,
                   estimator,
                   corrector,
                   maskdh_science,
                   gain,
                   Nbiter_corr,
                   Nbmode_corr,
                   Linesearch=False,
                   Linesearchmodes=None,
                   Search_best_Mode=False,
                   input_wavefront=0,
                   initial_DM_voltage=0.,
                   photon_noise=False,
                   nb_photons=0.,
                   silence=False):

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    nbiter = len(modevector)

    meancontrast = list()
    voltage_DMs = list()
    FP_Intensities = list()

    initialFP = testbed.todetector_Intensity(entrance_EF=input_wavefront,
                                             voltage_vector=initial_DM_voltage)
    initialFP_contrast = np.mean(initialFP[np.where(maskdh_science != 0)])

    voltage_DMs.append(initial_DM_voltage)
    meancontrast.append(initialFP_contrast)
    FP_Intensities.append(initialFP)

    if not silence:
        print("Initial contrast in DH: ", initialFP_contrast)
        plt.ion()
        plt.figure()

    for iteration, mode in enumerate(modevector):


        if Linesearch:
            # this is elegant but must be carefully done if we want to avoid infinite loop.
            bestcontrast, bestmode = CorrectionLoop(
                testbed,
                estimator,
                corrector,
                maskdh_science,
                gain,
                np.ones(len(Linesearchmodes), dtype=int),
                Linesearchmodes,
                Linesearch=False,
                Search_best_Mode=True,
                input_wavefront=input_wavefront,
                initial_DM_voltage=voltage_DMs[iteration],
                silence=True)
            print("Search Best Mode: ", bestmode, " contrast: ", bestcontrast)
            mode = bestmode

        if not silence:
            print("--------------------------------------------------")
            print("Iteration number: ", iteration, " SVD truncation: ", mode)

        # for now monochromatic estimation
        resultatestimation = estimator.estimate(
            testbed,
            voltage_vector=voltage_DMs[-1],
            entrance_EF=input_wavefront[testbed.wav_vec.tolist().index(testbed.wavelength_0)],
            wavelength=testbed.wavelength_0,
            photon_noise=photon_noise,
            nb_photons=nb_photons,
            perfect_estimation=Search_best_Mode)

        solution = -gain * corrector.toDM_voltage(testbed, resultatestimation,
                                                  mode)

        FP_Intensities.append(
            testbed.todetector_Intensity(entrance_EF=input_wavefront,
                                         voltage_vector=voltage_DMs[-1] + solution))

        if Search_best_Mode == False:
            #if we are only looking for the best mode, we do not update the DM shape
            # for the next iteration
            voltage_DMs.append(voltage_DMs[-1] + solution)

        meancontrast.append(
            np.mean(FP_Intensities[-1][np.where(maskdh_science != 0)]))

        if not silence:
            print("Mean contrast in DH: ", meancontrast[-1])
            print("")
            plt.clf()
            plt.imshow(np.log10(FP_Intensities[-1]),
                       vmin=-8,
                       vmax=-5)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.pause(0.01)

    if Search_best_Mode:
        return np.amin(meancontrast[1:]), modevector[np.argmin(meancontrast[1:])]

    else:
        # create an dictionnary to save all results
        CorrectionLoopResult = dict()
        CorrectionLoopResult["nb_total_iter"] = nbiter
        CorrectionLoopResult["SVDmodes"] = modevector
        CorrectionLoopResult["voltage_DMs"] = voltage_DMs
        CorrectionLoopResult["FP_Intensities"] = FP_Intensities
        CorrectionLoopResult["MeanDHContrast"] = meancontrast

        return CorrectionLoopResult


def Save_loop_results(CorrectionLoopResult, config, testbed, result_dir):

    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

    FP_Intensities = CorrectionLoopResult["FP_Intensities"]
    meancontrast = CorrectionLoopResult["MeanDHContrast"]
    voltage_DMs = CorrectionLoopResult["voltage_DMs"]
    nb_total_iter = CorrectionLoopResult["nb_total_iter"]

    ## SAVING...
    header = from_param_to_header(config)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_FocalPlane_Intesities" +
                 ".fits",
                 np.array(FP_Intensities),
                 header,
                 overwrite=True)

    fits.writeto(result_dir + current_time_str + "_Mean_Contrast_DH" + ".fits",
                 np.array(meancontrast),
                 header,
                 overwrite=True)

    DM_phases = np.zeros(
        (len(testbed.name_of_DMs), nb_total_iter, testbed.dim_overpad_pupil,
         testbed.dim_overpad_pupil))

    for i in range(nb_total_iter):
        allDMphases = testbed.voltage_to_phases(voltage_DMs[i])
        for j in range(len(testbed.name_of_DMs)):
            DM_phases[j, i, :, :] = allDMphases[j]

    for j, DM_name in enumerate(testbed.name_of_DMs):
        fits.writeto(result_dir + current_time_str + DM_name + "_phases" +
                     ".fits",
                     DM_phases[j],
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
