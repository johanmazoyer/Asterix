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
                   input_wavefront=0,
                   initial_DM_voltage=0.,
                   photon_noise=False,
                   nb_photons=0.,
                   plot_iter=False):

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

    print("Initial contrast in DH: ", initialFP_contrast)

    if plot_iter:
        plt.ion()
        plt.figure()
    for iteration, mode in enumerate(modevector):
        print("--------------------------------------------------")
        print("Iteration number: ", iteration, " SVD truncation: ", mode)

        resultatestimation = estimator.estimate(
            testbed,
            entrance_EF=input_wavefront,
            voltage_vector=voltage_DMs[iteration],
            wavelength=testbed.wavelength_0,
            photon_noise=photon_noise,
            nb_photons=nb_photons)

        solution = -gain * corrector.toDM_voltage(testbed, resultatestimation,
                                                  mode)

        voltage_DMs.append(voltage_DMs[iteration] + solution)

        FP_Intensities.append(
            testbed.todetector_Intensity(entrance_EF=input_wavefront,
                                         voltage_vector=voltage_DMs[iteration +
                                                                    1]))

        meancontrast.append(
            np.mean(FP_Intensities[iteration +
                                  1][np.where(maskdh_science != 0)]))
        print("Mean contrast in DH: ", meancontrast[iteration + 1])

        if plot_iter:
            plt.clf()
            plt.imshow(np.log10(FP_Intensities[iteration + 1]),
                       vmin=-8,
                       vmax=-5)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.pause(0.01)

    # create an dictionnary to save all results
    CorrectionLoopResult = dict()
    CorrectionLoopResult["nb_total_iter"] = nbiter
    CorrectionLoopResult["SVDmodes"] = np.array(modevector)
    CorrectionLoopResult["voltage_DMs"] = np.array(voltage_DMs)
    CorrectionLoopResult["FP_Intensities"] = np.array(FP_Intensities)
    CorrectionLoopResult["MeanDHContrast"] = np.array(meancontrast)

    return CorrectionLoopResult


# if Linesearch is True:
#     search_best_contrast = list()
#     perfectestimate = estim.estimate(thd2,
#                                      entrance_EF=input_wavefront,
#                                      voltage_vector=0.,
#                                      wavelength=testbed.wavelength_0,
#                                      perfect_estimation=True)

#     for modeLinesearch in Linesearchmode:

#         perfectsolution = -gain * correc.amplitudeEFC * correc.toDM_voltage(
#             thd2, perfectestimate, modeLinesearch)

#         tmpvoltage_DMs = voltage_DMs[iteration] + perfectsolution

#         imagedetector_tmp = testbed.todetector_Intensity(
#             entrance_EF=input_wavefront, voltage_vector=tmpvoltage_DMs)

#         search_best_contrast.append(
#             np.mean(imagedetector_tmp[np.where(MaskScience != 0)]))
#     bestcontrast = np.amin(search_best_contrast)
#     mode = Linesearchmode[np.argmin(search_best_contrast)]
#     print('Best contrast= ', bestcontrast, ' Best regul= ', mode)


def Save_loop_results(CorrectionLoopResult, config, testbed, result_dir):

    FP_Intensities = CorrectionLoopResult["FP_Intensities"]
    meancontrast = CorrectionLoopResult["MeanDHContrast"]
    voltage_DMs = CorrectionLoopResult["voltage_DMs"]
    nb_total_iter = CorrectionLoopResult["nb_total_iter"]

    ## SAVING...
    header = from_param_to_header(config)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_FocalPlane_Intesities" + ".fits",
                 FP_Intensities,
                 header,
                 overwrite=True)

    fits.writeto(result_dir + current_time_str + "_Mean_Contrast_DH" + ".fits",
                 meancontrast,
                 header,
                 overwrite=True)

    DM_phases = np.zeros((len(testbed.name_of_DMs),nb_total_iter,testbed.dim_overpad_pupil,testbed.dim_overpad_pupil))

    for i in range(nb_total_iter):
        allDMphases = testbed.voltage_to_phases(voltage_DMs[i])
        for j in range(len(testbed.name_of_DMs)):
            DM_phases[j,i,:,:] = allDMphases[j]

    for j, DM_name in enumerate(testbed.name_of_DMs):
        fits.writeto(result_dir + current_time_str + DM_name + "_phases" +
                     ".fits",
                     DM_phases[j],
                     header,
                     overwrite=True)

    if config["SIMUconfig"]["photon_noise"] == True:
        FP_Intensities_photonnoise = FP_Intensities * 0.
        for i in range(nb_total_iter):
            FP_Intensities_photonnoise[i] = np.random.poisson(
                FP_Intensities[i] * testbed.normPupto1 *
                config["SIMUconfig"]["nb_photons"])

        fits.writeto(result_dir + current_time_str + "_Photon_noise" +
                     ".fits",
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
