## Correction loop
import os
import datetime
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc

import Asterix.fits_functions as useful


def CorrectionLoop(testbed,
                   estimator,
                   corrector,
                   mask_dh,
                   Number_matrix,
                   gain,
                   Nbiter_corr,
                   Nbmode_corr,
                   Linesearch=False,
                   Linesearchmodes=None,
                   input_wavefront=0,
                   initial_DM_voltage=0.,
                   photon_noise=False,
                   nb_photons=0.,
                   silence=False):

    CorrectionLoopResult = dict()
    CorrectionLoopResult["nb_total_iter"] = 0
    CorrectionLoopResult["SVDmodes"] = list()
    CorrectionLoopResult["Nb_iter_per_mat"] = list()
    CorrectionLoopResult["voltage_DMs"] = list()
    CorrectionLoopResult["FP_Intensities"] = list()
    CorrectionLoopResult["EF_estim"] = list()
    CorrectionLoopResult["MeanDHContrast"] = list()

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
            gain,
            Nbiter_corr,
            Nbmode_corr,
            CorrectionLoopResult,
            Linesearch=Linesearch,
            Linesearchmodes=Linesearchmodes,
            input_wavefront=input_wavefront,
            initial_DM_voltage=initial_DM_voltage,
            photon_noise=photon_noise,
            nb_photons=nb_photons)

        min_contrast = min(CorrectionLoopResult["MeanDHContrast"])
        min_index = CorrectionLoopResult["MeanDHContrast"].index(min_contrast)
        initial_DM_voltage = Resultats_correction_loop["voltage_DMs"][
            min_index]

        print("end Matrix ", i)
        if i != Number_matrix:
            print("We will restart next matrix from contrast = ", min_contrast)

    return Resultats_correction_loop


def CorrectionLoop1Matrix(testbed,
                          estimator,
                          corrector,
                          maskdh_science,
                          gain,
                          Nbiter_corr,
                          Nbmode_corr,
                          CorrectionLoopResult,
                          Linesearch=False,
                          Linesearchmodes=None,
                          Search_best_Mode=False,
                          input_wavefront=1.,
                          initial_DM_voltage=0.,
                          photon_noise=False,
                          nb_photons=0.,
                          silence=False):

    ## Number of modes that is used as a function of the iteration cardinal
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

    initialFP_contrast = np.mean(initialFP[np.where(maskdh_science != 0)])

    nbiter = 1

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

    for iteration, mode in enumerate(modevector):

        if mode > corrector.total_number_modes:
            if not Search_best_Mode:
                print(
                    "You cannot use a cutoff mode ({:d}) larger than the total size of basis ({:d})"
                    .format(mode, corrector.Gmatrix.shape[1]))
                print("We skip this iteration")
            continue
        nbiter += 1

        if Linesearch:
            # this is elegant but must be carefully done if we want to avoid infinite loop.
            bestcontrast, bestmode = CorrectionLoop1Matrix(
                testbed,
                estimator,
                corrector,
                maskdh_science,
                gain,
                np.ones(len(Linesearchmodes), dtype=int),
                Linesearchmodes,
                list(),
                Linesearch=False,
                Search_best_Mode=True,
                input_wavefront=input_wavefront,
                initial_DM_voltage=thisloop_voltages_DMs[iteration],
                silence=True)
            print("Search Best Mode: ", bestmode, " contrast: ", bestcontrast)
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
            mode,
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
            np.mean(
                thisloop_FP_Intensities[-1][np.where(maskdh_science != 0)]))

        if Search_best_Mode == False:
            # if we are only looking for the best mode, we do not update the DM shape
            # for the next iteration
            thisloop_voltages_DMs.append(new_voltage)

        if not silence:
            print("Mean contrast in DH: ", thisloop_MeanDHContrast[-1])
            print("")
            plt.clf()
            plt.imshow(np.log10(thisloop_FP_Intensities[-1]), vmin=-8, vmax=-5)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.pause(0.01)

    if Search_best_Mode:
        return np.amin(thisloop_MeanDHContrast[1:]), modevector[np.argmin(
            thisloop_MeanDHContrast[1:])]

    else:
        # create a dictionnary to save all results

        modevector = [
            mode for mode in modevector if mode <= corrector.total_number_modes
        ]

        CorrectionLoopResult["nb_total_iter"] += nbiter
        CorrectionLoopResult["SVDmodes"].append(modevector)
        CorrectionLoopResult["Nb_iter_per_mat"].append(nbiter)

        CorrectionLoopResult["voltage_DMs"].extend(thisloop_voltages_DMs)
        CorrectionLoopResult["FP_Intensities"].extend(thisloop_FP_Intensities)
        CorrectionLoopResult["EF_estim"].extend(thisloop_EF_estim)
        CorrectionLoopResult["MeanDHContrast"].extend(thisloop_MeanDHContrast)
        if not silence:
            plt.close()

        return CorrectionLoopResult


def Save_loop_results(CorrectionLoopResult, config, testbed, result_dir):

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
