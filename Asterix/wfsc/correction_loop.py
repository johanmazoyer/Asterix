import os
import numpy as np
import matplotlib
from IPython import get_ipython
if get_ipython() is None:  # this matplotlib option is just in non-notebook case
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits

from Asterix.utils import plot_contrast_curves, from_param_to_header
from Asterix.optics import DeformableMirror, Testbed

import Asterix.wfsc.corrector as corrector_mod
import Asterix.wfsc.estimator as estimator_mod


def correction_loop(testbed: Testbed,
                    estimator: estimator_mod.Estimator,
                    corrector: corrector_mod.Corrector,
                    mask_dh,
                    Loopconfig,
                    SIMUconfig,
                    input_wavefront=1.,
                    initial_DM_voltage=0.,
                    silence=False,
                    **kwargs):
    """Run a full loop for several matrices.

    In each new iteration, we update the matrix and run correction_loop_1matrix() anew.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    testbed : OpticalSystem.Testbed
        Object which describes your testbed.
    estimator : Estimator
        Estimator object containing all information about the WF estimation.
    corrector : Corrector.
        Corrector object containing all information about the WF correction.
    mask_dh: 2d numpy array
        Binary array of size [dimScience, dimScience], the dark-hole mask.
    Loopconfig : dict
        Simulation parameters for the WFS&C loop.
    SIMUconfig : dict
        Simulation parameters.
    input_wavefront : float or 2d complex array or 3d complex array
        Initial wavefront at the beginning of this loop.
        Electrical Field which can be a:
            float=1 if there are no phase/amplitude aberrations (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic
        !!CAREFUL!!: Right now we do not use this wf to measure the matrix, although the update_matrices()
            method inside the Corrector allows it. Currently, each matrix is measured with a flat field in
            the entrance of the testbed (input_wavefront = 1).
            'input_wavefront' is only used in the loop once the matrix is calculated. This can be changed but be careful.
    initial_DM_voltage : float or 1D array
        Initial DM voltages at the beginning of this loop. The Matrix is measured using these initial DM voltages.
        Can be:
            float 0 if flat DMs (default)
            or 1D array of size testbed.number_act
    silence : boolean, default False.
        Whether to silence print outputs.

    Returns
    --------
    CorrectionLoopResult : dict
        A dictionary containing the results of all loops.
    """

    CorrectionLoopResult = dict()
    CorrectionLoopResult["nb_total_iter"] = 0
    CorrectionLoopResult["Nb_iter_per_mat"] = []
    CorrectionLoopResult["voltage_DMs"] = []
    CorrectionLoopResult["FP_Intensities"] = []
    CorrectionLoopResult["EF_estim"] = []
    CorrectionLoopResult["MeanDHContrast"] = []
    CorrectionLoopResult["SVDmodes"] = []

    # reading the simulation parameter files
    nb_photons = SIMUconfig["nb_photons"]

    if nb_photons > 1:
        CorrectionLoopResult["FP_Intensities_phot"] = []

    # reading the loop parameter files
    Nbiter_corr = list(Loopconfig["Nbiter_corr"])
    Number_matrix = Loopconfig["Number_matrix"]

    Nbmode_corr = []
    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
        Linesearch = Loopconfig["Linesearch"]
        gain = Loopconfig["gain"]
        if not Linesearch:
            Nbmode_corr = list(Loopconfig["Nbmode_corr"])
            if len(Nbiter_corr) != len(Nbmode_corr):
                raise ValueError(("In this correction mode and if Linesearch = False, "
                                  "the length of Nbmode_corr must match the length of Nbiter_corr"))

    else:
        Linesearch = None
        gain = 1.

    if corrector.correction_algorithm == 'sm':
        Linesearch = False

    no_aberr_FP = testbed.todetector_intensity()
    no_aberr_attenuation = np.max(no_aberr_FP)
    if not silence:
        print("Tesbed FP maximum contrast in the absence of aberrations: ", no_aberr_attenuation)

    for i in range(Number_matrix):

        if i > 0:
            corrector.update_matrices(testbed, initial_DM_voltage=initial_DM_voltage, silence=silence)

        Resultats_correction_loop = correction_loop_1matrix(testbed,
                                                            estimator,
                                                            corrector,
                                                            mask_dh,
                                                            Nbiter_corr,
                                                            CorrectionLoopResult,
                                                            gain=gain,
                                                            Nbmode_corr=Nbmode_corr,
                                                            Linesearch=Linesearch,
                                                            input_wavefront=input_wavefront,
                                                            initial_DM_voltage=initial_DM_voltage,
                                                            nb_photons=nb_photons,
                                                            silence=silence,
                                                            **kwargs)

        min_contrast = min(CorrectionLoopResult["MeanDHContrast"])
        min_index = CorrectionLoopResult["MeanDHContrast"].index(min_contrast)
        initial_DM_voltage = Resultats_correction_loop["voltage_DMs"][min_index]
        if not silence:
            if i != Number_matrix - 1:
                print("end Matrix ", i)
                print("We will restart next matrix from contrast = ", min_contrast)
            else:
                print("End correction")

    return Resultats_correction_loop


def correction_loop_1matrix(testbed: Testbed,
                            estimator: estimator_mod.Estimator,
                            corrector: corrector_mod.Corrector,
                            mask_dh,
                            Nbiter_corr,
                            CorrectionLoopResult,
                            gain=0.1,
                            Nbmode_corr=None,
                            Linesearch=False,
                            Search_best_Mode=False,
                            input_wavefront=1.,
                            initial_DM_voltage=0.,
                            nb_photons=0,
                            silence=False,
                            **kwargs):
    """Run a loop for a given interaction matrix.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    testbed : OpticalSystem.Testbed
        object which describes your testbed
    estimator : Estimator
        This contains all information about the estimation
    corrector : Corrector
        This contains all information about the correction
    mask_dh: 2d numpy array
        binary array of size [dimScience, dimScience] : dark hole mask
    Nbiter_corr : int or list of int
        number of iterations in the loop
    CorrectionLoopResult: dict
        Dictionary containing the result of the previous loop.
        This will be updated with the result of the current loop.
    gain : float between 0 and 1, default 0.1
        Control gain of the loop in EFC mode.
    Nbmode_corr : int or list of int
        Of same size as Nbiter_corr; SVD modes for each iteration.
    Linesearch: bool, default False
        If True, the function correction_loop_1matrix()
        will call itself at each iteration with Search_best_Mode=True to find
        the best SVD inversion mode among a few Linesearch modes.
    Search_best_Mode: bool, default False
        If true, the algorithm does not return the
        loop information, just the best mode and best contrast.
        This mode is used in Linesearch mode.
        Be careful when using this parameter, it can create an infinite loop.
    input_wavefront : float or 2d complex array or 3d complex array
        Initial wavefront at the beginning of this loop.
        Electrical Field which can be a:
            float=1 if no phase/amplitude aberrations present (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic
    initial_DM_voltage : float or 1D array
        Initial DM voltages at the beginning of this loop. The Matrix is measured using this initial DM voltages.
        Can be:
            float 0 if flat DMs (default)
            or 1D array of size testbed.number_act
    nb_photons : float, optional, default 0
        Number of photons entering the pupil. If 0, no photon noise.
    silence : boolean, default False.
        Whether to silence print outputs.

    Returns
    --------
    if Search_best_Mode == True, return [bestMode, bestContrast]
    else return CorrectionLoopResult dictionary updated with the results from this loop
    """

    if Search_best_Mode:
        # This is to prevent an infinite loop
        Linesearch = False

    thisloop_expected_iteration_number = sum(Nbiter_corr)

    # Number of modes that is used as a function of the iteration cardinal in the EFC case.

    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
        modevector = []
        if not Linesearch:
            for i in np.arange(len(Nbiter_corr)):
                modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    initialFP = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                             voltage_vector=initial_DM_voltage,
                                             nb_photons=0,
                                             **kwargs)

    initialFP_contrast = np.mean(initialFP[np.where(mask_dh != 0)])

    thisloop_voltages_DMs = []
    thisloop_FP_Intensities = []
    thisloop_FP_Intensities_phot = []
    thisloop_MeanDHContrast = []
    thisloop_EF_estim = []
    thisloop_actual_modes = []

    thisloop_voltages_DMs.append(initial_DM_voltage)
    thisloop_FP_Intensities.append(initialFP)
    thisloop_MeanDHContrast.append(initialFP_contrast)

    if nb_photons > 1:
        initialFP_phot = testbed.add_photon_noise(initialFP, nb_photons)
        thisloop_FP_Intensities_phot.append(initialFP_phot)

    if not silence:
        print("Initial contrast in DH: ", initialFP_contrast)
        plt.ion()
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        im = ax.imshow(np.log10(initialFP), vmin=-8, vmax=-5)
        ax.figure.colorbar(im)
        plt.pause(0.0001)

    # we start at 1 because we count the initial state as an iteration of the loop
    iteration_number = 1

    for iteration in range(thisloop_expected_iteration_number):

        if corrector.correction_algorithm in ['efc', 'em', 'steepest']:

            if Linesearch:

                # we search the best cutoff mode among 10 different ones evenly separated
                Linesearchmodes = 10 * (np.arange(0.2, 0.6, 0.03) * corrector.total_number_modes / 10).astype(int)

                print("Search Best Mode among ", Linesearchmodes)
                # if we are just trying to find the best mode, we just call the function itself
                # on the Linesearchmodes but without updating the results.
                # this is elegant but must be carefully done if we want to avoid infinite loop.
                bestcontrast, bestmode = correction_loop_1matrix(testbed,
                                                                 estimator,
                                                                 corrector,
                                                                 mask_dh,
                                                                 np.ones(len(Linesearchmodes), dtype=int),
                                                                 dict(),
                                                                 gain=gain,
                                                                 Nbmode_corr=Linesearchmodes,
                                                                 Search_best_Mode=True,
                                                                 nb_photons=0,
                                                                 input_wavefront=input_wavefront,
                                                                 initial_DM_voltage=thisloop_voltages_DMs[iteration],
                                                                 silence=silence,
                                                                 **kwargs)

                print("Best Mode is ", bestmode, " with contrast: ", bestcontrast)
                mode = bestmode
            else:
                mode = modevector[iteration]

            if mode > corrector.total_number_modes:
                if not Search_best_Mode:
                    print(f"You cannot use a cutoff mode ({mode:d}) larger than" +
                          f"the total size of basis ({corrector.Gmatrix.shape[1]:d})")
                    print("We skip this iteration")
                continue

        else:
            mode = 1

        if not silence:
            if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
                print("Iteration number " + corrector.correction_algorithm + ": ", iteration + 1, " SVD truncation: ",
                      mode)
            else:
                print("Iteration number " + corrector.correction_algorithm + ": ", iteration + 1)

        probed_images = estimator.probe(testbed,
                                        voltage_vector=thisloop_voltages_DMs[-1],
                                        entrance_EF=input_wavefront,
                                        perfect_estimation=Search_best_Mode,
                                        nb_photons=nb_photons,
                                        **kwargs)

        resultatestimation = estimator.estimate(probed_images,
                                                perfect_estimation=Search_best_Mode,
                                                dtype_complex=testbed.dtype_complex,
                                                testbed=testbed,
                                                **kwargs)

        solution = corrector.toDM_voltage(testbed,
                                          resultatestimation,
                                          mode=mode,
                                          ActualCurrentContrast=thisloop_MeanDHContrast[-1],
                                          silence=silence)

        if isinstance(solution, str) and solution == "StopTheLoop":
            # for each correction algorithm, we can break the loop by
            # the string "StopTheLoop" instead of a correction vector
            if not silence:
                print("we stop the correction")
            break

        if isinstance(solution, str) and solution == "RebootTheLoop":
            # for each correction algorithm, we can break the loop by
            # the string "RebootTheLoop" instead of a correction vector
            if not silence:
                print("we go back to last best correction")
            ze_arg_of_ze_best = np.argmin(thisloop_MeanDHContrast)
            new_voltage = thisloop_voltages_DMs[ze_arg_of_ze_best]

        else:
            new_voltage = thisloop_voltages_DMs[-1] + gain * solution

        thisloop_FP_Intensities.append(
            testbed.todetector_intensity(entrance_EF=input_wavefront, voltage_vector=new_voltage, nb_photons=0,
                                         **kwargs))
        thisloop_FP_Intensities_phot.append(testbed.add_photon_noise(thisloop_FP_Intensities[-1], nb_photons))
        thisloop_EF_estim.append(resultatestimation)

        # the contrast cannot be measured on a photon noise image, because at some point a lot of values
        # are at 0 and it will artificially lower the contrast. Photon noise is only used for the pw images.
        thisloop_MeanDHContrast.append(np.mean(thisloop_FP_Intensities[-1][np.where(mask_dh != 0)]))

        if not Search_best_Mode:
            # if we are only looking for the best mode, we do not update the DM shape
            # for the next iteration
            thisloop_voltages_DMs.append(new_voltage)

        iteration_number += 1
        if not silence:
            print("Mean contrast in DH: ", thisloop_MeanDHContrast[-1])
            print("--------------------------------------------------")
            print("")
            im.set_data(np.log10(thisloop_FP_Intensities[-1]))
            fig.canvas.flush_events()

        thisloop_actual_modes.append(mode)

    if Search_best_Mode:
        # in Search_best_Mode mode we return the mode that gives the best contrast
        return np.amin(thisloop_MeanDHContrast[1:]), modevector[np.argmin(thisloop_MeanDHContrast[1:])]

    else:
        # create a dictionary to save all results

        CorrectionLoopResult["nb_total_iter"] += iteration_number
        CorrectionLoopResult["Nb_iter_per_mat"].append(iteration_number)

        CorrectionLoopResult["SVDmodes"].append(thisloop_actual_modes)

        CorrectionLoopResult["voltage_DMs"].extend(thisloop_voltages_DMs)
        CorrectionLoopResult["FP_Intensities"].extend(thisloop_FP_Intensities)
        if nb_photons > 1:
            CorrectionLoopResult["FP_Intensities_phot"].extend(thisloop_FP_Intensities_phot)
        CorrectionLoopResult["EF_estim"].extend(thisloop_EF_estim)
        CorrectionLoopResult["MeanDHContrast"].extend(thisloop_MeanDHContrast)

        if not silence:
            plt.close()
            plt.ioff()

        if Linesearch:
            if not silence:
                print("Linesearch Mode. In this loop, we found the following modes to dig the contrast best:")
                print(thisloop_actual_modes)
                print("")

        return CorrectionLoopResult


def save_loop_results(CorrectionLoopResult, config, testbed: Testbed, MaskScience, result_dir, silence=False):
    """Save the results from a correction loop into the directory 'result_dir'.

    All fits files have all parameters in their header. The configfile is also saved, to an .ini file.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    CorrectionLoopResult : dict
        Dictionary containing the results from correction_loop_1matrix() or correction_loop().
    config : dict
        Dictionary holding the configuration from the input parameter file.
    testbed : OpticalSystem
        An OpticalSystem object which describes your testbed.
    MaskScience : 2d numpy array
        Binary array of size [dimScience, dimScience]: dark hole mask.
    result_dir : string
        Directory to save the results to.
    silence : boolean, default False.
        Whether to silence print outputs.
    """

    if not os.path.exists(result_dir):
        if not silence:
            print("Creating directory " + result_dir)
        os.makedirs(result_dir)

    FP_Intensities = CorrectionLoopResult["FP_Intensities"]
    meancontrast = CorrectionLoopResult["MeanDHContrast"]
    voltage_DMs = CorrectionLoopResult["voltage_DMs"]
    nb_total_iter = CorrectionLoopResult["nb_total_iter"]
    EF_estim = CorrectionLoopResult["EF_estim"]

    # SAVING...
    header = from_param_to_header(config)

    fits.writeto(os.path.join(result_dir, "FocalPlane_Intensities.fits"),
                 np.array(FP_Intensities),
                 header,
                 overwrite=True)

    if config["SIMUconfig"]["nb_photons"] > 1:
        FP_Intensities_phot = CorrectionLoopResult["FP_Intensities_phot"]
        fits.writeto(os.path.join(result_dir, "FocalPlane_Intensities_photnoise.fits"),
                     np.array(FP_Intensities_phot),
                     header,
                     overwrite=True)

    print("Final contrast in DH: ", meancontrast[-1])

    fits.writeto(os.path.join(result_dir, "Mean_Contrast_DH.fits"), np.array(meancontrast), header, overwrite=True)

    fits.writeto(os.path.join(result_dir, "estimationFP_RE.fits"), np.real(np.array(EF_estim)), header, overwrite=True)

    fits.writeto(os.path.join(result_dir, "estimationFP_IM.fits"), np.imag(np.array(EF_estim)), header, overwrite=True)

    voltage_DMs_nparray = np.zeros((nb_total_iter, testbed.number_act))

    DM_phases = np.zeros((len(testbed.name_of_DMs), nb_total_iter, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil))

    for i in range(len(voltage_DMs)):
        allDMphases = testbed.voltage_to_phases(voltage_DMs[i])

        if isinstance(voltage_DMs[i], (int, float)):
            voltage_DMs_nparray[i, :] += float(voltage_DMs[i])
        else:
            voltage_DMs_nparray[i, :] = voltage_DMs[i]

        for j, DM_name in enumerate(testbed.name_of_DMs):
            DM_phases[j, i, :, :] = allDMphases[j]

    DMstrokes = DM_phases * testbed.wavelength_0 / (2 * np.pi * 1e-9) / 2

    plt.figure()

    for j, DM_name in enumerate(testbed.name_of_DMs):
        DM: DeformableMirror = vars(testbed)[DM_name]
        if DM.active:

            fits.writeto(os.path.join(result_dir, f"{DM_name}_phases.fits"), DM_phases[j], header, overwrite=True)

            fits.writeto(os.path.join(result_dir, f"{DM_name}_strokes.fits"), DMstrokes[j], header, overwrite=True)

            voltage_DMs_tosave = testbed.testbed_voltage_to_indiv_DM_voltage(voltage_DMs_nparray, DM_name)

            fits.writeto(os.path.join(result_dir, f"{DM_name}_voltages.fits"),
                         voltage_DMs_tosave,
                         header,
                         overwrite=True)

            plt.plot(np.std(DMstrokes[j], axis=(1, 2)), label=DM_name + " RMS")
            plt.plot(np.max(DMstrokes[j], axis=(1, 2)) - np.min(DMstrokes[j], axis=(1, 2)), label=DM_name + " PV")

    plt.xlabel("Number of iterations")
    plt.ylabel("DM Strokes (nm)")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "DM_Strokes" + ".pdf"))
    plt.close()

    config.filename = os.path.join(result_dir, "Simulation_parameters.ini")
    config.write()

    plt.figure()
    plt.plot(meancontrast)
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean contrast in Dark Hole")
    plt.savefig(os.path.join(result_dir, "Mean_Contrast_DH.pdf"))
    plt.close()
    plot_contrast_curves(np.asarray(FP_Intensities),
                         delta_raddii=3,
                         numberofpix_per_loD=config["modelconfig"]["Science_sampling"],
                         type_of_contrast='mean',
                         mask_DH=MaskScience,
                         path=result_dir)
