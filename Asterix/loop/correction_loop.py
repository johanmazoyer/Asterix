# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from Asterix.optics import Testbed
from Asterix.wfsc import Corrector, Estimator, MaskDH


def correction_loop(testbed: Testbed,
                    estimator: Estimator,
                    corrector: Corrector,
                    mask_dh: MaskDH,
                    Loopconfig,
                    SIMUconfig,
                    input_wavefront=0,
                    initial_DM_voltage=0.,
                    silence=False,
                    **kwargs):
    """ --------------------------------------------------
    Run a full loop for several Matrix. at each iteration, we update the matrix and
    run correction_loop_1matrix().

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
        
    testbed: OpticalSystem.Testbed
            object which describes your testbed
    
    estimator: Estimator 
            This contains all information about the estimation
    
    corrector: Corrector. 
            This contains all information about the correction
    
    mask_dh: 2d numpy array
        binary array of size [dimScience, dimScience] : dark hole mask
    
    Loopconfig: dict
            simulation parameters containing the loop parameters

    SIMUconfig: dict
            simulation parameters containing the Simulation parameters
    
    input_wavefront: float or 2d complex array or 3d complex array
    initial wavefront at the beginning of this loop.
        Electrical Field which can be a :
            float 1. if no phase / amplitude (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic   
        
        !!CAREFUL!!: right now we do not use this wf to measure the matrix, although update_matrices
                    function allows it. Currently each matrix is measured with a flat field in 
                    entrance of the testbed (input_wavefront = 1). 
                    input_wavefront is only used in the loop once the matrix is calcultated   
                    This can be changed but be careful.             

    initial_DM_voltage: float or 1D array
            initial DM voltages at the beginning of this loop. The Matrix is measured 
            using this initial DM voltages. Can be:
            float 0 if flat DMs (default)
            or 1D array of size testbed.number_act

    
    silence=False: Boolean, default False
                if False, print and plot results as the loop runs

    Returns
    ------
    CorrectionLoopResult : dict
                a dictionnary containing the results of all loops

    
    -------------------------------------------------- """

    CorrectionLoopResult = dict()
    CorrectionLoopResult["nb_total_iter"] = 0
    CorrectionLoopResult["Nb_iter_per_mat"] = list()
    CorrectionLoopResult["voltage_DMs"] = list()
    CorrectionLoopResult["FP_Intensities"] = list()
    CorrectionLoopResult["EF_estim"] = list()
    CorrectionLoopResult["MeanDHContrast"] = list()
    CorrectionLoopResult["SVDmodes"] = list()

    # reading the simulation parameter files
    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]

    # reading the loop parameter files
    Nbiter_corr = list(Loopconfig["Nbiter_corr"])
    Number_matrix = Loopconfig["Number_matrix"]

    Nbmode_corr = []
    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
        Linesearch = Loopconfig["Linesearch"]
        gain = Loopconfig["gain"]
        if Linesearch == False:
            Nbmode_corr = list(Loopconfig["Nbmode_corr"])
            if len(Nbiter_corr) != len(Nbmode_corr):
                raise Exception("""In this correction mode and if Linesearch = False, 
                the length of Nbmode_corr must match the length of Nbiter_corr""")

    else:
        Linesearch = None
        gain = 1.

    if corrector.correction_algorithm == 'sm':
        Linesearch = False

    for i in range(Number_matrix):

        if i > 0:
            # the first matrix is done during initialization
            corrector.update_matrices(testbed, estimator, initial_DM_voltage=initial_DM_voltage)

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
                                                            photon_noise=photon_noise,
                                                            nb_photons=nb_photons,
                                                            silence=silence,
                                                            **kwargs)

        min_contrast = min(CorrectionLoopResult["MeanDHContrast"])
        min_index = CorrectionLoopResult["MeanDHContrast"].index(min_contrast)
        initial_DM_voltage = Resultats_correction_loop["voltage_DMs"][min_index]

        if i != Number_matrix - 1:
            print("end Matrix ", i)
            print("We will restart next matrix from contrast = ", min_contrast)
        else:
            print("End correction")

    return Resultats_correction_loop


def correction_loop_1matrix(testbed: Testbed,
                            estimator: Estimator,
                            corrector: Corrector,
                            mask_dh: MaskDH,
                            Nbiter_corr,
                            CorrectionLoopResult,
                            gain=0.1,
                            Nbmode_corr=None,
                            Linesearch=False,
                            Search_best_Mode=False,
                            input_wavefront=1.,
                            initial_DM_voltage=0.,
                            silence=False,
                            **kwargs):
    """ --------------------------------------------------
    Run a loop for a given interaction matrix.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
        
    testbed: OpticalSystem.Testbed
            object which describes your testbed
    
    estimator: Estimator 
            This contains all information about the estimation
    
    corrector: Corrector. 
            This contains all information about the correction
    
    mask_dh: 2d numpy array
        binary array of size [dimScience, dimScience] : dark hole mask
    
    Nbiter_corr: int or list of int
            number of iterations in the loop
    
    CorrectionLoopResult: dict
                dictionnary containing the result of the previous loop. 
                This will be updated with the result of this loop
    
    gain:  float between 0 and 1, default 0.1
            gain of the loop in EFC mode. 
            
    
    Nbmode_corr: int or list of int of same size as Nbiter_corr,
                    SVD modes for each iteration
    
    Linesearch: bool, default False. 
                If True, In this mode, the function correction_loop_1matrix()
                will call itself at each iteration with Search_best_Mode= True to find 
                the best SVD inversion mode among a few Linesearchmodes. 

    Search_best_Mode: bool, default False. 
                    If true, the algorithm does not return the
                    loop information, just the best mode and best contrast. 
                    This mode is used in Linesearch mode
                    Be careful when using this parameter, it can create an infinite loop
    
    input_wavefront: float or 2d complex array or 3d complex array
        initial wavefront at the beginning of this loop.
        Electrical Field which can be a :
            float 1. if no phase / amplitude (default)
            2D complex array, of size phase_abb.shape if monochromatic
            or 3D complex array of size [self.nb_wav,phase_abb.shape] if polychromatic                   

    initial_DM_voltage: float or 1D array
            initial DM voltages at the beginning of this loop. The Matrix is measured 
            using this initial DM voltages. Can be:
            float 0 if flat DMs (default)
            or 1D array of size testbed.number_act
            
    silence: Boolean, default False
                if False, print and plot results as the loop runs

    Returns
    ------
    if Search_best_Mode == True, return [bestMode, bestContrast]
    else return CorrectionLoopResult dictionnary updated with the results from this loop
    

    -------------------------------------------------- """

    if Search_best_Mode:
        # This is to prevent an infinite loop
        Linesearch = False

    thisloop_expected_iteration_number = sum(Nbiter_corr)

    ## Number of modes that is used as a function of the iteration cardinal
    # in the EFC case

    if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
        modevector = []
        if not Linesearch:
            for i in np.arange(len(Nbiter_corr)):
                modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    initialFP = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                             voltage_vector=initial_DM_voltage,
                                             save_all_planes_to_fits=False,
                                             dir_save_all_planes='/Users/jmazoyer/Desktop/test/',
                                             **kwargs)

    estim_init = estimator.estimate(testbed,
                                    voltage_vector=initial_DM_voltage,
                                    entrance_EF=input_wavefront,
                                    wavelength=testbed.wavelength_0,
                                    **kwargs)

    initialFP_contrast = np.mean(initialFP[np.where(mask_dh != 0)])

    thisloop_voltages_DMs = list()
    thisloop_FP_Intensities = list()
    thisloop_MeanDHContrast = list()
    thisloop_EF_estim = list()
    thisloop_actual_modes = list()

    thisloop_voltages_DMs.append(initial_DM_voltage)
    thisloop_FP_Intensities.append(initialFP)
    thisloop_EF_estim.append(estim_init)
    thisloop_MeanDHContrast.append(initialFP_contrast)

    if not silence:
        print("Initial contrast in DH: ", initialFP_contrast)
        plt.ion()
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        im = ax.imshow(np.log10(initialFP), vmin=-8, vmax=-5)
        plt.gca().invert_yaxis()
        ax.figure.colorbar(im)
        plt.pause(0.0001)

    # we start at 1 because we count the initial state as an iteration of the loop
    iteration_number = 1

    for iteration in range(thisloop_expected_iteration_number):

        if corrector.correction_algorithm in ['efc', 'em', 'steepest']:

            if Linesearch:

                # we search the best cutoff mode among 10 different ones evenly separated
                Linesearchmodes = 10 * (np.arange(0.2, 0.6, 0.03) * corrector.total_number_modes /
                                        10).astype(int)

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
                                                                 input_wavefront=input_wavefront,
                                                                 initial_DM_voltage=thisloop_voltages_DMs[iteration],
                                                                 silence=True,
                                                                 **kwargs)

                print("Best Mode is ", bestmode, " with contrast: ", bestcontrast)
                mode = bestmode
            else:
                mode = modevector[iteration]

            if mode > corrector.total_number_modes:
                if not Search_best_Mode:
                    print("You cannot use a cutoff mode ({:d}) larger than the total size of basis ({:d})".
                          format(mode, corrector.Gmatrix.shape[1]))
                    print("We skip this iteration")
                continue

        else:
            mode = 1

        if not silence:
            if corrector.correction_algorithm in ['efc', 'em', 'steepest']:
                print("Iteration number " + corrector.correction_algorithm + ": ", iteration + 1,
                      " SVD truncation: ", mode)
            else:
                print("Iteration number " + corrector.correction_algorithm + ": ", iteration + 1)

        # for now monochromatic estimation
        resultatestimation = estimator.estimate(testbed,
                                                voltage_vector=thisloop_voltages_DMs[-1],
                                                entrance_EF=input_wavefront,
                                                wavelength=testbed.wavelength_0,
                                                perfect_estimation=Search_best_Mode,
                                                **kwargs)

        solution = corrector.toDM_voltage(testbed,
                                          resultatestimation,
                                          mode=mode,
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
            new_voltage = thisloop_voltages_DMs[-1] + gain * solution

        thisloop_FP_Intensities.append(
            testbed.todetector_intensity(entrance_EF=input_wavefront,
                                         voltage_vector=new_voltage,
                                         save_all_planes_to_fits=False,
                                         dir_save_all_planes='/Users/jmazoyer/Desktop/test_roundpup/',
                                         **kwargs))
        thisloop_EF_estim.append(resultatestimation)
        thisloop_MeanDHContrast.append(np.mean(thisloop_FP_Intensities[-1][np.where(mask_dh != 0)]))

        if Search_best_Mode == False:
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
        # create a dictionnary to save all results

        CorrectionLoopResult["nb_total_iter"] += iteration_number
        CorrectionLoopResult["Nb_iter_per_mat"].append(iteration_number)

        CorrectionLoopResult["SVDmodes"].append(thisloop_actual_modes)

        CorrectionLoopResult["voltage_DMs"].extend(thisloop_voltages_DMs)
        CorrectionLoopResult["FP_Intensities"].extend(thisloop_FP_Intensities)
        CorrectionLoopResult["EF_estim"].extend(thisloop_EF_estim)
        CorrectionLoopResult["MeanDHContrast"].extend(thisloop_MeanDHContrast)

        if not silence:
            plt.close()
            plt.ioff()

        if Linesearch:
            print("Linesearch Mode. In this loop, we found the following modes to dig the contrast best:")
            print(thisloop_actual_modes)
            print("")

        return CorrectionLoopResult
