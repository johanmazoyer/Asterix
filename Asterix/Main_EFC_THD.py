__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'
# pylint: disable=invalid-name

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from configobj import ConfigObj
from validate import Validator

import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.WSC_functions as wsc

import Asterix.Optical_System_functions as OptSy
import Asterix.fits_functions as useful

from Asterix.MaskDH import MaskDH
from Asterix.estimator import Estimator
from Asterix.corrector import Corrector

__all__ = ["create_interaction_matrices", "correctionLoop"]

#######################################################
#######################################################
######## Interaction/control matrices for PW and EFC


def create_interaction_matrices(parameter_file,
                                NewMODELconfig={},
                                NewDMconfig={},
                                NewCoronaconfig={},
                                NewEstimationconfig={},
                                NewCorrectionconfig={},
                                NewSIMUconfig={}):

    ### CONFIGURATION FILE
    configspec_file = os.path.join(OptSy.Asterix_root, "Param_configspec.ini")
    config = ConfigObj(parameter_file,
                       configspec=configspec_file,
                       default_encoding="utf8")
    _ = config.validate(Validator(), copy=True)

    if not os.path.exists(parameter_file):
        raise Exception("The parameter file " + parameter_file +
                        " cannot be found")

    if not os.path.exists(configspec_file):
        raise Exception("The parameter config file " + configspec_file +
                        " cannot be found")

    ### CONFIG
    Data_dir = config["Data_dir"]

    ##################
    ##################
    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

    #On bench or numerical simulation
    onbench = modelconfig["onbench"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)
    DM1_creating_pushact = DMconfig["DM1_creating_pushact"]
    DM3_creating_pushact = DMconfig["DM3_creating_pushact"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ##################
    ##################
    ### PW CONFIG
    Estimationconfig = config["Estimationconfig"]
    Estimationconfig.update(NewEstimationconfig)

    ##################
    ##################
    ###EFC CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)

    ##############################################################################
    ### Initialization all the directories
    ##############################################################################

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)

    intermatrix_dir = os.path.join(Data_dir,
                                   "Interaction_Matrices") + os.path.sep
    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    if onbench == True:
        Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    # Initialize thd:
    pup_round = OptSy.pupil(modelconfig)
    DM1 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  load_fits=not DM1_creating_pushact,
                                  save_fits=DM3_creating_pushact,
                                  Name_DM='DM1',
                                  Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  load_fits=not DM3_creating_pushact,
                                  save_fits=DM3_creating_pushact,
                                  Name_DM='DM3',
                                  Model_local_dir=Model_local_dir)

    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                         ["entrancepupil", "DM1", "DM3", "corono"])

    # initialize the estimator
    estim = Estimator(Estimationconfig,
                      thd2,
                      matrix_dir=intermatrix_dir,
                      save_for_bench=onbench,
                      realtestbed_dir=Labview_dir)

    #initalize the DH masks
    mask_dh = MaskDH(Correctionconfig)

    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       thd2,
                       mask_dh,
                       estim,
                       matrix_dir=intermatrix_dir,
                       save_for_bench=onbench,
                       realtestbed_dir=Labview_dir)


#######################################################
#######################################################
######## Simulation of a correction loop


def correctionLoop(parameter_file,
                   NewMODELconfig={},
                   NewDMconfig={},
                   NewCoronaconfig={},
                   NewEstimationconfig={},
                   NewCorrectionconfig={},
                   NewSIMUconfig={}):


    ### CONFIGURATION FILE
    configspec_file = OptSy.Asterix_root + os.path.sep + "Param_configspec.ini"
    config = ConfigObj(parameter_file,
                       configspec=configspec_file,
                       default_encoding="utf8")
    _ = config.validate(Validator(), copy=True)
    # copy=True for copying the comments

    if not os.path.exists(parameter_file):
        raise Exception("The parameter file " + parameter_file +
                        " cannot be found")

    if not os.path.exists(configspec_file):
        raise Exception("The parameter config file " + configspec_file +
                        " cannot be found")

    ### CONFIG
    Data_dir = config["Data_dir"]

    ##################
    ##################
    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

    #Lambda over D in pixels
    wavelength_0 = modelconfig["wavelength_0"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM1_misregistration = DMconfig["DM1_misregistration"]
    DM3_misregistration = DMconfig["DM3_misregistration"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ##################
    ##################
    ### Estimation CONFIG
    Estimationconfig = config["Estimationconfig"]
    Estimationconfig.update(NewEstimationconfig)

    ##################
    ##################
    ###EFC CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)

    ##################
    ##################
    ###SIMU CONFIG
    SIMUconfig = config["SIMUconfig"]
    SIMUconfig.update(NewSIMUconfig)
    Name_Experiment = SIMUconfig["Name_Experiment"]
    set_amplitude_abb = SIMUconfig["set_amplitude_abb"]
    ampl_abb_filename = SIMUconfig["ampl_abb_filename"]
    set_random_ampl = SIMUconfig["set_random_ampl"]
    ampl_rms = SIMUconfig["ampl_rms"]
    ampl_rhoc = SIMUconfig["ampl_rhoc"]
    ampl_slope = SIMUconfig["ampl_slope"]
    set_phase_abb = SIMUconfig["set_phase_abb"]
    set_random_phase = SIMUconfig["set_random_phase"]
    phase_rms = SIMUconfig["phase_rms"]
    phase_rhoc = SIMUconfig["phase_rhoc"]
    phase_slope = SIMUconfig["phase_slope"]
    phase_abb_filename = SIMUconfig["phase_abb_filename"]
    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]
    correction_algorithm = SIMUconfig["correction_algorithm"]
    Linearization = SIMUconfig["Linearization"]
    Nbiter_corr = [int(i) for i in SIMUconfig["Nbiter_corr"]]
    Nbmode_corr = [int(i) for i in SIMUconfig["Nbmode_corr"]]
    Linesearch = SIMUconfig["Linesearch"]
    Linesearchmode = SIMUconfig["Linesearchmode"]
    Linesearchmode = [int(i) for i in Linesearchmode]
    gain = SIMUconfig["gain"]

    ##THEN DO

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]
    Asterix_model_dir = os.path.join(OptSy.Asterix_root, "Model") + os.path.sep

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep

    # Initialize thd:
    pup_round = OptSy.pupil(modelconfig)
    DM1 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  Name_DM='DM1',
                                  Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  Name_DM='DM3',
                                  Model_local_dir=Model_local_dir)

    # we also need to "clear" the apod plane because the THD2 is like that
    Coronaconfig.update({'filename_instr_apod': "ClearPlane"})
    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                         ["entrancepupil", "DM1", "DM3", "corono"])

    # TODO Because the code is currently setup heavily on the
    # 'default testbed' beeing thd2 having those elements, VS code thinks
    # there is an error if it thinks they are not defined (although in practice
    # they are).
    thd2.entrancepupil = pup_round
    thd2.DM1 = DM1
    thd2.DM3 = DM3
    # In practice this is done inside the Testbed initialization already !
    # and these lines are useless and I only put them to calm
    # To be removed when the correction and estimation class are done
    # in which case these class will be able when these things exists or not

    result_dir = os.path.join(Data_dir, "Results",
                              Name_Experiment) + os.path.sep
    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)


    intermatrix_dir = os.path.join(Data_dir,
                                   "Interaction_Matrices") + os.path.sep

    ## Initialize Estimation
    estim = Estimator(Estimationconfig, thd2, matrix_dir=intermatrix_dir)

    #initalize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    MaskEstim = mask_dh.creatingMaskDH(estim.dimEstim, estim.Estim_sampling)
    MaskScience = mask_dh.creatingMaskDH(thd2.dimScience,
                                         thd2.Science_sampling)


    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       thd2,
                       mask_dh,
                       estim,
                       matrix_dir=intermatrix_dir)


    ## Phase map and amplitude map for the static aberrations
    if set_phase_abb == True:
        if phase_abb_filename == '':
            phase_abb_filename = "phase_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                int(phase_rms * 1e9), int(phase_slope), phase_rhoc, thd2.prad)

        if set_random_phase is False and os.path.isfile(Model_local_dir +
                                                        phase_abb_filename +
                                                        ".fits") == True:
            phase_up = fits.getdata(Model_local_dir + phase_abb_filename +
                                    ".fits")

        else:
            phase_up = thd2.entrancepupil.random_phase_map(
                phase_rms, phase_rhoc, phase_slope)
            if set_random_phase is False:  # save it for next time
                fits.writeto(Model_local_dir + phase_abb_filename + ".fits",
                             phase_up,
                             overwrite=True)

        phase_up = phase_up * 2 * np.pi / wavelength_0  # where should we do that ? here ?
    else:
        phase_up = 0

    if set_amplitude_abb == True:
        if ampl_abb_filename != '' and os.path.isfile(
                Model_local_dir + ampl_abb_filename +
                ".fits") == True and set_random_ampl is False:
            ampfinal = phase_ampl.scale_amplitude_abb(
                Asterix_model_dir + ampl_abb_filename + ".fits", thd2.prad,
                thd2.entrancepupil.pup)
        else:
            ampl_abb_filename = "ampl_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                int(ampl_rms), int(ampl_slope), ampl_rhoc, thd2.prad)

            if set_random_ampl is False and os.path.isfile(Model_local_dir +
                                                           ampl_abb_filename +
                                                           ".fits") == True:
                ampfinal = fits.getdata(Model_local_dir + ampl_abb_filename +
                                        ".fits")
            else:
                ampfinal = thd2.entrancepupil.random_phase_map(
                    ampl_rms / 100., ampl_rhoc, ampl_slope)
            if set_random_ampl is False:  # save it for next time
                fits.writeto(Model_local_dir + ampl_abb_filename + ".fits",
                             ampfinal,
                             overwrite=True)
    else:
        ampfinal = 0

    amplitude_abb_up = ampfinal
    phase_abb_up = phase_up

    # ## To convert in photon flux
    # contrast_to_photons = 1 / thd2.transmission(
    # ) * nb_photons * thd2.maxPSF / thd2.sumPSF

    ## Adding error on the DM model?
    if DM3_misregistration == True:
        print("DM3 Misregistration!")
        pushactonDM3 = thd2.DM3.creatingpushact(DMconfig,
                                                load_fits=False,
                                                save_fits=False)
    else:
        pushactonDM3 = thd2.DM3.DM_pushact

    # not really defined for DM1 but it is there.
    if thd2.DM1.active == True:
        if DM1_misregistration == True:
            print("DM1 Misregistration!")
            pushactonDM1 = thd2.DM1.creatingpushact(DMconfig,
                                                    load_fits=False,
                                                    save_fits=False)
        else:
            pushactonDM1 = thd2.DM1.DM_pushact

    ## Correction loop
    nbiter = len(modevector)
    imagedetector = np.zeros((nbiter + 1, thd2.dimScience, thd2.dimScience))

    phaseDM3 = np.zeros(
        (nbiter + 1, thd2.dim_overpad_pupil, thd2.dim_overpad_pupil))
    phaseDM1 = np.zeros(
        (nbiter + 1, thd2.dim_overpad_pupil, thd2.dim_overpad_pupil))
    meancontrast = np.zeros(nbiter + 1)

    voltage_DM1 = [0.]  # initialize with no voltage
    voltage_DM3 = [0.]  # initialize with no voltage

    # Initial wavefront in pupil plane
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=amplitude_abb_up)
    imagedetector[0] = thd2.todetector_Intensity(entrance_EF=input_wavefront)

    if photon_noise == True:
        photondetector = np.zeros(
            (nbiter + 1, thd2.dimScience, thd2.dimScience))
        photondetector[0] = np.random.poisson(imagedetector[0] *
                                              thd2.normPupto1 * nb_photons)

    meancontrast[0] = np.mean(imagedetector[0][np.where(MaskScience != 0)])
    print("Mean contrast in DH: ", meancontrast[0])

    plt.ion()
    plt.figure()
    previousmode = modevector[0]
    k = 0
    for mode in modevector:
        print("--------------------------------------------------")
        print("Iteration number: ", k, " EFC truncation: ", mode)

        resultatestimation = estim.estimate(thd2,
                                            entrance_EF=input_wavefront,
                                            DM1phase=phaseDM1[k],
                                            DM3phase=phaseDM3[k],
                                            wavelength=thd2.wavelength_0,
                                            photon_noise=photon_noise,
                                            nb_photons=nb_photons)

        # Find the solution of actuator motions to apply
        if correction_algorithm == "EFC":

            if Linearization == True:

                # Calculate the control matrix for the current aberrations
                # (needed because of linearization of the problem?)
                if thd2.DM1.active == True:
                    correc.Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        thd2,
                        estim.dimEstim,
                        np.concatenate(
                            (thd2.DM3.DM_pushact, thd2.DM1.DM_pushact_inpup)) *
                        correc.amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0,
                        MaskEstim,
                        np.concatenate(
                            (thd2.DM3.WhichInPupil,
                             thd2.DM3.number_act + thd2.DM1.WhichInPupil)),
                        otherbasis=correc.DM3_otherbasis,
                        basisDM3=correc.DM3_basis)
                else:
                    correc.Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        thd2,
                        estim.dimEstim,
                        thd2.DM3.DM_pushact * correc.amplitudeEFC * 2 * np.pi * 1e-9 /
                        wavelength_0,
                        MaskEstim,
                        thd2.DM3.WhichInPupil,
                        otherbasis=correc.DM3_otherbasis,
                        basisDM3=correc.DM3_basis)

            # Use the control matrix simulated for a null input aberration
            if Linesearch is False:
                if mode != previousmode or k == 0:
                    invertGDH = wsc.invertSVD(
                        correc.Gmatrix,
                        mode,
                        goal="c",
                        visu=False,
                        regul=correc.regularization,
                        otherbasis=correc.DM3_otherbasis,
                        basisDM3=correc.DM3_basis,
                    )[2]

            else:
                # Look for the best number of modes for the control matrix
                # (uses the interaction matrix calculated for a null input aberration)
                meancontrasttemp = np.zeros(len(Linesearchmode))
                b = 0
                for mode in Linesearchmode:

                    SVD, SVD_trunc, invertGDH = wsc.invertSVD(
                        correc.Gmatrix,
                        mode,
                        goal="c",
                        visu=False,
                        regul=correc.regularization,
                        otherbasis=correc.DM3_otherbasis,
                        basisDM3=correc.DM3_basis,
                    )
                    if thd2.DM1.active == True:

                        solution1 = wsc.solutionEFC(
                            MaskEstim, resultatestimation, invertGDH,
                            np.concatenate(
                                (thd2.DM3.WhichInPupil,
                                 thd2.DM3.number_act + thd2.DM1.WhichInPupil)),
                            thd2.DM3.number_act + thd2.DM1.number_act)

                        apply_on_DM1 = solution1[DM3.number_act:] * (
                            -gain * correc.amplitudeEFC)
                        voltage_DM1_tmp = voltage_DM1[k] + apply_on_DM1
                        phaseDM1_tmp = thd2.DM1.voltage_to_phase(
                            voltage_DM1_tmp, wavelength=thd2.wavelength_0)

                    else:
                        solution1 = wsc.solutionEFC(MaskEstim,
                                                    resultatestimation,
                                                    invertGDH,
                                                    thd2.DM3.WhichInPupil,
                                                    thd2.DM3.number_act)

                        phaseDM1_tmp = 0.

                    # Phase to apply on DM3

                    apply_on_DM3 = solution1[0:DM3.number_act] * (-gain *
                                                                  correc.amplitudeEFC)
                    # Phase to apply on DM3
                    voltage_DM3_tmp = voltage_DM3[k] + apply_on_DM3
                    phaseDM3_tmp = thd2.DM3.voltage_to_phase(
                        voltage_DM3_tmp, wavelength=thd2.wavelength_0)

                    imagedetectortemp = thd2.todetector_Intensity(
                        entrance_EF=input_wavefront,
                        DM1phase=phaseDM1_tmp,
                        DM3phase=phaseDM3_tmp)

                    meancontrasttemp[b] = np.mean(
                        imagedetectortemp[np.where(MaskScience != 0)])

                    print('contraste moyen avec regul ', mode, '=',
                          meancontrasttemp[b])

                    b = b + 1

                bestcontrast = np.amin(meancontrasttemp)
                bestregul = Linesearchmode[np.argmin(meancontrasttemp)]
                print('Meilleur contraste= ', bestcontrast, ' Best regul= ',
                      bestregul)

                invertGDH = wsc.invertSVD(
                    correc.Gmatrix,
                    bestregul,
                    goal="c",
                    visu=False,
                    regul=correc.regularization,
                    otherbasis=correc.DM3_otherbasis,
                    basisDM3=correc.DM3_basis,
                )[2]

            if thd2.DM1.active == True:
                solution1 = wsc.solutionEFC(
                    MaskEstim, resultatestimation, invertGDH,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil,
                         thd2.DM3.number_act + thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act + thd2.DM1.number_act)
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEFC(MaskEstim, resultatestimation,
                                            invertGDH, thd2.DM3.WhichInPupil,
                                            thd2.DM3.number_act)

        if correction_algorithm == "EM":
            if mode != previousmode or k == 0:
                invertM0 = wsc.invertSVD(
                    correc.M0,
                    mode,
                    goal="c",
                    visu=False,
                    regul=correc.regularization,
                    otherbasis=correc.DM3_otherbasis,
                    basisDM3=correc.DM3_basis,
                )[2]

            # Concatenate can be done in the THD2 structure
            if thd2.DM1.active == True:
                solution1 = wsc.solutionEM(
                    MaskEstim, resultatestimation, invertM0, correc.G,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil,
                         thd2.DM3.number_act + thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act + thd2.DM1.number_act)
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEM(MaskEstim, resultatestimation,
                                           invertM0, correc.G, thd2.DM3.WhichInPupil,
                                           thd2.DM3.number_act)

        if correction_algorithm == "steepest":
            if thd2.DM1.active == True:
                solution1 = wsc.solutionSteepest(
                    MaskEstim, resultatestimation, correc.M0, correc.G,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil,
                         thd2.DM3.number_act + thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act + thd2.DM1.number_act)
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionSteepest(MaskEstim, resultatestimation,
                                                 correc.M0, correc.G, thd2.DM3.WhichInPupil,
                                                 thd2.DM3.number_act)

        if thd2.DM1.active == True:
            # Phase to apply on DM1
            apply_on_DM1 = solution1[thd2.DM3.number_act:] * (-gain *
                                                              correc.amplitudeEFC)
            voltage_DM1.append(voltage_DM1[k] + apply_on_DM1)
            phaseDM1[k + 1] = thd2.DM1.voltage_to_phase(
                voltage_DM1[k + 1], wavelength=thd2.wavelength_0)

        apply_on_DM3 = solution1[0:thd2.DM3.number_act] * (-gain *
                                                           correc.amplitudeEFC)
        # Phase to apply on DM3
        voltage_DM3.append(voltage_DM3[k] + apply_on_DM3)
        phaseDM3[k + 1] = thd2.DM3.voltage_to_phase(
            voltage_DM3[k + 1], wavelength=thd2.wavelength_0)

        imagedetector[k + 1] = thd2.todetector_Intensity(
            entrance_EF=input_wavefront,
            DM1phase=phaseDM1[k + 1],
            DM3phase=phaseDM3[k + 1])

        meancontrast[k + 1] = np.mean(
            imagedetector[k + 1][np.where(MaskScience != 0)])
        print("Mean contrast in DH: ", meancontrast[k + 1])

        if photon_noise == True:
            photondetector[k + 1] = np.random.poisson(
                imagedetector[k + 1] * thd2.normPupto1 * photon_noise)

        plt.clf()
        plt.imshow(np.log10(imagedetector[k + 1]), vmin=-8, vmax=-5)
        plt.colorbar()
        plt.pause(0.01)
        previousmode = mode
        k = k + 1

    plt.show()

    ## SAVING...
    header = useful.from_param_to_header(config)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_Detector_Images" + ".fits",
                 imagedetector,
                 header,
                 overwrite=True)
    if thd2.DM1.active == True:
        fits.writeto(result_dir + current_time_str + "_Phase_on_DM1" + ".fits",
                     phaseDM1,
                     header,
                     overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Phase_on_DM3" + ".fits",
                 phaseDM3,
                 header,
                 overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Mean_Contrast_DH" + ".fits",
                 meancontrast,
                 header,
                 overwrite=True)
    config.filename = result_dir + current_time_str + "_Simulation_parameters" + ".ini"
    config.write()

    if photon_noise == True:
        fits.writeto(result_dir + current_time_str + "_Photon_counting" +
                     ".fits",
                     photondetector,
                     header,
                     overwrite=True)

    plt.clf()
    plt.plot(meancontrast)
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean contrast in Dark Hole")

    return input_wavefront, imagedetector