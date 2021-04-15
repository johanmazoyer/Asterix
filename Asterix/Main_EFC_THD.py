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
                                  Name_DM='DM1',
                                  Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
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

    Nbiter_corr = [int(i) for i in Correctionconfig["Nbiter_corr"]]
    Nbmode_corr = [int(i) for i in Correctionconfig["Nbmode_corr"]]
    Linesearch = Correctionconfig["Linesearch"]
    Linesearchmode = [int(i) for i in Correctionconfig["Linesearchmode"]]
    gain = Correctionconfig["gain"]

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


    ##############################################################################
    ### Initialization all the directories
    ##############################################################################
    Asterix_model_dir = os.path.join(OptSy.Asterix_root, "Model") + os.path.sep


    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)

    intermatrix_dir = os.path.join(Data_dir,
                                   "Interaction_Matrices") + os.path.sep
    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    result_dir = os.path.join(Data_dir, "Results",
                              Name_Experiment) + os.path.sep
    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

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



    ## Initialize Estimation
    estim = Estimator(Estimationconfig, thd2, matrix_dir=intermatrix_dir)

    #initalize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    MaskScience = mask_dh.creatingMaskDH(thd2.dimScience,
                                         thd2.Science_sampling)


    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       thd2,
                       mask_dh,
                       estim,
                       matrix_dir=intermatrix_dir)


    ## Adding error on the DM model. Now that the matrix is measured, we can
    # introduce a small movememnt on one DM or the other. By changeing DM_pushact
    # we are changeing the position of the actuator and therre the phase of the
    # DM for a given voltage when using DM.voltage_to_phase

    # TODO can be automatized in a for loop over all DMs nad maybe put in
    # correction config
    if thd2.DM3.misregistration == True:
        print("DM3 Misregistration!")
        thd2.DM3.DM_pushact = thd2.DM3.creatingpushact(DMconfig)

    if thd2.DM1.misregistration == True:
        print("DM1 Misregistration!")
        thd2.DM1.DM_pushact = thd2.DM1.creatingpushact(DMconfig)



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



    ## Correction loop

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

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
    for iteration, mode in enumerate(modevector):
        print("--------------------------------------------------")
        print("Iteration number: ", iteration, " SVD truncation: ", mode)

        resultatestimation = estim.estimate(thd2,
                                            entrance_EF=input_wavefront,
                                            DM1phase=phaseDM1[iteration],
                                            DM3phase=phaseDM3[iteration],
                                            wavelength=thd2.wavelength_0,
                                            photon_noise=photon_noise,
                                            nb_photons=nb_photons)

        if Linesearch is True:
            search_best_contrast = list()
            perfectestimate = estim.estimate(thd2,
                            entrance_EF=input_wavefront,
                            DM1phase=phaseDM1[iteration],
                            DM3phase=phaseDM3[iteration],
                            wavelength=thd2.wavelength_0,
                            perfect_estimation= True)

            for modeLinesearch in Linesearchmode:

                perfectsolution = - gain * correc.amplitudeEFC* correc.toDM_voltage(thd2, perfectestimate, modeLinesearch)

                #### to be removed #### #### #### #### #### #### #### #### ####

                if thd2.DM1.active == True:
                    separation_DM1_DM3 = len(thd2.DM1.WhichInPupil)

                    # Phase to apply on DM1
                    apply_on_DM1 = perfectsolution[:thd2.DM1.number_act]
                    phaseDM1_tmp = thd2.DM1.voltage_to_phase(
                        voltage_DM1[iteration] + apply_on_DM1, wavelength=thd2.wavelength_0)
                else:
                    separation_DM1_DM3 = 0
                    phaseDM1_tmp = 0

                apply_on_DM3 = perfectsolution[separation_DM1_DM3: separation_DM1_DM3+ thd2.DM3.number_act]
                # Phase to apply on DM3
                phaseDM3_tmp = thd2.DM3.voltage_to_phase(
                    voltage_DM3[iteration] + apply_on_DM3, wavelength=thd2.wavelength_0)
                ######## #### #### #### #### #### #### #### #### #### #### ####

                imagedetector_tmp = thd2.todetector_Intensity(
                            entrance_EF=input_wavefront,
                            DM1phase=phaseDM1_tmp,
                            DM3phase=phaseDM3_tmp)

                search_best_contrast.append(np.mean(imagedetector_tmp[np.where(MaskScience != 0)]))
            bestcontrast = np.amin(search_best_contrast)
            mode = Linesearchmode[np.argmin(search_best_contrast)]
            print('Best contrast= ', bestcontrast, ' Best regul= ', mode)



        # TODO is amplitudeEFC really useful ? with the small phase hypothesis done when
        # measuring the matrix, everything is linear !
        solution = - gain * correc.amplitudeEFC* correc.toDM_voltage(thd2, resultatestimation, mode)


        ### TODO Following lines should be removed  since we should put directly solution in
        # the testbed model
        # this step should be 3 simple lines :
        # estimation = estim.estimate(...)
        # DMVoltag += correc.toDM_voltage(estimation,...)
        # imagedetector = thd2.todetector_Intensity(DMVoltag,...)

        #### to be removed #### #### #### #### #### #### #### #### ####

        if thd2.DM1.active == True:
            # Phase to apply on DM1
            separation_DM1_DM3 = len(thd2.DM1.number_act)
            apply_on_DM1 = solution[:thd2.DM1.number_act]
            voltage_DM1.append(voltage_DM1[iteration] + apply_on_DM1)
            phaseDM1[iteration + 1] = thd2.DM1.voltage_to_phase(
                voltage_DM1[iteration + 1], wavelength=thd2.wavelength_0)
        else:
            phaseDM1[iteration + 1] = 0
            separation_DM1_DM3 = 0


        apply_on_DM3 = solution[separation_DM1_DM3: separation_DM1_DM3+ thd2.DM3.number_act]

        # Phase to apply on DM3
        voltage_DM3.append(voltage_DM3[iteration] + apply_on_DM3)
        phaseDM3[iteration + 1] = thd2.DM3.voltage_to_phase(
            voltage_DM3[iteration + 1], wavelength=thd2.wavelength_0)
        ########################################################################

        imagedetector[iteration + 1] = thd2.todetector_Intensity(
            entrance_EF=input_wavefront,
            DM1phase=phaseDM1[iteration + 1],
            DM3phase=phaseDM3[iteration + 1])

        meancontrast[iteration + 1] = np.mean(
            imagedetector[iteration + 1][np.where(MaskScience != 0)])
        print("Mean contrast in DH: ", meancontrast[iteration + 1])

        if photon_noise == True:
            photondetector[iteration + 1] = np.random.poisson(
                imagedetector[iteration + 1] * thd2.normPupto1 * photon_noise)

        plt.clf()
        plt.imshow(np.log10(imagedetector[iteration + 1]), vmin=-8, vmax=-5)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.pause(0.01)

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
