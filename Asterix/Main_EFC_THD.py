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

import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc

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

    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]

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

    result_dir = os.path.join(Data_dir, "Results",
                              Name_Experiment) + os.path.sep
    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

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

    # we also need to "clear" the apod plane because the THD2 is like that
    Coronaconfig.update({'filename_instr_apod': "ClearPlane"})
    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                         ["entrancepupil", "DM1", "DM3", "corono"])

    ## Initialize Estimation
    estim = Estimator(Estimationconfig,
                      thd2,
                      matrix_dir=intermatrix_dir,
                      save_for_bench=onbench,
                      realtestbed_dir=Labview_dir)

    #initalize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    MaskScience = mask_dh.creatingMaskDH(thd2.dimScience,
                                         thd2.Science_sampling)

    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       thd2,
                       mask_dh,
                       estim,
                       matrix_dir=intermatrix_dir,
                       save_for_bench=onbench,
                       realtestbed_dir=Labview_dir)

    # set initial phase and amplitude and wavefront
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig,
                                             Model_local_dir=Model_local_dir)
    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig,
                                           Model_local_dir=Model_local_dir)
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=ampl_abb_up)

    ## Correction loop

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    nbiter = len(modevector)
    imagedetector = np.zeros((nbiter + 1, thd2.dimScience, thd2.dimScience))

    meancontrast = np.zeros(nbiter + 1)

    voltage_DMs = [0.]  # initialize with no voltage

    # Initial wavefront in pupil plane

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

        resultatestimation = estim.estimate(
            thd2,
            entrance_EF=input_wavefront,
            voltage_vector=voltage_DMs[iteration],
            wavelength=thd2.wavelength_0,
            photon_noise=photon_noise,
            nb_photons=nb_photons)

        if Linesearch is True:
            search_best_contrast = list()
            perfectestimate = estim.estimate(thd2,
                                             entrance_EF=input_wavefront,
                                             voltage_vector=0.,
                                             wavelength=thd2.wavelength_0,
                                             perfect_estimation=True)

            for modeLinesearch in Linesearchmode:

                perfectsolution = -gain * correc.amplitudeEFC * correc.toDM_voltage(
                    thd2, perfectestimate, modeLinesearch)

                tmpvoltage_DMs = voltage_DMs[iteration] + perfectsolution

                imagedetector_tmp = thd2.todetector_Intensity(
                    entrance_EF=input_wavefront, voltage_vector=tmpvoltage_DMs)

                search_best_contrast.append(
                    np.mean(imagedetector_tmp[np.where(MaskScience != 0)]))
            bestcontrast = np.amin(search_best_contrast)
            mode = Linesearchmode[np.argmin(search_best_contrast)]
            print('Best contrast= ', bestcontrast, ' Best regul= ', mode)

        # TODO is amplitudeEFC really useful ? with the small phase hypothesis done when
        # measuring the matrix, everything is linear !
        solution = -gain * correc.amplitudeEFC * correc.toDM_voltage(
            thd2, resultatestimation, mode)

        voltage_DMs.append(voltage_DMs[iteration] + solution)

        imagedetector[iteration + 1] = thd2.todetector_Intensity(
            entrance_EF=input_wavefront,
            voltage_vector=voltage_DMs[iteration + 1])

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
