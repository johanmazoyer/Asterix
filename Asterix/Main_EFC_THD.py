__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'
# pylint: disable=invalid-name

import os


from configobj import ConfigObj
from validate import Validator

import Asterix.Optical_System_functions as OptSy

from Asterix.MaskDH import MaskDH
from Asterix.estimator import Estimator
from Asterix.corrector import Corrector
from Asterix.correction_loop import CorrectionLoop, Save_loop_results


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
    #On bench or numerical simulation
    onbench = config["onbench"]

    ##################
    ##################
    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

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

    matrix_dir = os.path.join(Data_dir,
                                   "Interaction_Matrices") + os.path.sep
    if not os.path.exists(matrix_dir):
        print("Creating directory " + matrix_dir + " ...")
        os.makedirs(matrix_dir)

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
                      matrix_dir=matrix_dir,
                      save_for_bench=onbench,
                      realtestbed_dir=Labview_dir)

    #initalize the DH masks
    mask_dh = MaskDH(Correctionconfig)

    #initalize the corrector
    correc = Corrector(Correctionconfig,
                       thd2,
                       mask_dh,
                       estim,
                       matrix_dir=matrix_dir,
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
    #On bench or numerical simulation
    onbench = config["onbench"]


    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ### Estimation CONFIG
    Estimationconfig = config["Estimationconfig"]
    Estimationconfig.update(NewEstimationconfig)

    ###EFC CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)


    ###SIMU CONFIG
    SIMUconfig = config["SIMUconfig"]
    SIMUconfig.update(NewSIMUconfig)

    Name_Experiment = SIMUconfig["Name_Experiment"]

    Nbiter_corr = [int(i) for i in SIMUconfig["Nbiter_corr"]]
    Nbmode_corr = [int(i) for i in SIMUconfig["Nbmode_corr"]]
    Linesearch = SIMUconfig["Linesearch"]
    Linesearchmodes = [int(i) for i in SIMUconfig["Linesearchmodes"]]
    gain = SIMUconfig["gain"]

    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]

    ##############################################################################
    ### Initialization all the directories
    ##############################################################################

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    matrix_dir = os.path.join(Data_dir,"Interaction_Matrices") + os.path.sep
    result_dir = os.path.join(Data_dir, "Results",Name_Experiment) + os.path.sep
    Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep


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
    # this can also be a parameter in Coronaconfig.

    corono = OptSy.coronagraph(modelconfig, Coronaconfig)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                         ["entrancepupil", "DM1", "DM3", "corono"])

    ## Initialize Estimation
    estim = Estimator(Estimationconfig,
                      thd2,
                      matrix_dir=matrix_dir,
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
                       matrix_dir=matrix_dir,
                       save_for_bench=onbench,
                       realtestbed_dir=Labview_dir)

    # set initial phase and amplitude and wavefront
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig,
                                             Model_local_dir=Model_local_dir)
    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig,
                                           Model_local_dir=Model_local_dir)
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=ampl_abb_up)

    Resultats_correction_loop = CorrectionLoop(thd2,
                                               estim,
                                               correc,
                                               MaskScience,
                                               gain,
                                               Nbiter_corr,
                                               Nbmode_corr,
                                               Linesearch= Linesearch,
                                               Linesearchmodes=Linesearchmodes,
                                               input_wavefront=input_wavefront,
                                               initial_DM_voltage=0,
                                               photon_noise=photon_noise,
                                               nb_photons=nb_photons)

    Save_loop_results(Resultats_correction_loop, config, thd2, result_dir)