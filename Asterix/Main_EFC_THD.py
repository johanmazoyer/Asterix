__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'
# pylint: disable=invalid-name

import os

import Asterix.fits_functions as useful
import Asterix.Optical_System_functions as OptSy

from Asterix.MaskDH import MaskDH
from Asterix.estimator import Estimator
from Asterix.corrector import Corrector
from Asterix.correction_loop import CorrectionLoop, Save_loop_results

__all__ = ["runthd2"]

#######################################################
#######################################################
######## Simulation of a correction loop


def runthd2(parameter_file,
            NewMODELconfig={},
            NewDMconfig={},
            NewCoronaconfig={},
            NewEstimationconfig={},
            NewCorrectionconfig={},
            NewSIMUconfig={}):

    ### CONFIGURATION FILE
    config = useful.read_parameter_file(parameter_file)

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
    matrix_dir = os.path.join(Data_dir, "Interaction_Matrices") + os.path.sep
    result_dir = os.path.join(Data_dir, "Results",
                              Name_Experiment) + os.path.sep
    Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep

    # Initialize thd:
    pup_round = OptSy.pupil(modelconfig, PupType=modelconfig['filename_instr_pup'])
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

    Resultats_correction_loop = CorrectionLoop(
                                                thd2,
                                                estim,
                                                correc,
                                                MaskScience,
                                                gain,
                                                Nbiter_corr,
                                                Nbmode_corr,
                                                Linesearch=Linesearch,
                                                Linesearchmodes=Linesearchmodes,
                                                input_wavefront=input_wavefront,
                                                initial_DM_voltage=0,
                                                photon_noise=photon_noise,
                                                nb_photons=nb_photons)

    Save_loop_results(Resultats_correction_loop, config, thd2, result_dir)