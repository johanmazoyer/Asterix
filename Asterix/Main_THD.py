__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'
# pylint: disable=invalid-name

import os

import Asterix.fits_functions as useful
import Asterix.Optical_System_functions as OptSy

from Asterix.MaskDH import MaskDH
from Asterix.estimator import Estimator
from Asterix.corrector import Corrector
from Asterix.correction_loop import CorrectionLoop, Save_loop_results

#######################################################
#######################################################
######## Simulation of a correction loop for thd2 testbed


def runthd2(parameter_file,
            NewMODELconfig={},
            NewDMconfig={},
            NewCoronaconfig={},
            NewEstimationconfig={},
            NewCorrectionconfig={},
            NewLoopconfig={},
            NewSIMUconfig={}):
    """ --------------------------------------------------
        Initialize the thd 2 testbed, the estimation method, 
        the correction method from parameter_file file
        Run the loop and save the results

        All NewXXXconfig can be used to update a single parameter in one of 
        the subsections of the parameter file. This will replace the value in the parameter file. 
        e.g. : NewCoronaconfig ={'paramXXX': YYY}

        AUTHOR : Johan Mazoyer


        Parameters
        ----------
        parameter_file: path 
            path to a .ini parameter file

        NewMODELconfig: dict
        NewDMconfig: dict
        NewCoronaconfig: dict
        NewEstimationconfig: dict
        NewCorrectionconfig: dict
        NewSIMUconfig: dict

        
        -------------------------------------------------- """

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

    ### Correction CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)

    ### Loop CONFIG
    Loopconfig = config["Loopconfig"]
    Loopconfig.update(NewLoopconfig)

    ###SIMU CONFIG
    SIMUconfig = config["SIMUconfig"]
    SIMUconfig.update(NewSIMUconfig)

    Name_Experiment = SIMUconfig["Name_Experiment"]

    ##############################################################################
    ### Initialization all the directories
    ##############################################################################

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    matrix_dir = os.path.join(Data_dir, "Interaction_Matrices") + os.path.sep
    result_dir = os.path.join(Data_dir, "Results",
                              Name_Experiment) + os.path.sep
    Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep

    # Initialize thd:
    entrance_pupil = OptSy.pupil(
        modelconfig,
        PupType=modelconfig['filename_instr_pup'],
        angle_rotation=modelconfig['entrance_pup_rotation'],
        Model_local_dir=Model_local_dir)

    DM1 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  Name_DM='DM1',
                                  Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                  DMconfig,
                                  Name_DM='DM3',
                                  Model_local_dir=Model_local_dir)

    corono = OptSy.coronagraph(modelconfig,
                               Coronaconfig,
                               Model_local_dir=Model_local_dir)
    # and then just concatenate
    thd2 = OptSy.Testbed([entrance_pupil, DM1, DM3, corono],
                         ["entrancepupil", "DM1", "DM3", "corono"])

    # The following line can be used to change the DM to make the PW probe,
    # including with a DM out of the pupil plane.
    # This is an unsual option so not in the param file and not well documented.
    # thd2.name_DM_to_probe_in_PW = "DM1"

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
    # phase up stream of the coronagraph (entrance pup)
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig,
                                             up_or_down='up',
                                             Model_local_dir=Model_local_dir)

    # phase down stream of the coronagraph (Lyot stop)
    phase_abb_do = thd2.generate_phase_aberr(SIMUconfig,
                                             up_or_down='do',
                                             Model_local_dir=Model_local_dir)

    # amplitude up stream of the coronagraph (entrance pup)

    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig,
                                           Model_local_dir=Model_local_dir)

    # WF in the testbed entrance pupil
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=ampl_abb_up)

    # aberrated WF in the testbed Lyot stop
    EF_aberrations_introduced_in_LS = thd2.EF_from_phase_and_ampl(
        phase_abb=phase_abb_do)

    Resultats_correction_loop = CorrectionLoop(
        thd2,
        estim,
        correc,
        MaskScience,
        Loopconfig,
        SIMUconfig,
        input_wavefront=input_wavefront,
        EF_aberrations_introduced_in_LS=EF_aberrations_introduced_in_LS,
        initial_DM_voltage=0,
        silence=False)

    Save_loop_results(Resultats_correction_loop, config, thd2, result_dir)
