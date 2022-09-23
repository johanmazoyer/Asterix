# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os

from Asterix.utils import read_parameter_file
from Asterix.optics import Pupil, Coronagraph, DeformableMirror, Testbed
from Asterix.wfsc import Estimator, Corrector, MaskDH, correction_loop, save_loop_results

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
            Can be used to directly change a parameter if needed, outside of the param file    
        
        -------------------------------------------------- """

    ### CONFIGURATION FILE
    config = read_parameter_file(parameter_file,
                                 NewMODELconfig=NewMODELconfig,
                                 NewDMconfig=NewDMconfig,
                                 NewCoronaconfig=NewCoronaconfig,
                                 NewEstimationconfig=NewEstimationconfig,
                                 NewCorrectionconfig=NewCorrectionconfig,
                                 NewLoopconfig=NewLoopconfig,
                                 NewSIMUconfig=NewSIMUconfig)

    ### CONFIG
    Data_dir = config["Data_dir"]
    #On bench or numerical simulation
    onbench = config["onbench"]

    ### MODEL CONFIG
    modelconfig = config["modelconfig"]

    ### DM CONFIG
    DMconfig = config["DMconfig"]

    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]

    ### Estimation CONFIG
    Estimationconfig = config["Estimationconfig"]

    ### Correction CONFIG
    Correctionconfig = config["Correctionconfig"]

    ### Loop CONFIG
    Loopconfig = config["Loopconfig"]

    ###SIMU CONFIG
    SIMUconfig = config["SIMUconfig"]
    Name_Experiment = SIMUconfig["Name_Experiment"]

    ##############################################################################
    ### Initialization all the directories
    ##############################################################################

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    matrix_dir = os.path.join(Data_dir, "Interaction_Matrices") + os.path.sep
    result_dir = os.path.join(Data_dir, "Results", Name_Experiment) + os.path.sep
    Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep

    # Initialize thd:
    entrance_pupil = Pupil(modelconfig,
                           PupType=modelconfig['filename_instr_pup'],
                           angle_rotation=modelconfig['entrance_pup_rotation'],
                           Model_local_dir=Model_local_dir)

    DM1 = DeformableMirror(modelconfig, DMconfig, Name_DM='DM1', Model_local_dir=Model_local_dir)

    DM3 = DeformableMirror(modelconfig, DMconfig, Name_DM='DM3', Model_local_dir=Model_local_dir)

    corono = Coronagraph(modelconfig, Coronaconfig, Model_local_dir=Model_local_dir)
    # and then just concatenate
    thd2 = Testbed([entrance_pupil, DM1, DM3, corono], ["entrancepupil", "DM1", "DM3", "corono"])

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
    MaskScience = mask_dh.creatingMaskDH(thd2.dimScience, thd2.Science_sampling)

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
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig, up_or_down='up', Model_local_dir=Model_local_dir)

    # phase down stream of the coronagraph (Lyot stop)
    phase_abb_do = thd2.generate_phase_aberr(SIMUconfig, up_or_down='do', Model_local_dir=Model_local_dir)

    # amplitude up stream of the coronagraph (entrance pup)

    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig, Model_local_dir=Model_local_dir)

    # WF in the testbed entrance pupil
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up, ampl_abb=ampl_abb_up)

    # aberrated WF in the testbed Lyot stop
    EF_aberrations_introduced_in_LS = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_do)

    Resultats_correction_loop = correction_loop(
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

    save_loop_results(Resultats_correction_loop, config, thd2, MaskScience, result_dir)
