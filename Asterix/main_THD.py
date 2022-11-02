import os

from Asterix.utils import create_experiment_dir, get_data_dir, read_parameter_file
from Asterix.optics import THD2
from Asterix.wfsc import Estimator, Corrector, MaskDH, correction_loop, save_loop_results


def runthd2(parameter_file,
            NewMODELconfig={},
            NewDMconfig={},
            NewCoronaconfig={},
            NewEstimationconfig={},
            NewCorrectionconfig={},
            NewLoopconfig={},
            NewSIMUconfig={},
            silence=False,
            **kwargs):
    """
    Run a simulation of a correction loop for the THD2 testbed.

    Initialize the THD2 testbed, the estimation method, the correction method from parameter_file.
    Run the loop and save the results.

    All NewXXXconfig input variables can be used to update a single parameter in one of
    the subsections of the parameter file. This will replace only the overwritten
    value in the parameter file.
    e.g.: NewCoronaconfig = {'paramXXX': YYY}

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    parameter_file: string
        Absolute path to a .ini parameter file
    NewMODELconfig: dict, optional
        Can be used to directly change a parameter in the MODELconfig section of the input parameter file.
    NewDMconfig: dict, optional
        Can be used to directly change a parameter in the DMconfig section of the input parameter file.
    NewCoronaconfig: dict, optional
        Can be used to directly change a parameter in the Coronaconfig section of the input parameter file.
    NewEstimationconfig: dict, optional
        Can be used to directly change a parameter in the Estimationconfig section of the input parameter file.
    NewCorrectionconfig: dict, optional
        Can be used to directly change a parameter in the Correctionconfig section of the input parameter file.
    NewSIMUconfig: dict, optional
        Can be used to directly change a parameter in the SIMUconfig section of the input parameter file.
    silence : boolean
        Whether to silence correction loop outputs; default False.
    """

    # Load configuration file
    config = read_parameter_file(parameter_file,
                                 NewMODELconfig=NewMODELconfig,
                                 NewDMconfig=NewDMconfig,
                                 NewCoronaconfig=NewCoronaconfig,
                                 NewEstimationconfig=NewEstimationconfig,
                                 NewCorrectionconfig=NewCorrectionconfig,
                                 NewLoopconfig=NewLoopconfig,
                                 NewSIMUconfig=NewSIMUconfig)

    Data_dir = get_data_dir(config_in=config["Data_dir"])
    onbench = config["onbench"]
    Estimationconfig = config["Estimationconfig"]
    Correctionconfig = config["Correctionconfig"]
    Loopconfig = config["Loopconfig"]
    SIMUconfig = config["SIMUconfig"]
    Name_Experiment = create_experiment_dir(append=SIMUconfig["Name_Experiment"])

    # Initialize all directories
    model_local_dir = os.path.join(Data_dir, "Model_local")
    matrix_dir = os.path.join(Data_dir, "Interaction_Matrices")
    result_dir = os.path.join(Data_dir, "Results", Name_Experiment)
    labview_dir = os.path.join(Data_dir, "Labview")

    # Concatenate into the full testbed optical system
    thd2 = THD2(parameter_file, NewMODELconfig, NewDMconfig, NewCoronaconfig)

    # The following line can be used to change the DM which applies PW probes. This could be used to use the DM out of
    # the pupil plane.
    # This is an unusual option so not in the param file and not well documented.
    # thd2.name_DM_to_probe_in_PW = "DM1"

    # Initialize the estimation
    estimator = Estimator(Estimationconfig,
                          thd2,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench,
                          realtestbed_dir=labview_dir)

    # Initialize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    science_mask_dh = mask_dh.creatingMaskDH(thd2.dimScience, thd2.Science_sampling, **kwargs)

    # Initialize the corrector
    corrector = Corrector(Correctionconfig,
                          thd2,
                          mask_dh,
                          estimator,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench,
                          realtestbed_dir=labview_dir)

    ### Set initial phase and amplitude
    # Phase upstream of the coronagraph (entrance pup)
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig, up_or_down='up', Model_local_dir=model_local_dir)

    # Phase downstream of the coronagraph (Lyot stop)
    phase_abb_do = thd2.generate_phase_aberr(SIMUconfig, up_or_down='do', Model_local_dir=model_local_dir)

    # Amplitude upstream of the coronagraph (entrance pup)
    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig, Model_local_dir=model_local_dir)

    ### Create the wavefronts including the phase and amplitude aberrations
    # WF in the testbed entrance pupil
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up, ampl_abb=ampl_abb_up)

    # WF in the testbed Lyot stop
    wavefront_in_LS = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_do)

    # Run the WFS&C loop
    results = correction_loop(thd2,
                              estimator,
                              corrector,
                              science_mask_dh,
                              Loopconfig,
                              SIMUconfig,
                              input_wavefront=input_wavefront,
                              EF_aberrations_introduced_in_LS=wavefront_in_LS,
                              initial_DM_voltage=0,
                              silence=silence,
                              **kwargs)

    save_loop_results(results, config, thd2, science_mask_dh, result_dir)
