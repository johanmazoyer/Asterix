import os

from Asterix.utils import create_experiment_dir, get_data_dir, read_parameter_file
from Asterix.optics import Pupil, Coronagraph, DeformableMirror, Testbed
from Asterix.wfsc import Estimator, Corrector, MaskDH, correction_loop, save_loop_results


class THD2(Testbed):
    """Testbed object configured for THD2, from input configfile.

    AUTHOR : ILa, Nov 2022

    Attributes
    ----------
    config : dict
        A read-in .ini parameter file.
    """

    def __init__(self, config, model_local_dir):
        """
        Parameters
        ----------
        config: dict
            Parameter dictionary. Must at least contains parameter for Entrance Pupil, Coronagraphs and DM(s).
        Model_local_dir: string
                path directory to save things you can measure yourself and can save to save time
        """
        
        model_config = config["modelconfig"]
        dm_config = config["DMconfig"]
        corona_config = config["Coronaconfig"]

        # Create all optical elements of the THD
        entrance_pupil = Pupil(model_config,
                               PupType=model_config["filename_instr_pup"],
                               angle_rotation=model_config["entrance_pup_rotation"],
                               Model_local_dir=model_local_dir)
        dm1 = DeformableMirror(model_config, dm_config, Name_DM="DM1", Model_local_dir=model_local_dir)
        dm3 = DeformableMirror(model_config, dm_config, Name_DM="DM3", Model_local_dir=model_local_dir)
        corono = Coronagraph(model_config, corona_config, Model_local_dir=model_local_dir)

        # Concatenate into the full testbed optical system
        super().__init__([entrance_pupil, dm1, dm3, corono], ["entrancepupil", "DM1", "DM3", "corono"])


def runthd2(parameter_file_path,
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

    Initialize the THD2 testbed, the estimation method, the correction method from parameter_file_path.
    Run the loop and save the results.

    All NewXXXconfig input variables can be used to update a single parameter in one of
    the subsections of the parameter file. This will replace only the overwritten
    value in the parameter file.
    e.g.: NewCoronaconfig = {'paramXXX': YYY}

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    parameter_file_path: string
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
    config = read_parameter_file(parameter_file_path,
                                 NewMODELconfig=NewMODELconfig,
                                 NewDMconfig=NewDMconfig,
                                 NewCoronaconfig=NewCoronaconfig,
                                 NewEstimationconfig=NewEstimationconfig,
                                 NewCorrectionconfig=NewCorrectionconfig,
                                 NewLoopconfig=NewLoopconfig,
                                 NewSIMUconfig=NewSIMUconfig)

    data_dir = get_data_dir(config_in=config["Data_dir"])
    onbench = config["onbench"]
    Estimationconfig = config["Estimationconfig"]
    Correctionconfig = config["Correctionconfig"]
    Loopconfig = config["Loopconfig"]
    SIMUconfig = config["SIMUconfig"]
    name_experiment = create_experiment_dir(append=SIMUconfig["Name_Experiment"])

    # Initialize all directories
    model_local_dir = os.path.join(data_dir, "Model_local")
    matrix_dir = os.path.join(data_dir, "Interaction_Matrices")
    result_dir = os.path.join(data_dir, "Results", name_experiment)
    labview_dir = os.path.join(data_dir, "Labview")

    # Concatenate into the full testbed optical system
    thd2 = THD2(config, model_local_dir)

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
