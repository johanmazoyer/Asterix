import os
import numpy as np

from Asterix.utils import create_experiment_dir, get_data_dir, get_git_description, read_parameter_file
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

    def __init__(self, config, model_local_dir, silence=False):
        """
        Parameters
        ----------
        config : dict
            Parameter dictionary. Must at least contain parameters for general model [modelconfig],
            coronagraph ([Coronaconfig]) and DM(s) ([DMconfig]).
        Model_local_dir : string
            Directory output path for model-related files created on the file for later reuse.
        silence : boolean, default False.
            Whether to silence print outputs.
        """

        model_config = config["modelconfig"]
        dm_config = config["DMconfig"]
        corona_config = config["Coronaconfig"]

        # The following line is there to force the correction / estimation wavelength to be part of
        # the wavelengths that are simulated for the testbed. We use numpy.unique to have only unique
        # values from the List.
        model_config['mandatory_wls'] = sorted(
            np.unique(np.array(model_config['mandatory_wls'] + config["Estimationconfig"]["estimation_wls"])).tolist())

        # The following line can be added to change the precision of the complex number. complex64 is faster but can be
        # slightly different at the 10-10 contrast level.
        # model_config['complex_precision'] = 'complex64'

        # Create all optical elements of the THD
        self.entrance_pupil = Pupil(model_config,
                               PupType=model_config["filename_instr_pup"],
                               angle_rotation=model_config["entrance_pup_rotation"],
                               Model_local_dir=model_local_dir,
                               silence=silence)
        self.dm1 = DeformableMirror(model_config, dm_config, Name_DM="DM1", Model_local_dir=model_local_dir, silence=silence)
        self.dm2 = DeformableMirror(model_config, dm_config, Name_DM="DM2", Model_local_dir=model_local_dir, silence=silence)
        self.corono = Coronagraph(model_config, corona_config, Model_local_dir=model_local_dir, silence=silence)

        self.config_file = config

        # Concatenate into the full testbed optical system
        super().__init__([self.entrance_pupil, self.dm1, self.dm2, self.corono], ["entrancepupil", "DM1", "DM2", "corono"])


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
    NewMODELconfig : dict, optional
        Can be used to directly change a parameter in the MODELconfig section of the input parameter file.
    NewDMconfig : dict, optional
        Can be used to directly change a parameter in the DMconfig section of the input parameter file.
    NewCoronaconfig : dict, optional
        Can be used to directly change a parameter in the Coronaconfig section of the input parameter file.
    NewEstimationconfig : dict, optional
        Can be used to directly change a parameter in the Estimationconfig section of the input parameter file.
    NewCorrectionconfig : dict, optional
        Can be used to directly change a parameter in the Correctionconfig section of the input parameter file.
    NewSIMUconfig : dict, optional
        Can be used to directly change a parameter in the SIMUconfig section of the input parameter file.
    silence : boolean, default False.
        Whether to silence print outputs.
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

    # Get git commit hash
    commit = get_git_description()
    hardware_dir = os.path.join(data_dir, "Hardware_mat",
                                create_experiment_dir(append=config["Coronaconfig"]["corona_type"]) + f"_{commit}")

    # Concatenate into the full testbed optical system
    thd2 = THD2(config, model_local_dir, silence=silence)

    # Initialize the estimation
    estimator = Estimator(Estimationconfig,
                          thd2,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench,
                          realtestbed_dir=hardware_dir,
                          silence=silence)

    # Initialize the DH masks
    mask_dh = MaskDH(Correctionconfig)

    # mask for the estimation used in correction loop
    estim_mask_dh = mask_dh.creatingMaskDH(estimator.dimEstim, estimator.Estim_sampling)

    # Mask in science detector size for contrast estimation at each iteration.
    # if we used a fits file to do the dh mask, the fits must be of dimEstim size and
    # we do not use a science dark-hole mask.
    if mask_dh.DH_shape.lower().endswith('.fits'):
        science_mask_dh = np.ones((thd2.dimScience, thd2.dimScience))
    else:
        science_mask_dh = mask_dh.creatingMaskDH(thd2.dimScience, thd2.Science_sampling, **kwargs)

    # Initialize the corrector
    corrector = Corrector(Correctionconfig,
                          thd2,
                          estimator.dimEstim,
                          maskEstim=estim_mask_dh,
                          wav_vec_estim=estimator.wav_vec_estim,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench,
                          realtestbed_dir=hardware_dir,
                          silence=silence)

    ### Write configfile to Labview-style matrix directory
    if onbench:
        os.makedirs(hardware_dir, exist_ok=True)
        config.filename = os.path.join(hardware_dir, "Simulation_parameters.ini")
        config.write()

    ### Set initial phase and amplitude
    # Phase upstream of the coronagraph (entrance pup)
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig,
                                             up_or_down='up',
                                             Model_local_dir=model_local_dir,
                                             silence=silence)

    # Phase downstream of the coronagraph (Lyot stop)
    phase_abb_do = thd2.generate_phase_aberr(SIMUconfig,
                                             up_or_down='do',
                                             Model_local_dir=model_local_dir,
                                             silence=silence)

    # Amplitude upstream of the coronagraph (entrance pup)
    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig, Model_local_dir=model_local_dir, silence=silence)

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

    save_loop_results(results, config, thd2, science_mask_dh, result_dir, silence=silence)
