import os
import numpy as np

from Asterix.main_THD import THD2
from Asterix import Asterix_root
from Asterix.utils import create_experiment_dir, get_data_dir, read_parameter_file
from Asterix.wfsc import Estimator, Corrector, MaskDH, correction_loop, save_loop_results


def test_1dm_correction():
    # Load the test parameter file
    parameter_file_test = os.path.join(Asterix_root, 'tests', "param_file_tests.ini")

    # Load configuration file
    config = read_parameter_file(parameter_file_test,
                                 NewDMconfig={'DM1_active': False},
                                 NewEstimationconfig={'estimation': 'pw'},
                                 NewCorrectionconfig={
                                     'DH_side': "Right",
                                     'correction_algorithm': "efc",
                                     'Nbmodes_OnTestbed': 330
                                 },
                                 NewLoopconfig={
                                     'Nbiter_corr': [5],
                                     "Nbmode_corr": [320]
                                 })


    data_dir = get_data_dir(config_in=config["Data_dir"])
    onbench = config["onbench"]
    model_config = config["modelconfig"]
    dm_config = config["DMconfig"]
    corona_config = config["Coronaconfig"]
    Estimationconfig = config["Estimationconfig"]
    Correctionconfig = config["Correctionconfig"]
    Loopconfig = config["Loopconfig"]
    SIMUconfig = config["SIMUconfig"]

    # Initialize all directories
    model_local_dir = os.path.join(data_dir, "Model_local")
    matrix_dir = os.path.join(data_dir, "Interaction_Matrices")

    # Concatenate into the full testbed optical system
    thd2 = THD2(model_config, dm_config, corona_config, model_local_dir)

    # The following line can be used to change the DM which applies PW probes. This could be used to use the DM out of
    # the pupil plane.
    # This is an unusual option so not in the param file and not well documented.
    # thd2.name_DM_to_probe_in_PW = "DM1"

    # Initialize the estimation
    estimator = Estimator(Estimationconfig,
                          thd2,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench)

    # Initialize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    science_mask_dh = mask_dh.creatingMaskDH(thd2.dimScience, thd2.Science_sampling)

    # Initialize the corrector
    corrector = Corrector(Correctionconfig,
                          thd2,
                          mask_dh,
                          estimator,
                          matrix_dir=matrix_dir,
                          save_for_bench=onbench)

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
                              silence=True)

    best_contrast = np.min(results["MeanDHContrast"])

    assert best_contrast < 1e-8

    # save_loop_results(results, config, thd2, science_mask_dh, result_dir)
