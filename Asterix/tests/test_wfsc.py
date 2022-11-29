import os
import numpy as np

from Asterix.main_THD import THD2
from Asterix import Asterix_root
from Asterix.utils import create_experiment_dir, get_data_dir, read_parameter_file
from Asterix.wfsc import Estimator, Corrector, MaskDH, correction_loop, save_loop_results


def quick_run_no_save(config):
    data_dir = get_data_dir(config_in=config["Data_dir"])
    Estimationconfig = config["Estimationconfig"]
    Correctionconfig = config["Correctionconfig"]
    Loopconfig = config["Loopconfig"]
    SIMUconfig = config["SIMUconfig"]

    # Initialize all directories
    model_local_dir = os.path.join(data_dir, "Model_local")
    matrix_dir = os.path.join(data_dir, "Interaction_Matrices")

    # Concatenate into the full testbed optical system
    thd2 = THD2(config, model_local_dir)

    # Initialize the estimation
    estimator = Estimator(Estimationconfig, thd2, matrix_dir=matrix_dir)

    # Initialize the DH masks
    mask_dh = MaskDH(Correctionconfig)
    science_mask_dh = mask_dh.creatingMaskDH(thd2.dimScience, thd2.Science_sampling)

    # Initialize the corrector
    corrector = Corrector(Correctionconfig, thd2, mask_dh, estimator, matrix_dir=matrix_dir)

    ### Set initial phase and amplitude
    # Phase upstream of the coronagraph (entrance pup)
    phase_abb_up = thd2.generate_phase_aberr(SIMUconfig, up_or_down='up')

    # Amplitude upstream of the coronagraph (entrance pup)
    ampl_abb_up = thd2.generate_ampl_aberr(SIMUconfig, Model_local_dir=model_local_dir)

    ### Create the wavefronts including the phase and amplitude aberrations
    # WF in the testbed entrance pupil
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up, ampl_abb=ampl_abb_up)

    # Run the WFS&C loop
    results = correction_loop(thd2,
                              estimator,
                              corrector,
                              science_mask_dh,
                              Loopconfig,
                              SIMUconfig,
                              input_wavefront=input_wavefront,
                              initial_DM_voltage=0,
                              silence=True)

    best_contrast = np.min(results["MeanDHContrast"])
    return best_contrast


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
    best_contrast_1DM = quick_run_no_save(config)
    assert best_contrast_1DM < 1e-8, "best contrast 1DM should be < 1e-8"

    # Load configuration file
    config = read_parameter_file(parameter_file_test,
                                 NewDMconfig={'DM1_active': True},
                                 NewEstimationconfig={'estimation': 'perfect'},
                                 NewCorrectionconfig={
                                     'DH_side': "Full",
                                     'correction_algorithm': "sm",
                                     'Nbmodes_OnTestbed': 600
                                 },
                                 NewLoopconfig={
                                     'Nbiter_corr': [5],
                                     'Nbmode_corr': [250]
                                 })
    best_contrast_2DM = quick_run_no_save(config)

    assert best_contrast_2DM < 1e-8, "best contrast 2DM should be < 1e-8"
