import os
from Asterix import main_THD
from Asterix.tests.test_utils import read_test_results_from_file


asterix_root = os.path.dirname(os.path.realpath(__file__))   #TODO: the result of this is the 'test' folder
print(asterix_root)
param_file_path = asterix_root + os.path.sep + 'Example_param_file.ini'
regression_path = os.path.join(asterix_root, )   #TODO: need oto define this


def test_perfect_estimate_with_strokemin(tmpdir):   #TODO: need to be able to overwrite the data_dir in the WFS loop

    # Known results for this setup on a full DH
    regression_results_full_dh = read_test_results_from_file("perfect_estim_with_strokemin_FullDH.txt")

    # Run on full DH
    results_full_dh = main_THD.runthd2(param_file_path,
                                       NewDMconfig={'DM1_active': True},
                                       NewEstimationconfig={'estimation': 'perfect'},
                                       NewCorrectionconfig={
                                           'DH_side': "Full",
                                           'correction_algorithm': "sm",
                                           'Nbmodes_OnTestbed': 600
                                       },
                                       NewLoopconfig={
                                           'Nbiter_corr': [20],
                                           'Nbmode_corr': [250]
                                       },
                                       NewSIMUconfig={'Name_Experiment': "My_fourth_experiment"})

    # Known results for this setup on a right-sided half-DH
    regression_results_half_dh = read_test_results_from_file("perfect_estim_with_strokemin_RightDH.txt")

    # Run on half-DH
    results_half_dh = main_THD.runthd2(param_file_path,
                                       NewDMconfig={'DM1_active': True},
                                       NewEstimationconfig={'estimation': 'perfect'},
                                       NewCorrectionconfig={
                                           'DH_side': "Right",
                                           'correction_algorithm': "sm",
                                           'Nbmodes_OnTestbed': 600
                                       },
                                       NewLoopconfig={
                                           'Nbiter_corr': [20],
                                           'Nbmode_corr': [250]
                                       },
                                       NewSIMUconfig={'Name_Experiment': "My_fourth_experiment"})
    for result_item in ['nb_total_iter', 'Nb_iter_per_mat', 'MeanDHContrast']:
        assert results_full_dh[result_item] == regression_results_full_dh[result_item],\
            f"Error in '{result_item}' during WFS&C regression test with perfect estimation and strokemin controller," \
            f"in a *full DH*."
        assert results_half_dh[result_item] == regression_results_half_dh[result_item],\
            f"Error in '{result_item}' during WFS&C regression test with perfect estimation and strokemin controller," \
            f"in a *half-DH*."


def test_pairwise_and_strokemin():
    # Full DH and half DH
    pass


def test_pairwise_with_efc():
    # Full DH and half DH
    pass
