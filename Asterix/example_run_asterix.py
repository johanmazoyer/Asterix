# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import time
from Asterix import main_THD, Asterix_root

start_time = time.time()
main_THD.runthd2(Asterix_root + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': False},
                 NewEstimationconfig={'estimation': 'pw'},
                 NewCorrectionconfig={
                     'DH_side': "Right",
                     'correction_algorithm': "efc",
                     'Nbmodes_OnTestbed': 330
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [5, 10],
                     "Nbmode_corr": [320, 340]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_first_experiment"})

print('time correction 1DM perfect estim efc', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
main_THD.runthd2(Asterix_root + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': False},
                 NewEstimationconfig={'estimation': 'perfect'},
                 NewCorrectionconfig={
                     'DH_side': "Right",
                     'correction_algorithm': "efc",
                     'Nbmodes_OnTestbed': 330
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [5, 10],
                     "Nbmode_corr": [320, 340]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_second_experiment"})
print('time correction 1DM perfect estim efc', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
main_THD.runthd2(Asterix_root + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': True},
                 NewEstimationconfig={'estimation': 'perfect'},
                 NewCorrectionconfig={
                     'DH_side': "Full",
                     'correction_algorithm': "sm",
                     'Nbmodes_OnTestbed': 600
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [20],
                     "Nbmode_corr": [250]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_fourth_experiment"})
print('time correction 2DM perfect estim sm', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
main_THD.runthd2(Asterix_root + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': True},
                 NewEstimationconfig={'estimation': 'pw'},
                 NewCorrectionconfig={
                     'DH_side': "Full",
                     'correction_algorithm': "efc",
                     'Nbmodes_OnTestbed': 600
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [5, 1, 1, 1, 3, 2, 1, 2, 4, 3],
                     "Nbmode_corr": [500, 800, 500, 1000, 700, 900, 1000, 900, 700, 900]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_fifth_experiment"})
print('time correction 2DM pw efc', time.time() - start_time)
print("")
print("")
print("")
