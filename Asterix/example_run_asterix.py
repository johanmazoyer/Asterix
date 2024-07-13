import os
import time
from Asterix import main_THD, Asterix_root

# please replace with your own parameter file
your_directory = Asterix_root
your_parameter_file_name = 'Example_param_file.ini'

parameter_file_path = os.path.join(your_directory, your_parameter_file_name)

# start_time = time.time()
# main_THD.runthd2(parameter_file_path,
#                  NewDMconfig={'DM1_active': False},
#                  NewEstimationconfig={'estimation': 'pwp'},
#                  NewCorrectionconfig={
#                      'DH_side': "Right",
#                      'correction_algorithm': "efc",
#                      'Nbmodes_OnTestbed': 330
#                  },
#                  NewLoopconfig={
#                      'Nbiter_corr': [5, 10],
#                      "Nbmode_corr": [330, 340]
#                  },
#                  NewSIMUconfig={'Name_Experiment': "My_first_experiment"},
#                  dir_save_all_planes=None)

# print('time correction 1DM perfect estim efc', time.time() - start_time)
# print("")
# print("")
# print("")

# start_time = time.time()
# main_THD.runthd2(parameter_file_path,
#                  NewDMconfig={'DM1_active': False},
#                  NewEstimationconfig={'estimation': 'perfect'},
#                  NewCorrectionconfig={
#                      'DH_side': "Right",
#                      'correction_algorithm': "efc",
#                      'Nbmodes_OnTestbed': 330
#                  },
#                  NewLoopconfig={
#                      'Nbiter_corr': [5, 10],
#                      "Nbmode_corr": [330, 340]
#                  },
#                  NewSIMUconfig={'Name_Experiment': "My_second_experiment"})
# print('time correction 1DM perfect estim efc', time.time() - start_time)
# print("")
# print("")
# print("")

# start_time = time.time()
# main_THD.runthd2(parameter_file_path,
#                  NewDMconfig={'DM1_active': True},
#                  NewEstimationconfig={'estimation': 'perfect'},
#                  NewCorrectionconfig={
#                      'DH_side': "Full",
#                      'correction_algorithm': "sm",
#                  },
#                  NewLoopconfig={'Nbiter_corr': [20]},
#                  NewSIMUconfig={'Name_Experiment': "My_third_experiment"})
# print('time correction 2DM perfect estim sm', time.time() - start_time)
# print("")
# print("")
# print("")

start_time = time.time()
main_THD.runthd2(parameter_file_path,
                 NewDMconfig={'DM1_active': True},
                 NewEstimationconfig={'estimation': 'btp'},
                 NewCorrectionconfig={
                     'DH_side': "Full",
                     'correction_algorithm': "efc",
                     'Nbmodes_OnTestbed': 600
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [5, 1, 1, 1, 3, 2, 1, 2, 4, 3],
                     'Nbmode_corr': [600, 800, 500, 1000, 700, 900, 1000, 900, 700, 900]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_fourth_experiment"})
print('time correction 2DM pw efc', time.time() - start_time)
print("")
print("")
print("")
