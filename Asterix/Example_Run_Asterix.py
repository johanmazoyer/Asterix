# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os

from Asterix import Main_THD
import time

Asterixroot = os.path.dirname(os.path.realpath(__file__))

start_time = time.time()
Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': False},
                 NewEstimationconfig={'estimation': 'perfect'},
                 NewCorrectionconfig={
                     'DH_side': "Right",
                     'correction_algorithm': "efc",
                 },
                 NewLoopconfig={
                     'Nbiter_corr': ["5", "5", "5"],
                     "Nbmode_corr": ["350", "380", "400"]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_first_experiment"})
print('time correction 1DM perfect estim efc', time.time() - start_time)
print("")
print("")
print("")


# start_time = time.time()
# Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                  NewDMconfig={'DM1_active': False},
#                  NewEstimationconfig={'estimation': 'pw'},
#                  NewCorrectionconfig={
#                      'DH_side': "Right",
#                      'correction_algorithm': "efc",
#                  },
#                  NewLoopconfig={
#                      'Nbiter_corr': ["5", "5", "5"],
#                      "Nbmode_corr": ["350", "380", "400"]
#                  },
#                  NewSIMUconfig={'Name_Experiment': "My_second_experiment"})
# print('time correction 1DM perfect estim efc', time.time() - start_time)
# print("")
# print("")
# print("")


# start_time = time.time()
# Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                  NewDMconfig={'DM1_active': False},
#                  NewEstimationconfig={'estimation': 'perfect'},
#                  NewCorrectionconfig={
#                      'DH_side': "Right",
#                      'correction_algorithm': "efc",
#                  },
#                  NewLoopconfig={
#                      'Nbiter_corr': ["10"],
#                      'Linesearch': True
#                  },
#                  NewSIMUconfig={'Name_Experiment': "My_third_experiment"})
# print('time correction 1DM perfect estim (Linesearch) efc',
#       time.time() - start_time)
# print("")
# print("")
# print("")


start_time = time.time()
Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                 NewDMconfig={'DM1_active': True},
                 NewEstimationconfig={'estimation': 'perfect'},
                 NewCorrectionconfig={
                     'DH_side': "Full",
                     'correction_algorithm': "sm",                 },
                 NewLoopconfig={
                     'Nbiter_corr': ["8"],
                     "Nbmode_corr": ["250"]
                 },
                 NewSIMUconfig={'Name_Experiment': "My_fourth_experiment"})
print('time correction 2DM perfect estim sm', time.time() - start_time)
print("")
print("")
print("")

# start_time = time.time()
# Main_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                  NewDMconfig={'DM1_active': True},
#                  NewEstimationconfig={'estimation': 'perfect'},
#                  NewCorrectionconfig={
#                      'DH_side': "Full",
#                      'correction_algorithm': "efc",
#                  },
#                  NewLoopconfig={
#                      'Nbiter_corr':
#                      ["10"],
#                      "Nbmode_corr": [
#                          "500"
#                      ]
#                  },
#                  NewSIMUconfig={'Name_Experiment': "My_fifth_experiment"})
# print('time correction 2DM pw efc', time.time() - start_time)
# print("")
# print("")
# print("")
