import os
from Asterix import Main_EFC_THD
import time

Asterixroot = os.path.dirname(os.path.realpath(__file__))

# These five cases need to converge before pull request !

## CORRECTION 1DM IN PW + EFC
# start_time = time.time()
# Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                      NewDMconfig={'DM1_active': False},
#                      NewEstimationconfig={'estimation': 'perfect'},
#                      NewCorrectionconfig={'DH_side': "top"},
#                      NewSIMUconfig={
#                          'Nbiter_corr': ["8", "1", "1"],
#                          "Nbmode_corr": ["350", "380", "400"]
#                      })
# print('time correction 1DM pw', time.time() - start_time)
# print("")
# print("")
# print("")


# # start_time = time.time()
# Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                      NewDMconfig={'DM1_active': False},
#                      NewEstimationconfig={'estimation': 'Perfect'},
#                      NewCorrectionconfig={'DH_side': "right"},
#                      NewSIMUconfig={
#                          'Nbiter_corr': ["5", "5", "5"],
#                          "Nbmode_corr": ["350", "380", "400"]
#                      })
# print('total time 1DM perfect HLC', time.time() - start_time)
# print("")
# print("")
# print("")

# ### CORRECTION 2DM IN PW + EFC
# start_time = time.time()
# Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
#                      NewDMconfig={'DM1_active': False},
#                      NewEstimationconfig={'estimation': 'pw'},
#                      NewCorrectionconfig={'DH_side': "Full"},
#                      NewSIMUconfig={
#                          'Nbiter_corr':
#                          ["5", "1", "1", "1", "3", "2", "1", "2", "4", "3"],
#                          'Nbmode_corr': [
#                              "500", "800", "500", "1000", "700", "900", "1000",
#                              "900", "700", "900"
#                          ]
#                      })
# print('time correction 2DM pw', time.time() - start_time)
# print("")
# print("")
# print("")

### CORRECTION 2DM IN PW + EFC
start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini')
print('time correction 2DM pw', time.time() - start_time)
print("")
print("")
print("")
asd


start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewCoronaconfig={'corona_type': 'knife'},
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'Perfect'},
                     NewCorrectionconfig={'DH_side': "right"},
                     NewSIMUconfig={
                         'Nbiter_corr': ["1", "1", "1"],
                         "Nbmode_corr": ["350", "380", "400"]
                     })
print('total time 1DM perfect knife', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewCoronaconfig={'corona_type': 'hlc'},
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'Perfect'},
                     NewCorrectionconfig={'DH_side': "right"},
                     NewSIMUconfig={
                         'Nbiter_corr': ["5", "5", "5"],
                         "Nbmode_corr": ["350", "380", "400"]
                     })
print('total time 1DM perfect knife', time.time() - start_time)
print("")
print("")
print("")