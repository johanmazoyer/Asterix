import os

from astropy.io.fits.column import ASCII_DEFAULT_WIDTHS
from Asterix import Main_EFC_THD
import time

Asterixroot = os.path.dirname(os.path.realpath(__file__))

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'perfect'},
                     NewCorrectionconfig={
                         'DH_side': "right",
                         'correction_algorithm': "efc",
                         "MatrixType": "Perfect"
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "knife_perf_SPmat_four",
                         'Nbiter_corr': ["10"],
                         "Nbmode_corr": ["250"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'perfect'},
                     NewCorrectionconfig={
                         'DH_side': "right",
                         'correction_algorithm': "efc",
                         "MatrixType": "SmallPhase"
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "knife_perf_perfmat_four",
                         'Nbiter_corr': ["10"],
                         "Nbmode_corr": ["250"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'pw'},
                     NewCorrectionconfig={
                         'DH_side': "right",
                         'correction_algorithm': "efc",
                         "MatrixType": "Perfect"
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "lyot_pw_SPmat_four",
                         'Nbiter_corr': ["10"],
                         "Nbmode_corr": ["250"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'pw'},
                     NewCorrectionconfig={
                         'DH_side': "right",
                         'correction_algorithm': "efc",
                         "MatrixType": "SmallPhase"
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "lyot_pw_SPmat_four",
                         'Nbiter_corr': ["10"],
                         "Nbmode_corr": ["250"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': False},
                     NewEstimationconfig={'estimation': 'pw'},
                     NewCorrectionconfig={
                         'DH_side': "top",
                         'correction_algorithm': "efc",
                         "MatrixType": "Perfect"
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "lyot_pw_perfmat_four",
                         'Nbiter_corr': ["5"],
                         "Nbmode_corr": ["250"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': True},
                     NewEstimationconfig={'estimation': 'pw'},
                     NewCorrectionconfig={
                         'DH_side': "Full",
                         'correction_algorithm': "sm",
                         "MatrixType": "SmallPhase",
                         'DM_basis': 'fourier'
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "test1",
                         'Nbiter_corr': ["15"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': True},
                     NewEstimationconfig={'estimation': 'pw'},
                     NewCorrectionconfig={
                         'DH_side': "Full",
                         'correction_algorithm': "sm",
                         "MatrixType": "Perfec",
                         'DM_basis': 'fourier'
                     },
                     NewSIMUconfig={
                         'Name_Experiment': "test1",
                         'Nbiter_corr': ["15"]
                     })
print('time correction 1DM pw', time.time() - start_time)
print("")
print("")
print("")

### CORRECTION 2DM IN PW + EFC
start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': True},
                     NewCorrectionconfig={
                         'DH_side': "Full",
                         'correction_algorithm': "efc",
                         "MatrixType": "SmallPhase",
                         'DM_basis': 'fourier'
                     },
                     NewSIMUconfig={
                         'Nbiter_corr':
                         ["5", "1", "1", "1", "3", "2", "1", "2", "4", "3"],
                         'Nbmode_corr': [
                             "500", "800", "500", "1000", "700", "900", "1000",
                             "900", "700", "900"
                         ]
                     })
print('time correction 2DM pw', time.time() - start_time)
print("")
print("")
print("")

start_time = time.time()
Main_EFC_THD.runthd2(Asterixroot + os.path.sep + 'Example_param_file.ini',
                     NewDMconfig={'DM1_active': True},
                     NewCorrectionconfig={
                         'DH_side': "Full",
                         'correction_algorithm': "efc",
                         "MatrixType": "SmallPhase",
                         'DM_basis': 'fourier'
                     },
                     NewSIMUconfig={
                         'Nbiter_corr':
                         ["5", "1", "1", "1", "3", "2", "1", "2", "4", "3"],
                         'Nbmode_corr': [
                             "500", "800", "500", "1000", "700", "900", "1000",
                             "900", "700", "900"
                         ]
                     })
print('time correction 2DM pw', time.time() - start_time)
print("")
print("")
print("")