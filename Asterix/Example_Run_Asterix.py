import os
from Asterix import Main_EFC_THD
import time

Asterixroot = os.path.dirname(os.path.realpath(__file__))

# These four cases need to converge before pull request !

# INITIALISATION 1DM IN PW + EFC
start_time = time.time()
Main_EFC_THD.create_interaction_matrices(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': False},
    NewEstimationconfig={'estimation': 'pw'},
    NewCorrectionconfig={
        'circ_side': "right",
        'Nbiter_corr': "5,5,5",
        "Nbmode_corr": "350, 380, 400"
    })
print('time for 1DM initialization', time.time() - start_time)

# CORRECTION 1DM IN PW + EFC
start_time = time.time()
phase, im = Main_EFC_THD.correctionLoop(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': False},
    NewEstimationconfig={'estimation': 'pw'},
    NewCorrectionconfig={
        'circ_side': "right",
        'Nbiter_corr': "5,5,5",
        "Nbmode_corr": "350, 380, 400"
    })
print('time correction 1DM pw', time.time() - start_time)

# CORRECTION 1DM IN Perfect + EFC
start_time = time.time()
phase, im = Main_EFC_THD.correctionLoop(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': False},
    NewEstimationconfig={'estimation': 'Perfect'},
    NewCorrectionconfig={
        'circ_side': "right",
        'Nbiter_corr': "5,5,5",
        "Nbmode_corr": "350, 380, 400"
    })
print('total time 1DM perfect', time.time() - start_time)

# INITIALISATION 2DM IN PW + EFC
start_time = time.time()
Main_EFC_THD.create_interaction_matrices(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': True},
    NewEstimationconfig={'estimation': 'pw'},
    NewCorrectionconfig={
        'circ_side': "Full",
        'Nbiter_corr': '2  ,1  ,1  ,1   ,3  ,2  ,1  ,2  ,4  ,3',
        'Nbmode_corr': '500,800,500,1000,700,900,1000,900,700,900'
    })
print('time for 2DM initialization', time.time() - start_time)

# CORRECTION 2DM IN PW + EFC
start_time = time.time()
phase, im = Main_EFC_THD.correctionLoop(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': True},
    NewEstimationconfig={'estimation': 'pw'},
    NewCorrectionconfig={
        'circ_side': "Full",
        'Nbiter_corr': '2  ,1  ,1  ,1   ,3  ,2  ,1  ,2  ,4  ,3',
        'Nbmode_corr': '500,800,500,1000,700,900,1000,900,700,900'
    })
print('time correction 2DM pw', time.time() - start_time)

# CORRECTION 2DM IN Perfect + EFC
start_time = time.time()
phase, im = Main_EFC_THD.correctionLoop(
    Asterixroot + os.path.sep + 'Example_param_file.ini',
    NewDMconfig={'DM1_active': True},
    NewEstimationconfig={'estimation': 'Perfect'},
    NewCorrectionconfig={
        'circ_side': "Full",
        'Nbiter_corr': '2  ,1  ,1  ,1   ,3  ,2  ,1  ,2  ,4  ,3',
        'Nbmode_corr': '500,800,500,1000,700,900,1000,900,700,900'
    })
print('total time correction 2DM perfect', time.time() - start_time)