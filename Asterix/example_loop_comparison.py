import os
import time
from Asterix import main_THD, Asterix_root


# please replace with your own parameter file
your_directory = Asterix_root
your_parameter_file_name = 'thd2_setups/Example_param_file_10jul24.ini'

parameter_file_path = os.path.join(your_directory, your_parameter_file_name)

modesnb = 300  # 1587

amp = 24

### SINGLE  RUN
start_time = time.time()
main_THD.runthd2(parameter_file_path,
                 NewDMconfig={'DM1_active': True},
                 NewEstimationconfig={'estimation': 'pw',
                                      'posprobes': (297, 298, 266),
                                      'amplitudePW': amp},
                 NewCorrectionconfig={
                     'DH_side': "Full",
                     'correction_algorithm': "efc",
                     'Nbmodes_OnTestbed': modesnb
                 },
                 NewLoopconfig={
                     'Nbiter_corr': [50],
                     'Nbmode_corr': [modesnb]
                 },
                 NewSIMUconfig={'Name_Experiment': f"HLC_783nm_actuators-shifted_{modesnb}modes_{amp}"})
                 # dir_save_all_planes='/Users/ilaginja/asterix_data/Results/all_planes')
print('time correction 2DM pw efc', time.time() - start_time)
