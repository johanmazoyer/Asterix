import itertools
import os
import time
from astropy.io import fits
import numpy as np
from Asterix import main_THD, Asterix_root

# please replace with your own parameter file
your_directory = Asterix_root
your_parameter_file_name = 'thd2_setups/Example_param_file_10jul24.ini'

parameter_file_path = os.path.join(your_directory, your_parameter_file_name)

modesnb = 400  # 1587

dm3_actuators_array = fits.getdata('/Users/ilaginja/asterix_data/Labview/20240903_16-34-07_HLC_fb06a12d/Base_Matrix_DM3.fits')
dm3_act_list = np.sum(dm3_actuators_array, axis=0)
active_actuators = np.where(dm3_act_list != 0)
actuator_pairs = itertools.combinations_with_replacement(active_actuators[0], r=2)

for pair in actuator_pairs:
    print(f"Actuator pair: {pair}")
    print(f'{pair[0]},{pair[1]}')

    start_time = time.time()
    main_THD.runthd2(parameter_file_path,
                     NewDMconfig={'DM1_active': True},
                     NewEstimationconfig={'estimation': 'pw',
                                          'posprobes': (pair[0], pair[1])},
                     NewCorrectionconfig={
                         'DH_side': "Full",
                         'correction_algorithm': "efc",
                         'Nbmodes_OnTestbed': modesnb
                     },
                     NewLoopconfig={
                         'Nbiter_corr': [50],
                         'Nbmode_corr': [modesnb]
                     },
                     NewSIMUconfig={'Name_Experiment': f"HLC_783nm_{pair[0]}-{pair[1]}"})
    print('time correction 2DM pw efc', time.time() - start_time)
    print("")
