import os
from Asterix import Main_EFC_THD
import time

start_time = time.time()
Asterixroot = os.path.dirname(os.path.realpath(__file__))
Main_EFC_THD.create_interaction_matrices(Asterixroot + os.path.sep+ 'Example_param_file.ini')
print('time for initialization', time.time() - start_time)
phase,im=Main_EFC_THD.correctionLoop(Asterixroot + os.path.sep+ 'Example_param_file.ini')#,NewSIMUconfig=dict
print('total time', time.time() - start_time)