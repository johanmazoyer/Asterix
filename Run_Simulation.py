import os
from Main_EFC_THD import *
import Main_EFC_THD as main

#dict={'Name_Experiment': 'Experiment1'}
main.create_interaction_matrices(os.getcwd()+'/parameter_files/First_test_params.ini')
main.phase,im=CorrectionLoop(os.getcwd()+'/parameter_files/First_test_params.ini')#,NewSIMUconfig=dict)
