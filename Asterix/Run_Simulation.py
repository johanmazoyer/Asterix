import os
from Main_EFC_THD import *
import Main_EFC_THD as main

#dict={'Name_Experiment': 'Experiment1'}
main.create_interaction_matrices(os.getcwd()+'/Test_param.ini')
main.phase,im=CorrectionLoop(os.getcwd()+'/Test_param.ini')#,NewSIMUconfig=dict
