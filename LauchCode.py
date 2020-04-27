import os
from Main_EFC_THDv2 import *
import Main_EFC_THDv2 as main

dict={'Name_Experiment': 'Experiment1'}
main.create_interaction_matrices(os.getcwd()+'/Essai_param2.ini')
main.phase,im=CorrectionLoop(os.getcwd()+'/Essai_param2.ini',NewSIMUconfig=dict)
