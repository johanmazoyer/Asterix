import os
from Asterix.Main_EFC_THD import *
import Asterix.Main_EFC_THD as main

#dict={'Name_Experiment': 'Experiment1'}
Asterixroot = os.path.dirname(os.path.realpath(__file__))

main.create_interaction_matrices(Asterixroot+os.path.sep+'Test_param.ini')
main.phase,im=main.correctionLoop(Asterixroot+os.path.sep+'Test_param.ini')#,NewSIMUconfig=dict
