import os
from Asterix import Main_EFC_THD

Asterixroot = os.path.dirname(os.path.realpath(__file__))
Main_EFC_THD.create_interaction_matrices(Asterixroot + os.path.sep+ 'Example_param_file.ini')
phase,im=Main_EFC_THD.correctionLoop(Asterixroot + os.path.sep+ 'Example_param_file.ini')#,NewSIMUconfig=dict