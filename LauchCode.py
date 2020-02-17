<<<<<<< HEAD
import os
from Main_EFC_THDv2 import *
import Main_EFC_THDv2 as main

main.create_interaction_matrices(os.getcwd()+'/Essai_param2.ini')

main.phase,im=CorrectionLoop(os.getcwd()+'/Essai_param2.ini')
=======
from Main_EFC_THDv2 import *
import Main_EFC_THDv2 as main

#main.create_interaction_matrices('/home/apotier/Documents/Recherche/SimuPython/GitHub/Essai/Essai_param2.ini')

main.phase,im=CorrectionLoop('/home/apotier/Documents/Recherche/SimuPython/GitHub/Essai/Essai_param2.ini')
>>>>>>> 676cb01dd9df3008835ba4e79fd6cfe8d7a2ead7
