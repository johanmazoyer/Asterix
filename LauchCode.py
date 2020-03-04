import os
from Main_EFC_THDv2 import *
import Main_EFC_THDv2 as main

for PWamp in [34]:
    dict={'amplitudePW': PWamp}
    main.create_interaction_matrices(os.getcwd()+'/Essai_param2.ini',NewPWconfig=dict)
    main.phase,im=CorrectionLoop(os.getcwd()+'/Essai_param2.ini',NewPWconfig=dict)
