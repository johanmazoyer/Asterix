import os

import numpy as np
from configobj import ConfigObj
from validate import Validator

import Asterix.InstrumentSimu_functions as instr
import Asterix.fits_functions as useful

import Asterix.phase_amplitude_functions as phase_ampl


Asterixroot = os.path.dirname(os.path.realpath(__file__))

### CONFIGURATION FILE
parameter_file = Asterixroot + os.path.sep + "Example_param_file.ini"
configspec_file = Asterixroot + os.path.sep + "Param_configspec.ini"
config = ConfigObj(parameter_file,
                    configspec=configspec_file,
                       default_encoding="utf8")
vtor = Validator()
checks = config.validate(vtor, copy=True)


### LOAD CONFIG
Data_dir = config["Data_dir"]
modelconfig = config["modelconfig"]
DMconfig = config["DMconfig"]
Coronaconfig = config["Coronaconfig"]

# LOAD DIR
model_dir = Asterixroot + os.path.sep + "Model" + os.path.sep
        

print(modelconfig)

# we initialize a pupil

# Roman pupil of radius prad as define in the parameter file
pup_roman = instr.pupil(modelconfig, model_dir=model_dir, filename="roman_pup_1002pix_center4pixels.fits")

# Clear pupil of radius prad as define in the parameter file
pup_round = instr.pupil(modelconfig)

# Clear pupil of radius equal to 100 pixel
pup_round_100 = instr.pupil(modelconfig, prad=100)

# these are Opitcal System objects, so we can easily access their PSF for example:
psf_roman = pup_roman.todetector_Intensity()
useful.quickfits(psf_roman)

# if you want to access the pupil itself as an array, you can with
numpy_array_pup = pup_roman.pup
useful.quickfits(numpy_array_pup)


# all Opitcal System objects,have a tranmission, which is the ratio of flux after the system, compared to a clear 
# aperture of self.prad radius:
transmission_roman_pup = pup_roman.transmission()
print("transmission pup roman = ", transmission_roman_pup)


# Now lets initialie a coronagraph 
corono = instr.coronagraph(modelconfig,Coronaconfig,model_dir=model_dir)

#we can also update some of the parameter on the spot if we wish. 
# For example the coronagraph define int he parameter file does not have an entrance pupil because 
# it is in the "THD2" way, but we can put one, for exemple roman:
Coronaconfig.update({'filename_instr_apod':"roman_pup_1002pix_center4pixels.fits"})
# WE can also cahnge the Bandwidth
Coronaconfig.update({    'Delta_wav': '50e-9'})
# Lets reinitialize the coronagraph
corono = instr.coronagraph(modelconfig,Coronaconfig,model_dir=model_dir)

# lets create a phase:
phaserms = 50e-9 #in meter
rhoc_phase = 4.3
slope_phase = 3
phase = phase_ampl.random_phase_map(corono.dim_overpad_pupil, phaserms, rhoc_phase, slope_phase, pup_round.pup)

# fron this phase we create a electrical field
Entrance_EF = corono.EF_from_phase_and_ampl(phase_abb=phase)

# we measure the off axis PSF 
No_mask_PSF = np.max(corono.todetector_Intensity(center_on_pixel=True, noFPM = True))

# we measure the FP intensity of the corono
FP_after_corono_in_contrast =  corono.todetector_Intensity(entrance_EF=Entrance_EF)/ No_mask_PSF
useful.quickfits(FP_after_corono_in_contrast)

# If you want to understand something, you can plot all planes in the propagation
# This is nuclear option, it can produce tens of fits:
FP_after_corono_in_contrast =  corono.todetector_Intensity(entrance_EF=Entrance_EF, save_all_planes_to_fits=True, dir_save_fits='/Users/jmazoyer/Desktop/toto')/ No_mask_PSF

