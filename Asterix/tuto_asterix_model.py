import os

import numpy as np
from configobj import ConfigObj
from validate import Validator

import Asterix.Optical_System_functions as OptSy
import Asterix.fits_functions as useful

### CONFIGURATION FILE
parameter_file = OptSy.Asterix_root + os.path.sep + "Example_param_file.ini"
configspec_file = OptSy.Asterix_root + os.path.sep + "Param_configspec.ini"
config = ConfigObj(parameter_file,
                   configspec=configspec_file,
                   default_encoding="utf8")
vtor = Validator()
checks = config.validate(vtor, copy=True)

### LOAD CONFIG
modelconfig = config["modelconfig"]
DMconfig = config["DMconfig"]
Coronaconfig = config["Coronaconfig"]
SIMUconfig = config["SIMUconfig"]

# please modify
Data_dir = '.'

# DIR to save the .fits files that you need to simulate but that you can re-calculate yourself
# Should not be in the git.
Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
if not os.path.exists(Model_local_dir):
    print("Creating directory " + Model_local_dir + " ...")
    os.makedirs(Model_local_dir)

# where to save the results
result_dir = os.path.join(Data_dir, "Results") + os.path.sep
if not os.path.exists(result_dir):
    print("Creating directory " + result_dir + " ...")
    os.makedirs(result_dir)

#we put a very small number of pixel in the pup to go super fast in this tutorial
modelconfig.update({'diam_pup_in_pix': 80})

# we start the tutorial initializing a pupil
# Clear pupil of radius prad as define in the parameter file (diameter of the telescope)
pup_round = OptSy.pupil(modelconfig)

# If you have a file, you can initialize complex pupil at the right size
# Roman pupil of radius prad as define in the parameter file.
pup_roman = OptSy.pupil(modelconfig, PupType="RomanPup")

# by default all pupil have the diameter of the telescope but smaller / larger pupil
# can be defined. This can be useful for Lyot stop
# Clear pupil of radius equal to 100 pixel
pup_round_100 = OptSy.pupil(modelconfig, prad=100)

# # Careful, pup_roman is not an array, it is an Optical System object.
# if you want to access the pupil itself as an array, you can with
numpy_array_pup = pup_roman.pup
useful.quickfits(numpy_array_pup, dir=result_dir, name="numpy_array_pup")

# One you defined an Optical System object, so we can easily access sereval feature

#  measure the electrical field (EF) through the system:
EF_though_roman = pup_roman.EF_through(entrance_EF=1.)

# measure the EF in the next focal plane after the system.
# Default is monochromatic at center wavelenght
EF_though_roman = pup_roman.EF_through(entrance_EF=1.)

#  On the associated PSF. Default is polychromatic.
psf_roman = pup_roman.todetector_Intensity()
useful.quickfits(psf_roman, dir=result_dir, name="psf_roman")

#chromaticity of the source is define in all Opitcla System, with three parameter:
print("central wavelength: ", pup_roman.wavelength_0)
print("bandwidth: ", pup_roman.Delta_wav)
print("number of sub wavelength: ", pup_roman.nb_wav)

# aalso ll Opitcal System objects,have a tranmission, which is the ratio of flux after the system, compared to a clear
# aperture of self.prad radius:
transmission_roman_pup = pup_roman.transmission()
print("transmission pup roman = ", transmission_roman_pup)

#if you know what your doing you can update the parameter files on the go:
# WE can also cahnge the Bandwidth.

modelconfig.update({'Delta_wav': 50e-9})
# but then files have to be reinitialize:
pup_roman_poly = OptSy.pupil(modelconfig,
                             filename="roman_pup_1002pix_center4pixels.fits")

# Be careful when you change modelconfig because Optical Systems are designed to work together
# but that only works if they are defined with the same base config

#let's go back to monochromatic for now.
modelconfig.update({'Delta_wav': 0})

# Now lets initialize a coronagraph
# a coronagraph is a system composed of 3 planes. A apodization plane (PP), a FPM (FP) and a Lyot stop (PP)
# The coronagraph currently int he parameter file does not have an apodization pupil
# because there is no such plane on the THD2 bench, but we can put one, that is why I put an apod.
Coronaconfig.update({'filename_instr_apod': "RoundPup"})
corono = OptSy.coronagraph(modelconfig, Coronaconfig)


# For the coronagraph, we can measure 2 types of PSF: with or without mask
No_mask_PSF = corono.todetector_Intensity(center_on_pixel=True, noFPM=True)
# This allow us to normalize the images
Max_No_mask_PSF = np.max(No_mask_PSF)


# set initial phase and amplitude and wavefront
phase = corono.generate_phase_aberr(SIMUconfig,
                                            Model_local_dir=Model_local_dir)

# # from this phase we create a electrical field. This is a general
# aberrated field before any pupil multiplication.
aberrated_EF = corono.EF_from_phase_and_ampl(phase_abb=phase)

# By default this is the normal coronagraphic PSF (with FPM)
coronagraphic_PSF = corono.todetector_Intensity(entrance_EF=aberrated_EF)

#which we can now normalize:
normalized_coronagraphic_PSF = coronagraphic_PSF / Max_No_mask_PSF
useful.quickfits(normalized_coronagraphic_PSF,
                 dir=result_dir,
                 name="Normalized_coronagraphic_PSF")

# If you want to debug something, you can plot all planes in the propagation

plot_all_fits_dir = os.path.join(result_dir, "plot_all_fits") + os.path.sep
if not os.path.exists(plot_all_fits_dir):
    os.makedirs(plot_all_fits_dir)

# This is nuclear option, it can produce tens of fits, especially in a correction loop:
FP_after_corono_in_contrast = corono.todetector_Intensity(
    entrance_EF=aberrated_EF,
    save_all_planes_to_fits=True,
    dir_save_all_planes=plot_all_fits_dir) / No_mask_PSF

# Finally we can initialize DMs. DMs can be in pupil or off pupil.
# in the default parameter file, this one is in pupil
DM3 = OptSy.deformable_mirror(modelconfig,
                              DMconfig,
                              Name_DM='DM3',
                              Model_local_dir=Model_local_dir)

# DMs are also optical systems but have another parameter to control them, DMphase
# measure the EF throught the DM
EF_though_DM = DM3.EF_through(entrance_EF=1., DMphase=0.)
# plus all normal function of optical systems (todetector, todetector_Intensity)

# for example, something like can be funny:
EF_though_DM = DM3.EF_through(entrance_EF=aberrated_EF, DMphase=phase)

# Now that we have all these Optical Systems defined, we can play with it and concatenate them.
# The concatenate function takes 2 parameters:
#                               - a list of Optical Systems
#                               - A list of the same size of the name of those system so that you can access it
# The list order is from the first optics system to the last in the
# path of the light (so usually from entrance pupil to Lyot pupil)
testbed_1DM = OptSy.Testbed([pup_round, DM3, corono],
                            ["entrancepupil", "DM3", "corono"])

# each of the subsystem can now be access individually with the name you gave it:
# testbed_1DM.entrancepupil, testbed.DM3, etc

# to avoid any confustion in case of multiple DM, the command to access DMs is now XXXphase, where XXX is nameDM
PSF_after_testbed = testbed_1DM.todetector_Intensity(entrance_EF=aberrated_EF,
                                                     DM3phase=phase)

# and the confusing DM3phase is removed so this will send an error:
# PSF_after_testbed = testbed_1DM.todetector_Intensity(entrance_EF=aberrated_EF, DMphase = phase)

# we can now play with all the things we define up to now. for example:
testbed_1DM_romanpup = OptSy.Testbed([pup_roman, DM3, corono],
                                     ["entrancepupil", "DM3", "corono"])

# if you have DMs in your system, these are saved in the structure so that you can access it:
print("number of DMs in testbed_1DM_romanpup:",
      testbed_1DM_romanpup.number_DMs)
print("name of the DMs: ", testbed_1DM_romanpup.name_of_DMs)

# if we want to define exactly the thd2, we need to add a second DM off pupil plane.
# This can take som time to initialize becasue the DM is off pupil.

# We need to increase the number of pixel in the pupil if we add another DM.
# I'll put it at the minimum to go faster
# because of numerical sampling for the Fresnel propagation
modelconfig.update({'diam_pup_in_pix': 400})

# Once we change modelconfig, all the previously defined systems are of the wrong dimensions so they cannot
# be concatenated adn must be reclacultated
del pup_round, DM3, corono
pup_round = OptSy.pupil(modelconfig)

DM3 = OptSy.deformable_mirror(modelconfig,
                              DMconfig,
                              Name_DM='DM3',
                              Model_local_dir=Model_local_dir)

DMconfig.update({'DM1_active': True})
DM1 = OptSy.deformable_mirror(modelconfig,
                              DMconfig,
                              Name_DM='DM1',
                              Model_local_dir=Model_local_dir)
# we also need to "clear" the apod plane because  there
# is no apod plane on the thd2 bench
Coronaconfig.update({'filename_instr_apod': "ClearPlane"})
corono_thd = OptSy.coronagraph(modelconfig, Coronaconfig)

# and then just concatenate
thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono_thd],
                     ["entrancepupil", "DM1", "DM3", "corono"])

# if you have DMs in your system, these are saved in the structure so that you can access it:
print("number of DMs in thd2:", thd2.number_DMs)
print("name of the DMs: ", thd2.name_of_DMs)

# And Now that we have all the tools, we can concatenate define even more complicated system
# let's define a third DM, similar to DM1, but off pupil in the other dimension
DMconfig.update({'DM1_z_position': -15e-2})  # meter
DMconfig.update({'DM1_active': True})
DMnew = OptSy.deformable_mirror(modelconfig,
                                DMconfig,
                                Name_DM='DM1',
                                Model_local_dir=Model_local_dir)
# the Name_DM in this function is to be understand as the type of DM you want to use (DM3 is a BMC32x32 type DM
# and DM1 is a BMC34x34) but hte real name in the system is to be defined in the concatenation

# We also want to add a pupil in between all these DM. Lets make is a round pupil for now, but we could imagine
# putting an apodizer here.
pupil_inbetween_DM = OptSy.pupil(modelconfig)

# and a roman entrance pupil
pup_roman = OptSy.pupil(modelconfig, PupType="RomanPup")

#lets concatenate everything !
testbed_3DM = OptSy.Testbed(
    [pup_roman, DM1, DM3, pupil_inbetween_DM, DMnew, corono_thd],
    ["entrancepupil", "DM1", "DM3", "pupil_inbetween_DM", "DMnew", "corono"])

print("number of DMs in testbed_3DM:", testbed_3DM.number_DMs)
print("name of the DMs: ", testbed_3DM.name_of_DMs)
