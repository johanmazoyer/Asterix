import os

import numpy as np
from configobj import ConfigObj
from validate import Validator

import Asterix.optical_systems as OptSy
import Asterix.save_and_read as useful


# Set the path to your configuration file of choice
parameter_file = OptSy.Asterix_root + os.path.sep + "Example_param_file.ini"

# Check the picked configuration file against a template of the configuraiton file and load it
configspec_file = OptSy.Asterix_root + os.path.sep + "Param_configspec.ini"
config = ConfigObj(parameter_file, configspec=configspec_file, default_encoding="utf8")
vtor = Validator()
checks = config.validate(vtor, copy=True)   #TODO: what is this supposed to do?

# Load the configuration parameters from the configuration file
modelconfig = config["modelconfig"]
DMconfig = config["DMconfig"]
Coronaconfig = config["Coronaconfig"]
SIMUconfig = config["SIMUconfig"]

# Set the main directory to which output data will be saved
Data_dir = '.'

# Subdirectory to save the .fits files that you need to run simulations but that you can always re-calculate yourself.
# Should not be committed to the repository.
Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
if not os.path.exists(Model_local_dir):
    print(f"Creating directory {Model_local_dir} ...")
    os.makedirs(Model_local_dir)

# Subdirectory to save the results
result_dir = os.path.join(Data_dir, "Results") + os.path.sep
if not os.path.exists(result_dir):
    print(f"Creating directory {result_dir} ...")
    os.makedirs(result_dir)

# We put a very small number of pixels in the pupil to go super fast in this tutorial
modelconfig.update({'diam_pup_in_pix': 80})

# we start the tutorial by initializing a pupil
# Clear pupil of radius prad as defined by the diameter in pixels above
# (across the diameter of the telescope; usually in the configfile)
pup_round = OptSy.Pupil(modelconfig)

# If you have an input file, you can initialize a complex pupil with the right size.
# This loads the Roman pupil of same radius/diameter as defined above (usually in the configfile).
pup_roman = OptSy.Pupil(modelconfig, PupType="RomanPup")

# By default, all pupils have the diameter of the telescope but smaller/larger pupils
# can also be defined. This can be useful for example for the Lyot stop.
# Here, create a clear pupil with a radius of 100 pixels.
pup_round_100 = OptSy.Pupil(modelconfig, prad=100)

# Careful, pup_roman is not an array, it is an Optical System object.
# If you want to access the pupil itself as an attribute that is an array, and save it out if you like.
numpy_array_pup = pup_roman.pup
useful._quickfits(numpy_array_pup, dir=result_dir, name="numpy_array_pup")

# --> Once you defined an Optical System object, so we can easily access several feature of this class.

# Measure the electrical field (EF) right after the pupil optical element:
# Default is monochromatic light at a center wavelength
EF_through_roman = pup_roman.EF_through(entrance_EF=1.)

#  Calculate the associated PSF. Default is polychromatic.
psf_roman = pup_roman.todetector_intensity()
useful._quickfits(psf_roman, dir=result_dir, name="psf_roman")

# The chromaticity of the source is defined in all opitcal systems with three parameters:
print("Central wavelength: ", pup_roman.wavelength_0)
print("Bandwidth: ", pup_roman.Delta_wav)
print("Number of sub wavelengths: ", pup_roman.nb_wav)

# Also, all OpticalSystem objects have a transmission, which is the ratio of flux after the system, compared to a clear
# aperture of equivalent radius:
transmission_roman_pup = pup_roman.transmission()
print("transmission pup roman = ", transmission_roman_pup)

# If you know what you are doing you can update the parameters of configuration file on the go.

# We can also change the bandwidth.
modelconfig.update({'Delta_wav': 50e-9})
# But then Optical System object has to be reinitialized:
pup_roman_poly = OptSy.Pupil(modelconfig, PupType="RomanPup")

# Be careful when you change the "modelconfig" in the configuration file because Optical Systems are designed to work
# together but that only works if they are defined with the same baseline configuration.

# Let us go back to a monochromatic case for now.
modelconfig.update({'Delta_wav': 0})

# Now let us initialize a coronagraph.

# A coronagraph is a system composed of 3 planes. An apodization plane (PP), a FPM (FP) and a Lyot stop (PP).
# The coronagraph currently in the configuration file does not have an apodization pupil
# because there is no such plane on the THD2 bench, but we can put one in, which is what we do with RoundPup below.

Coronaconfig.update({'filename_instr_apod': "RoundPup"})

# Below, there is some stuff happening that is currently not part of this tutorial.

# modelconfig.update({'diam_pup_in_pix': 128})
# modelconfig.update({'overpadding_pupilplane_factor': 2.0})
# modelconfig.update({'dimScience': 192})
# modelconfig.update({'Science_sampling': 8.})

# Coronaconfig.update({'corona_type': "classicLyot"})
# Coronaconfig.update({'rad_lyot_fpm': 5.})
# Coronaconfig.update({'filename_instr_lyot': "RoundPup"})
# Coronaconfig.update({'diam_lyot_in_m': modelconfig["diam_pup_in_m"]*0.97})

corono = OptSy.Coronagraph(modelconfig, Coronaconfig)

# For the coronagraph, we can measure 2 types of PSFs: with or without the FPM
No_mask_PSF = corono.todetector_intensity(center_on_pixel=True, noFPM=True)
# This allows us to normalize the images.
Max_No_mask_PSF = np.max(No_mask_PSF)

# Set an initial phase and amplitude
phase = corono.generate_phase_aberr(SIMUconfig,
                                    Model_local_dir=Model_local_dir)

# From this phase we create an electrical field. This is a general
# aberrated field before any pupil multiplication.
aberrated_EF = corono.EF_from_phase_and_ampl(phase_abb=phase)

# By default, this is the normal coronagraphic PSF (with FPM)
coronagraphic_PSF = corono.todetector_intensity(entrance_EF=aberrated_EF)

# Which we can now normalize:
normalized_coronagraphic_PSF = coronagraphic_PSF / Max_No_mask_PSF
useful._quickfits(normalized_coronagraphic_PSF,
                  dir=result_dir,
                  name="Normalized_coronagraphic_PSF")

# If you want to debug something, you can save out all planes in the propagation, triggered by a keyword in most
# propagation functions and methods.
# First, create a directory to save these plots to.
plot_all_fits_dir = os.path.join(result_dir, "plot_all_fits") + os.path.sep
if not os.path.exists(plot_all_fits_dir):
    os.makedirs(plot_all_fits_dir)

# Be careful, this can produce tens of fits files and more, especially in a correction loop:
FP_after_corono_in_contrast = corono.todetector_intensity(entrance_EF=aberrated_EF,
                                                          save_all_planes_to_fits=True,
                                                          dir_save_all_planes=plot_all_fits_dir) / No_mask_PSF

# Finally, we can initialize some DMs. DMs can be in a pupil or outside a pupil.
# In the default configuraiton file, this one is in a pupil plane.
DM3 = OptSy.DeformableMirror(modelconfig,
                             DMconfig,
                             Name_DM='DM3',
                             Model_local_dir=Model_local_dir)

# DMs are also optical systems but have another parameter to control them, DMphase.
# Measure the EF through the DM:
EF_though_DM = DM3.EF_through(entrance_EF=1., DMphase=0.)
# plus all normal functions of optical systems (todetector, todetector_intensity)

# For example, something like the following can be funny:
EF_though_DM = DM3.EF_through(entrance_EF=aberrated_EF, DMphase=phase)

# Now that we have all these Optical Systems defined, we can play with them and concatenate them.
# The concatenate function takes 2 parameters:
#                               - A list of Optical Systems
#                               - A list of the same size containing the names of those systems so that you can access them
# The list order is from the first optical system to the last in the
# path of the light (so usually from entrance pupil to Lyot pupil).
testbed_1DM = OptSy.Testbed([pup_round, DM3, corono],
                            ["entrancepupil", "DM3", "corono"])

# Each of the subsystems can now be accessed individually with the name you have given it:
# --> testbed_1DM.entrancepupil, testbed.DM3, etc

# To avoid any confusion in case of multiple DMs, the command to access DMs is now XXXphase, where XXX is the name of the DM.
PSF_after_testbed = testbed_1DM.todetector_intensity(entrance_EF=aberrated_EF,
                                                     DM3phase=phase)

# And the confusing DM3phase is removed so this will send an error:
# PSF_after_testbed = testbed_1DM.todetector_intensity(entrance_EF=aberrated_EF, DMphase = phase)

# We can now play with all the things we defined up to now, for example:
testbed_1DM_romanpup = OptSy.Testbed([pup_roman, DM3, corono],
                                     ["entrancepupil", "DM3", "corono"])

# If you have DMs in your system, these are saved in the structure so that you can access them:
print("Number of DMs in testbed_1DM_romanpup:", testbed_1DM_romanpup.number_DMs)
print("Name of the DMs: ", testbed_1DM_romanpup.name_of_DMs)

# If we want to define exactly the thd2, we need to add a second DM outside the pupil plane.
# This can take som time to initialize exactly because the DM is outside the pupil.

# We need to increase the number of pixels in the pupil if we add another DM.
# I'll put it at the minimum to go faster because of the numerical sampling for the Fresnel propagation.
modelconfig.update({'diam_pup_in_pix': 200})

# Once we change the modelconfig secion of the configuration file/object, all the previously defined systems are of
# the wrong dimensions so they cannot be concatenated and must be recalculated.
del pup_round, DM3, corono
pup_round = OptSy.Pupil(modelconfig)

DM3 = OptSy.DeformableMirror(modelconfig, DMconfig, Name_DM='DM3', Model_local_dir=Model_local_dir)

DMconfig.update({'DM1_active': True})

DM1 = OptSy.DeformableMirror(modelconfig,
                             DMconfig,
                             Name_DM='DM1',
                             Model_local_dir=Model_local_dir)

# We also need to "clear" the apodizer plane because  there is no apodizer plane on the thd2 bench.
Coronaconfig.update({'filename_instr_apod': "Clear"})
corono_thd = OptSy.Coronagraph(modelconfig, Coronaconfig)

# And then just concatenate:
thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono_thd],
                     ["entrancepupil", "DM1", "DM3", "corono"])

# If you have DMs in your system, these are saved in the structure so that you can access them:
print("Number of DMs in thd2:", thd2.number_DMs)
print("Name of the DMs: ", thd2.name_of_DMs)

# And Now that we have all the tools, we can define even more complicated systems.
# Let's define a third DM, similar to DM1, but outside the pupil in the other dimension.
DMconfig.update({'DM1_z_position': -15e-2})  # meter
DMconfig.update({'DM1_active': True})

DMnew = OptSy.DeformableMirror(modelconfig,
                               DMconfig,
                               Name_DM='DM1',
                               Model_local_dir=Model_local_dir)
# The variable Name_DM in this function is to be understood as the type of DM you want to use (DM3 is a BMC32x32 type DM
# and DM1 is a BMC34x34) but the real name in the system is to be defined in the concatenation.

# We also want to add a pupil in between all these DMs. Let's make it a round pupil for now, but we could imagine
# putting an apodizer here.
pupil_inbetween_DM = OptSy.Pupil(modelconfig)

# And a roman entrance pupil
pup_roman = OptSy.Pupil(modelconfig, PupType="RomanPup")


# Let's concatenate everything!

testbed_3DM = OptSy.Testbed([pup_roman, DM1, DM3, pupil_inbetween_DM, DMnew, corono_thd],
                            ["entrancepupil", "DM1", "DM3", "pupil_inbetween_DM", "DMnew", "corono"])

print("Number of DMs in testbed_3DM:", testbed_3DM.number_DMs)
print("Name of the DMs: ", testbed_3DM.name_of_DMs)
