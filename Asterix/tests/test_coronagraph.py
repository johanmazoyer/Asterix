import os
import numpy as np
from configobj import ConfigObj
from validate import Validator

from Asterix import Asterix_root
from Asterix.optics import Coronagraph

def test_default_coronagraph():
    # Load the example parameter file
    parameter_file_ex = Asterix_root + "Example_param_file.ini"
    # Load the template parameter file
    configspec_file = Asterix_root + "Param_configspec.ini"
    # Load configuration - all three of the below lines are necessary
    config = ConfigObj(parameter_file_ex, configspec=configspec_file, default_encoding="utf8")
    vtor = Validator()
    checks = config.validate(vtor, copy=True)

    # Reassign the parameter groups to variables
    modelconfig = config["modelconfig"]
    Coronaconfig = config["Coronaconfig"]

    # Update the pixels across the pupil
    modelconfig.update({'diam_pup_in_pix': 80})
    # Define a round pupil in the apodization plane
    Coronaconfig.update({'filename_instr_apod': "RoundPup"})

    # Create the coronagraph
    corono = Coronagraph(modelconfig, Coronaconfig)
    coro_psf = corono.todetector_intensity(center_on_pixel=True)

    assert np.max(coro_psf) == 0.0, "A perfect coronagraph should return an empty array."
