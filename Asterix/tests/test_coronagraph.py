import os
import numpy as np

from Asterix import Asterix_root
from Asterix.utils import read_parameter_file
from Asterix.optics import Coronagraph


def test_default_coronagraph():
    # Load the example parameter file
    parameter_file_ex = os.path.join(Asterix_root, "Example_param_file.ini")
    config = read_parameter_file(parameter_file_ex)

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
