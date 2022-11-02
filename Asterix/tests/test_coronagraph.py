import os
import numpy as np

from Asterix import Asterix_root
from Asterix.utils import read_parameter_file
from Asterix.optics import Coronagraph, create_wrapped_vortex_mask, fqpm_mask


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


def test_wrapped_vortex_phase_mask():
    size = 1000

    thval = np.array([0, 3, 4, 5, 8]) * np.pi / 8
    phval = np.array([3, 0, 1, 2, 1]) * np.pi
    jump = np.array([2, 2, 2, 2]) * np.pi

    angles_1d, phase_1d = create_wrapped_vortex_mask(dim=size, thval=thval, phval=phval, jump=jump, return_1d=True)
    assert np.max(angles_1d) <= 2 * np.pi, "Angles go beyond 2pi."

    angles_2d, phase_2d = create_wrapped_vortex_mask(dim=size, thval=thval, phval=phval, jump=jump, return_1d=False)
    assert np.max(angles_2d) <= 2 * np.pi, "Angles go beyond 2pi."

    angles_2d_shifted, phase_2d_shifted = create_wrapped_vortex_mask(dim=size,
                                                                     thval=thval,
                                                                     phval=phval,
                                                                     jump=jump,
                                                                     return_1d=False,
                                                                     cen_shift=(10, 10))
    assert np.sum(phase_2d - phase_2d_shifted) != 0., "Shifting of phase mask does not work."


def test_fqpm_phase_mask():
    size = 1000

    fqpm = fqpm_mask(size)
    assert np.max(fqpm) == np.pi, "FQPM max is not pi."
    assert np.min(fqpm) == 0, "FQPM min is not 0."
    assert np.sum(np.cos(fqpm)) == 0

    dim = fqpm.shape[0]
    hsize = int(dim / 2)
    qsize = int(hsize / 2)
    assert fqpm[qsize, qsize] == np.pi, "Expected pi-quadrant is not pi."
    assert fqpm[-qsize, -qsize] == np.pi, "Expected pi-quadrant is not pi."
    assert fqpm[-qsize, qsize] == 0, "Expected zero-quadrant is not zero."
    assert fqpm[qsize, -qsize] == 0, "Expected zero-quadrant is not zero."
