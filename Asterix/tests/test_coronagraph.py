import os
import numpy as np

from Asterix import Asterix_root
from Asterix.utils import read_parameter_file
from Asterix.optics import Coronagraph, create_wrapped_vortex_mask, fqpm_mask


def test_all_coronagraphs():
    # Load the test parameter file
    parameter_file_test = os.path.join(Asterix_root, 'tests', "param_file_tests.ini")
    config = read_parameter_file(parameter_file_test)

    # Reassign the parameter groups to variables
    modelconfig = config["modelconfig"]
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update({"filename_instr_apod": "RoundPup"})

    # Set coronagraph to be tested
    coros_to_test = ["fqpm", "wrapped_vortex", "classiclyot", "knife", "hlc", "vortex"]
    expected_attenuation = [1e-20, 1e-8, 5e-3, 1e-2, 1e-3, 5e-9]  # Note that these are for the 200 px pupil
    atols = [0, 1e-14, 1e-14, np.nan, 1e-14, 1e-14]  # zeros are for perfect coronagraphs

    for i, coro in enumerate(coros_to_test):
        Coronaconfig.update({"corona_type": coro})

        # Create the coronagraph
        corono = Coronagraph(modelconfig, Coronaconfig)
        coro_psf = corono.todetector_intensity(in_contrast=True)
        print(f"Attenuation of '{coro}' is { np.max(coro_psf)}")

        assert np.max(
            coro_psf) < expected_attenuation[i], f"Attenuation of '{coro}' not below expected {expected_attenuation[i]}."
        if coro != 'knife':
            assert np.allclose(coro_psf, np.transpose(coro_psf), atol=atols[i],
                               rtol=0), f"Coronagraphic image is not symmetric in transpose for '{coro}'."


def test_all_coronagraphs_polychromatic():
    # Load the test parameter file
    parameter_file_test = os.path.join(Asterix_root, 'tests', "param_file_tests.ini")
    config = read_parameter_file(parameter_file_test)

    config["modelconfig"]["Delta_wav"] = 30e-9
    config["modelconfig"]["nb_wav"] = 3

    # Reassign the parameter groups to variables
    modelconfig = config["modelconfig"]
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update({"filename_instr_apod": "RoundPup"})
    Coronaconfig.update({"achrom_phase_coro": False})

    # Set coronagraph to be tested
    coros_to_test = ["fqpm", "wrapped_vortex", "classiclyot", "knife", "hlc", "vortex"]

    # We do not add any performance tests because most coronagraphs aren't great
    # in polychromatic, but it should at least run.
    for coro in coros_to_test:
        Coronaconfig.update({"corona_type": coro})

        # Create the coronagraph
        corono = Coronagraph(modelconfig, Coronaconfig)
        coro_psf = corono.todetector_intensity(in_contrast=True)


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
    assert np.real(np.sum(np.exp(1j * fqpm))) == 0

    dim = fqpm.shape[0]
    hsize = int(dim / 2)
    qsize = int(hsize / 2)
    assert fqpm[qsize, qsize] == np.pi, "Expected pi-quadrant is not pi."
    assert fqpm[-qsize, -qsize] == np.pi, "Expected pi-quadrant is not pi."
    assert fqpm[-qsize, qsize] == 0, "Expected zero-quadrant is not zero."
    assert fqpm[qsize, -qsize] == 0, "Expected zero-quadrant is not zero."
