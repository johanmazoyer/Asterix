from Asterix.optics import roundpupil
import numpy as np


def test_roundpupil():
    dim = 10
    rad = 4.3

    pupil = roundpupil(dim, rad, grey_pup_bin_factor=2, center_pos='b')
    assert (pupil == np.transpose(pupil)).all(), f"{dim}-pixel pupil with center_pos='b' is not centered."
    assert (pupil == np.flip(pupil, axis=0)).all(), f"{dim}-pixel pupil with center_pos='b' is not centered."

    pupil = roundpupil(dim + 1, rad, grey_pup_bin_factor=3, center_pos='p')
    assert (pupil == np.transpose(pupil)).all(), f"{dim}-pixel pupil with center_pos='p' is not centered."
    assert (pupil == np.flip(pupil, axis=0)).all(), f"{dim}-pixel pupil with center_pos='p' is not centered."
