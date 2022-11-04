from Asterix.optics import roundpupil
import numpy as np


def test_roundpupil():
    dim = 10
    rad = 4.3

    pupil = roundpupil(dim, rad, grey_pup_bin_factor=2, center_pos='b')
    centerb = (pupil == np.transpose(pupil)).all() and (pupil == np.flip(pupil, axis=0)).all() and (pupil == np.flip(
        pupil, axis=1)).all()

    assert centerb, f"{dim}-pixel pupil with center_pos='b' is not centered."

    pupil = roundpupil(dim + 1, rad, grey_pup_bin_factor=3, center_pos='p')
    centerp = (pupil == np.transpose(pupil)).all() and (pupil == np.flip(pupil, axis=0)).all() and (pupil == np.flip(
        pupil, axis=1)).all()

    assert centerp, f"{dim}-pixel pupil with center_pos='p' is not centered."
