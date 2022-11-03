from Asterix.optics import roundpupil
import numpy as np


def test_roundpupil():
    dim = 10
    rad = 4.3

    pupil = roundpupil(dim, rad, grey_pup_bin_factor=2, center_pos='b')
    centerb = (pupil == np.transpose(pupil)).all() and (pupil == np.flip(pupil, axis=0)).all() and (pupil == np.flip(
        pupil, axis=1)).all()

    pupil = roundpupil(dim + 1, rad, grey_pup_bin_factor=3, center_pos='p')
    centerp = (pupil == np.transpose(pupil)).all() and (pupil == np.flip(pupil, axis=0)).all() and (pupil == np.flip(
        pupil, axis=1)).all()

    assert centerb and centerp, f"{dim}-pixel pupil is not centered between pixels, in center of array."
