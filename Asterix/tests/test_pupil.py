from Asterix.optics import roundpupil
from Asterix.optics import roundpupil_shifted
import numpy as np
import pytest


def test_roundpupil():
    dim = 10
    rad = 4.3

    pupil = roundpupil(dim, rad, grey_pup_bin_factor=2, center_pos='b')
    assert (pupil == np.transpose(pupil)).all(), f"{dim}-pixel pupil with center_pos='b' is not centered."
    assert (pupil == np.flip(pupil, axis=0)).all(), f"{dim}-pixel pupil with center_pos='b' is not centered."

    pupil = roundpupil(dim + 1, rad, grey_pup_bin_factor=3, center_pos='p')
    assert (pupil == np.transpose(pupil)).all(), f"{dim}-pixel pupil with center_pos='p' is not centered."
    assert (pupil == np.flip(pupil, axis=0)).all(), f"{dim}-pixel pupil with center_pos='p' is not centered."


@pytest.mark.parametrize('dim,factor,center', [(10, 1, 'b'), (11, 1, 'b'),
                                             (10, 1, 'p'), (11, 1, 'p'),
                                             (10, 2, 'b'), (11, 3, 'p'), (84, 10, 'b')])
def test_roundpupil_shifted_centered(dim, factor, center):
    np.testing.assert_allclose(roundpupil_shifted(dim, 4.3, factor, center),
                               roundpupil(dim, 4.3, factor, center), rtol=0, atol=1e-15)


def test_roundpupil_shifted_translation():
    nominal = roundpupil_shifted(24, 4.3, 10, (11.5, 11.5))
    shifted = roundpupil_shifted(24, 4.3, 10, (13.5, 10.5))
    np.testing.assert_array_equal(shifted, np.roll(nominal, (-1, 2), axis=(0, 1)))
    # An off-array pupil is clipped, never wrapped to the opposite edge.
    assert not roundpupil_shifted(24, 4.3, 10, (-10, 11.5)).any()


@pytest.mark.parametrize('dim,factor', [(24, 3), (25, 4), (84, 10)])
def test_roundpupil_shifted_fractional_center(dim, factor):
    cx, cy, radius = dim / 2 + 0.3, dim / 2 - 0.7, 4.3
    pupil = roundpupil_shifted(dim, radius, factor, (cx, cy))
    y, x = np.indices(pupil.shape)
    assert abs((pupil * x).sum() / pupil.sum() - cx) < 0.1
    assert abs((pupil * y).sum() / pupil.sum() - cy) < 0.1
    assert abs(pupil.sum() - np.pi * radius**2) < 1
    assert np.all((pupil >= 0) & (pupil <= 1))
    assert np.any((pupil > 0) & (pupil < 1))
