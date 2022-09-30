import numpy as np
from Asterix.optics import mft, roundpupil


def test_mft_centering():
    """
    Test default centering of MFT, which in Asterix isi defined to be from between pixels to between pixels.
    """
    pdim = 8
    rad = pdim / 2
    pup = roundpupil(pdim, rad)

    samp = 4
    efield = mft(pup, real_dim_input=pdim, dim_output=pdim, nbres=samp)
    img = np.abs(efield) ** 2

    four_equal_pixels = (img[3,3] == img[3,4]) and (img[3,3] == img[4,3]) and (img[4,3] == img[4,3]) and (img[4,4] == img[4,4])
    assert four_equal_pixels, "PSF from MFT is not symmetric in four center pixels."
    