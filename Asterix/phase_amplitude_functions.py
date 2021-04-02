import numpy as np
import skimage.transform
from astropy.io import fits
import Asterix.processing_functions as proc


##############################################
##############################################
### Pupil
def roundpupil(dim_im, prad1):
    """ --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.

    Parameters
    ----------
    dim_im : int
        Size of the image (in pixels)
    prad1 : float
        Size of the pupil radius (in pixels)

    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil

    AUTHOR : Axel Pottier
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(dim_im) - (dim_im) / 2,
        np.arange(dim_im) - (dim_im) / 2)
    rr = np.hypot(yy + 1 / 2, xx + 1 / 2)
    pupilnormal = np.zeros((dim_im, dim_im))
    pupilnormal[rr <= prad1] = 1.0
    return pupilnormal


def shift_phase_ramp(dim_im, a, b):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_im,dim_im) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = real(fft(ifft(im)*exp(i phase ramp)))

    Parameters
    ----------
    dim_im : int
        Size of the phase ramp (in pixels)
    a : float
        Shift desired in the x direction (in pixels)
    b : float
        Shift desired in the y direction (in pixels)

    Returns
    ------
    masktot : 2D array
        Phase ramp
    -------------------------------------------------- """
    if (a == 0) & (b == 0):
        ramp = 1
    else:
        maska = np.linspace(-np.pi * a, np.pi * a, dim_im)
        maskb = np.linspace(-np.pi * b, np.pi * b, dim_im)
        xx, yy = np.meshgrid(maska, maskb)
        ramp = np.exp(-1j * xx) * np.exp(-1j * yy)
    return ramp


def scale_amplitude_abb(filename, prad, pupil):
    """ --------------------------------------------------
    Scale the map of a saved amplitude map

    Parameters
    ----------
    filename : str
            filename of the amplitude map in the pupil plane

    prad : float
            radius of the pupil in pixel

    pupil : 2D array
            binary pupil array

    Returns
    ------
    ampfinal : 2D array (float)
            amplitude aberrations (in amplitude, not intensity)

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    Revision 1.1  2021-02-18 Raphael Galicher
    Initial revision

    -------------------------------------------------- """

    #File with amplitude aberrations in amplitude (not intensity)
    # centered on the pixel dim/2+1, dim/2 +1 with dim = 2*[dim/2]
    # diameter of the pupil is 148 pixels in this image
    amp = np.fft.fftshift(fits.getdata(filename))

    #Rescale to the pupil size
    amp1 = skimage.transform.rescale(amp,
                                     2 * prad / 148 * 1.03,
                                     preserve_range=True,
                                     anti_aliasing=True,
                                     multichannel=False)
    # Shift to center between 4 pixels
    #bidouille entre le grandissement 1.03 à la ligne au-dessus et le -1,-1 au lieu
    #de -.5,-.5 C'est pour éviter un écran d'amplitude juste plus petit que la pupille
    tmp_phase_ramp = np.fft.fftshift(shift_phase_ramp(amp1.shape[0], -1., -1.))
    amp1 = np.real(
        np.fft.fftshift(np.fft.fft2(np.fft.ifft2(amp1) * tmp_phase_ramp)))

    # Create the array with same size as the pupil

    ampfinal = proc.crop_or_pad_image(amp1, pupil.shape[1])

    #Set the average to 0 inside entrancepupil
    ampfinal = (ampfinal / np.mean(ampfinal[np.where(pupil != 0)]) - np.ones(
        (pupil.shape[1], pupil.shape[1]))) * pupil
    return ampfinal