import numpy as np
import skimage.transform
from astropy.io import fits
import Asterix.processing_functions as proc
import Asterix.fits_functions as useful

##############################################
##############################################
### Pupil
def roundpupil(dim_pp, prad, no_pixel = False):
    """ --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
        Size of the pupil radius (in pixels)

    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil

    AUTHOR : Axel Pottier
    Modified by J Mazoyer to remove the pixellization
    -------------------------------------------------- """

    if no_pixel == True:
        dim_pp_small = np.copy(dim_pp)
        dim_pp = 6000
        prad = prad/dim_pp_small*6000

    xx, yy = np.meshgrid(
        np.arange(dim_pp) - (dim_pp) / 2,
        np.arange(dim_pp) - (dim_pp) / 2)
    rr = np.hypot(yy + 1 / 2, xx + 1 / 2)
    pupilnormal = np.zeros((dim_pp, dim_pp))
    pupilnormal[rr <= prad] = 1.0

    if no_pixel == True:
        pupilnormal = np.array(skimage.transform.rescale(pupilnormal,
                                     dim_pp_small/dim_pp,
                                     preserve_range=True,
                                     anti_aliasing=True,
                                     multichannel=False))


    return pupilnormal


def shift_phase_ramp(dim_pp, shift_x, shift_y):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_pp,dim_pp) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = real(fft(ifft(im)*exp(i phase ramp)))

    Parameters
    ----------
    dim_pp : int
                Size of the phase ramp (in pixels)
    shift_x : float
                Shift desired in the x direction (in pixels)
    shift_y : float
                Shift desired in the y direction (in pixels)

    Returns
    ------
    masktot : 2D array
        Phase ramp
    -------------------------------------------------- """
    if (shift_x == 0) & (shift_y == 0):
        ramp = 1
    else:
        maskx = np.linspace(-np.pi * shift_x, np.pi * shift_x, dim_pp)
        masky = np.linspace(-np.pi * shift_y, np.pi * shift_y, dim_pp)
        xx, yy = np.meshgrid(maskx, masky)
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