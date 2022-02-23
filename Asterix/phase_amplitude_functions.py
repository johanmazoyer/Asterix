import numpy as np
import skimage.transform
from astropy.io import fits
import Asterix.propagation_functions as prop
import Asterix.fits_functions as useful


##############################################
##############################################
### Pupil
def roundpupil(dim_pp, prad, no_pixel=False):
    """ --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.
    no_pixel mode is a way to create very ovsersampled pupil that are then rescale.
    no_pixel is currently not well tested. 

    AUTHOR : Axel Pottier.
    Modified by J Mazoyer to remove the pixel crenellation

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
        Size of the pupil radius (in pixels)
    no_pixel : boolean (defaut false).
                If true, the pupil is first defined at a very large
                scale (dim_pp = 6000) and then rescale to the given parameter dim_pp.
                This limits the pixel crenellation in pupil for small pupils

    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    
    
    -------------------------------------------------- """

    if no_pixel == True:
        dim_pp_small = np.copy(dim_pp)
        dim_pp = 6000
        prad = prad / dim_pp_small * 6000

    xx, yy = np.meshgrid(
        np.arange(dim_pp) - (dim_pp) / 2,
        np.arange(dim_pp) - (dim_pp) / 2)
    rr = np.hypot(yy + 1 / 2, xx + 1 / 2)
    pupilnormal = np.zeros((dim_pp, dim_pp))
    pupilnormal[rr <= prad] = 1.0

    if no_pixel == True:
        pupilnormal = np.array(
            skimage.transform.rescale(pupilnormal,
                                      dim_pp_small / dim_pp,
                                      preserve_range=True,
                                      anti_aliasing=True,
                                      multichannel=False))

    return pupilnormal


def shift_phase_ramp(dim_pp, shift_x, shift_y):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_pp,dim_pp) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = real(fft(ifft(im)*exp(i phase ramp)))
    
    AUTHOR: Axel Potier

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


def random_phase_map(pupil_rad, dim_image, phaserms, rhoc, slope):
    """ --------------------------------------------------
    Create a random phase map, whose PSD decrease in f^(-slope)
    average is null and stadard deviation is phaserms

    AUTHOR: Axel Potier

    Parameters
    ----------
    pupil_rad: int
        radius of the pupil on which the phaserms will be measured

    dim_image: int
        size of the output (can be different than 2*pupil_rad)

    phaserms : float
        standard deviation of aberration

    rhoc : float
        See Borde et Traub 2006

    slope : float
        Slope of the PSD. See Borde et Traub 2006

    Returns
    ------
    phase : 2D array
        Static random phase map (or OPD) generated
    
    
    -------------------------------------------------- """

    # create a circular pupil of the same radius of the given pupil
    # this will be the pupil over which phase rms = phaserms
    pup = roundpupil(pupil_rad, dim_image)

    xx, yy = np.meshgrid(
        np.arange(dim_image) - dim_image / 2,
        np.arange(dim_image) - dim_image / 2)
    rho = np.hypot(yy, xx)
    PSD0 = 1
    PSD = PSD0 / (1 + (rho / rhoc)**slope)
    sqrtPSD = np.sqrt(2 * PSD)

    randomphase = np.random.randn(
        dim_image, dim_image) + 1j * np.random.randn(dim_image, dim_image)
    phase = np.real(np.fft.ifft2(np.fft.fftshift(sqrtPSD * randomphase)))
    phase = phase - np.mean(phase[np.where(pup == 1.)])
    phase = phase / np.std(phase[np.where(pup == 1.)]) * phaserms
    return phase



def SinCosBasis(Nact1D):
    """ --------------------------------------------------
    For a given number of actuator accross the DM, create coefficients for the sin/cos basis

    TODO Check with Pierre that this is equivalent to what is done on the testbed
    TODO Ask Pierre what he thinks: is it possible to do the basis only for the actuators in the pup
        in which case, the important number would be the number of act in the pup ?
    
    AUTHOR: Johan Mazoyer
    
    Parameters
    ----------
    Nact1D : float
        Numnber of actuators of a square DM in one of the principal direction
    
    Returns
    ------
    SinCosBasis : 3D array 
                Coefficient to apply to DMs to obtain sine and cosine phases.
                size :[(Nact1D)^2,Nact1D,Nact1D] if even
                size :[(Nact1D)^2 -1 ,Nact1D,Nact1D] if odd (to remove piston)
    
    
    -------------------------------------------------- """

    TFCoeffs = np.zeros((Nact1D**2, Nact1D, Nact1D), dtype=complex)
    SinCos = np.zeros((Nact1D**2, Nact1D, Nact1D))

    for Coeff_SinCos in range(Nact1D**2):
        Coeffs = np.zeros((Nact1D, Nact1D), dtype=complex)
        #  the First half of basis are cosine and the second half are sines

        # Lets start with the cosines
        if Coeff_SinCos < Nact1D**2 // 2:
            i = Coeff_SinCos // Nact1D
            j = Coeff_SinCos % Nact1D
            Coeffs[i, j] = 1 / 2
            Coeffs[Nact1D - i - 1, Nact1D - j - 1] = 1 / 2

        # # Lets do the sines
        else:
            i = (Coeff_SinCos - Nact1D**2 // 2) // Nact1D
            j = (Coeff_SinCos - Nact1D**2 // 2) % Nact1D
            Coeffs[i, j] = 1 / (2 * 1j)
            Coeffs[Nact1D - i - 1, Nact1D - j - 1] = -1 / (2 * 1j)
        TFCoeffs[Coeff_SinCos] = Coeffs

        SinCos[Coeff_SinCos] = np.real(
            prop.mft(TFCoeffs[Coeff_SinCos],
                     Nact1D,
                     Nact1D,
                     Nact1D,
                     X_offset_input=-0.5,
                     Y_offset_input=-0.5))

    if Nact1D % 2 == 1:
        # in the odd case the last one is a piston
        SinCos = SinCos[0:Nact1D**2 - 1]

    return SinCos
