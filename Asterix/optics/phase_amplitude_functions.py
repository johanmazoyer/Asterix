import numpy as np
from Asterix.optics.propagation_functions import mft
from Asterix.utils import crop_or_pad_image, rebin


def roundpupil(dim_pp, prad, no_pixel=False, center_pos='b'):
    """Create a circular pupil.

    With no_pixel=True, this is a way to create a very oversampled pupil that is then rescaled using rebin.
    See also Asterix.optics.pupil.grey_pupil().

    AUTHORS : Axel Pottier, Johan Mazoyer
    7/9/22 Modified by J Mazoyer to remove the pixel crenellation with rebin and add a better center option

    Parameters
    ----------
    dim_pp : int
        Size of the image array (in pixels).
    prad : float
        Pupil radius within the image array (in pixels).
    no_pixel : boolean (default False).
        If true, the pupil is first defined at a very large
        scale (prad = 10*prad) and then rescaled to the given parameter 'prad'.
        This limits the pixel crenellation in the pupil for small pupils.
        See also Asterix.optics.pupil.grey_pupil().
    center_pos : string (optional, default 'b')
        Option for the center pixel.
        If 'p', center on the pixel dim_pp//2.
        If 'b', center in between pixels dim_pp//2 -1 and dim_pp//2, for 'dim_pp' odd or even.

    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    """

    if no_pixel:
        factor_bin = int(10)
        pup_large = roundpupil(int(2 * prad) * factor_bin, factor_bin * prad, no_pixel=False)
        return crop_or_pad_image(rebin(pup_large, factor=factor_bin, center_on_pixel=False), dim_pp)

    else:
        xx, yy = np.meshgrid(np.arange(dim_pp) - dim_pp // 2, np.arange(dim_pp) - dim_pp // 2)

        if center_pos.lower() == 'b':
            offset = 1 / 2
        elif center_pos.lower() == 'p':
            offset = 0
        else:
            raise Exception("center_pos can only be 'p' or 'b'")

        rr = np.hypot(yy + offset, xx + offset)
        pupilnormal = np.zeros((dim_pp, dim_pp))
        pupilnormal[rr <= prad] = 1.0

        return pupilnormal


def shift_phase_ramp(dim_pp, shift_x, shift_y):
    """
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

    """
    if (shift_x == 0) & (shift_y == 0):
        ramp = 1
    else:
        maskx = np.linspace(-np.pi * shift_x, np.pi * shift_x, dim_pp, endpoint=False)
        masky = np.linspace(-np.pi * shift_y, np.pi * shift_y, dim_pp, endpoint=False)
        xx, yy = np.meshgrid(maskx, masky)
        ramp = np.exp(-1j * xx) * np.exp(-1j * yy)
    return ramp


def random_phase_map(pupil_rad, dim_image, phaserms, rhoc, slope):
    """Create a random phase map, whose PSD decrease in f^(-slope) average is
    null and stadard deviation is phaserms.

    AUTHOR: Axel Potier

    Parameters
    ----------
    pupil_rad: float
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
    """

    # create a circular pupil of the same radius of the given pupil
    # this will be the pupil over which phase rms = phaserms
    pup = roundpupil(dim_image, pupil_rad)

    xx, yy = np.meshgrid(np.arange(dim_image) - dim_image / 2, np.arange(dim_image) - dim_image / 2)
    rho = np.hypot(yy, xx)
    PSD0 = 1
    PSD = PSD0 / (1 + (rho / rhoc)**slope)
    sqrtPSD = np.sqrt(2 * PSD)

    randomphase = np.random.randn(dim_image, dim_image) + 1j * np.random.randn(dim_image, dim_image)
    phase = np.real(np.fft.ifft2(np.fft.fftshift(sqrtPSD * randomphase)))
    phase = phase - np.mean(phase[np.where(pup == 1.)])
    phase = phase / np.std(phase[np.where(pup == 1.)]) * phaserms
    return phase


def sine_cosine_basis(Nact1D):
    """For a given number of actuator across the DM, create coefficients for
    the sin/cos basis.

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
    SinCos : 3D array
        Coefficient to apply to DMs to obtain sine and cosine phases.
        size :[(Nact1D)^2,Nact1D,Nact1D] if even
        size :[(Nact1D)^2 -1 ,Nact1D,Nact1D] if odd (to remove piston)
    """

    TFCoeffs = np.zeros((Nact1D**2, Nact1D, Nact1D), dtype=complex)
    SinCos = np.zeros((Nact1D**2, Nact1D, Nact1D))

    AA, BB, norm0 = mft(TFCoeffs[0],
                        real_dim_input=Nact1D,
                        dim_output=Nact1D,
                        nbres=Nact1D,
                        inverse=False,
                        norm='backward',
                        returnAABB=True)

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

        SinCos[Coeff_SinCos] = np.real(mft(TFCoeffs[Coeff_SinCos], AA=AA, BB=BB, norm0=norm0, only_mat_mult=True))

    if Nact1D % 2 == 1:
        # in the odd case the last one is a piston
        SinCos = SinCos[0:Nact1D**2 - 1]

    return SinCos
