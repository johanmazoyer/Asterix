import numpy as np
from Asterix.optics.propagation_functions import mft
from Asterix.utils import crop_or_pad_image, rebin


def roundpupil(dim_pp, prad, grey_pup_bin_factor=1, center_pos='b'):
    """Create a circular pupil.

    With grey_pup_bin_factor >1, this creates an oversized pupil that is then rescaled using rebin to dim_pp

    AUTHORS : Axel Pottier, Johan Mazoyer
    7/9/22 Modified by J Mazoyer to remove the pixel crenellation with rebin and add a better center option
    2/10/22 Modified by J Mazoyer to enhance pixel grey_pup_bin_factor factor

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
       Pupil radius within the image array (in pixels)
    grey_pup_bin_factor : int (default, 1)
        If grey_pup_bin_factor > 1, the pupil is first defined at a very large scale
        (prad = grey_pup_bin_factor*prad) and then rebinned to the given parameter 'prad'.
        This limits the pixel crenellation in the pupil for small pupils.
        If this option is activated (grey_pup_bin_factor>1) the pupil has to be perfectly centered on
        the array because binning while keeping the centering is tricky:
            -if center_pos is 'p', dimpp and grey_pup_bin_factor must both be odd
            -if center_pos is 'b', dimpp and grey_pup_bin_factor must both be even
    center_pos : string (optional, default 'b')
        Option for the center pixel.
        If 'p', center on the pixel dim_pp//2.
        If 'b', center in between pixels dim_pp//2 -1 and dim_pp//2, for 'dim_pp' odd or even.

    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    """

    if grey_pup_bin_factor > 1:
        if not isinstance(grey_pup_bin_factor, int):
            raise ValueError(f"grey_pup_bin_factor must be an integer, currently it is {grey_pup_bin_factor}")

        if center_pos.lower() == 'p' and (dim_pp % 2 == 0 or grey_pup_bin_factor % 2 == 0):
            raise ValueError(("if grey_pup_bin_factor>1, the pupil has to be perfectly centered:",
                              "if center is 'p', dimpp and grey_pup_bin_factor must be odd"))

        if center_pos.lower() == 'b' and (dim_pp % 2 == 1 or grey_pup_bin_factor % 2 == 1):
            raise ValueError(("if grey_pup_bin_factor>1, the pupil has to be perfectly centered:",
                              "if center is 'b', dimpp and grey_pup_bin_factor must be even"))
        # we add valueError conditions because it is very hard to maintain the same centering after the
        # rebin in all conditions

        if center_pos.lower() == 'b':
            dimpp_pup_large = (2 * int(np.ceil(prad))) * grey_pup_bin_factor
            center_on_pixel = False
        elif center_pos.lower() == 'p':
            dimpp_pup_large = (2 * int(np.ceil(prad)) + 1) * grey_pup_bin_factor
            center_on_pixel = True
        else:
            raise ValueError("center_pos must be 'p' (centered on pixel) or 'b' (centered in between 4 pixels)")

        pup_large = roundpupil(dimpp_pup_large, grey_pup_bin_factor * prad, grey_pup_bin_factor=1, center_pos=center_pos)
        return crop_or_pad_image(rebin(pup_large, factor=grey_pup_bin_factor, center_on_pixel=center_on_pixel), dim_pp)

    else:
        xx, yy = np.meshgrid(np.arange(dim_pp) - dim_pp // 2, np.arange(dim_pp) - dim_pp // 2)

        if center_pos.lower() == 'b':
            xx = xx + 1 / 2
            yy = yy + 1 / 2
        elif center_pos.lower() == 'p':
            pass
        else:
            raise ValueError("center_pos must be 'p' (centered on pixel) or 'b' (centered in between 4 pixels)")
        rr = np.hypot(yy, xx)

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
    """
    Create a random phase map, whose PSD decreases as f^(-slope).

    The average is null and the standard deviation is 'phaserms'.

    AUTHOR: Axel Potier

    Parameters
    ----------
    pupil_rad: float
        Radius of the pupil on which the phase rms will be measured.
    dim_image: int
        Size of the output (can be different from 2*pupil_rad).
    phaserms : float
        Standard deviation of the aberration.
    rhoc : float
        See Borde et Traub 2006.
    slope : float
        Slope of the PSD. See Borde et Traub 2006.

    Returns
    ------
    phase : 2D array
        Static random phase map (or OPD)
    """

    # create a circular pupil of the same radius of the given pupil
    # this will be the pupil over which phase rms = phaserms
    # TODO if grey_pupils = True, this is not a grey_pupil like the others. To chance, we need to pass directly
    # the pupil instead of the pupil radius, or to pass grey pupil.
    # This is not very important because it will have a very small impact on the phase level (prob less that 1%)

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
    """
    For a given number of actuator across the DM, create coefficients for the sin/cos basis.

    TODO Check with Pierre that this is equivalent to what is done on the testbed
    TODO Ask Pierre what he thinks: is it possible to do the basis only for the actuators in the pup
        in which case, the important number would be the number of act in the pup ?

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    Nact1D : float
        Number of actuators of a square DM in one of the principal directions.

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
