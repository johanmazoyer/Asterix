import numpy as np
from Asterix.optics.propagation_functions import mft
from Asterix.utils import crop_or_pad_image, rebin

from Asterix.utils.save_and_read import quickfits

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


def make_apodizer(dim_pp, prad, apodizer_profile, grey_pup_bin_factor=1, center_pos='b'):
    """
    Return a generic apodizer, by apllying a given transmission profil on the round pupil.
    The transmission profile must be a fonction of the radial coordinate.
    The radial coordinate is 0 at the center and 1 on the outer edge, namely at prad.

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
       Pupil radius within the image array (in pixels)
    apodizer_profile : function.
        Apodizer radial transmission. It is a function of the radial coordinate.
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
        apodizer_pupil : 2D array.
            Apodizer pupil
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

        apodizer_pupil_large = make_apodizer(dimpp_pup_large,
                                             grey_pup_bin_factor * prad,
                                             apodizer_profile,
                                             grey_pup_bin_factor=1,
                                             center_pos=center_pos)
        return crop_or_pad_image(rebin(apodizer_pupil_large,
                                       factor=grey_pup_bin_factor,
                                       center_on_pixel=center_on_pixel),
                                       dim_pp)

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

        apodizer_pupil = np.zeros((dim_pp, dim_pp))
        apodizer_pupil[rr <= prad] = apodizer_profile(rr[rr <= prad] / prad)

        return apodizer_pupil


def make_spider(dim_pp, starting_point, finishing_point, w_spiders, center_pos='b'):
    """
    Make a unique spider from starting_point to finishing point, of width w_spiders

    AUTHORS : Johan Mazoyer, heavily inspired by Emiel Por in HCIpy

    Parameters
    ----------
    dim_pp : int
        Size of the pupil plane (in pixels)
    starting_point: tupple of float
        (x0,y0) the starting point of the spider in pixel
    finishing_point: tupple of float
        (x0,y0) the finishing point of the spider in pixel
    w_spiders: float
        width of the spider in pixel
    center_pos : string (optional, default 'b')
        Option for the center pixel.
        If 'p', center on the pixel dim_pp//2.
        If 'b', center in between pixels dim_pp//2 -1 and dim_pp//2, for 'dim_pp' odd or even.

    Returns
    ------
    spider_map : 2D bool array
        spider boolean array
    """

    delta = np.array(finishing_point) - np.array(starting_point)
    shift = delta / 2 + np.array(starting_point)

    spider_angle = np.arctan2(delta[1], delta[0])
    spider_length = np.sqrt(delta[0]**2 + delta[1]**2)

    xx, yy = np.meshgrid(np.arange(dim_pp) - dim_pp // 2, np.arange(dim_pp) - dim_pp // 2)

    if center_pos.lower() == 'b':
        xx = xx + 1 / 2 - shift[0]
        yy = yy + 1 / 2 - shift[1]
    elif center_pos.lower() == 'p':
        xx = xx - shift[0]
        yy = yy - shift[1]
    else:
        raise Exception("center_pos can only be 'p' or 'b'")

    xx_rot = xx * np.cos(spider_angle) + yy * np.sin(spider_angle)
    yy_rot = yy * np.cos(spider_angle) - xx * np.sin(spider_angle)

    spider_map = xx_rot <= (spider_length / 2)
    spider_map *= xx_rot >= (-spider_length / 2)
    spider_map *= yy_rot <= (w_spiders / 2)
    spider_map *= yy_rot >= (-w_spiders / 2)

    return spider_map


def make_VLT_pup(dim_pp,
                 prad,
                 pupangle=0,
                 spiders=True,
                 grey_pup_bin_factor=1,
                 center_pos='b',
                 reduce_outer_radius=0,
                 add_central_obs=0,
                 add_spider_thickness=0):
    """
    Return VLT pup, heavily inspired by HCIpy.

    AUTHORS : Johan Mazoyer, heavily inspired by Emiel Por in HCIpy and with help from C. Goulas
    I used the number in a slide given by Anthony available here:
    https://www.dropbox.com/s/so0wpq58wh5i5o2/pupil_VLT.pdf?dl=1

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
       Pupil radius within the image array (in pixels)
    pupangle : float
        pupil rotation angle in deg
    spiders : bool, (default True)
        if False, return the VLT pupil without spiders
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
    reduce_outer_radius : float (default 0)
        reduced diameter of outer radius in fraction of the diameter
    add_central_obs : float (default 0)
        additional diameter of central obstruction in fraction of the diameter
    add_spider_thickness : float (default 0)
    additional spiders width in fraction of the diameter

    Returns
    ------
    VLTpupil : 2D numpy array
        VLT transmission pupil of shape (pupdiam, pupdiam), filled with 0 and 1
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

        return crop_or_pad_image(
            rebin(make_VLT_pup(dimpp_pup_large,
                               grey_pup_bin_factor * prad,
                               pupangle=pupangle,
                               spiders=spiders,
                               grey_pup_bin_factor=1,
                               center_pos=center_pos,
                               reduce_outer_radius=reduce_outer_radius,
                               add_central_obs=add_central_obs,
                               add_spider_thickness=add_spider_thickness),
                  factor=grey_pup_bin_factor,
                  center_on_pixel=center_on_pixel), dim_pp)

    pupil_diameter = 8.0  # meter
    central_obscuration_diam = (1.1 / pupil_diameter + add_central_obs) * 2 * prad  # pix

    spider_width = (0.040 / pupil_diameter + add_spider_thickness) * 2 * prad  # pix
    spider_offset = 0.4045 / pupil_diameter * 2 * prad  # pix
    spider_outer_radius = 4.2197 / pupil_diameter * 2 * prad  # pix
    angle_between_spiders = 101  # degrees

    VLTpupil = np.zeros((dim_pp, dim_pp))

    pup_map = roundpupil(dim_pp, prad * (1 - reduce_outer_radius / 2), center_pos=center_pos) == 1
    VLTpupil[pup_map] = 1
    central_obs_map = 1 - roundpupil(dim_pp, central_obscuration_diam / 2, center_pos=center_pos) == 0
    VLTpupil[central_obs_map] = 0

    if spiders:
        pupangle_rad = np.deg2rad(pupangle)

        spider_inner_radius = spider_offset / np.cos(np.radians(45 - (angle_between_spiders - 90) / 2))

        spider_start_1 = -spider_inner_radius * np.array(
            [np.cos(np.pi / 4 + pupangle_rad), np.sin(np.pi / 4 + pupangle_rad)])
        spider_end_1 = spider_outer_radius * np.array([np.cos(np.pi + pupangle_rad), np.sin(np.pi + pupangle_rad)])

        spider_start_2 = -spider_inner_radius * np.array(
            [np.cos(np.pi / 4 + pupangle_rad), np.sin(np.pi / 4 + pupangle_rad)])
        spider_end_2 = spider_outer_radius * np.array(
            [np.cos(-np.pi / 2 + pupangle_rad), np.sin(-np.pi / 2 + pupangle_rad)])

        spider_start_3 = spider_inner_radius * np.array(
            [np.cos(np.pi / 4 + pupangle_rad), np.sin(np.pi / 4 + pupangle_rad)])
        spider_end_3 = spider_outer_radius * np.array([np.cos(0 + pupangle_rad), np.sin(0 + pupangle_rad)])

        spider_start_4 = spider_inner_radius * np.array(
            [np.cos(np.pi / 4 + pupangle_rad), np.sin(np.pi / 4 + pupangle_rad)])
        spider_end_4 = spider_outer_radius * np.array(
            [np.cos(np.pi / 2 + pupangle_rad), np.sin(np.pi / 2 + pupangle_rad)])

        spider_map1 = make_spider(dim_pp, spider_start_1, spider_end_1, spider_width)
        spider_map2 = make_spider(dim_pp, spider_start_2, spider_end_2, spider_width)
        spider_map3 = make_spider(dim_pp, spider_start_3, spider_end_3, spider_width)
        spider_map4 = make_spider(dim_pp, spider_start_4, spider_end_4, spider_width)

        VLTpupil[spider_map1] = 0
        VLTpupil[spider_map2] = 0
        VLTpupil[spider_map3] = 0
        VLTpupil[spider_map4] = 0

    return VLTpupil


def sphere_apodizer_radial_profile(x):
    """
    Compute the transmission radial profile of the SPHERE APO1 apodizer.
    x is the radial coordinate inside the pupil
    x = 0 at the center of the pupil and x = 1 on the outer edge
    This profile has been estimated with a five order polynomial fit.
    Don't go inside the central obstruction, namely x < 0.14,
    as the fit is no longer reliable.

    Parameters
    ----------
    x : float or array
        distance to the pupil center, in fraction of the pupil radius

    Returns
    ------
    profile : float or array
        corresponding transmission
    """
    a = 0.16544446129778326
    b = 4.840243632913415
    c = -12.02291052479871
    d = 7.549499000031292
    e = -1.031115714037546
    f = 0.8227341447351052
    profile = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    return profile


def make_sphere_apodizer(dim_pp, prad, grey_pup_bin_factor=1, center_pos='b'):
    """
    Return the SPHERE APO1 apodizer pupil.

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
    sphere_apodizer : 2D array
        sphere APO1 apodizer pupil
    """

    sphere_apodizer = make_apodizer(dim_pp,
                                    prad,
                                    sphere_apodizer_radial_profile,
                                    grey_pup_bin_factor=grey_pup_bin_factor,
                                    center_pos=center_pos)
    sphere_apodizer *= make_VLT_pup(dim_pp,
                                    prad,
                                    pupangle=0,
                                    spiders=False,
                                    grey_pup_bin_factor=grey_pup_bin_factor,
                                    center_pos=center_pos)
    return sphere_apodizer


def make_sphere_lyot(dim_pp, prad, pupangle=0, spiders=True, grey_pup_bin_factor=1, center_pos='b'):
    """ 
    Return SPHERE Lyot stop aperture

    values of additional central obstruction, spiders size and
    outer edge obstruction have been estimated by eye on the real lyot stop
    warning : this lyot stop does not feature the dead actuators patches

    AUTHORS : Johan Mazoyer from C. Goulas

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
       Pupil radius within the image array (in pixels). Careful this is not the radius of the lyot, 
        but the pupil associated to this Lyot.
    pupangle : float
        pupil rotation angle in deg
    spiders : bool, (default True)
        if False, return the VLT pupil without spiders
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
    SPHERELyotStop : 2D numpy array
        SPHERE Lyot Stop transmission pupil of shape (dim_pp, dim_pp), filled with 0 and 1
    """

    addCentralObs = 7.3 / 100
    addSpiderObs = 3.12 / 100
    lyotOuterEdgeObs = 1.8 / 100

    return make_VLT_pup(dim_pp,
                        prad,
                        pupangle=pupangle,
                        spiders=spiders,
                        grey_pup_bin_factor=grey_pup_bin_factor,
                        center_pos=center_pos,
                        reduce_outer_radius=lyotOuterEdgeObs,
                        add_central_obs=addCentralObs,
                        add_spider_thickness=addSpiderObs)