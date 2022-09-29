import numpy as np
from Asterix.utils import crop_or_pad_image


def mft(image,
        real_dim_input=4,
        dim_output=4,
        nbres=1,
        inverse=False,
        norm='backward',
        X_offset_input=0,
        Y_offset_input=0,
        X_offset_output=0,
        Y_offset_output=0,
        only_mat_mult=False,
        AA=None,
        BB=None,
        norm0=None,
        returnAABB=False):
    """Return the Matrix Direct Fourier transform (MFT) of a 2D image.

    Based on Matrix Direct Fourier transform (MFT) from R. Galicher (cf.Soummer et al. 2007, OSA).
    Can deal with any size, any position of the 0-frequency...

    This function measures 2 matrices AA and BB, and the normalization factor 'norm0'. The MFT itself is
    the matrix multiplication norm0 * ((AA @ image) @ BB) (where @ is a matrix multiplication)

    Can be used in a classical way:
    ZeMFT = mft(image,
                real_dim_input=real_dim_input,
                dim_output=dim_output,
                nbres=nbres,
                only_mat_mult=False,
                returnAABB=False)

    or you can measure AA, BB, and norm0 and / or only do the matrix multiplication:
    AA,BB,norm0 = mft(image,
                      real_dim_input=real_dim_input,
                      dim_output=dim_output,
                      nbres=nbres,
                      only_mat_mult=False,
                      returnAABB=True)
    ZeMFT = mft(image,
                AA=AA,
                BB=BB,
                norm0 = norm0,
                only_mat_mult=True,
                returnAABB=False)

    By separating those 2 steps you can save a lot of time. If you are doing a lot of MFTs with the same
    input and output dimension parameters, only the second step (with only_mat_mult=True) need to be done.

    AUTHORS: Baudoz, Galicher, Mazoyer

    REVISION HISTORY :
        -Revision 1.1  2011  Initial revision. Raphaël Galicher (from Soummer, in IDL)
        -Revision 2.0  2012-04-12 P. Baudoz (IDL version): added pup offset
        -Revision 3.0  2020-03-10 J. Mazoyer (to python). Replace the MFT with no input offset option
        -Revision 4.0  2020-04-20 J. Mazoyer. change the normalization. Change dim_pup name to be more
                                        coherent. Made better parameter format check
        -Revision 5.0  2022-03-09 J. Mazoyer. 1/2 pixel error in xx0, xx1, uu0 and uu1. Now MFT of clear
                                        pup if fully real.
        -Revision 6.0  2022-10-11 J. Mazoyer. Introduced the option to do only the measurement of AA and BB
                                        and the option to do only the matrix multiplication. Matrix multiplication
                                        itself is be done separately in mat_mult_mft which allowed GPU maybe.
                                        I tried using numba here to save so time but no improvement

    Parameters
    ----------
        image : 2D array
            Entrance image (entrance size in x and y can be different)
        real_dim_input : int or tuple of ints of dim 2, default 4
            Diameter of the support in pup (can differ from image.shape)
            Example: real_dim_input = diameter of the pupil in pixel for a padded pupil
        dim_output : int or tuple of int of dim 2, default 4
            Dimension of the output in pixels (square if int, rectangular if (int, int).
        nbres: float or tuple of float of dim 2, default 1
            Number of spatial resolution elements (same in both directions if float).
        inverse : bool, default False
            Direction of the MFT.
            If inverse=False, direct mft (default value).
            If inverse=True, indirect mft.
        norm : string default 'backward'
                'backward', 'forward' or 'ortho'. this is the same paramter as in numpy.fft functions
                https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
                if 'backward' no normalisation is done on MFT(inverse = False) and normalisation 1/N is
                done in MFT(inverse = True)
                if 'forward' 1/N normalisation is done on MFT(inverse = False) and no normalisation is
                done in MFT(inverse = True)
                if 'ortho' 1/sqrt(N) normalisation is done in both directions.
                Note that norm = 'ortho' allows you to conserve energy between a focal plane and pupil plane
                The default is 'backward' to be consistent with numpy.fft.fft2 and numpy.fft.ifft2
        X_offset_input : float default 0
                position of the 0-frequency pixel in x for the entrance
                image with respect to the center of the entrance image (real position
                of the 0-frequency pixel on dim_input_x/2+x0)
        Y_offset_input : float default 0
                position of the 0-frequency pixel in Y for the entrance
                image with respect to the center of the entrance image (real position
                of the 0-frequency pixel on dim_input_y/2+y0)
        X_offset_output : float default 0
                position of the 0-frequency pixel in x for the output
                image with respect to the center of the output image (real position
                of the 0-frequency pixel on dim_output_x/2+x1)
        Y_offset_output : float default 0
                position of the 0-frequency pixel in Y for the output
                image with respect to the center of the output image (real position
                of the 0-frequency pixel on dim_output_y/2+y1)
        only_mat_mult: boolean, default False
                if True, we only do the matrix multiplication, but we need AA, BB and norm0 to be provided.
                    in that case all other parameters are not used. Careful, in this mode, it is assumed that
                    the user is 'expert' and no specific error message will be thrown if parameters are wrong.
                    e.g. it will crash if image, AA and BB dimensions are not compatibles
                if False : classical MFT, AA, BB and norm0 parameters are not used
        AA: complex numpy array, default None
                Matrix multiplied in norm0 * ((AA @ image) @ BB). This parameter is only used if only_mat_mult = True
        BB: complex numpy array, default None
                Matrix multiplied in norm0 * ((AA @ image) @ BB). This parameter is only used if only_mat_mult = True
        norm0: float, default None
                Normalization value in matrix multiplication norm0 * ((AA @ image) @ BB).
                This parameter is only used if only_mat_mult = True
        returnAABB: boolean, default False
                if False, the normal MFT(image is returned)
                if True, we return AA, BB, norm0 used to do norm0 * ((AA @ image) @ BB).

    Returns
    ------
        if returnAABB is False:
            MFT of the image : complex 2D array.

        else:
            AA, BB, norm0 : complex 2D array, complex 2D array, float
                terms used in MFT matrix multiplication norm0 * ((AA @ image) @ BB).
    """

    if only_mat_mult:
        return mat_mult_mft(image.astype('complex64'), AA, BB, norm0)

    # check dimensions and type of real_dim_input
    error_string_real_dim_input = "'dimpup' must be an int (square input pupil) or tuple of ints of dimension 2"
    if np.isscalar(real_dim_input):
        if isinstance(real_dim_input, int):
            real_dim_input_x = real_dim_input
            real_dim_input_y = real_dim_input
        else:
            raise TypeError(error_string_real_dim_input)
    elif isinstance(real_dim_input, tuple):
        if all(isinstance(dims, int) for dims in real_dim_input) & (len(real_dim_input) == 2):
            real_dim_input_x = real_dim_input[0]
            real_dim_input_y = real_dim_input[1]
        else:
            raise TypeError(error_string_real_dim_input)
    else:
        raise TypeError(error_string_real_dim_input)

    # check dimensions and type of dim_output
    error_string_dim_output = "'dim_output' must be an int (square output) or tuple of ints of dimension 2"
    if np.isscalar(dim_output):
        if isinstance(dim_output, int):
            dim_output_x = dim_output
            dim_output_y = dim_output
        else:
            raise TypeError(error_string_dim_output)
    elif isinstance(dim_output, tuple):
        if all(isinstance(dims, int) for dims in dim_output) & (len(dim_output) == 2):
            dim_output_x = dim_output[0]
            dim_output_y = dim_output[1]
        else:
            raise TypeError(error_string_dim_output)
    else:
        raise TypeError(error_string_dim_output)

    # check dimensions and type of nbres
    error_string_nbr = "'nbres' must be an float or int (square output) or tuple of float or int of dimension 2"
    if np.isscalar(nbres):
        if isinstance(nbres, (float, int)):
            nbresx = float(nbres)
            nbresy = float(nbres)
        else:
            raise TypeError(error_string_nbr)
    elif isinstance(nbres, tuple):
        if all(isinstance(nbresi, (float, int)) for nbresi in nbres) & (len(nbres) == 2):
            nbresx = float(nbres[0])
            nbresy = float(nbres[1])
        else:
            raise TypeError(error_string_nbr)
    else:
        raise TypeError(error_string_nbr)

    dim_input_x = image.shape[0]
    dim_input_y = image.shape[1]

    nbresx = nbresx * dim_input_x / real_dim_input_x
    nbresy = nbresy * dim_input_y / real_dim_input_y

    X0 = dim_input_x / 2 + X_offset_input
    Y0 = dim_input_y / 2 + Y_offset_input

    X1 = X_offset_output
    Y1 = Y_offset_output

    # image0 = dcomplex(image)
    xx0 = ((np.arange(dim_input_x) - X0 + 1 / 2) / dim_input_x)  # Entrance image
    xx1 = ((np.arange(dim_input_y) - Y0 + 1 / 2) / dim_input_y)  # Entrance image
    uu0 = ((np.arange(dim_output_x) - X1 + 1 / 2) / dim_output_x - 1 / 2) * nbresx  # Fourier plane
    uu1 = ((np.arange(dim_output_y) - Y1 + 1 / 2) / dim_output_y - 1 / 2) * nbresy  # Fourier plane

    norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
    if not inverse:
        if norm == 'backward':
            norm0 = 1.
        elif norm == 'forward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = -1

    else:
        if norm == 'backward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'forward':
            norm0 = 1.
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y)
        sign_exponential = 1

    AA = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(uu0, xx0)).astype('complex64')
    BB = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(xx1, uu1)).astype('complex64')

    if returnAABB:
        return AA, BB, norm0

    return mat_mult_mft(image.astype('complex64'), AA, BB, norm0)


def mat_mult_mft(image, AA, BB, norm0):
    """Perform only the Matrix multiplication for the MFT.

    It is be done separately in to allow this single line function to be sped up.
    I tried using the numba compiler on this function (https://numba.pydata.org/)
    to optimize it, but no improvements. This can probably be optimized with GPU here.

    complex64 type is not mandatory but helps greatly speed up the code. I
    realized that previously (before 2022-10-11) image was float64/complex64, while AA and BB where complex128.
    Therefore the results of the MFT was artificially extended to complex128.

    AUTHOR : Johan Mazoyer
        2022-10-11 Creation from MFT

    Parameters
    ----------
    image : 2D numpy array (complex64)
            Entrance image
    AA: 2D numpy array (complex64)
            Matrix multiplied in norm0 * ((AA @ image) @ BB).
    BB: 2D numpy array (complex64)
            Matrix multiplied in norm0 * ((AA @ image) @ BB).
    norm0: float
            Normalization value in matrix multiplication norm0 * ((AA @ image) @ BB).

    Returns
    ------
    norm0 * ((AA @ image) @ BB) :  2D numpy array (complex64)
    """

    return norm0 * ((AA @ image) @ BB)


def prop_fresnel(pup, lam, z, rad, prad, retscale=0):
    """Fresnel propagation of electric field along a distance z in a collimated
    beam and in free space.

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    - Revision 1.1  2020-01-22 Raphael Galicher Initial revision

    Parameters
    ----------
    pup : 2D array (complex or real)
        if retscale == 0
            electric field at z=0
            CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
            where 'dimpup' is the pup array dimension
        else:
            dim of the input array that will be used for pup
    lam : float
         wavelength in meter
    z : float
         distance of propagation
    rad : float
         if z>0: entrance beam radius in meter
         if z<0: output beam radius in meter
    prad : float
         if z>0: entrance beam radius in pixel
         if z<0: output beam radius in pixel
    retscale : int 0 or 1:
        if not 0, the function returns the scales
        of the input and output arrays
        if 0, the function returns the output
        electric field (see Returns)

    Returns
    ------
    if retscale is 0:
        pup_z : 2D array (complex)
                electric field after propagating in free space along
                a distance z
        dxout : float
                lateral sampling in the output array

    else:
        dx : float
                lateral sampling in the input array
        dxout : float
                lateral sampling in the output array
    """
    # dimension of the input array
    if retscale == 0:
        dim = pup.shape[0]
    else:
        dim = pup

    # if z<0, we consider we go back wrt the real path of the light
    if np.sign(z) == 1:
        sign = 1
        # Sampling in the input dim x dim array if FFT
        dx = rad / prad
        # Sampling in the output dim x dim array if FFT
        dxout = np.abs(lam * z / (dx * dim))
        inverse_mft = False
    # Zoom factor to get the same spatial scale in the input and output array
    # fac = dx/dxout
    else:
        sign = -1
        # Sampling in the output dim x dim array if FFT
        dxout = rad / prad
        # Sampling in the input dim x dim array if FFT
        dx = np.abs(lam * z / (dxout * dim))
        inverse_mft = True
    # Zoom factor to get the same spatial scale in the input and output array
    # fac = dxout/dx

    if retscale != 0:
        return dx, dxout

    # The fac option is removed: not easy to use (aliasing and so on)
    fac = 1

    # create a 2D-array of distances from the central pixel

    u, v = np.meshgrid(np.arange(dim) - dim / 2, np.arange(dim) - dim / 2)
    rho = np.hypot(v, u)
    # Fresnel factor that applies before Fourier transform
    H = np.exp(1j * sign * np.pi * rho**2 / dim * dx / dxout)

    if np.abs(fac) > 1.2:
        print('need to increase lam or z or 1/dx')
        return -1

    # Fourier transform using MFT
    result = mft(pup * H, real_dim_input=2 * prad, dim_output=dim, nbres=2 * prad * fac, inverse=inverse_mft)

    # Fresnel factor that applies after Fourier transform
    result = result * np.exp(1j * sign * np.pi * rho**2 / dim * dxout / dx)

    if sign == -1:
        result = result / fac**2
    return result, dxout


def prop_angular_spectrum(pup, lam, z, rad, prad, gamma=2):
    """Angular spectrum propagation of electric field along a distance z in a
    collimated beam and in free space in close field (small z).

    AUTHOR : Johan Mazoyer

    REVISION HISTORY :
    - Revision 1.0  2022-02-15 Johan Mazoyer Initial revision

    Parameters
    ----------
    pup : 2D array (complex or real)
        electric field at z=0
        CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
        where 'dimpup' is the pup array dimension
    lam : float
         wavelength in meter
    z : float
         distance of propagation in meter
    rad : float
        entrance beam radius in meter
    prad : float
        entrance beam radius in pixel
    gamma : int >=2
        factor of oversizing in the fourier plane in diameter of the pupil
        (gamma*2*prad is the output dim)
        optional: default = 2

    Returns
    ------
    pup_z : 2D array (complex) of size [2*gamma*prad,2*gamma*prad]
        electric field after propagating in free space along
        a distance z
    """

    diam_pup_in_m = 2 * rad
    diam_pup_in_pix = 2 * prad

    Nfourier = gamma * diam_pup_in_pix
    cycles = diam_pup_in_pix

    four = np.fft.fft2(crop_or_pad_image(pup, Nfourier), norm='ortho')
    u, v = np.meshgrid(np.arange(Nfourier) - Nfourier / 2, np.arange(Nfourier) - Nfourier / 2)

    rho2D = np.fft.fftshift(np.hypot(v, u)) * (cycles / diam_pup_in_m) / Nfourier

    angular = np.exp(-1j * np.pi * z * lam * (rho2D**2))
    return np.fft.ifft2(angular * four, norm='ortho')


def fft_choosecenter(image, inverse=False, center_pos='bb', norm='backward'):
    """FFT Computation. IDL "FFT" routine uses coordinates origin at pixel.

    This function uses a coordinate origin at any pixel [k,l],
    thanks to multiplication by adequate array before using numpy routine
    "FFT". Keywords allow convenient origins either at central pixel ('p') or
    between the 4 central pixels ('b')

    AUTHORS: L.Mugnier, M.Kourdourli, J. Mazoyer

    07/09/2022 : Introduction in asterix (Kourdourli's version. Based on fftshift2.pro from ONERA's
                IDL library by Laurent Mugnier
    07/09/2022 : works for non square array / non even dimensions array Mazoyer

    Parameters
    ----------
    input : 2D numpy array
            initial array.
    inverse : bool (optional, default False)
            direction of the FFT,
            inverse == False for direct FFT,
            inverse == True for inverse FFT.
    center_pos : string (optional, default 'bb')
                      option for the origin. Shorthand for specifying
                      the origin center in direct and fourier spaces when
                      manipulating centered arrays.
                       Direct space             Fourier space
               pp     Central pix              Central pix
               pb     Central pix              Between 4 central pix
               bp     Between 4 central pix    Central pix
               bb     Between 4 central pix    Between 4 central pix
               if dim_i (i = x or y) is even or odd :
                    Central pix = dim_i // 2
                    Between 4 central pix: between dim_i // 2 - 1 and dim_i // 2
                with // the euclidian division.
    norm : string default 'backward'
                'backward', 'forward' or 'ortho'. this is the same paramter as in numpy.fft functions
                https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
                if 'backward' no normalisation is done on MFT(inverse = False) and normalisation 1/N
                is done in MFT(inverse = True)
                if 'forward' 1/N normalisation is done on MFT(inverse = False) and no normalisation
                is done in MFT(inverse = True)
                if 'ortho' 1/sqrt(N) normalisation is done in both directions.
                Note that norm = 'ortho' allows you to conserve energy between a focal plane and pupil plane
                The default is 'backward' to be consistent with numpy.fft.fft2 and numpy.fft.ifft2

    Returns
    ------
    FFT_array : 2D numpy array
        FFT of input array with respect to the input centering parameters.
    """

    Nx = np.shape(image)[0]
    Ny = np.shape(image)[1]
    if inverse:
        sens = 1
    else:
        sens = -1

    if center_pos.lower() not in ['pp', 'pb', 'bp', 'bb']:
        raise ValueError("center_pos parameter must be 'pp', 'pb', 'bp', or 'bb' only")

    if center_pos.lower()[0] == 'p':
        direct = np.array([Nx // 2, Ny // 2])
    else:
        direct = np.array([Nx // 2 - 1 / 2., Ny // 2 - 1 / 2.])

    if center_pos.lower()[1] == 'p':
        fourier = np.array([Nx // 2, Ny // 2])
    else:
        fourier = np.array([Nx // 2 - 1 / 2., Ny // 2 - 1 / 2.])

    X, Y = np.meshgrid(np.linspace(0, Ny, Ny, endpoint=False), np.linspace(0, Nx, Nx, endpoint=False))

    # shift in Fourier space, i.e. multiplication in direct space, and computation of FFT
    if not inverse:
        farray = np.fft.fft2(image * np.exp((-sens) * 2. * np.pi * 1j * (fourier[0] * X / Nx + fourier[1] * Y / Ny)),
                             norm=norm)
    else:
        farray = np.fft.ifft2(image * np.exp((-sens) * 2. * np.pi * 1j * (fourier[0] * X / Nx + fourier[1] * Y / Ny)),
                              norm=norm)

    # shift in direct space, i.e. multiplication in fourier space, and computation of FFT
    farray *= np.exp((-sens) * 2. * np.pi * 1j * (direct[0] * X / Nx + direct[1] * Y / Ny))

    # normalisation
    farray *= np.exp(sens * (2. * 1j * np.pi / np.sqrt(Nx * Ny)) * np.sum(direct * fourier))

    return farray


def butterworth_circle(dim, size_filter, order=5, xshift=0, yshift=0):
    """
    Return a circular Butterworth filter.

    AUTHOR: Raphaël Galicher (in IDL)
            ILa (to Python)

    Parameters
    ----------
    dim : int
        Dimension of 2D output array in pixels. If even, filter will be centered on a pixel, but can be shifted to
        between pixels by using xshift=-0.5 and yshift=-0.5. If uneven, filter will be centered between pixels.
    size_filter : int
        Inverse size of the filter.
    order : int
        Order of the filter.
    xshift : float
        Shift of filter with respect to its array in the x direction, in pixels.
    yshift : float
        Shift of filter with respect to its array in the y direction, in pixels.

    Returns
    -------
    butterworth : array
    """
    ty = (np.arange(dim) - yshift - dim / 2)
    tx = (np.arange(dim) - xshift - dim / 2)
    xx, yy = np.meshgrid(ty, tx)

    butterworth = 1 / np.sqrt(1 + (np.sqrt(xx ** 2 + yy ** 2) / np.abs(size_filter) * 2) ** (2. * order))

    return butterworth


def prop_fpm_regional_sampling(pup, fpm, nbres=np.arange(1, 11), samp_outer=2, filter_order=15, alpha=1.5):
    """
    Calculate the coronagraphic electric field in th Lyot plane by using varying sampling in different parts of the FPM.

    Starting from the electric field in an input pupil plane, propagate through a phase-mask coronagraph with
    different sampling values in different rings around the FPM center. Each propagation goes from the pupil to the FPM
    and then to the Lyot plane (without Lyot stop applied), using MFTs, while a (combination of) Butterworth filters
    marks annular regions around the FPM center that are considered in the propagation in each step. The electric field
    contribution from each step is then summed up to yield the total electric field in the Lyot plane (before the LS).

    AUTHOR: R. Galicher (in IDL)
            ILa (to Python)

    Parameters
    ----------
    pup : 2D array
        Input mage array containing the entrance pupil of the optical system.
    fpm : 2D array
        Complex focal-plane mask.
    nbres : 1D array or list
        List of the number of resolution elements in the total image plane for all propagation layers.
    samp_outer : float
        Sampling in the outermost layer of propagations.
    filter_order : int
        Order of the Butterworth filter.
    alpha : float
        Scale factor for the filter size. The larger this number, the smaller the filter size with respect to the
        input array.

    Returns
    -------
    array : E-field before the Lyot stop.
    """
    dim = pup.shape[0]

    # Innermost part of the focal plane
    but_inner = butterworth_circle(dim, dim / alpha, filter_order, -0.5, -0.5)
    efield_before_fpm_inner = mft(pup, real_dim_input=dim, dim_output=dim, nbres=nbres[0])
    efield_before_ls = mft(efield_before_fpm_inner * fpm * but_inner, real_dim_input=dim, dim_output=dim,
                           nbres=nbres[0], inverse=True)

    # From inner to outer part of FPM
    for k in range(nbres.shape[0] - 1):
        # Butterworth filter in each layer
        sizebut_here = dim / alpha * nbres[k] / nbres[k + 1]
        but = (1 - butterworth_circle(dim, sizebut_here, filter_order, xshift=-0.5, yshift=-0.5)) * butterworth_circle(dim,
                                                                                                        dim / alpha,
                                                                                                        filter_order,
                                                                                                        xshift=-0.5,
                                                                                                        yshift=-0.5)
        ef_pre_fpm = mft(pup, real_dim_input=dim, dim_output=dim, nbres=nbres[k + 1])
        ef_pre_ls = mft(ef_pre_fpm * fpm * but, real_dim_input=dim, dim_output=dim, nbres=nbres[k + 1], inverse=True)

        # Sum up E-field contributions before the LS
        efield_before_ls += ef_pre_ls

    # Outer part of the FPM
    nbres_outer = dim / samp_outer
    sizebut_outer = dim / alpha * nbres[-1] / nbres_outer
    but_outer = 1 - butterworth_circle(dim, sizebut_outer, filter_order, xshift=-0.5, yshift=-0.5)

    ef_pre_fpm_outer = mft(pup, real_dim_input=dim, dim_output=dim, nbres=nbres_outer, inverse=True)
    ef_pre_ls_outer = mft(ef_pre_fpm_outer * fpm * but_outer, real_dim_input=dim, dim_output=dim, nbres=nbres_outer,
                          inverse=True)

    # Total E-field before the LS
    efield_before_ls += ef_pre_ls_outer

    return efield_before_ls
