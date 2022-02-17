import numpy as np
import processing_functions as proc
import Asterix.fits_functions as useful


def mft(image,
        real_dim_input,
        dim_output,
        nbres,
        inverse=False,
        norm='backward',
        X_offset_input=0,
        Y_offset_input=0,
        X_offset_output=0,
        Y_offset_output=0):
    """
    Based on Matrix Direct Fourier transform (MFT) from R. Galicher
    (cf. Soummer et al. 2007, OSA)
        - Return the Matrix Direct Fourier transform (MFT) of a 2D image
        - Can deal with any size, any position of
            the 0-frequency...
    
    AUTHORS: Baudoz, Galicher, Mazoyer


    REVISION HISTORY :
        -Revision 1.1  2011  Initial revision. Raphaël Galicher (from soummer, in IDL)
        -Revision 2.0  2012-04-12 P. Baudoz (IDL version): added pup offset
        -Revision 3.0  2020-03-10 J. Mazoyer (to python). Replace the MFT with no input offset option
        -Revision 4.0  2020-04-20 J. Mazoyer. change the normalization. Change dim_pup name to be more
                                        coherent. Made better parameter format check

    Parameters
    ----------
        image : 2D array
            Entrance image (entrance size in x and y can be different)

        real_dim_input : int or tupple of int of dim 2
                Diameter of the support in pup (can differ from image.shape)
                Example : real_dim_input = diameter of the pupil in pixel for a padded pupil

        dim_output : int or tupple of int of dim 2
                Dimension of the output in pixels (square if int, rectangular if (int, int)

        nbres: float or tupple of float of dim 2
                Number of spatial resolution elements (same in both directions if float)

        inverse : bool, default False
                direction of the MFT
                inverse = False, direct mft (default value)
                inverse = True, indirect mft

        norm : string default 'backward'
                'backward', 'forward' or 'ortho'. this is the same paramter as in numpy.fft functions
                https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
                if 'backward' no normalisation is done on MFT(inverse = False) and normalisation 1/N is done in MFT(inverse = True)
                if 'forward' 1/N normalisation is done on MFT(inverse = False) and no normalisation is done in MFT(inverse = True)
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

    Returns
    ------
        result : complex 2D array
            Output is a complex array dimft x dimft with the position of the
            0-frequency on the (dim_output_x/2+x1,dim_output_y/2+y1) pixel

    """

    # check dimensions and type of real_dim_input
    error_string_real_dim_input = """dimpup must be an int (square input pupil)
                             or tupple of int of dimension 2"""
    if np.isscalar(real_dim_input):
        if isinstance(real_dim_input, int):
            real_dim_input_x = real_dim_input
            real_dim_input_y = real_dim_input
        else:
            raise Exception(error_string_real_dim_input)
    elif isinstance(real_dim_input, tuple):
        if all(isinstance(dims, int)
               for dims in real_dim_input) & (len(real_dim_input) == 2):
            real_dim_input_x = real_dim_input[0]
            real_dim_input_y = real_dim_input[1]
        else:
            raise Exception(error_string_real_dim_input)
    else:
        raise Exception(error_string_real_dim_input)

    # check dimensions and type of dim_output
    error_string_dim_output = """dim_output must be an int (square output)
                                or tupple of int of dimension 2"""
    if np.isscalar(dim_output):
        if isinstance(dim_output, int):
            dim_output_x = dim_output
            dim_output_y = dim_output
        else:
            raise Exception(error_string_dim_output)
    elif isinstance(dim_output, tuple):
        if all(isinstance(dims, int)
               for dims in dim_output) & (len(dim_output) == 2):
            dim_output_x = dim_output[0]
            dim_output_y = dim_output[1]
        else:
            raise Exception(error_string_dim_output)
    else:
        raise Exception(error_string_dim_output)

    # check dimensions and type of nbres
    error_string_nbr = """nbres must be an float or int (square output)
                                or tupple of float or int of dimension 2"""
    if np.isscalar(nbres):
        if isinstance(nbres, (float, int)):
            nbresx = float(nbres)
            nbresy = float(nbres)
        else:
            raise Exception(error_string_nbr)
    elif isinstance(nbres, tuple):
        if all(isinstance(nbresi, (float, int))
               for nbresi in nbres) & (len(nbres) == 2):
            nbresx = float(nbres[0])
            nbresy = float(nbres[1])
        else:
            raise Exception(error_string_nbr)
    else:
        raise Exception(error_string_nbr)

    dim_input_x = image.shape[0]
    dim_input_y = image.shape[1]

    nbresx = nbresx * dim_input_x / real_dim_input_x
    nbresy = nbresy * dim_input_y / real_dim_input_y

    X0 = dim_input_x / 2 + X_offset_input
    Y0 = dim_input_y / 2 + Y_offset_input

    X1 = X_offset_output
    Y1 = Y_offset_output

    # image0 = dcomplex(image)
    xx0 = ((np.arange(dim_input_x) - X0) / dim_input_x)  #Entrance image
    xx1 = ((np.arange(dim_input_y) - Y0) / dim_input_y)  #Entrance image
    uu0 = ((np.arange(dim_output_x) - X1) / dim_output_x -
           1 / 2) * nbresx  #Fourier plane
    uu1 = ((np.arange(dim_output_y) - Y1) / dim_output_y -
           1 / 2) * nbresy  #Fourier plane

    norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y /
                    dim_output_x / dim_output_y)
    if inverse == False:
        if norm == 'backward':
            norm0 = 1.
        elif norm == 'forward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y /
                            dim_output_x / dim_output_y)
        sign_exponential = -1

    elif inverse == True:
        if norm == 'backward':
            norm0 = nbresx * nbresy / dim_input_x / dim_input_y / dim_output_x / dim_output_y
        elif norm == 'forward':
            norm0 = 1.
        elif norm == 'ortho':
            norm0 = np.sqrt(nbresx * nbresy / dim_input_x / dim_input_y /
                            dim_output_x / dim_output_y)
        sign_exponential = 1

    AA = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(uu0, xx0))
    BB = np.exp(sign_exponential * 1j * 2 * np.pi * np.outer(xx1, uu1))
    result = norm0 * np.matmul(np.matmul(AA, image), BB)

    return result


def prop_fresnel(pup, lam, z, rad, prad, retscale=0):
    """ --------------------------------------------------
    Fresnel propagation of electric field along a distance z
    in a collimated beam and in Free space

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    - Revision 1.1  2020-01-22 Raphael Galicher Initial revision

    Parameters
    ----------
    pup : 2D array (complex or real)
        IF retscale == 0
            electric field at z=0
            CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
            where dimpup is the pup array dimension
        ELSE
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

    retscale int 0 or 1:
        IF NOT 0, the function returns the scales
        of the input and output arrays
        IF 0, the function returns the output
        electric field (see Returns)

    Returns
    ------
    IF retscale is 0
        pup_z : 2D array (complex)
                electric field after propagating in free space along
                a distance z
        dxout : float
                lateral sampling in the output array

    ELSE
        dx : float
                lateral sampling in the input array

        dxout : float
                lateral sampling in the output array



    -------------------------------------------------- """
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
    #fac = dx/dxout
    else:
        sign = -1
        # Sampling in the output dim x dim array if FFT
        dxout = rad / prad
        # Sampling in the input dim x dim array if FFT
        dx = np.abs(lam * z / (dxout * dim))
        inverse_mft = True
    # Zoom factor to get the same spatial scale in the input and output array
    #fac = dxout/dx

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
    result = mft(pup * H, 2 * prad, dim, 2 * prad * fac, inverse=inverse_mft)

    # Fresnel factor that applies after Fourier transform
    result = result * np.exp(1j * sign * np.pi * rho**2 / dim * dxout / dx)

    if sign == -1:
        result = result / fac**2
    return result, dxout


def prop_angular_spectrum(pup, lam, z, rad, prad, gamma=2):
    """ --------------------------------------------------
    Angular spectrum propagation of electric field along a distance z
    in a collimated beam and in Free space in close field (small z).

    AUTHOR : Johan Mazoyer

    REVISION HISTORY :
    - Revision 1.0  2022-02-15 Johan Mazoyer Initial revision

    Parameters
    ----------
    pup : 2D array (complex or real)
            electric field at z=0
            CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
            where dimpup is the pup array dimension

    lam : float
         wavelength in meter

    z : float
         distance of propagation in meter

    rad : float
        entrance beam radius in meter
        
    prad : float
        entrance beam radius in pixel
    
    gamma : int >=2
        factor of oversizing in the fourrier plane in diameter of the pupil 
        (gamma*2*prad is the output dim)
        optionnal: default = 2

    Returns
    ------

    pup_z : 2D array (complex) of size [2*gamma*prad,2*gamma*prad]
            electric field after propagating in free space along
            a distance z

    -------------------------------------------------- """

    diam_pup_in_m = 2 * rad
    diam_pup_in_pix = 2 * prad

    Nfourier = gamma * diam_pup_in_pix
    cycles = diam_pup_in_pix

    four = np.fft.fft2(proc.crop_or_pad_image(pup, Nfourier), norm='ortho')
    u, v = np.meshgrid(
        np.arange(Nfourier) - Nfourier / 2,
        np.arange(Nfourier) - Nfourier / 2)
    rho2D = np.fft.fftshift(np.hypot(v,
                                     u)) * (cycles / diam_pup_in_m) / Nfourier

    angular = np.exp(-1j * np.pi * z * lam * (rho2D**2))
    return np.fft.ifft2(angular * four, norm='ortho')
