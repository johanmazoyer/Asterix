import numpy as np
import scipy.optimize as opt
import Asterix.fits_functions as useful
from astropy.io import fits

def twoD_Gaussian(xy,
                  amplitude,
                  sigma_x,
                  sigma_y,
                  xo,
                  yo,
                  theta,
                  h,
                  flatten=True):
    """ --------------------------------------------------
    Create a gaussian in 2D.

    Author : Axel Potier

    Parameters
    ----------
    xy: Tuple object (2,dim1,dim2)  
        which can be created with:
        x, y = np.mgrid[0:dim1, 0:dim2]
        xy=(x,y)

    amplitude: float
        Peak of the gaussian function

    sigma_x: float
        Standard deviation of the gaussian function in the x direction

    sigma_y: float
        Standard deviation of the gaussian function in the y direction
    xo: float
        Position of the Gaussian peak in the x direction
    yo: float
        Position of the Gaussian peak in the y direction
    h: float
        Floor amplitude
    flatten : bool, default True
        if True (default), the 2D-array is flatten into 1D-array

    Returns
    ------
    gauss: 2d numpy array
        2D gaussian function



    -------------------------------------------------- """
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**
                                                 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(
        2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**
                                                 2) / (2 * sigma_y**2)
    g = (amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) *
                              (y - yo) + c * ((y - yo)**2))) + h)
    if flatten == True:
        g = g.flatten()
    return g


def gauss2Dfit(data):
    """ --------------------------------------------------
    Return the parameter of the 2D-Gaussian that best fits data

    Parameters
    ----------
    data: 2D array
        input image

    Returns
    ------
    popt: tupple of float
        parameter of the gaussian: max, sig_x, sig_y, x_cen, y_cen, angle, offset

    -------------------------------------------------- """
    # 2D-Gaussian fit
    popt = np.zeros(8)
    w, h = data.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x, y)

    # Fit 2D Gaussian with fixed parameters
    initial_guess = (np.amax(data), 1, 1, len(data) / 2, len(data) / 2, 0, 0)

    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian,
                                   xy,
                                   data.flatten(),
                                   p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed")

    return popt


def resizing(image, new):
    """ --------------------------------------------------
    Resample the focal plane image to create a 2D array with new dimensions

    - v1.0 2020 A. Potier
    - v2.0 19/03/21 J Mazoyer clean names + if image is real, result is real.
    - v3.0 05/2021 J Mazoyer Replacing currenly with standard pyhton function scipy.ndimage.zoom
    - v4.0 06/2022 J Mazoyer Replacing with the rebin and crop function following discussion with L. Mugnier
    - v5.0 08/2022 J Mazoyer Rename to resizing


    Parameters
    ----------
    image: 2D array
        input image
    new: int
        Size of the output image after resizing, in pixels

    Returns
    ------
    Gvector: 2D array
        image resampled into new dimensions


    -------------------------------------------------- """

    dimScience = len(image)
    dimEstim = new

    Estim_bin_factor = int(np.round(dimScience/dimEstim))

    # if the image was not orinigally a factor of Estim_bin_factor we crop a few raws
    slightly_crop_image = crop_or_pad_image(image,dimEstim*Estim_bin_factor)

    resized_image = resize_crop_bin(slightly_crop_image, dimEstim)

    return resized_image


def cropimage(img, ctr_x, ctr_y, newsizeimg):
    """ --------------------------------------------------
    Crop an image to create a 2D array with new dimensions
    AUTHOR: Axel Potier

    Parameters
    ----------
    img: 2D array
        input image, can be non squared
    ctr_x: float
        Center of the input image in the x direction around which you make the cut
    ctr_y: float
        Center of the input image in the y direction around which you make the cut
    
    newsizeimg: int
        Size of the new image

    Returns
    ------
    Gvector: 2D array
        squared image cropped into new dimensions


    -------------------------------------------------- """
    newimgs2 = newsizeimg / 2
    return img[int(ctr_x - newimgs2):int(ctr_x + newimgs2),
               int(ctr_y - newimgs2):int(ctr_y + newimgs2), ]


def crop_or_pad_image(image, dimout):
    """ --------------------------------------------------
    crop or padd with zero to a 2D image depeding: 
        - if dimout < dim : cropped image around pixel (dim/2,dim/2)
        - if dimout > dim : image around pixel (dim/2,dim/2) surrounded by 0

    AUTHOR: Raphael Galicher

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim x dim array

    dimout : int
         dimension of the output array

    Returns
    ------
    im_out : 2D array (float)
        resized image


    -------------------------------------------------- """
    if float(dimout) < image.shape[0]:
        im_out = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        im_out = image[int((image.shape[0] - dimout) /
                           2):int((image.shape[0] + dimout) / 2),
                       int((image.shape[1] - dimout) /
                           2):int((image.shape[1] + dimout) / 2)]
    elif dimout > image.shape[0]:
        im_out = np.zeros((dimout, dimout), dtype=image.dtype)
        im_out[int((dimout - image.shape[0]) /
                   2):int((dimout + image.shape[0]) / 2),
               int((dimout - image.shape[1]) /
                   2):int((dimout + image.shape[1]) / 2)] = image
    else:
        im_out = image
    return im_out


def rebin(image, factor = 4, center_on_pixel = False):

    """ --------------------------------------------------
    bin the image by a factor. The dimension dim MUST be divisible by factor
    or it will raise an error. It this is not the case, use function resize_crop_bin

    we add a center_on_pixel option . If false we shift the image to put 0 freq in the corner 
    before binning and then shift back.
    The goal is if you have a PSF center in between 4 pixel this property is conserved.
    If center_on_pixel= True there is no way to conserve this property unless we 
    are binning by odd number

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim1 x dim2 array with dim1 and dim2 are divisible by factor

    factor : int
         factor of bin 

    
    center_on_pixe :bool (optional, default: False). 
                If False the PSf is shifhted before bining

    Returns
    ------
    im_out : 2D array (float)
        resized image of size dim1 // 4 x dim2//4

    -------------------------------------------------- """

    dim1, dim2 = image.shape

    if (dim1 % factor != 0) or (dim2 % factor != 0):
            raise Exception(
                "Image in Bin function must be divisible by factor of bin")

    shape = (dim1//factor, factor,
             dim2//factor, factor)
    
    if center_on_pixel is False:
        return np.fft.fftshift(np.fft.fftshift(image).reshape(shape).mean(-1).mean(1))
    else:
        return image.reshape(shape).mean(-1).mean(1)

def resize_crop_bin(image, new_dim, center_on_pixel = False):

    """ --------------------------------------------------
    resize the imge by : 
        - cropping entrance image to nearest multiplicative number of new_dim
        - bin the image to new_dim size

    we add a center_on_pixel option . If false we shift the image to put 0 freq in the corner 
    before binning and then shift back.
    The goal is if you have a PSF center in between 4 pixel this property is conserved.
    If center_on_pixel= True there is no way to conserve this property unless we 
    are binning by odd number

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim1 x dim2 array with dim1 and dim2 are divisible by factor

    new_dim : int
         dimenstion of output image. new_dim must be samller than dim of the entrance image

    center_on_pixe :bool (optional, default: False). 
                If False the PSf is shifhted before bining

    Returns
    ------
    im_out : 2D array (float)
        resized image of size new_dim x new_dim

    -------------------------------------------------- """

    dim1, dim2 = image.shape

    if (dim1 < new_dim) or (dim2 < new_dim):
            raise Exception(
                "new_dim must be samller than dimensions of the entrance image")

    
    # check closest multiplicative factor
    dim_smaller = min(dim1,dim2)
    factor = dim_smaller//new_dim

    # crop at the right size. Careful with the centering @TODO check
    return_image = cropimage(image,dim1//2, dim2//2, factor*new_dim)

    # Bin at the right size
    return_image = rebin(return_image, factor, center_on_pixel = center_on_pixel)

    return return_image

def actuator_position(measured_grid, measured_ActuN, ActuN,
                      sampling_simu_over_measured):
    """ --------------------------------------------------
    Convert the measred positions of actuators to positions for numerical simulation
    
    AUTHOR: Axel Potier
    
    Parameters
    ----------
    measured_grid : 2D array (float) 
                    array of shape 2 x Nb_actuator
                    x and y measured positions for each actuator (unit = pixel)
    measured_ActuN: 1D array (float) 
                    arrayof shape 2. x and y positions of actuator ActuN same unit as measured_grid
    ActuN:          int
                    Index of the actuator ActuN (corresponding to measured_ActuN)
    sampling_simu_over_measured : float
                    Ratio of sampling in simulation grid over sampling in measured grid
    
    
    Returns
    ------
    simu_grid : 2D array 
                Array of shape is 2 x Nb_actuator
                x and y positions of each actuator for simulation
                same unit as measured_ActuN


    -------------------------------------------------- """
    simu_grid = measured_grid * 0
    for i in np.arange(measured_grid.shape[1]):
        simu_grid[:, i] = measured_grid[:, i] - measured_grid[:, int(
            ActuN)] + measured_ActuN
    simu_grid = simu_grid * sampling_simu_over_measured

    useful._quickfits(simu_grid)
    return simu_grid


def generic_actuator_position(Nact1D, pitchDM, diam_pup_in_m, diam_pup_in_pix):
    """ --------------------------------------------------
    Create a grid of position of actuators for generic  DM.
    The DM will then be automatically defined as squared with Nact1D x Nact1D actuators
    and the pupil centered on this DM.
    
    We need N_act1D > diam_pup_in_m / DM_pitch, so that the DM is larger than the pupil.

    at the end compare to the result of actuator_position in case of DM3, it 
    should be relatively close. If we can, we should try that actu 0 is 
    relatively at the same pos. Test with huge DM pitch (2 actus in the pup)


    AUTHOR: Johan Mazoyer
    
    Parameters
    ----------
    Nact1D : int 
            Numnber of actuators of a square DM in one of the principal direction
    pitchDM: float
            Pitch of the DM (distance between actuators),in meter
    diam_pup_in_m : float
            Diameter of the pupil in meter
    diam_pup_in_pix : int 
            Diameter of the pupil in pixel
    
    Returns
    ------
    simu_grid : 2D array 
                Array of shape is 2 x Nb_actuator
                x and y positions of each actuator for simulation
    
    
    -------------------------------------------------- """
    if Nact1D * pitchDM < diam_pup_in_m:
        raise Exception(
            """Nact1D*pitchDM < diam_pup_in_m: The DM is smaller than the pupil"""
        )

    pitchDM_in_pix = pitchDM * diam_pup_in_pix / diam_pup_in_m

    pos_actu_in_pitch = np.zeros((2, Nact1D**2))
    for i in range(Nact1D**2):
        pos_actu_in_pitch[:, i] = np.array([i // Nact1D, i % Nact1D])

    # relative positions in pixel of the actuators
    pos_actu_in_pix = pos_actu_in_pitch * pitchDM_in_pix

    if Nact1D % 2 == 1:
        # if Nact1D if odd, then the center of the DM is the
        # actuator number (Nact1D**2 -1) /2
        #
        # 20 21 22 23 24
        # 15 16 17 18 19
        # 10 11 12 13 14
        # 5  6  7  8  9
        # 0  1  2  3  4
        #
        # 6 7 8
        # 3 4 5
        # 0 1 2
        pos_actu_center_pos = np.copy(pos_actu_in_pix[:, (Nact1D**2 - 1) // 2])
        center_pup = np.array([0.5, 0.5])

        for i in range(Nact1D**2):
            pos_actu_in_pix[:,
                            i] = pos_actu_in_pix[:,
                                                 i] - pos_actu_center_pos + center_pup

    else:

        #if Nact1D is even, the center of the DM is in between 4 actuators
        # (Nact1D -2) //2 * (Nact1D) +  Nact1D//2 -1    is in (-1/2 act, -1/2 act)
        # (Nact1D -2) //2 * (Nact1D) +  Nact1D//2       is in (-1/2 act, +1/2 act)

        # Nact1D //2 * Nact1D +  Nact1D//2 - 1          is in (+1/2 act, -1/2 act)
        # Nact1D //2 * Nact1D +  Nact1D//2              is in (+1/2 act, +1/2 act)

        #  30 31 32 33 34 35
        #  24 25 26 27 28 29
        #  18 19 20 21 22 23
        #  12 13 14 15 16 17
        #  6  7  8  9  10 11
        #  0  1  2  3  4  5

        # 12 13 14 15
        # 8  9  10 11
        # 4  5  6  7
        # 0  1  2  3

        # 2 3
        # 0 1

        pos_actuhalfactfromcenter = np.copy(
            pos_actu_in_pix[:, Nact1D // 2 * Nact1D + Nact1D // 2])
        halfactfromcenter = np.array(
            [0.5 * pitchDM_in_pix, 0.5 * pitchDM_in_pix])

        center_pup = np.array([0.5, 0.5])

        for i in range(Nact1D**2):
            toto = np.copy(pos_actu_in_pix[:, i])
            pos_actu_in_pix[:,
                            i] = pos_actu_in_pix[:,
                                                 i] - pos_actuhalfactfromcenter + halfactfromcenter + center_pup
            pos_actu_in_pix[0, i] *= -1
    return pos_actu_in_pix




def ft_subpixel_shift(image, xshift, yshift, fourier=False):
    """
    ft_subpixel_shift :
    This function returns an image shifted by a non-integer amount via a
    Fourier domain computation.

    The IMAGE must be square and of even width.

    (Based on subpixel_shift.pro from ONERA's IDL library by Laurent Mugnier)
    Renamed into ft_subpixel_shift to be clear on its purpose by Johan Mazoyer

    AUTHORS: L.Mugnier, M.Kourdourli

    image (2D array) : (input) amount of desired shift in X direction.

    xshift (float) : (input) amount of desired shift in X direction.

    yshift (float) : (input) amount of desired shift in Y direction.

    fourier (bool) : (optional input) if this keyword is "True", then the input
               IMAGE is assumed to be already Fourier transformed, i.e. the input is FFT^-1(image).

    return (2D array) : (output) shifted array with respect to the xshift and yshift used as input.
    """
    sz = np.shape(image)
    NP = sz[0]
    NL = sz[1]
    if (NL != NP) or (NP % 2 != 0):
        raise Exception("This routine require square input array of even width")
    if fourier == True:
        ft_image = image
    else:
        ft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))

    x_ramp = np.outer(np.ones(NP), np.arange(NP) - NP / 2)
    y_ramp = np.outer(np.arange(NP) - NP / 2, np.ones(NP))

    # tilt describes the phase term in exp(i*phi) we will use to shift the image
    # by multiplying in the Fourier space and convolving in the direct space

    tilt = (-2 * np.pi / NP) * (xshift * x_ramp + yshift * y_ramp)
    # shift -> exp(i*phi)
    shift = np.cos(tilt) + 1j * np.sin(tilt)
    # inverse FFT to go back to the real space
    shifted_image = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ft_image * shift))))

    return shifted_image