import numpy as np
import scipy.optimize as opt
import scipy.ndimage as nd
import Asterix.propagation_functions as prop
import Asterix.fits_functions as useful


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


def resampling(image, new):
    """ --------------------------------------------------
    Resample the focal plane image to create a 2D array with new dimensions

    - v1.0 2020 A. Potier
    - v2.0 19/03/21 J Mazoyer clean names + if image is real, result is real.
    - v3.0 05/2021 J Mazoyer Replacing currenly with standard pyhton function scipy.ndimage.zoom

    Parameters
    ----------
    image: 2D array
        input image
    new: int
        Size of the output image after resampling in pixels

    Returns
    ------
    Gvector: 2D array
        image resampled into new dimensions


    -------------------------------------------------- """

    dimScience = len(image)

    # THe old function is decentering the PSF (it is not centered between 4 pixels) !!
    # Replacing currenly with standard pyhton function scipy.ndimage.zoom
    # TODO Check that this is equivalent to what is done on the testbed !

    # fftimage_cropped = cropimage(
    #     np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image))), dimScience / 2,
    #     dimScience / 2, new)
    # resized_image = np.fft.fftshift(
    #     np.fft.fft2(np.fft.ifftshift(fftimage_cropped)))

    # if np.isrealobj(image):
    #     resized_image = np.real(resized_image)

    resized_image = nd.zoom(image, new / dimScience)

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


def generic_simu_grid(Nact1D, pitchDM, diam_pup_in_m, diam_pup_in_pix,
                      dim_array):
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
            Number of actuators of the square DM in 1 direction
    pitchDM: float
            pitch of the DM (distance between actuators),in meter
    diam_pup_in_m : float
            diameter of the pupil in meter
    diam_pup_in_pix : int 
            diameter of the pupil in pixel
    dim_array : int
            total pupil array in pixel
    
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

    # the pupil is centered in between 4 pixels and dim_array is always even number
    pupilcenter = [dim_array // 2 + 1 / 2, dim_array // 2 + 1 / 2]

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
        actu_center_pos = pos_actu_in_pix[:, (Nact1D**2 - 1) / 2]
        pos_actu_in_pix = pos_actu_in_pix - actu_center_pos + pupilcenter

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

        actu_05actfromcenter = pos_actu_in_pix[:, Nact1D // 2 * Nact1D +
                                               Nact1D // 2]
        for i in range(Nact1D**2):
            pos_actu_in_pix[:, i] += pupilcenter - actu_05actfromcenter + [
                0.5 * pitchDM_in_pix, 0.5 * pitchDM_in_pix
            ]
    return pos_actu_in_pix


def SinCosBasis(sqrtNbActuators):
    """ --------------------------------------------------
    For a given number of actuator accross the pupil, create coefficients for the sin/cos basis
    Currently works only for a even number of actuator accross the pupil 

    TODO Check that this is equivalent to what is done on the testbed !
    
    AUTHOR: Johan Mazoyer
    
    Parameters
    ----------
    sqrtNbActuators : float
        Numnber of actuator accross the pupil
    

    Returns
    ------
    SinCosBasis : 3D array 
                Coefficient to apply to DMs to obtain sinus and cosinus.
                size :[(sqrtNbActuators)^2,sqrtNbActuators,sqrtNbActuators]
    

    -------------------------------------------------- """

    TFCoeffs = np.zeros((sqrtNbActuators**2, sqrtNbActuators, sqrtNbActuators),
                        dtype=complex)
    SinCos = np.zeros((sqrtNbActuators**2, sqrtNbActuators, sqrtNbActuators))

    for Coeff_SinCos in range(sqrtNbActuators**2):
        Coeffs = np.zeros((sqrtNbActuators, sqrtNbActuators), dtype=complex)

        # It's a cosinus
        if Coeff_SinCos < sqrtNbActuators**2 // 2:
            i = Coeff_SinCos // sqrtNbActuators
            j = Coeff_SinCos % sqrtNbActuators
            Coeffs[i, j] = 1 / 2
            Coeffs[sqrtNbActuators - i - 1, sqrtNbActuators - j - 1] = 1 / 2

        # It's a sinus
        else:
            i = (Coeff_SinCos - sqrtNbActuators**2 // 2) // sqrtNbActuators
            j = (Coeff_SinCos - sqrtNbActuators**2 // 2) % sqrtNbActuators
            Coeffs[i, j] = 1 / (2 * 1j)
            Coeffs[sqrtNbActuators - i - 1,
                   sqrtNbActuators - j - 1] = -1 / (2 * 1j)
        TFCoeffs[Coeff_SinCos] = Coeffs

        SinCos[Coeff_SinCos] = np.real(
            prop.mft(TFCoeffs[Coeff_SinCos],
                     sqrtNbActuators,
                     sqrtNbActuators,
                     sqrtNbActuators,
                     X_offset_input=-0.5,
                     Y_offset_input=-0.5))

    return SinCos