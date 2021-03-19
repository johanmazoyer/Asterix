__author__ = 'Axel Potier'

import numpy as np
import scipy.optimize as opt

# def butterworth(image, order, length):
#     """ --------------------------------------------------
#     Multiply the image by a butterworth

#     Parameters:
#     ----------
#     image: 2D-array, input image
#     order: butterworth order
#     length: butterworth length

#     Return:
#     ------
#     image*butt: 2D array, same dimension as input frame
#     The input image is multiplied by the butterworth
#     -------------------------------------------------- """

#     dim_im = len(image)
#     xx, yy = np.meshgrid(np.arange(dim_im) - dim_im / 2, np.arange(dim_im) - dim_im / 2)
#     rr = np.hypot(yy, xx)
#     butt = 1 / (1 + (np.sqrt(2) - 1) * (rr / length)**(2 * order))
#     return image * butt


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
    Create a gaussian in 2D 
    
    Parameters:
    ----------
    xy: Tuple object (2,dim1,dim2)  which can be created with:
        x, y = np.mgrid[0:dim1, 0:dim2]
        xy=(x,y)    
    amplitude: Peak of the gaussian function
    sigma_x: Standard deviation of the gaussian function in the x direction
    sigma_y: Standard deviation of the gaussian function in the y direction
    xo: Position of the Gaussian peak in the x direction
    yo: Position of the Gaussian peak in the y direction
    h: Floor amplitude
    flatten : if True (default), the 2D-array is flatten into 1D-array

    Return:
    ------
    The array is the created 2D gaussian function
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
    
    Parameters:
    ----------
    data: 2D array, input image
     
    Return:
    ------
    popt: max, sig_x, sig_y, x_cen, y_cen, angle, offset
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
    Crop and then resample the focal plane image to create a 2D array with new dimensions
    
    Parameters:
    ----------
    image: 2D array, input image
    reechpup: Size of the cropped image before resempling in pixels
    new: Size of the output image after resampling in pixels
    
    Return:
    ------
    Gvector: 2D array, image resampled into new dimensions
    v1.0 2020 A. Potier
    v2.0 19/030/21 J Mazoyer clean names + if image is real, result is real. 
    -------------------------------------------------- """

    dim_im = len(image)

    fftimage_cropped = cropimage(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image))), dim_im / 2,
        dim_im / 2, new)
    resized_image = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(fftimage_cropped)))

    if not np.iscomplexobj(image):
        resize_image = np.real(resized_image)

    return resized_image


def cropimage(img, ctr_x, ctr_y, newsizeimg):
    """ --------------------------------------------------
    Crop an image to create a 2D array with new dimensions
    
    Parameters:
    ----------
    img: 2D array, input image, can be non squared
    ctr_x: Center of the input image in the x direction around which you make the cut
    ctr_y: Center of the input image in the y direction around which you make the cut
    newsizeimg: int
    
    Return:
    ------
    Gvector: 2D array, squared image resampled into new dimensions
    -------------------------------------------------- """
    newimgs2 = newsizeimg / 2
    return img[int(ctr_x - newimgs2):int(ctr_x + newimgs2),
               int(ctr_y - newimgs2):int(ctr_y + newimgs2), ]


def crop_or_pad_image(image, dimout):
    """ --------------------------------------------------
    crop or padd with zero to a 2D image

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim x dim array

    dimout : int
         dimension of the output array

    Returns
    ------
    im_out : 2D array (float)
            if dimout < dim : cropped image around pixel (dim/2,dim/2)
            if dimout > dim : image around pixel (dim/2,dim/2) surrounded by 0

    AUTHOR : Raphaël Galicher

    REVISION HISTORY :
    Revision 1.1  2021-02-10 Raphaël Galicher Initial revision
    Revision 2.0  2021-02-24. JM Rename because cut_image was innacurate
    

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
    Parameters
    ----------
    measured_grid : 2D array (float) of shape is 2 x Nb_actuator
                    x and y measured positions for each actuator (unit = pixel)
    measured_ActuN: 1D array (float) of shape 2
                    x and y positions of actuator ActuN same unit as measured_grid
    ActuN:          int
                    Index of the actuator ActuN (corresponding to measured_ActuN) 
    sampling_simu_over_measured : float
                    Ratio of sampling in simulation grid over sampling in measured grid 
    Returns
    ------
    simu_grid : 2D array of shape is 2 x Nb_actuator
                x and y positions of each actuator for simulation
                same unit as measured_ActuN
    -------------------------------------------------- """
    simu_grid = measured_grid * 0
    for i in np.arange(measured_grid.shape[1]):
        simu_grid[:, i] = measured_grid[:, i] - measured_grid[:, int(
            ActuN)] + measured_ActuN
    simu_grid = simu_grid * sampling_simu_over_measured
    return simu_grid

