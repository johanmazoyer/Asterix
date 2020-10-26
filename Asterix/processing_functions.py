# Version 29 Janvier 2020
import numpy as np
import scipy.optimize as opt
import cv2

# Raccourcis conversions angles
dtor = np.pi / 180.0  # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor


def rotate_frame(img, angle, interpolation="lanczos4", cyx=None):
    """
    Rotates the input frame by the given angle (in degrees), around the center
    of the frame by default, or around the given cxy coordinates.
    
    Parameters
    ----------
    img: 2D-array
        input frame
    angle: float
        angle for the rotation
    interpolation: string
        interpolation method. Can be:
        'cubic', 'linear', 'nearest', 'area', 'lanczos4'
        'lanczos4' is the default as it seems to produce the least artifacts.
    cyx: tuple (cy,cx)
        center of the rotation. If None is given, rotation is performed around
        the center of the frame.
        
    Returns
    -------
    rotated_img: 2D-array
        Rotated frame, same dimensions as inpute frame.
    """
    ny, nx = img.shape

    cx = nx / 2.0
    cy = ny / 2.0

    if not cyx:
        cx = nx / 2.0
        cy = ny / 2.0

    # interpolation type
    if interpolation == "cubic":
        intp = cv2.INTER_CUBIC
    elif interpolation == "linear":
        intp = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        intp = cv2.INTER_NEAREST
    elif interpolation == "area":
        intp = cv2.INTER_AREA
    elif interpolation == "lanczos4":
        intp = cv2.INTER_LANCZOS4

    if np.abs(angle) > 0:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        rotated_img = cv2.warpAffine(img.astype(np.float32), M, (nx, ny), flags=intp)
    else:
        rotated_img = img

    return rotated_img


def butterworth(image, order, length):
    """ --------------------------------------------------
    Multiply the image by a butterworth 
    
    Parameters:
    ----------
    image: 2D-array, input image
    order: butterworth order
    length: butterworth length
    
    
    Return:
    ------
    image*butt: 2D array, same dimension as input frame
    The input image is multiplied by the butterworth
    -------------------------------------------------- """

    isz = len(image)
    xx, yy = np.meshgrid(np.arange(isz) - isz / 2, np.arange(isz) - isz / 2)
    rr = np.hypot(yy, xx)
    butt = 1 / (1 + (np.sqrt(2) - 1) * (rr / length) ** (2 * order))
    return image * butt


def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, h):
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
    
    
    Return:
    ------
    g.flatten(): 1D array
    The array is the created 2D gaussian function
    -------------------------------------------------- """
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    theta = 0
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (
        4 * sigma_y ** 2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (
        2 * sigma_y ** 2
    )
    g = (
        amplitude
        * np.exp(
            -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
        )
        + h
    )
    return g.flatten()


def gauss2Dfit(data):
    """ --------------------------------------------------
    Fit a flattened - 2D gaussian on the input image
    
    Parameters:
    ----------
    data: 2D array, input image
     
    Return:
    ------
    popt[3],popt[4]: x and y position of the gaussian peak
    -------------------------------------------------- """
    # 2D-Gaussian fit
    popt = np.zeros(8)
    w, h = data.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x, y)

    # Fit 2D Gaussian with fixed parameters
    initial_guess = (np.amax(data), 1, 1, len(data) / 2, len(data) / 2, 0)

    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, xy, data.flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed")

    return popt[3], popt[4]


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
    -------------------------------------------------- """
    isz = len(image)
    Gvectorbis = np.fft.ifftshift(image)
    Gvectorbis = np.fft.ifft2(Gvectorbis)
    Gvectorbis = np.fft.fftshift(Gvectorbis)
    Gvector = cropimage(Gvectorbis, isz / 2, isz / 2, new)
    Gvector = np.fft.ifftshift(Gvector)
    Gvector = np.fft.fft2(Gvector)
    Gvector = np.fft.fftshift(Gvector)
    return Gvector


def cropimage(img, ctr_x, ctr_y, newsizeimg):
    """ --------------------------------------------------
    Crop an image to create a 2D array with new dimensions
    
    Parameters:
    ----------
    img: 2D array, input image, can be non squared
    ctr_x: Center of the input image in the x direction
    ctr_y: Center of the input image in the y direction
    newsizeimg: 
    
    Return:
    ------
    Gvector: 2D array, squared image resampled into new dimensions
    -------------------------------------------------- """
    newimgs2 = newsizeimg / 2
    return img[
        int(ctr_x - newimgs2) : int(ctr_x + newimgs2),
        int(ctr_y - newimgs2) : int(ctr_y + newimgs2),
    ]

