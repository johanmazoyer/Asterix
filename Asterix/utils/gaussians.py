import numpy as np
import scipy.optimize as opt


def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, theta, h, flatten=True):
    """Create a gaussian in 2D.

    Author : Axel Potier

    Parameters
    ----------
    xy: Tuple object (2,dim1,dim2)
        which can be created with:
        x, y = np.mgrid[0:dim1, 0:dim2]
        xy=(x,y)
    amplitude : float
        Peak of the gaussian function
    sigma_x : float
        Standard deviation of the gaussian function in the x direction
    sigma_y : float
        Standard deviation of the gaussian function in the y direction
    xo : float
        Position of the Gaussian peak in the x direction
    yo : float
        Position of the Gaussian peak in the y direction
    h : float
        Floor amplitude
    flatten : bool, default True
        if True (default), the 2D-array is flatten into 1D-array

    Returns
    --------
    gauss : 2d numpy array
        2D gaussian function
    """
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = (amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2))) + h)
    if flatten:
        g = g.flatten()
    return g


def gauss2Dfit(data):
    """Return the parameter of the 2D-Gaussian that best fits data.

    Parameters
    ----------
    data : 2D array
        input image

    Returns
    --------
    popt : tuple of floats
        Parameters of the gaussian: max, sig_x, sig_y, x_cen, y_cen, angle, offset.
    """
    # 2D-Gaussian fit
    popt = np.zeros(8)
    w, h = data.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x, y)

    # Fit 2D Gaussian with fixed parameters
    initial_guess = (np.amax(data), 1, 1, len(data) / 2, len(data) / 2, 0, 0)

    try:
        popt, _ = opt.curve_fit(twoD_Gaussian, xy, data.flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed")

    return popt
