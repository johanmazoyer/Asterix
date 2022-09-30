import matplotlib.pyplot as plt
import numpy as np


def display_complex(efield):
    """
    Plot the intensity, phase, real part and imaginary part of an electric field in a 2x2 panel plot.

    This function does not invoke saving to disk or the actual display, so you will want to add plt.show() or
    plt.savefigure() (or anything else) after this function, depending on what you are trying to do.

    Parameters
    ----------
    efield : complex 2D array
        The electric field to plot.
    """
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(np.abs(efield)**2, cmap='inferno', origin='lower')
    plt.title('Intensity')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.angle(efield), cmap='RdBu', origin='lower')
    plt.title('Phase')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(np.real(efield), origin='lower')
    plt.title('Real part')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(np.imag(efield), origin='lower')
    plt.title('Imaginary part')
    plt.colorbar()
