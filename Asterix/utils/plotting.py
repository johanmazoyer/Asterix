import os
import random
import matplotlib.pyplot as plt
import numpy as np


def display_complex(efield):
    """
    Plot the intensity, phase, real part and imaginary part of an electric field in a 2x2 panel plot.

    This function does not invoke saving to disk or the actual display, so you will want to add plt.show() or
    plt.savefigure() (or anything else) after this function, depending on what you are trying to do.

    AUTHOR: ILa

    Parameters
    ----------
    efield : complex 2D array
        The electric field to plot.
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(efield)**2, cmap='inferno', origin='lower')
    plt.title('Intensity')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(np.angle(efield), cmap='RdBu', origin='lower')
    plt.title('Phase')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(np.real(efield), origin='lower')
    plt.title('Real part')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(np.imag(efield), origin='lower')
    plt.title('Imaginary part')
    plt.colorbar()


def quickshow(tab):
    """
    Function to quickly show an array.

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    tab : array
        Array to be shown.
    """

    tmp = np.copy(tab)
    plt.axis('off')
    plt.imshow(tmp, origin='lower', cmap='gray')
    plt.show()
    plt.close()


def quickpng(tab, folder='', name='tmp'):
    """
    Function to quickly save a figure as png file.

    By default, it will save to the desktop with a random name.

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    tab : array
        Array to be saved.
    folder : string, optional
        Directory where to save the png file to.
    name : string, optional
        Name of the png file saved to disk.  By default tmpXX.png where xx is a random number.
    """
    if folder == '':
        desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        bureau = os.path.join(os.path.join(os.path.expanduser('~')), 'Bureau')
        if os.path.exists(desktop):
            folder = desktop
        elif os.path.exists(bureau):
            # You are french, are you?
            folder = bureau
        else:
            raise FileNotFoundError("I cannot find your desktop, please give me a folder to save the png file to.")

    plt.figure(figsize=(10, 10))
    tmp = tab
    plt.axis('off')
    plt.imshow(tmp, origin='lower', cmap='gray')
    if name == 'toto':
        name = name + str(int(random.random() * 100))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, name + '.png'), dpi=300)
    plt.close()
