import numpy as np


def mft(pup, dimpup, dimft, nbres, xshift=0, yshift=0, inv=-1):
    """ --------------------------------------------------
    MFT  - Return the Matrix Direct Fourier transform (MFT) of pup
    (cf. Soummer et al. 2007, OSA)

    Parameters
    ----------
    pup : 2D array (complex or real)
         Entrance pupil.
         CAUTION : pup has to be centered on (dim0/2+1,dim0/2+1)
         where dim0 is the pup array dimension

    dimpup : integer
            Diameter of the support in pup (can differ from dim0)
            Example : dimpup = diameter of the pupil in pixel

    dimft : integer
           Dimension of the output

    nbres : float
           Number of spatial resolution elements

    xshift : float
            center of the output array in the x direction

    yshift : float
            center of the output array in the y direction    

    inv : integer
            direct MFT if 1
            indirect MFT if -1 (default)
    

    Returns
    ------
    result : 2D array (complex)
            MFT of pup centered on the pixel (dimft/2D+1+xhift,dimft/2D+1+yxhift)
            dimension is dimft x dimft

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-22 Raphaël Galicher
    Initial revision (from MFT.pro written in IDL)

    -------------------------------------------------- """
    dim0 = pup.shape[0]
    nbres = nbres * dim0 / dimpup

    xx0 = np.arange(dim0) / dim0 - 0.5
    uu0 = ((np.arange(dimft) - xshift) / dimft - 0.5) * nbres
    uu1 = ((np.arange(dimft) - yshift) / dimft - 0.5) * nbres

    if inv == 1:
        norm0 = 1
    else:
        norm0 = ((1. * nbres)**2 / (1. * dimft)**2 / (1. * dim0)**2)

    AA = np.exp(-inv * 1j * 2 * np.pi * np.outer(uu0, xx0))
    BB = np.exp(-inv * 1j * 2 * np.pi * np.outer(xx0, uu1))
    result = norm0 * np.matmul(np.matmul(AA, pup), BB)

    return result


def prop_fresnel(pup, lam, z, rad, prad, retscale=0):
    """ --------------------------------------------------
    Fresnel propagation of electric field along a distance z
    in a collimated beam and in Free space

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

    retscale :
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

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-22 Raphael Galicher
    Initial revision

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
    # Zoom factor to get the same spatial scale in the input and output array
    #fac = dx/dxout
    else:
        sign = -1
        # Sampling in the output dim x dim array if FFT
        dxout = rad / prad
        # Sampling in the input dim x dim array if FFT
        dx = np.abs(lam * z / (dxout * dim))
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
    result = mft(pup * H, 2 * prad, dim, 2 * prad * fac, inv=sign)

    # Fresnel factor that applies after Fourier transform
    result = result * np.exp(1j * sign * np.pi * rho**2 / dim * dxout / dx)

    if sign == -1:
        result = result / fac**2
    return result, dxout