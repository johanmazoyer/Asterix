__author__ = 'Axel Potier'

import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import Asterix.processing_functions as proc

# Raccourcis conversions angles
dtor = np.pi / 180.0  # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor


def shift_phase_ramp(dim_im, a, b):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_im,dim_im) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = fft(im*exp(i phase ramp))
    
    Parameters
    ----------
    dim_im : int
        Size of the phase ramp (in pixels)
    a : float
        Shift desired in the x direction (in pixels)
    b : float
        Shift desired in the y direction (in pixels)
    
    Returns
    ------
    masktot : 2D array
        Phase ramp
    -------------------------------------------------- """
    # Verify this function works
    maska = np.linspace(-np.pi * a, np.pi * a, dim_im)
    maskb = np.linspace(-np.pi * b, np.pi * b, dim_im)
    xx, yy = np.meshgrid(maska, maskb)
    return np.exp(-1j * xx) * np.exp(-1j * yy)


##############################################
##############################################
### CORONAGRAPHS


def FQPM(dim_im, err=0):
    """ --------------------------------------------------
    Create a perfect Four Quadrant Phase Mask coronagraph of size (dim_im,dim_im)
    
    Parameters
    ----------
    dim_im : int
        Size of the coronagraph (in pixels)
    err : phase error on the pi phase-shift in rad (default=0)

    Returns
    ------
    FQPM : 2D array giving the complex transmission of the
        FQPM mask, centered at the four edges of the image
    -------------------------------------------------- """
    phase = np.zeros((dim_im, dim_im))
    for i in np.arange(dim_im):
        for j in np.arange(dim_im):
            if i < dim_im / 2 and j < dim_im / 2:
                phase[i, j] = np.pi + err
            if i >= dim_im / 2 and j >= dim_im / 2:
                phase[i, j] = np.pi + err
    return np.exp(1j * phase)


def KnifeEdgeCoro(dim_im, position, shiftinldp, ld_p):
    """ --------------------------------------------------
    Create a Knife edge coronagraph of size (dim_im,dim_im)
    
    Parameters
    ----------
    dim_im : int
        Size of the coronagraph (in pixels)
    position : string
        Can be 'left', 'right', 'top' or 'bottom' to define the orientation of the coronagraph
    shiftinldp : int 
        Position of the edge, with respect to the image center, in number of pixels per resolution element
    ld_p : float
        Number of pixels per resolution element
    
    Returns
    ------
    shift(Knife) : 2D array
        Knife edge coronagraph, located at the four edges of the image
    -------------------------------------------------- """
    Knife = np.zeros((dim_im, dim_im))
    for i in np.arange(dim_im):
        if position == "left":
            if i > dim_im / 2 + shiftinldp * ld_p:
                Knife[:, i] = 1
        if position == "right":
            if i < dim_im / 2 - shiftinldp * ld_p:
                Knife[:, i] = 1
        if position == "top":
            if i > dim_im / 2 + shiftinldp * ld_p:
                Knife[i, :] = 1
        if position == "bottom":
            if i < dim_im / 2 - shiftinldp * ld_p:
                Knife[i, :] = 1
    return np.fft.fftshift(Knife)


##############################################
##############################################
### Pupil


def roundpupil(dim_im, prad1):
    """ --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.
    
    Parameters
    ----------
    dim_im : int  
        Size of the image (in pixels)
    prad1 : float 
        Size of the pupil radius (in pixels)
    
    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(dim_im) - (dim_im) / 2,
        np.arange(dim_im) - (dim_im) / 2)
    rr = np.hypot(yy + 1 / 2, xx + 1 / 2)
    pupilnormal = np.zeros((dim_im, dim_im))
    pupilnormal[rr <= prad1] = 1.0
    return pupilnormal


##############################################
##############################################
### Propagation through coronagraph


def pupiltodetector(input_wavefront,
                    coro_mask,
                    lyot_mask,
                    perfect_coro=False,
                    perfect_entrance_pupil=0):  # aberrationphase,prad1,prad2
    """ --------------------------------------------------
    Propagate a wavefront through a high-contrast imaging instrument, from pupil plane to focal plane.
    The image is then cropped and resampled.
    
    Parameters
    ----------
    input_wavefront : 2D array,can be complex.  
        Input wavefront,can be complex.
    coro_mask : 2D array, can be complex. 
        Coronagraphic mask
    lyot_mask : 2D array 
        Lyot mask
    perfect_coro : bool, optional
        Set to True if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    perfect_entrance_pupil : 2D array, optional 
        Entrance pupil which should be nulled by the used coronagraph
    
    Returns
    ------
    shift(sqrtimage) : 2D array, 
        Focal plane electric field created by 
        the input wavefront through the high-contrast instrument.
    -------------------------------------------------- """

    maskshifthalfpix = shift_phase_ramp(len(input_wavefront), 0.5, 0.5)
    # Focal plane 1
    if perfect_coro == True:
        input_wavefront = input_wavefront - perfect_entrance_pupil

    focal1end = np.fft.fft2(np.fft.fftshift(input_wavefront *
                                            maskshifthalfpix))

    # Lyot plane
    pupil2end = np.fft.ifft2(focal1end * coro_mask)

    # Focal plane 2
    focal2end = np.fft.fftshift(
        np.fft.fft2(pupil2end * np.fft.fftshift(lyot_mask)))

    return focal2end


##############################################
##############################################
### Deformable mirror


def pushact_function(which,
                     grilleact,
                     actshapeinpupilresized,
                     xycent,
                     xy309,
                     xerror=0,
                     yerror=0,
                     angerror=0,
                     gausserror=0):
    """ --------------------------------------------------
    Push the desired DM actuator in the pupil
    
    Parameters
    ----------
    which : int 
        Index of the individual actuator to push
    grilleact: 2D array 
        x and y position of all the DM actuator in the pupil
    actshapeinpupilresized: 2D array
        Well-sampled actuator to be translated to its true position
    xycent : numpy array
        Position of the actuator in actshapeinpupilresized before translation
    xy309 : numpy array
        Position of the actuator 309 in pixels
    xerror : float 
        Size of the error in pixels for translation in x-direction
    yerror : float 
        Size of the error in pixels for translation in y-direction
    angerror : float 
        Size of the rotation error in degrees 
    gausserror : float 
        Error on the Influence function size (1 = 100% error)
    
    Returns
    ------
    Psivector : 2D array
        Pupil plane phase with the opd created by the poke of the desired actuator
    -------------------------------------------------- """
    dim_im = len(actshapeinpupilresized)
    xact = grilleact[0, which] + (xy309[0] - grilleact[0, 309])
    yact = grilleact[1, which] + (xy309[1] - grilleact[1, 309])

    if gausserror == 0:
        Psivector = nd.interpolation.shift(
            actshapeinpupilresized,
            (yact - xycent + yerror, xact - xycent + xerror))

        if angerror != 0:
            Psivector = nd.rotate(Psivector, angerror, order=5, cval=0)
    else:
        Psivector = nd.interpolation.shift(actshapeinpupilresized,
                                           (yact - xycent, xact - xycent))

        xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
        x, y = np.mgrid[0:dim_im, 0:dim_im]
        xy = (x, y)
        Psivector = proc.twoD_Gaussian(xy,
                                       1,
                                       1 + gausserror,
                                       1 + gausserror,
                                       xo,
                                       yo,
                                       0,
                                       0,
                                       flatten=False)
    Psivector[np.where(Psivector < 1e-4)] = 0

    return Psivector


def creatingpushact(
        model_dir,
        dim_im,
        pdiam,
        prad,
        xy309,
        pitchDM=0.3e-3,
        filename_actu309="",
        filename_grid_actu="Grid_actu.fits",
        filename_actu_infl_fct="Actu_DM32_field=6x6pitch_pitch=22pix.fits",
        xerror=0,
        yerror=0,
        angerror=0,
        gausserror=0):
    """ --------------------------------------------------
    Push the desired DM actuator in the pupil
    
    Parameters
    ----------
    model_dir :
    xy309 : center of the actuator #309 in pixels
    pitchDM : pitch of the DM in meter
    filename_actu309 : filename of estimated phase when poking actuator #309
    filename_grid_actu : filename of the grid of actuator positions
    filename_actu_infl_fct : filename of the actuator influence function

    Error on the model of the DM
        xerror : x-direction translation in pixel
        yerror : y-direction translation in pixel
        angerror : rotation in degree
        gausserror : influence function size (1=100% error)

    Returns
    ------
    pushact : 
    -------------------------------------------------- """
    # TODO It may not work at the moment. Pour le faire
    if filename_actu309 != "":
        im309size = len(fits.getdata(model_dir + filename_actu309))
        act309 = np.zeros((dim_im, dim_im))
        act309[int(dim_im / 2 - im309size / 2):int(dim_im / 2 + im309size / 2),
               int(dim_im / 2 -
                   im309size / 2):int(dim_im / 2 + im309size /
                                      2), ] = fits.getdata(model_dir +
                                                           filename_actu309)
        y309, x309 = np.unravel_index(np.abs(act309).argmax(), act309.shape)
        # shift by (0.5,0.5) pixel because the pupil is centered between pixels
        xy309 = [x309 - 0.5, y309 - 0.5]

    grille = fits.getdata(model_dir + filename_grid_actu)
    actshape = fits.getdata(model_dir + filename_actu_infl_fct)
    resizeactshape = skimage.transform.rescale(actshape,
                                               2 * prad / pdiam * pitchDM / 22,
                                               order=1,
                                               preserve_range=True,
                                               anti_aliasing=True,
                                               multichannel=False)

    # Gauss2Dfit for centering the rescaled influence function
    tmp = proc.gauss2Dfit(resizeactshape)
    dx = tmp[3]
    dy = tmp[4]
    xycent = len(resizeactshape) / 2
    resizeactshape = nd.interpolation.shift(resizeactshape,
                                            (xycent - dx, xycent - dy))

    # Put the centered influence function inside a larger array (400x400)
    actshapeinpupil = np.zeros((dim_im, dim_im))
    actshapeinpupil[
        0:len(resizeactshape),
        0:len(resizeactshape)] = resizeactshape / np.amax(resizeactshape)

    pushact = np.zeros((grille.shape[1], dim_im, dim_im))
    for i in np.arange(pushact.shape[0]):
        pushact[i] = pushact_function(i,
                                      grille,
                                      actshapeinpupil,
                                      xycent,
                                      xy309,
                                      xerror=xerror,
                                      yerror=yerror,
                                      angerror=angerror,
                                      gausserror=gausserror)
    return pushact


##############################################
##############################################
### Difference of images for Pair-Wise probing


def createdifference(aberramp,
                     aberrphase,
                     posprobes,
                     pushact,
                     amplitude,
                     entrancepupil,
                     coro_mask,
                     lyot_mask,
                     PSF,
                     dimimages,
                     wavelength,
                     perfect_coro=False,
                     perfect_entrance_pupil=0,
                     noise=False,
                     numphot=1e30):
    """ --------------------------------------------------
    Simulate the acquisition of probe images (actuator pokes) and create their differences
    
    Parameters
    ----------
    aberramp : 0 or 2D-array 
        Upstream amplitude aberration
    aberrphase : 0 or 2D-array 
        Upstream phase aberration
    posprobes : 1D-array
        Index of the actuators to push and pull for pair-wise probing
    pushact : 3D-array
        OPD created by the pokes of all actuators in the DM.
    amplitude : float
        amplitude of the actuator pokes for pair(wise probing in nm
    entrancepupil : 2D-array
        Entrance pupil shape
    coro_mask : 2D array, can be complex
        Coronagraphic mask
    lyot_mask : 2D array
        Lyot mask
    dimimages : int
        Size of the output image after resampling in pixels
    wavelength : float
        Wavelength of the  incoming flux in meter
    perfect_coro : bool, optional
        Set if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    perfect_entrance_pupil: 2D array, optional
        Entrance pupil which should be nulled by the used coronagraph
    noise : boolean, optional
        If True, add photon noise. 
    numphot : int, optional
        Number of photons entering the pupil
    
    Returns
    ------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing
    -------------------------------------------------- """
    dim_im = len(entrancepupil)
    Ikmoins = np.zeros((dim_im, dim_im))
    Ikplus = np.zeros((dim_im, dim_im))
    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    maxPSF = np.amax(PSF)

    contrast_to_photons = (np.sum(entrancepupil) / np.sum(lyot_mask) *
                           numphot * maxPSF / np.sum(PSF))

    k = 0
    for i in posprobes:
        probephase = amplitude * pushact[i]
        probephase = 2 * np.pi * probephase * 1e-9 / wavelength
        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase - 1 * probephase)))
        Ikmoins = (np.abs(
            pupiltodetector(input_wavefront, coro_mask, lyot_mask,
                            perfect_coro, perfect_entrance_pupil))**2 / maxPSF)

        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase + 1 * probephase)))
        Ikplus = (np.abs(
            pupiltodetector(input_wavefront, coro_mask, lyot_mask,
                            perfect_coro, perfect_entrance_pupil))**2 / maxPSF)

        if noise == True:
            Ikplus = (np.random.poisson(Ikplus * contrast_to_photons) /
                      contrast_to_photons)
            Ikmoins = (np.random.poisson(Ikmoins * contrast_to_photons) /
                       contrast_to_photons)

        Ikplus = np.abs(proc.resampling(Ikplus, dimimages))
        Ikmoins = np.abs(proc.resampling(Ikmoins, dimimages))
        # print(np.sum(contrast_to_photons*Ikplus*1e-9),np.sum(entrancepupil)/np.sum(lyot_mask))

        Difference[k] = Ikplus - Ikmoins
        k = k + 1

    return Difference


##############################################
##############################################
### Phase screen


def random_phase_map(dim_im, phaserms, rhoc, slope):
    """ --------------------------------------------------
    Create a random phase map, whose PSD decrease in f^(-slope)
    
    Parameters
    ----------
    dim_im : integer
        Size of the generated phase map
    phaserms : float
        Level of aberration
    rhoc : float
        See Borde et Traub 2006
    slope : float
        Slope of the PSD
    
    Returns
    ------
    phase : 2D array
        Static random phase map (or OPD) generated 
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(dim_im) - dim_im / 2,
        np.arange(dim_im) - dim_im / 2)
    rho = np.hypot(yy, xx)
    PSD0 = 1
    PSD = PSD0 / (1 + (rho / rhoc)**slope)
    sqrtPSD = np.sqrt(2 * PSD)
    randomphase = 2 * np.pi * (np.random.rand(dim_im, dim_im) - 0.5)
    product = np.fft.fftshift(sqrtPSD * np.exp(1j * randomphase))
    phase = np.real(np.fft.ifft2(product))
    phase = phase / np.std(phase) * phaserms
    return phase
