__author__ = 'Axel Potier'

import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import Asterix.processing_functions as proc

# Raccourcis conversions angles
dtor = np.pi / 180.0  # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor


def translationFFT(isz, a, b):
    """ --------------------------------------------------
    Create a phase ramp of size (isz,isz) which can be multiply by an image 
    to shift its fourier transform by (a,b) pixels
    
    Parameters
    ----------
    isz : int
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
    maska = np.linspace(-np.pi * a, np.pi * a, isz)
    maskb = np.linspace(-np.pi * b, np.pi * b, isz)
    xx, yy = np.meshgrid(maska, maskb)
    masktot = np.exp(-1j * xx) * np.exp(-1j * yy)
    return masktot


def FQPM(isz):
    """ --------------------------------------------------
    Create a perfect Four Quadrant Phase Mask coronagraph of size (isz,isz)
    
    Parameters
    ----------
    isz : int
        Size of the coronagraph (in pixels)
    
    Returns
    ------
    FQPM : 2D array
        Four quadrant phase mask coronagraph
    -------------------------------------------------- """
    phase = np.zeros((isz, isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            if i < isz / 2 and j < isz / 2:
                phase[i, j] = np.pi
            if i >= isz / 2 and j >= isz / 2:
                phase[i, j] = np.pi

    FQPM = np.exp(1j * phase)
    return FQPM


def KnifeEdgeCoro(isz, position, shiftinldp, ld_p):
    """ --------------------------------------------------
    Create a Knife edge coronagraph of size (isz,isz)
    
    Parameters
    ----------
    isz : int
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
    Knife = np.zeros((isz, isz))
    for i in np.arange(isz):
        if position == "left":
            if i > isz / 2 + shiftinldp * ld_p:
                Knife[:, i] = 1
        if position == "right":
            if i < isz / 2 - shiftinldp * ld_p:
                Knife[:, i] = 1
        if position == "top":
            if i > isz / 2 + shiftinldp * ld_p:
                Knife[i, :] = 1
        if position == "bottom":
            if i < isz / 2 - shiftinldp * ld_p:
                Knife[i, :] = 1
    return np.fft.fftshift(Knife)


def roundpupil(isz, prad1):
    """ --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.
    
    Parameters
    ----------
    isz : int  
        Size of the image (in pixels)
    prad1 : float 
        Size of the pupil radius (in pixels)
    
    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(isz) - (isz) / 2,
        np.arange(isz) - (isz) / 2)
    rr = np.hypot(yy + 1 / 2, xx + 1 / 2)
    pupilnormal = np.zeros((isz, isz))
    pupilnormal[rr <= prad1] = 1.0
    return pupilnormal


def pupiltodetector(input_wavefront,
                    coro_mask,
                    lyot_mask,
                    perfect_coro=False,
                    perfect_entrance_pupil=0):  # aberrationphase,prad1,prad2
    """ --------------------------------------------------
    Propagate a wavefront in a pupil plane through a high-contrast imaging instrument, until the science detector.
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
    isz = len(input_wavefront)
    masktot = translationFFT(isz, 0.5, 0.5)
    # Focale1
    focal1end = np.fft.fftshift(input_wavefront * masktot)
    if perfect_coro == True:
        focal1end = focal1end - np.fft.fftshift(
            perfect_entrance_pupil * masktot)

    focal1end = np.fft.fft2(focal1end)

    # Pupille2
    pupil2end = focal1end * coro_mask
    pupil2end = np.fft.ifft2(pupil2end)  # /shift(masktot)

    # Intensite en sortie de Lyot
    focal2end = pupil2end * np.fft.fftshift(lyot_mask)
    focal2end = np.fft.fft2(focal2end)
    focal2end = np.fft.fftshift(focal2end)

    return focal2end


def pushact_function(which,
                     grilleact,
                     actshapeinpupilresized,
                     xycent,
                     xy309,
                     modelerror="NoError",
                     error=0):
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
    modelerror : string
        Can be 'NoError', 'translationxy', 'translationx', 
        'translationy', 'rotation', 'influence_function'
    error : float 
        Size of the error in pixels or in degrees
    
    Returns
    ------
    Psivector : 2D array
        Pupil plane phase with the opd created by the poke of the desired actuator
    -------------------------------------------------- """
    isz = len(actshapeinpupilresized)
    x309 = xy309[0]
    y309 = xy309[1]
    xact = grilleact[0, which] + (x309 - grilleact[0, 309])
    yact = grilleact[1, which] + (y309 - grilleact[1, 309])

    if modelerror == "NoError":
        Psivector = nd.interpolation.shift(actshapeinpupilresized,
                                           (yact - xycent, xact - xycent))
    if modelerror == "translationxy":
        Psivector = nd.interpolation.shift(
            actshapeinpupilresized,
            (
                yact - xycent + np.sqrt(error**2 / 2),
                xact - xycent + np.sqrt(error**2 / 2),
            ),
        )
    if modelerror == "translationx":
        Psivector = nd.interpolation.shift(
            actshapeinpupilresized, (yact - xycent, xact - xycent + error))
    if modelerror == "translationy":
        Psivector = nd.interpolation.shift(
            actshapeinpupilresized, (yact - xycent + error, xact - xycent))
    if modelerror == "rotation":
        Psivector = proc.rotate_frame(Psivector,
                                      devx,
                                      interpolation="nearest",
                                      cyx=None)
    if modelerror == "influence_function":
        Psivector = nd.interpolation.shift(actshapeinpupilresized,
                                           (yact - xycent, xact - xycent))
        x, y = np.mgrid[0:isz, 0:isz]
        xy = (x, y)
        xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
        Psivector = proc.twoD_Gaussian(xy, 1, 1 + devx, 1 + devx, xo, yo, 0, 0)
        # devx peut etre a remplacer par error pour la rotation

    Psivector[np.where(Psivector < 1e-4)] = 0

    return Psivector


def creatingpushact(model_dir,
                    file309,
                    x309,
                    y309,
                    findxy309byhand,
                    isz,
                    prad,
                    pdiam,
                    modelerror="NoError",
                    error=0):
    """ --------------------------------------------------
    Push the desired DM actuator in the pupil
    
    Parameters
    ----------
    model_dir :
    file309 :
    x309 :
    y309 :
    findxy309byhand :
    modelerror : , optional
    error : , optional
    
    Returns
    ------
    pushact : 
    -------------------------------------------------- """
    # TODO It may not work at the moment. Pour le faire
    if findxy309byhand == False:
        file309 = "Phase_estim_Act309_v20200107.fits"
        im309size = len(fits.getdata(model_dir + file309))
        act309 = np.zeros((isz, isz))
        act309[int(isz / 2 - im309size / 2):int(isz / 2 + im309size / 2),
               int(isz / 2 -
                   im309size / 2):int(isz / 2 + im309size /
                                      2), ] = fits.getdata(model_dir + file309)
        y309, x309 = np.unravel_index(np.abs(act309).argmax(), act309.shape)

    # shift by (0.5,0.5) pixel because the pupil is centerd between pixels
    y309 = y309 - 0.5
    x309 = x309 - 0.5
    xy309 = [x309, y309]

    grille = fits.getdata(model_dir + "Grid_actu.fits")
    actshape = fits.getdata(model_dir +
                            "Actu_DM32_field=6x6pitch_pitch=22pix.fits")
    resizeactshape = skimage.transform.rescale(actshape,
                                               2 * prad / pdiam * 0.3e-3 / 22,
                                               order=1,
                                               preserve_range=True,
                                               anti_aliasing=True,
                                               multichannel=False)

    # Gauss2Dfit for centering the rescaled influence function
    dx, dy = proc.gauss2Dfit(resizeactshape)
    xycent = len(resizeactshape) / 2
    resizeactshape = nd.interpolation.shift(resizeactshape,
                                            (xycent - dx, xycent - dy))

    # Put the centered influence function inside a larger array (400x400)
    actshapeinpupil = np.zeros((isz, isz))
    actshapeinpupil[
        0:len(resizeactshape),
        0:len(resizeactshape)] = resizeactshape / np.amax(resizeactshape)

    pushact = np.zeros((1024, isz, isz))
    for i in np.arange(1024):
        pushact[i] = pushact_function(i, grille, actshapeinpupil, xycent,
                                      xy309, modelerror, error)
    return pushact


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
    isz = len(entrancepupil)
    Ikmoins = np.zeros((isz, isz))
    Ikplus = np.zeros((isz, isz))
    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    squaremaxPSF = np.amax(PSF)

    contrast_to_photons = (np.sum(entrancepupil) / np.sum(lyot_mask) *
                           numphot * squaremaxPSF**2 / np.sum(PSF)**2)

    k = 0
    for i in posprobes:
        probephase = amplitude * pushact[i]
        probephase = 2 * np.pi * probephase * 1e-9 / wavelength
        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase - 1 * probephase)))
        Ikmoins = (np.abs(
            pupiltodetector(
                input_wavefront,
                coro_mask,
                lyot_mask,
                perfect_coro,
                perfect_entrance_pupil,
            ))**2 / squaremaxPSF**2)
        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase + 1 * probephase)))
        Ikplus = (np.abs(
            pupiltodetector(input_wavefront, coro_mask, lyot_mask,
                            perfect_coro, perfect_entrance_pupil))**2 /
                  squaremaxPSF**2)

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


def random_phase_map(isz, phaserms, rhoc, slope):
    """ --------------------------------------------------
    Create a random phase map, whose PSD decrease in f^(-slope)
    
    Parameters
    ----------
    isz : integer
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
    xx, yy = np.meshgrid(np.arange(isz) - isz / 2, np.arange(isz) - isz / 2)
    rho = np.hypot(yy, xx)
    PSD0 = 1
    PSD = PSD0 / (1 + (rho / rhoc)**slope)
    sqrtPSD = np.sqrt(2 * PSD)
    randomphase = 2 * np.pi * (np.random.rand(isz, isz) - 0.5)
    product = np.fft.fftshift(sqrtPSD * np.exp(1j * randomphase))
    phase = np.real(np.fft.ifft2(product))
    phase = phase / np.std(phase) * phaserms
    return phase
