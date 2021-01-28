__author__ = 'Axel Potier'

import os
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform
import Asterix.processing_functions as proc
import Asterix.fits_functions as useful

# Raccourcis conversions angles
dtor = np.pi / 180.0  # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor

##############################################
##############################################
### CORONAGRAPHS


class coronagraph:
    
    
    def __init__(self, model_dir, modelconfig, coroconfig):
        """ --------------------------------------------------
        Initialize a coronograph objects : pupil, mask and Lyot stop
        
        Parameters
        ----------
        model_dir : is needed, we load the mask in this directory
        modelconfig : general configuration parameters (sizes and dimensions)
        coroconfig : coronagraph parameters

        -------------------------------------------------- """

        #Image
        dim_im = modelconfig["dim_im"]  #image size on detector
        #pupil and Lyot stop
        pdiam = modelconfig["pdiam"]
        lyotdiam = modelconfig["lyotdiam"]
        filename_instr_pup = modelconfig["filename_instr_pup"]
        filename_instr_lyot = modelconfig["filename_instr_lyot"]

        #Lambda over D in pixels in the pupil plane
        science_sampling = modelconfig["science_sampling"]

        ## define important measure of the coronagraph
        lyotrad = dim_im / 2 / science_sampling
        prad = int(np.ceil(lyotrad * pdiam / lyotdiam))
        lyotrad = int(np.ceil(lyotrad))

        #coronagraph
        self.corona_type = coroconfig["coronagraph"]
        self.coro_position = coroconfig["coro_position"]
        self.knife_coro_offset = coroconfig["knife_coro_offset"]
        self.err_fqpm = coroconfig["err_fqpm"]
        self.perfect_coro = coroconfig["perfect_coro"]
        self.prop_lyot2science = coroconfig["prop_lyot2science"]

        self.dim_im = dim_im
        self.pdiam = pdiam
        self.lyotdiam = lyotdiam
        self.science_sampling = science_sampling
        self.lyotrad = lyotrad
        self.prad = prad

        ## transmission of the phase mask (exp(i*phase))
        ## centered on pixel [0.5,0.5]
        if self.corona_type == "fqpm":
            self.FPmsk = self.FQPM()
            self.perfect_coro = True
        elif self.corona_type == "knife":
            self.FPmsk = self.KnifeEdgeCoro()
            self.perfect_coro = False
        elif self.corona_type == "vortex":
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True

        if filename_instr_pup != "" and filename_instr_lyot != "":
            self.entrancepupil = fits.getdata(model_dir + filename_instr_pup)
            self.lyot_pup = fits.getdata(model_dir + filename_instr_lyot)
        else:
            self.entrancepupil = roundpupil(dim_im, prad)
            self.lyot_pup = roundpupil(dim_im, lyotrad)

        if self.perfect_coro:
            self.perfect_Lyot_pupil = self.pupiltolyot(self.entrancepupil)

    def FQPM(self):
        """ --------------------------------------------------
        Create a perfect Four Quadrant Phase Mask coronagraph of size (dim_im,dim_im)
        

        Returns
        ------
        FQPM : 2D array giving the complex transmission of the
            FQPM mask, centered at the four edges of the image
        -------------------------------------------------- """
        phase = np.zeros((self.dim_im, self.dim_im))
        for i in np.arange(self.dim_im):
            for j in np.arange(self.dim_im):
                if i < self.dim_im / 2 and j < self.dim_im / 2:
                    phase[i, j] = np.pi + self.err_fqpm
                if i >= self.dim_im / 2 and j >= self.dim_im / 2:
                    phase[i, j] = np.pi + self.err_fqpm
        return np.exp(1j * phase)

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dim_im,dim_im)
    
        
        Returns
        ------
        shift(Knife) : 2D array
            Knife edge coronagraph, located at the four edges of the image
        -------------------------------------------------- """

        position = self.coro_position # Can be 'left', 'right', 'top' or 'bottom' to define the orientation of the coronagraph
        shiftinldp = self.knife_coro_offset #  Position of the edge, with respect to the image center, in number of pixels per resolution element
        ld_p = self.science_sampling * self.lyotdiam / self.pdiam #  Number of pixels per resolution element

        Knife = np.zeros((self.dim_im, self.dim_im))
        for i in np.arange(self.dim_im):
            if position == "left":
                if i > self.dim_im / 2 + shiftinldp * ld_p:
                    Knife[:, i] = 1
            if position == "right":
                if i < self.dim_im / 2 - shiftinldp * ld_p:
                    Knife[:, i] = 1
            if position == "top":
                if i > self.dim_im / 2 + shiftinldp * ld_p:
                    Knife[i, :] = 1
            if position == "bottom":
                if i < self.dim_im / 2 - shiftinldp * ld_p:
                    Knife[i, :] = 1
        return np.fft.fftshift(Knife)

    ##############################################
    ##############################################
    ### Propagation through coronagraph

    def lyottodetector(self,
                       Lyot_plane_after_Lyot):  # aberrationphase,prad1,prad2
        """ --------------------------------------------------
        Propagate the electric field from Lyot plane after Lyot to Science focal plane.
        The output is cropped and resampled.
        
        Parameters
        ----------
        Lyot_plane_after_Lyot : 2D array,can be complex.  
            Input wavefront,can be complex.
        
        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """

        science_focal_plane = np.fft.fftshift(
            np.fft.fft2(Lyot_plane_after_Lyot))

        return science_focal_plane

    def pupiltolyot(self, input_wavefront):  # aberrationphase,prad1,prad2
        """ --------------------------------------------------
        Propagate the electric field from entrance pupil to Lyot plane after Lyot pupil

        Parameters
        ----------
        input_wavefront : 2D array,can be complex.  
            Input wavefront,can be complex.
        
        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """

        maskshifthalfpix = shift_phase_ramp(len(input_wavefront), 0.5, 0.5)

        corono_focal_plane = np.fft.fft2(
            np.fft.fftshift(input_wavefront * maskshifthalfpix))

        # Focal plane
        lyotplane_before_lyot = np.fft.ifft2(corono_focal_plane * self.FPmsk)

        # Lyot plane
        lyotplane_after_lyot = lyotplane_before_lyot * np.fft.fftshift(
            self.lyot_pup)

        return lyotplane_after_lyot

    def pupiltodetector(self, input_wavefront):  # aberrationphase,prad1,prad2
        """ --------------------------------------------------
        Propagate the electric field through a high-contrast imaging instrument,
        from pupil plane to focal plane.
        The output is cropped and resampled.
        
        Parameters
        ----------
        input_wavefront : 2D array,can be complex.  
            Input wavefront,can be complex.
        
        Returns
        ------
        shift(sqrtimage) : 2D array, 
            Focal plane electric field created by 
            the input wavefront through the high-contrast instrument.
        -------------------------------------------------- """

        lyotplane_after_lyot = self.pupiltolyot(input_wavefront)

        if self.perfect_coro:
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        # Science_focal_plane
        science_focal_plane = self.lyottodetector(lyotplane_after_lyot)

        return science_focal_plane


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
    Phase map induced in the DM plane for one pushed actuator

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
    Phase map induced in the DM plane for each actuator

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
    if filename_actu309 != "":
        im309size = len(fits.getdata(model_dir + filename_actu309))
        act309 = np.zeros((dim_im, dim_im))
        act309[int(dim_im / 2 - im309size / 2):int(dim_im / 2 + im309size / 2),
               int(dim_im / 2 -
                   im309size / 2):int(dim_im / 2 + im309size /
                                      2)] = fits.getdata(model_dir +
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
                     corona_struct,
                     PSF,
                     dimimages,
                     wavelength,
                     noise=False,
                     numphot=1e30):
    """ --------------------------------------------------
    Simulate the acquisition of probe images using Pair-wise
    and calculate the difference of images [I(+probe) - I(-probe)]
    
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
    corona_struct: coronagraph structure
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

    contrast_to_photons = (np.sum(entrancepupil) /
                           np.sum(corona_struct.lyot_pup) * numphot * maxPSF /
                           np.sum(PSF))

    k = 0
    for i in posprobes:
        probephase = amplitude * pushact[i]
        probephase = 2 * np.pi * probephase * 1e-9 / wavelength
        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase - 1 * probephase)))
        Ikmoins = (np.abs(corona_struct.pupiltodetector(input_wavefront))**2 /
                   maxPSF)

        input_wavefront = (entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase + 1 * probephase)))
        Ikplus = (np.abs(corona_struct.pupiltodetector(input_wavefront))**2 /
                  maxPSF)

        if noise == True:
            Ikplus = (np.random.poisson(Ikplus * contrast_to_photons) /
                      contrast_to_photons)
            Ikmoins = (np.random.poisson(Ikmoins * contrast_to_photons) /
                       contrast_to_photons)

        Ikplus = np.abs(proc.resampling(Ikplus, dimimages))
        Ikmoins = np.abs(proc.resampling(Ikmoins, dimimages))
        # print(np.sum(contrast_to_photons*Ikplus*1e-9),np.sum(entrancepupil)/np.sum(lyot_pup))

        Difference[k] = Ikplus - Ikmoins
        k = k + 1

    return Difference


##############################################
##############################################
### Phase screen


def shift_phase_ramp(dim_im, a, b):
    """ --------------------------------------------------
    Create a phase ramp of size (dim_im,dim_im) that can be used as follow
    to shift one image by (a,b) pixels : shift_im = real(fft(ifft(im)*exp(i phase ramp)))
    
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


def mft(pup, dimft, nbres, xshift=0, yshift=0, inv=-1, pupil_center = 'pixel'):
    """ --------------------------------------------------
    MFT  - Return the Matrix Direct Fourier transform (MFT) of pup
    (cf. Soummer et al. 2007, OSA)

    Parameters
    ----------
    pup : 2D array (complex or real)
         Entrance pupil.
         CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
         where dimpup is the pup array dimension

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
    
    pupil_center: string
            'pixel': pupil is centered on pixel 
                    pup has to be centered on (dimpup/2+1,dimpup/2+1)
                    where dimpup is the pup array dimension
            'nopixel' : pupil is centered between 4 pixels

    Returns
    ------
    result : 2D array (complex)
            MFT of pup centered on the pixel (dimft/2D+1+xhift,dimft/2D+1+yxhift)
            dimension is dimft x dimft

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-22 Raphaël Galicher
    Initial revision (from MFT.pro written in IDL)
    Revision 1.1  2020-01-27 Johan Mazoyer
    added option for centered pupil

    -------------------------------------------------- """

    dimpup = pup.shape[0]
    if pupil_center == 'pixel':
        centering_value = 0.5
    elif pupil_center == 'nopixel':
        centering_value = 0.
    else:
        raise Exception("mft: keyword pupil_center must be pixel or nopixel")

    xx0 = np.arange(dimpup) / dimpup - centering_value
    uu0 = ((np.arange(dimft) - xshift) / dimft - centering_value) * nbres
    uu1 = ((np.arange(dimft) - yshift) / dimft - centering_value) * nbres

    if inv == 1:
        norm0 = (nbres / dimpup)**2
    else:
        norm0 = ((1. * nbres)**2 / (1. * dimft)**2 / (1. * dimpup)**2)


    AA = np.exp(-inv * 1j * 2 * np.pi * np.outer(uu0, xx0))
    BB = np.exp(-inv * 1j * 2 * np.pi * np.outer(xx0, uu1))
    result = norm0 * np.matmul(np.matmul(AA, pup), BB)
    return result


def prop_fresnel(pup, lam, z, rad, prad):
    """ --------------------------------------------------
    Fresnel propagation of pup along a distance z in a collimated beam
    and in Free space

    Parameters
    ----------
    pup : 2D array (complex or real)
         electric field at z=0
         CAUTION : pup has to be centered on (dimpup/2+1,dimpup/2+1)
         where dimpup is the pup array dimension

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

    Returns
    ------
    pup_z : 2D array (complex)
            electric field after propagating in free space along
            a distance z

    AUTHOR : Raphael Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-22 Raphael Galicher
    Initial revision

    -------------------------------------------------- """
    # dimension of the input array
    dim = pup.shape[0]

    # if z<0, we consider we go back wrt the real path of the light
    if np.sign(z) == 1:
        sign=1
    # Sampling in the input dim x dim array if FFT
        dx = rad/prad
    # Sampling in the output dim x dim array if FFT
        dxout = np.abs(lam*z/(dx*dim))
    # Zoom factor to get the same spatial scale in the input and output array
        fac = dx/dxout
    else:
        sign=-1
    # Sampling in the output dim x dim array if FFT
        dxout = rad/prad
    # Sampling in the input dim x dim array if FFT
        dx = np.abs(lam*z/(dxout*dim))
    # Zoom factor to get the same spatial scale in the input and output array
        fac = dxout/dx
    
   # print(dx,dxout,fac)
   # print(300e-6/dx, 300e-6/dxout)

    # create a 2D-array of distances from the central pixel
    u, v = np.meshgrid(np.arange(dim) - dim / 2, np.arange(dim) - dim / 2)
    rho = np.hypot(v, u)
    # Fresnel factor that applies before Fourier transform
    H = np.exp(1j*sign*np.pi * rho**2/dim *dx/dxout)

    if np.abs(fac) > 1.2:
        print('need to increase lam or z or 1/dx')
        return -1

    # Fourier transform using MFT
#    result = mft(pup * H, dim, np.abs(dim * fac),inv=np.sign(z))
#    if sign == 1:
#        result = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup*H)))
#    else:
#        result = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(pup*H)))
    # Fresnel factor that applies after Fourier transform
    result = mft(pup * H, dim, dim*fac,inv=sign)

    result = result * np.exp(1j *sign* np.pi * rho**2/dim *dxout/dx)
    if sign == -1:
        result = result / fac**2
    return result

def create_binary_pupil(direct,filename, dim, prad):
    """ --------------------------------------------------
    Create a binary pupil from a Fits file or create a round pupil

    Parameters
    ----------
    direct : string
         name of the directory where filename is

    filename : string
         name of the Fits file

    dim : int
         dimension in pixels of the output array

    prad : int
         radius in pixels of the round pupil mask

    Returns
    ------
    pup_z : 2D array (float)
            Binary pupil (used for entrance pupil and Lyot stop)

    AUTHOR : Raphaël Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-26 Raphaël Galicher
    Initial revision

    -------------------------------------------------- """

    if filename != "":
        pupil = fits.getdata(direct+filename)
    else:
        pupil = roundpupil(dim, prad)

    return pupil

def psfNoAberr(pup):
    """ --------------------------------------------------
    Generate the PSF if no aberrations for a given pupil

    Parameters
    ----------
    
    pup : 2D array (complex)
            pupil plane

    Returns
    ------
    PSF : 2D array (float)
            Intensity

    AUTHOR : Raphaël Galicher

    REVISION HISTORY :
    Revision 1.1  2020-01-26 Raphaël Galicher
    Initial revision

    -------------------------------------------------- """
    return np.abs(np.fft.fftshift(np.fft.fft2(
                        np.fft.fftshift(pup))))**2