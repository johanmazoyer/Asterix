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
        model_dir : if needed, we load the mask in this directory
        modelconfig : general configuration parameters (sizes and dimensions)
        coroconfig : coronagraph parameters

        -------------------------------------------------- """

        #Image
        dim_im = modelconfig["dim_im"]  #image size on detector
        #pupil and Lyot stop
        diam_pup_in_m = modelconfig["diam_pup_in_m"]
        diam_lyot_in_m = modelconfig["diam_lyot_in_m"]
        filename_instr_pup = modelconfig["filename_instr_pup"]
        filename_instr_lyot = modelconfig["filename_instr_lyot"]

        #Lambda over D in pixels in the pupil plane
        science_sampling = modelconfig["science_sampling"]

        ## define important measure of the coronagraph
        lyotrad = dim_im / 2 / science_sampling
        prad = int(np.ceil(lyotrad * diam_pup_in_m / diam_lyot_in_m))
        lyotrad = int(np.ceil(lyotrad))
        prev_science_sampling = science_sampling
        science_sampling = dim_im / 2 / lyotrad
        print(
            "Pupil resolution: 'Science Sampling' has been rounded up from {:.3f} to {:.3f} l/D"
            .format(prev_science_sampling, science_sampling))

        #coronagraph
        self.corona_type = coroconfig["corona_type"]
        self.coro_position = coroconfig["coro_position"]
        self.knife_coro_offset = coroconfig["knife_coro_offset"]
        self.err_fqpm = coroconfig["err_fqpm"]
        self.prop_lyot2science = coroconfig["prop_lyot2science"]

        self.dim_im = dim_im
        self.diam_pup_in_m = diam_pup_in_m
        self.diam_lyot_in_m = diam_lyot_in_m
        self.science_sampling = science_sampling
        self.lyotrad = lyotrad
        self.prad = prad

        ## transmission of the phase mask (exp(i*phase))
        ## centered on pixel [0.5,0.5]
        if self.corona_type == "fqpm":
            self.FPmsk = self.FQPM()
            self.perfect_coro = True
            self.prop_apod2lyot = 'fft'
        elif self.corona_type == "knife":
            self.FPmsk = self.KnifeEdgeCoro()
            self.perfect_coro = False
            self.prop_apod2lyot = 'fft'
        elif self.corona_type == "vortex":
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True
            self.prop_apod2lyot = 'fft'

        # Maybe should remove the entrance pupil from the coronostructure,
        # this is "before the DMs" so probably not relevant here.
        self.entrancepupil = create_binary_pupil(model_dir, filename_instr_pup,
                                                 dim_im, prad)

        #right now to be closer to THD2, the apodisation plane (entrance of the coronagraph)
        # is not define, but can be changed
        self.apod_pup = 1
        # self.apod_pup = create_binary_pupil(model_dir, filename_instr_apod,
        #                                     dim_im, prad)

        self.lyot_pup = create_binary_pupil(model_dir, filename_instr_lyot,
                                            2 * lyotrad, lyotrad)

        if self.perfect_coro:
            # do a propagation once with self.perfect_Lyot_pupil = 0 to
            # measure the Lyot pupil that will be removed after
            self.perfect_Lyot_pupil = 0

            self.perfect_Lyot_pupil = self.apodtolyot(self.entrancepupil)

        # Measure the PSF and store max and Sum
        self.maxPSF, self.sumPSF = self.max_sum_PSF()

    def max_sum_PSF(self):
        """ --------------------------------------------------
        Measure the non-coronagraphic PSF with no focal plane mask and return max and sum
        Returns
        ------
        np.amax(PSF): max of the non-coronagraphic PSF
        np.sum(PSF): sum of the non-coronagraphic PSF
        -------------------------------------------------- """
        PSF = np.abs(self.apodtodetector(self.entrancepupil, noFPM=True))**2
        # useful.quickfits(PSF, name='fftpsf')
        # asd
        return np.amax(PSF), np.sum(PSF)

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

        position = self.coro_position  # Can be 'left', 'right', 'top' or 'bottom' to define the orientation of the coronagraph
        shiftinldp = self.knife_coro_offset  #  Position of the edge, with respect to the image center, in number of pixels per resolution element
        ld_p = self.science_sampling * self.diam_lyot_in_m / self.diam_pup_in_m  #  Number of pixels per resolution element

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


    def apodtodetector(self,
                       input_wavefront,
                       noFPM=False):
        """ --------------------------------------------------
        Propagate the electric field through a high-contrast imaging instrument,
        from the entrance of the coronagraph (pupil plane before apodization pupil) to final detector focal plane.
        The output is cropped and resampled.
        
        Parameters
        ----------
        input_wavefront : 2D array,can be complex.  
            Input wavefront,can be complex.
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        
        Returns
        ------
        shift(sqrtimage) : 2D array, 
            Focal plane electric field created by 
            the input wavefront through the high-contrast instrument.
        -------------------------------------------------- """

        lyotplane_after_lyot = self.apodtolyot(input_wavefront, noFPM)

        # Science_focal_plane
        science_focal_plane = self.lyottodetector(lyotplane_after_lyot)

        return science_focal_plane


    def apodtolyot(self,
                   input_wavefront,
                   noFPM=False):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod pupil to Lyot plane after Lyot pupil

        Parameters
        ----------
        input_wavefront : 2D array,can be complex.  
            Input wavefront,can be complex.
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        
        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """

        if noFPM:
            FPmsk = 1.
            maskshifthalfpix = 1.
        else:
            FPmsk = self.FPmsk
            maskshifthalfpix = shift_phase_ramp(len(input_wavefront), 0.5, 0.5)

        input_wavefront_after_apod = input_wavefront * self.apod_pup
            

        corono_focal_plane = np.fft.fft2(
            np.fft.fftshift(input_wavefront_after_apod * maskshifthalfpix))

        # Focal plane to Lyot plane
        lyotplane_before_lyot_pad = np.fft.fftshift(
            np.fft.ifft2(corono_focal_plane * FPmsk))

        # Lyot mask
        
        lyotplane_before_lyot = proc.cropimage(lyotplane_before_lyot_pad,
                                               self.dim_im / 2,
                                               self.dim_im / 2,
                                               2 * self.lyotrad)

        lyotplane_after_lyot = lyotplane_before_lyot * self.lyot_pup

        if (self.perfect_coro) & (not noFPM):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        return lyotplane_after_lyot


    def lyottodetector(self,
                       Lyot_plane_after_Lyot,
                       propagation_method=None,
                       dim_focal_plane=None,
                       sampling_focal_plane=None):
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
        if propagation_method == None:
            propagation_method = self.prop_lyot2science

        if dim_focal_plane == None:
            dim_focal_plane = self.dim_im

        if sampling_focal_plane == None:
            sampling_focal_plane = self.science_sampling

        if propagation_method == "mft":

            # TODO here, be careful if the pupil is center between 4 pixels or on a pixel.
            # For the moment, only in between 4 pixels, but can be a pb
            science_focal_plane = mft(Lyot_plane_after_Lyot,
                                      dim_focal_plane,
                                      dim_focal_plane / sampling_focal_plane,
                                      inv=1)

        elif propagation_method == "fft":
            # in this case, the Lyot pupil is not padded, lets pad it before propagate
            # TODO here, be careful if the pupil is center between 4 pixels or on a pixel.
            # For the moment, only in between 4 pixels.
            # TODO To test, this is a rare case but not sure it works...

            dim_lyot = Lyot_plane_after_Lyot.shape[0]
            Lyot_plane_after_Lyot_padded = np.pad(Lyot_plane_after_Lyot,
                                                  int(dim_focal_plane / 2 -
                                                      dim_lyot / 2),
                                                  mode='constant',
                                                  constant_values=0)

            science_focal_plane = np.fft.fftshift(
                np.fft.fft2(np.fft.fftshift(Lyot_plane_after_Lyot_padded)))
        else:
            raise Exception(
                propagation_method +
                " is not a valid Lyot to Science plane propagation method")
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
    sampling_simu_over_meaasured : float
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


def creatingpushactv2(model_dir,
                      diam_pup_in_m,
                      prad,
                      DMconfig,
                      which_DM=3,
                      xerror=0,
                      yerror=0,
                      angerror=0,
                      gausserror=0):
    """ --------------------------------------------------
    Phase map induced in the DM plane for each actuator

    Parameters
    ----------
    model_dir :
    xy_ActuN : center of the actuator #__ActuN in pixels
    pitchDM : pitch of the DM in meter
    filename_ActuN : filename of estimated phase when poking actuator #ActuN
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
    if which_DM == 3:
        namDM = "DM3_"
    elif which_DM == 1:
        namDM = "DM1_"
    else:
        namDM = "DM3_"

    pitchDM = DMconfig[namDM + "pitch"]
    filename_ActuN = DMconfig[namDM + "filename_ActuN"]
    filename_grid_actu = DMconfig[namDM + "filename_grid_actu"]
    filename_actu_infl_fct = DMconfig[namDM + "filename_actu_infl_fct"]
    ActuN = DMconfig[namDM + "ActuN"]
    y_ActuN = DMconfig[namDM + "y_ActuN"]
    x_ActuN = DMconfig[namDM + "x_ActuN"]
    xy_ActuN = [x_ActuN, y_ActuN]

    #Measured positions for each actuator in pixel
    measured_grid = fits.getdata(model_dir + filename_grid_actu)
    #Ratio: pupil radius in the measured position over
    # pupil radius in the numerical simulation
    sampling_simu_over_meaasured = prad / fits.getheader(
        model_dir + filename_grid_actu)['PRAD']

    #dimension of the pushact array = size of the pupil
    # plus 20% of margin in case the pupil is smaller than the DM
    dim_pushact = int(pitchDM * np.sqrt(measured_grid.shape[1]) /
                      diam_pup_in_m * prad * 1.2) * 2

    if filename_ActuN != "":
        im_ActuN = fits.getdata(model_dir + filename_ActuN)
        im_ActuN_dim = np.zeros((dim_pushact, dim_pushact))
        im_ActuN_dim[int(dim_pushact / 2 -
                         len(im_ActuN) / 2):int(dim_pushact / 2 +
                                                len(im_ActuN) / 2),
                     int(dim_pushact / 2 -
                         len(im_ActuN) / 2):int(dim_pushact / 2 +
                                                len(im_ActuN) / 2)] = im_ActuN
        ytmp, xtmp = np.unravel_index(
            np.abs(im_ActuN_dim).argmax(), im_ActuN_dim.shape)
        # shift by (0.5,0.5) pixel because the pupil is
        # centered between pixels
        xy_ActuN = [xtmp - 0.5, ytmp - 0.5]

    #Position for each actuator in pixel for the numerical simulation
    simu_grid = actuator_position(measured_grid, xy_ActuN, ActuN,
                                  sampling_simu_over_meaasured)
    # Influence function and the pitch in pixels
    actshape = fits.getdata(model_dir + filename_actu_infl_fct)
    pitch_actshape = fits.getheader(model_dir +
                                    filename_actu_infl_fct)['PITCH']

    # Scaling the influence function to the desired dimension
    # for numerical simulation
    resizeactshape = skimage.transform.rescale(actshape,
                                               2 * prad / diam_pup_in_m *
                                               pitchDM / pitch_actshape,
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

    # Put the centered influence function inside an array (2*prad x 2*prad)
    actshapeinpupil = np.zeros((dim_pushact, dim_pushact))
    if len(resizeactshape) < dim_pushact:
        actshapeinpupil[
            0:len(resizeactshape),
            0:len(resizeactshape)] = resizeactshape / np.amax(resizeactshape)
        xycenttmp = len(resizeactshape) / 2
    else:
        actshapeinpupil = resizeactshape[
            0:dim_pushact, 0:dim_pushact] / np.amax(resizeactshape)
        xycenttmp = prad

    # Fill an array with the influence functions of all actuators
    pushact = np.zeros((simu_grid.shape[1], dim_pushact, dim_pushact))
    for i in np.arange(pushact.shape[0]):
        if gausserror == 0:
            Psivector = nd.interpolation.shift(
                actshapeinpupil,
                (simu_grid[1, i] + dim_pushact / 2 - xycenttmp + yerror,
                 simu_grid[0, i] + dim_pushact / 2 - xycenttmp + xerror))

            # Add an error on the orientation of the grid
            if angerror != 0:
                Psivector = nd.rotate(Psivector,
                                      angerror,
                                      order=5,
                                      cval=0,
                                      reshape=False)[0:dim_pushact,
                                                     0:dim_pushact]
        else:
            # Add an error on the sizes of the influence functions
            Psivector = nd.interpolation.shift(
                actshapeinpupil,
                (simu_grid[1, i] + dim_pushact / 2 - xycenttmp,
                 simu_grid[0, i] + dim_pushact / 2 - xycenttmp))

            xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
            x, y = np.mgrid[0:dim_pushact, 0:dim_pushact]
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

        pushact[i] = Psivector

    return pushact


##############################################
##############################################
### Difference of images for Pair-Wise probing


def createdifference(aberramp,
                     aberrphase,
                     posprobes,
                     pushact,
                     amplitude,
                     corona_struct,
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
    corona_struct: coronagraph structure
    dimimages : int
        Size of the output image after resampling in pixels
    wavelength : float
        Wavelength of the  incoming flux in meter
    perfect_coro : bool, optional
        Set if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    noise : boolean, optional
        If True, add photon noise. 
    numphot : int, optional
        Number of photons entering the pupil
    
    Returns
    ------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing
    -------------------------------------------------- """
    Ikmoins = np.zeros((corona_struct.dim_im, corona_struct.dim_im))
    Ikplus = np.zeros((corona_struct.dim_im, corona_struct.dim_im))
    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    contrast_to_photons = (np.sum(corona_struct.entrancepupil) /
                           np.sum(corona_struct.lyot_pup) * numphot *
                           corona_struct.maxPSF / corona_struct.PSF)

    dim_pup = corona_struct.apod_pup.shape[1]
    dimpush = pushact.shape[1]
    probephase = np.zeros((dim_pup, dim_pup))

    k = 0
    for i in posprobes:
        probephase[dim_pup / 2 - dimpush / 2:dim_pup / 2 + dimpush / 2,
                   dim_pup / 2 - dimpush / 2:dim_pup / 2 + dimpush /
                   2] = amplitude * 1e-9 * pushact[i] * 2 * np.pi / wavelength

        input_wavefront = (corona_struct.entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase - 1 * probephase)))
        Ikmoins = (np.abs(corona_struct.apodtodetector(input_wavefront))**2 /
                   corona_struct.maxPSF)

        input_wavefront = (corona_struct.entrancepupil * (1 + aberramp) *
                           np.exp(1j * (aberrphase + 1 * probephase)))
        Ikplus = (np.abs(corona_struct.apodtodetector(input_wavefront))**2 /
                  corona_struct.maxPSF)

        if noise == True:
            Ikplus = (np.random.poisson(Ikplus * contrast_to_photons) /
                      contrast_to_photons)
            Ikmoins = (np.random.poisson(Ikmoins * contrast_to_photons) /
                       contrast_to_photons)

        Ikplus = np.abs(proc.resampling(Ikplus, dimimages))
        Ikmoins = np.abs(proc.resampling(Ikmoins, dimimages))

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


def random_phase_map(dim_im, phaserms, rhoc, slope, prad):
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
    dim_pup = 2 * int(prad)
    # dim_pup = dim_im # if we un comment, this will be previous version 
    xx, yy = np.meshgrid(
        np.arange(dim_pup) - dim_pup / 2,
        np.arange(dim_pup) - dim_pup / 2)
    rho = np.hypot(yy, xx)
    PSD0 = 1
    PSD = PSD0 / (1 + (rho / rhoc)**slope)
    sqrtPSD = np.sqrt(2 * PSD)
    randomphase = 2 * np.pi * (np.random.rand(dim_pup, dim_pup) - 0.5)
    product = np.fft.fftshift(sqrtPSD * np.exp(1j * randomphase))
    phase = np.real(np.fft.ifft2(product))
    phase = phase / np.std(phase) * phaserms

    phase_pad = np.pad(phase,
                       int(dim_im / 2 - dim_pup / 2),
                       mode='constant',
                       constant_values=0)

    return phase_pad


def mft(pup, dimft, nbres, xshift=0, yshift=0, inv=-1):
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
    dimpup = pup.shape[0]

    xx0 = np.arange(dimpup) / dimpup - 0.5
    uu0 = ((np.arange(dimft) - xshift) / dimft - 0.5) * nbres
    uu1 = ((np.arange(dimft) - yshift) / dimft - 0.5) * nbres

    if inv == 1:
        norm0 = (nbres / dimpup)**2
    else:
        norm0 = ((1. * nbres)**2 / (1. * dimft)**2 / (1. * dimpup)**2)

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
    result = mft(pup * H, dim, dim * fac, inv=sign)

    # Fresnel factor that applies after Fourier transform
    result = result * np.exp(1j * sign * np.pi * rho**2 / dim * dxout / dx)

    if sign == -1:
        result = result / fac**2
    return result, dxout


def create_binary_pupil(direct, filename, dim, prad):
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
        pupil = fits.getdata(direct + filename)
    else:
        pupil = roundpupil(dim, prad)

    return pupil
