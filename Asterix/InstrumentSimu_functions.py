__author__ = 'Axel Potier'

import os
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import propagation_functions as prop
import Asterix.processing_functions as proc
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.fits_functions as useful


##############################################
##############################################
### CORONAGRAPHS
# initilaze and describe behavior in the coronagraph system (in the apod plane to the Lyot plane)
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

        # wavelength
        self.wavelength_0, self.wav_vec = create_wave_ref_and_vec(modelconfig)

        #image size and resolution in detector
        self.dim_im = modelconfig["dim_im"]
        self.science_sampling = modelconfig["science_sampling"]
        #Lambda over D in pixels in the focal plane
        # at the central wavelength

        #pupil and Lyot stop in m
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]
        self.diam_lyot_in_m = modelconfig["diam_lyot_in_m"]

        #pupil and Lyot stop in pixel
        self.prad = int(modelconfig["diam_pup_in_pix"] / 2)
        self.lyotrad = int(self.prad * self.diam_lyot_in_m /
                           self.diam_pup_in_m)

        #coronagraph
        self.corona_type = coroconfig["corona_type"].lower()

        # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
        self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
        for i, wav in enumerate(self.wav_vec):
            self.dim_fp_fft[i] = int(
                np.ceil(self.prad * self.science_sampling * self.diam_lyot_in_m
                        / self.diam_pup_in_m * self.wavelength_0 / wav)) * 2
            # we take the ceil to be sure that we measure at least the good resolution
            # We do not need to be exact, the mft in science_focal_plane will be

        ## transmission of the phase mask (exp(i*phase))
        ## centered on pixel [0.5,0.5]
        if self.corona_type == "fqpm":
            self.prop_apod2lyot = 'fft'
            self.err_fqpm = coroconfig["err_fqpm"]
            self.achrom_fqpm = coroconfig["achrom_fqpm"]
            self.FPmsk = self.FQPM()
            self.perfect_coro = True

        elif self.corona_type == "classiclyot":
            self.prop_apod2lyot = 'mft-babinet'
            self.Lyot_fpm_sampling = 30  # hard coded for now, this is very internal cooking
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            self.FPmsk = self.ClassicalLyot()
            self.perfect_coro = False

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'fft'
            self.coro_position = coroconfig["coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'fft'
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True

        #radius of the pupil in pixel in DM1 plane
        #(updated in Main_EFC_THD) Should not be there
        self.pradDM1 = self.prad

        # hard coded for now. TODO Fix that.
        self.dim_overpad_pupil = int(self.prad * 1.25) * 2

        # Maybe should remove the entrance pupil from the coronostructure,
        # this is "before the DMs" so probably not relevant here.
        self.entrancepupil = phase_ampl.load_or_create_binary_pupil(
            model_dir, modelconfig["filename_instr_pup"],
            self.dim_overpad_pupil, self.prad)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        self.apod_pup = 1

        # Lyot plane
        self.lyot_pup = phase_ampl.load_or_create_binary_pupil(
            model_dir, modelconfig["filename_instr_lyot"],
            self.dim_overpad_pupil, self.lyotrad)

        if self.perfect_coro:
            # do a propagation once with self.perfect_Lyot_pupil = 0 to
            # measure the Lyot pupil that will be removed after
            self.perfect_Lyot_pupil = 0
            self.perfect_Lyot_pupil = self.apodtolyot(
                self.entrancepupil, wavelength=self.wavelength_0)

        # Measure the PSF and store max and Sum
        self.maxPSF, self.sumPSF = self.max_sum_PSF()

    def max_sum_PSF(self):
        """ --------------------------------------------------
        Measure the non-coronagraphic PSF with no focal plane mask
        and return max and sum
        
        Returns
        ------
        np.amax(PSF): max of the non-coronagraphic PSF
        np.sum(PSF): sum of the non-coronagraphic PSF
        -------------------------------------------------- """
        # PSF = self.entrancetodetector(0, 0, noFPM=True)
        PSF = np.zeros((self.dim_im, self.dim_im))

        for wav in self.wav_vec:
            PSF += np.abs(
                self.apodtodetector(self.entrancepupil,
                                    noFPM=True,
                                    wavelength=wav))**2
        return np.amax(PSF), np.sum(PSF)

    def FQPM(self):
        """ --------------------------------------------------
        Create a Four Quadrant Phase Mask coronagraph 
        

        Returns
        ------
        FQPM : list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            FQPM mask
        -------------------------------------------------- """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            raise Exception("FQPM shuold not be simuated wit MFT")

        xx, yy = np.meshgrid(
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2,
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2)

        fqpm_thick_vert = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_vert[np.where(xx < 0)] = 1
        fqpm_thick_hor = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_hor[np.where(yy >= 0)] = 1
        fqpm_thick = fqpm_thick_vert - fqpm_thick_hor

        fqpm = list()
        for i, wav in enumerate(self.wav_vec):
            phase = np.zeros((self.dim_fp_fft[i], self.dim_fp_fft[i]))
            fqpm_thick_cut = proc.crop_or_pad_image(fqpm_thick,
                                                    self.dim_fp_fft[i])
            phase[np.where(fqpm_thick_cut != 0)] = (np.pi + self.err_fqpm)
            if self.achrom_fqpm == False:
                phase = phase * self.wavelength_0 / wav
            fqpm.append(np.exp(1j * phase))

        return fqpm

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dim_im,dim_im)
    
        Returns
        ------
        shift(Knife) :list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            Knife edge coronagraph mask. A CHECKER YA PTET UN SOUCIS DE SHIFT
        -------------------------------------------------- """

        if len(self.wav_vec) == 1:
            raise Exception(
                "KnifeEdgeCoro only working in monochromatic as of now")
        maxdimension_array_fpm = np.max(self.dim_fp_fft)

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength
        ld_p_0 = self.science_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm),
                             np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(xx > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "left":
            Knife[np.where(xx < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(yy > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "top":
            Knife[np.where(yy < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1

        return [Knife]

    def ClassicalLyot(self):
        """ --------------------------------------------------
        Create a classical Lyot coronagraph of radius rad_LyotFP 0

        rad_LyotFP : int, radius of the Lyot focal plane
    
        Returns
        ------
         classical Lyot : 2D array
        -------------------------------------------------- """

        ld_p = self.Lyot_fpm_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        rad_LyotFP_pix = self.rad_lyot_fpm * ld_p

        self.dim_fpm = 2 * int(rad_LyotFP_pix)
        ClassicalLyotstop = 1 - phase_ampl.roundpupil(self.dim_fpm,
                                                      rad_LyotFP_pix)

        return [ClassicalLyotstop]

##############################################
##############################################
### Propagation through coronagraph

    def apodtodetector(self, input_wavefront, noFPM=False, wavelength=None):
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
        wavelength : current wavelength in m
        
        Returns
        ------
        shift(sqrtimage) : 2D array, 
            Focal plane electric field created by 
            the input wavefront through the high-contrast instrument.
        -------------------------------------------------- """

        lyotplane_after_lyot = self.apodtolyot(input_wavefront,
                                               noFPM=noFPM,
                                               wavelength=wavelength)

        # Science_focal_plane
        science_focal_plane = self.lyottodetector(lyotplane_after_lyot,
                                                  noFPM=noFPM,
                                                  wavelength=wavelength)

        return science_focal_plane

    def apodtolyot(self, input_wavefront, noFPM=False, wavelength=None):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

        Parameters
        ----------
        input_wavefront : 2D array,can be complex.  
            Input wavefront,can be complex.
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        wavelength : current wavelength in m. Default is the reference wavelength of the coronagraph

        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """

        if wavelength == None:
            wavelength = self.wavelength_0
        if noFPM:
            # Psf_offset = (0, 0)
            FPmsk = 1.
        else:
            # Psf_offset = (-0.5, -0.5)
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        lambda_ratio = wavelength / self.wavelength_0

        input_wavefront_after_apod = input_wavefront * self.apod_pup

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(
                wavelength)]
            input_wavefront_pad = proc.crop_or_pad_image(
                input_wavefront, dim_fp_fft_here)
            # Phase ramp to center focal plane between 4 pixels

            maskshifthalfpix = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, 0.5, 0.5)
            maskshifthalfpix_invert = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, -0.5, -0.5)

            #Apod plane to focal plane
            corono_focal_plane = np.fft.fft2(
                np.fft.fftshift(input_wavefront_pad * maskshifthalfpix))

            # Focal plane to Lyot plane
            lyotplane_before_lyot = np.fft.fftshift(
                np.fft.ifft2(
                    corono_focal_plane * FPmsk)) * maskshifthalfpix_invert

        elif self.prop_apod2lyot == "mft-babinet":
            #Apod plane to focal plane

            # in babinet, we have to shift also the initial pupil.
            # this is a bit absurd but this is the only way if we want
            # focal plane to be always centered between 4 pixels
            # if we only centered duing the PSF then in the subtraction in babinet's trick
            # we introduce a shift. Also this is not a perfect shift
            # We need to code a anti-shift for mft !

            maskshifthalfpix_fpm = phase_ampl.shift_phase_ramp(
                self.dim_overpad_pupil,
                0.5 / self.Lyot_fpm_sampling * lambda_ratio,
                0.5 / self.Lyot_fpm_sampling * lambda_ratio)

            maskshifthalfpix_fpm_inverse = phase_ampl.shift_phase_ramp(
                self.dim_overpad_pupil,
                -0.5 / self.Lyot_fpm_sampling * lambda_ratio,
                -0.5 / self.Lyot_fpm_sampling * lambda_ratio)

            input_wavefront_after_apod_shifted = input_wavefront_after_apod * maskshifthalfpix_fpm

            corono_focal_plane = prop.mft(
                input_wavefront_after_apod_shifted,
                self.dim_overpad_pupil,
                self.dim_fpm,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                inv=1)

            # Focal plane to Lyot plane
            lyotplane_before_lyot_central_part = prop.mft(
                corono_focal_plane * (1 - FPmsk),
                self.dim_fpm,
                self.dim_overpad_pupil,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                inv=-1)

            # Babinet's trick
            lyotplane_before_lyot = (input_wavefront_after_apod_shifted -
                                     lyotplane_before_lyot_central_part
                                     ) * maskshifthalfpix_fpm_inverse
            # this is ugly as sh*t but it works to be coherent with other convention in the code

        # elif self.prop_apod2lyot == "mft":
        #     #Apod plane to focal plane

        #     corono_focal_plane = prop.mft(input_wavefront_after_apod,
        #                                   self.dim_overpad_pupil,
        #                                   self.dim_im,
        #                                   self.dim_im / self.science_sampling *
        #                                   lambda_ratio,
        #                                   xshift=Psf_offset[0],
        #                                   yshift=Psf_offset[1],
        #                                   inv=1)

        #     # Focal plane to Lyot plane
        #     lyotplane_before_lyot = prop.mft(
        #         corono_focal_plane * FPmsk,
        #         self.dim_im,
        #         self.dim_overpad_pupil,
        #         self.dim_im / self.science_sampling * lambda_ratio,
        #         inv=-1)

        else:
            raise Exception(
                self.prop_apod2lyot +
                " is not a known prop_apod2lyot propagation mehtod")

        # crop to the Lyot stop size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(
            lyotplane_before_lyot, self.lyot_pup.shape[0])

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = lyotplane_before_lyot_crop * self.lyot_pup

        if (self.perfect_coro) & (not noFPM):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        return lyotplane_after_lyot

    def lyottodetector(self,
                       Lyot_plane_after_Lyot,
                       noFPM=False,
                       wavelength=None):
        """ --------------------------------------------------
        Propagate the electric field from Lyot plane after Lyot to Science focal plane.
        
        Parameters
        ----------
        Lyot_plane_after_Lyot : 2D array,can be complex.  
            Input wavefront,can be complex.

        wavelength : current wavelength in m. Default is the reference wavelength of the coronagraph

        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """
        if wavelength == None:
            wavelength = self.wavelength_0

        if noFPM:
            Psf_offset = (0, 0)
        else:
            Psf_offset = (-0.5, -0.5)

        lambda_ratio = wavelength / self.wavelength_0

        science_focal_plane = prop.mft(Lyot_plane_after_Lyot,
                                       self.lyotrad * 2,
                                       self.dim_im,
                                       self.dim_im / self.science_sampling *
                                       lambda_ratio,
                                       xshift=Psf_offset[0],
                                       yshift=Psf_offset[1],
                                       inv=1)

        return science_focal_plane

    def entrancetodetector(self,
                           ampl_abb,
                           phase_abb,
                           wavelength=None,
                           noFPM=False,
                           DM1_active=False,
                           phaseDM1=0,
                           DM3_active=False,
                           phaseDM3=0,
                           DM1_z_position=0):
        """ --------------------------------------------------
        Propagate the electric field through the entire instrument (from Entrance pupil plane to Science focal plane) 
        for a given wavelength
        
        Parameters
        ----------
        ampl_abb: amplitude aberrations
        phase_abb: phase aberrations
        wavelength : current wavelength in m. Default is the reference wavelength of the coronagraph
        noFPM : bool (default=False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        DM1_active : bool (default=False). If true we use DM1
        phase_DM1 : 2D array (default=0)
                Phase introduced by DM1
        DM3_active: bool (default=False). If true we use DM3
        phase_DM3 : 2D array(default=0)
                Phase introduced by DM3
        DM1_z_position : float
                distance between the pupil plane and DM1

        TODO
        # this function should not be part of the coronagraph class.
        # Coronagraph clases initilaze and describe behavior in the coronagraph system (from the apod plane to the Lyot plane)
        # should be in the form
        # def entrancetodetector(coronagraph_structure, DM_structure, input_wavefront, wavelength (optional), phaseDM (optional), phaseDM3 (optional))

        Returns
        ------
        science_focal_plane : 2D array, 
            Focal plane electric field in the focal plane
        -------------------------------------------------- """

        if wavelength == None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        # Entrance pupil
        input_wavefront = self.entrancepupil * (1 + ampl_abb) * np.exp(
            1j * phase_abb / lambda_ratio)

        if DM1_active == True:
            # Propagation in DM1 plane, add DM1 phase
            # and propagate to next pupil plane (DM3 plane)
            input_wavefront = prop_pup_DM1_DM3(input_wavefront,
                                               phaseDM1 / lambda_ratio,
                                               wavelength, DM1_z_position,
                                               self.diam_pup_in_m / 2,
                                               self.prad)
        if DM3_active == True:
            input_wavefront = input_wavefront * np.exp(
                1j * proc.crop_or_pad_image(phaseDM3 / lambda_ratio,
                                            self.entrancepupil.shape[1]))

        # Pupil to Lyot
        lyotplane_after_lyot = self.apodtolyot(input_wavefront,
                                               noFPM=noFPM,
                                               wavelength=wavelength)

        # Science_focal_plane
        science_focal_plane = self.lyottodetector(lyotplane_after_lyot,
                                                  wavelength=wavelength)

        return science_focal_plane

    def entrancetodetector_Intensity(self,
                                     ampl_abb,
                                     phase_abb,
                                     wavelengths='all',
                                     noFPM=False,
                                     DM3_active=False,
                                     phaseDM3=0,
                                     DM1_active=False,
                                     phaseDM1=0,
                                     DM1_z_position=0):
        """ --------------------------------------------------
        Propagate the electric field through the entire instrument (from Entrance pupil plane to Science focal plane) 
        for a given wavelength
        
        Parameters
        ----------
        ampl_abb: amplitude aberrations
        phase_abb: phase aberrations
        wavelengths : string. defautl 'all'
                    if 'all': all wavelength
                    if 'ref': only the reference wavelength
                            Default is a one element vector containing the reference wavelength of the coronagraph
        noFPM : bool (default=False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        DM1_active : bool (default=False). If true we use DM1
        phase_DM1 : 2D array (default=0)
                Phase introduced by DM1
        DM3_active: bool (default=False). If true we use DM3
        phase_DM3 : 2D array(default=0)
                Phase introduced by DM3
        DM1_z_position : float
                distance between the pupil plane and DM1

        TODO
        # Not super satisfied with this function. This should be generalized by a function doing Optical_system.to_Intensity(phase) function 

        Returns
        ------
        Intensity_science_focal_plane : 2D array, 
            Intensity in the focal plane
        -------------------------------------------------- """

        if wavelengths == 'all':
            wavelength_vec = self.wav_vec
        elif wavelengths == 'ref':
            wavelength_vec = [self.wavelength_0]
        else:
            raise Exception("'wavelengths' keyword can only be 'all' or 'ref'")

        Intensity = np.zeros((self.dim_im, self.dim_im))

        for wav in wavelength_vec:
            Intensity += np.abs(
                self.entrancetodetector(ampl_abb,
                                        phase_abb,
                                        wavelength=wav,
                                        noFPM=noFPM,
                                        DM3_active=DM3_active,
                                        phaseDM3=phaseDM3,
                                        DM1_active=DM1_active,
                                        phaseDM1=phaseDM1,
                                        DM1_z_position=DM1_z_position))**2

        return Intensity


##############################################
##############################################
### Deformable mirror
#TODO
# These functions should be part of a DMsystem class that inlc
# initilaze and describe behavior in the DM system (from entrance pupil to Apod plane)


def prop_pup_DM1_DM3(pupil_wavefront, phase_DM1, wavelength, DM1_z_position,
                     rad_pup_in_m, rad_pup_in_pixel):
    """ --------------------------------------------------
    Propagate the field towards an out-of-pupil plane (DM1 plane),
    add the DM1 phase, and propagate to the next pupil plane (DM3 plane)

    Parameters
    ----------
    pupil_wavefront : 2D array (float, double or complex)
                Wavefront in the pupil plane

    phase_DM1 : 2D array
                Phase introduced by DM1
    
    wavelength : float
                wavelength
    
    DM1_z_position : float
                distance between the pupil plane and DM1

    rad_pup_in_m : float
                radius of the pupil in meter

    rad_pup_in_pix : int
                radius of the pupil in pixel

    Returns
    ------
    UDM3 : 2D array (complex)
            Wavefront in the pupil plane after DM1
            (corresponds to DM3 plane on THD2 bench)

    AUTHOR : Raphaël Galicher

    REVISION HISTORY :
    Revision 1.1  2021-02-10 Raphaël Galicher
    Initial revision

    -------------------------------------------------- """

    # Propagation in DM1 plane
    UDM1, dxout = prop.prop_fresnel(pupil_wavefront, wavelength,
                                    DM1_z_position, rad_pup_in_m,
                                    rad_pup_in_pixel)
    # Add DM1 phase and propagate to next pupil plane (DM3 plane)
    UDM3, dxpup = prop.prop_fresnel(UDM1 * np.exp(1j * phase_DM1), wavelength,
                                    -DM1_z_position, rad_pup_in_m,
                                    rad_pup_in_pixel)
    return UDM3


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


def creatingpushact(model_dir,
                    diam_pup_in_m,
                    diam_pup_in_pix,
                    DMconfig,
                    dim_array,
                    which_DM=3,
                    xerror=0,
                    yerror=0,
                    angerror=0,
                    gausserror=0):
    """ --------------------------------------------------
    OPD map induced in the DM plane for each actuator

    Parameters
    ----------
    model_dir : directory where the influence function of the DM is recorded
    diam_pup_in_m : diameter of the pupil in meter
    diam_pup_in_pix : radius of the pupil in pixels
    DMconfig : structure with all information on DMs
    dim_array : dimension of the output array

    Error on the model of the DM
        xerror : x-direction translation in pixel
        yerror : y-direction translation in pixel
        angerror : rotation in degree
        gausserror : influence function size (1=100% error)

    Returns
    ------
    pushact : 
    -------------------------------------------------- """

    # this is not ideal if we want to have DMs with other names
    if which_DM == 1:
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
    sampling_simu_over_meaasured = diam_pup_in_pix / 2 / fits.getheader(
        model_dir + filename_grid_actu)['PRAD']
    if filename_ActuN != "":
        im_ActuN = fits.getdata(model_dir + filename_ActuN)
        im_ActuN_dim = proc.crop_or_pad_image(im_ActuN, dim_array)

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
    resizeactshape = skimage.transform.rescale(
        actshape,
        diam_pup_in_pix / diam_pup_in_m * pitchDM / pitch_actshape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
        multichannel=False)

    # Gauss2Dfit for centering the rescaled influence function
    Gaussian_fit_param = proc.gauss2Dfit(resizeactshape)
    dx = Gaussian_fit_param[3]
    dy = Gaussian_fit_param[4]
    xycent = len(resizeactshape) / 2
    resizeactshape = nd.interpolation.shift(resizeactshape,
                                            (xycent - dx, xycent - dy))

    # Put the centered influence function inside an array (2*prad x 2*prad)
    actshapeinpupil = np.zeros((dim_array, dim_array))
    if len(resizeactshape) < dim_array:
        actshapeinpupil[
            0:len(resizeactshape),
            0:len(resizeactshape)] = resizeactshape / np.amax(resizeactshape)
        xycenttmp = len(resizeactshape) / 2
    else:
        actshapeinpupil = resizeactshape[0:dim_array,
                                         0:dim_array] / np.amax(resizeactshape)
        xycenttmp = dim_array / 2

    # Fill an array with the influence functions of all actuators
    pushact = np.zeros((simu_grid.shape[1], dim_array, dim_array))
    for i in np.arange(pushact.shape[0]):
        if gausserror == 0:
            Psivector = nd.interpolation.shift(
                actshapeinpupil,
                (simu_grid[1, i] + dim_array / 2 - xycenttmp + yerror,
                 simu_grid[0, i] + dim_array / 2 - xycenttmp + xerror))

            # Add an error on the orientation of the grid
            if angerror != 0:
                Psivector = nd.rotate(Psivector,
                                      angerror,
                                      order=5,
                                      cval=0,
                                      reshape=False)[0:dim_array, 0:dim_array]
        else:
            # Add an error on the sizes of the influence functions
            Psivector = nd.interpolation.shift(
                actshapeinpupil, (simu_grid[1, i] + dim_array / 2 - xycenttmp,
                                  simu_grid[0, i] + dim_array / 2 - xycenttmp))

            xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
            x, y = np.mgrid[0:dim_array, 0:dim_array]
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


## Create the influence functions of an out-of-pupil DM
## in the pupil plane
def creatingpushact_inpup(DM_pushact, wavelength, corona_struct, z_position):
    """ --------------------------------------------------
    OPD map induced by out-of-pupil DM in the pupil plane for each actuator
    TODO: could be merged with creatingpushact to be more general for all z_position

    Parameters
    ----------
    DM_pushact : OPD map induced by the DM in the DM plane
    wavelength : wavelengtht in m
    corona_struct : coronagraph structure (includes entrancepupil
                    and the dimension of the pupil in meter)
    z_position : distance of DM from the pupil

    Returns
    ------
    pushact_inpup : Map of the complex phase induced in pupil plane
    -------------------------------------------------- """
    # Size of the array (diameter of the pupil * 125%)
    # dimtmp = int(corona_struct.prad*2*1.25)

    dim_entrancepupil = corona_struct.entrancepupil.shape[1]
    UDM1, dxout = prop.prop_fresnel(corona_struct.entrancepupil, wavelength,
                                    z_position,
                                    corona_struct.diam_pup_in_m / 2,
                                    corona_struct.prad)
    pushact_inpup = np.zeros(
        (DM_pushact.shape[0], dim_entrancepupil, dim_entrancepupil),
        dtype=complex)

    for i in np.arange(DM_pushact.shape[0]):
        Upup, dxpup = prop.prop_fresnel(
            UDM1 * proc.crop_or_pad_image(DM_pushact[i], dim_entrancepupil),
            wavelength, -z_position, corona_struct.diam_pup_in_m / 2,
            corona_struct.prad)
        pushact_inpup[i] = Upup

    return pushact_inpup


##############################################
##############################################
### Difference of images for Pair-Wise probing


def createdifference(input_wavefront,
                     posprobes,
                     pushact,
                     corona_struct,
                     dimimages,
                     noise=False,
                     numphot=1e30):
    """ --------------------------------------------------
    Simulate the acquisition of probe images using Pair-wise
    and calculate the difference of images [I(+probe) - I(-probe)]
    
    Parameters
    ----------
    input_wavefront : 2D-array (complex)
        Input wavefront in pupil plane
    posprobes : 1D-array
        Index of the actuators to push and pull for pair-wise probing
    pushact : 3D-array
        OPD created by the pokes of all actuators in the DM
        Unit = phase with the amplitude of the wished probe
    corona_struct: coronagraph structure
    dimimages : int
        Size of the output image after resampling in pixels
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
                           corona_struct.maxPSF / corona_struct.sumPSF)

    dim_pup = corona_struct.entrancepupil.shape[1]

    k = 0
    for i in posprobes:
        probephase = proc.crop_or_pad_image(pushact[i], dim_pup)

        Ikmoins = np.abs(
            corona_struct.apodtodetector(input_wavefront * np.exp(
                -1j * probephase)))**2 / corona_struct.maxPSF

        Ikplus = np.abs(
            corona_struct.apodtodetector(input_wavefront * np.exp(
                1j * probephase)))**2 / corona_struct.maxPSF

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


def create_wave_ref_and_vec(modelconfig):
    """ --------------------------------------------------
     from the parameter file return the "reference" wavelength and the wave vector containing all the wavelengths.
     As of now, reference is the central_wavelength

     If we want to change the reference (the smallest one or the largest one of the middle one) we can do it once here.

    
    Parameters
    ----------
    modelconfig : the config defined from the parameter file
    
    Returns
    ------
    wavelength_0, wavevec : the reference wavelength for the simulation
    1D vector with the wavelengths used in this simulation
    -------------------------------------------------- """

    Delta_wav = modelconfig["Delta_wav"]
    nb_wav = modelconfig["nb_wav"]
    wavelength_0 = modelconfig["wavelength_0"]

    if Delta_wav != 0:
        if (nb_wav % 2 == 0) or nb_wav < 2:
            raise Exception("please set nb_wav parameter to an odd number > 1")

        return wavelength_0, np.linspace(wavelength_0 - Delta_wav / 2,
                                         wavelength_0 + Delta_wav / 2,
                                         num=nb_wav,
                                         endpoint=True)
    else:
        return wavelength_0, np.array([wavelength_0])