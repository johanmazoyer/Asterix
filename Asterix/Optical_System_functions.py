# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import inspect
import copy
import time
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.fits_functions as useful

Asterix_root = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
model_dir = os.path.join(Asterix_root, "Model") + os.path.sep


##############################################
##############################################
### Optical_System
class Optical_System:
    """ --------------------------------------------------
    Super class Optical_System allows to pass parameters to all sub class.
    We can then creat blocks inside this super class. An Optical_System start and
    end in the pupil plane.
    The entrance and exit pupil plane must always of the same size (dim_overpad_pupil)
    With these convention, they can be easily assemble to create complex optical systems.


    AUTHOR : Johan Mazoyer

        -------------------------------------------------- """
    def __init__(self, modelconfig):
        """ --------------------------------------------------
        Initialize Optical_System objects
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)

        -------------------------------------------------- """

        #pupil in pixel
        self.prad = int(modelconfig["diam_pup_in_pix"] / 2)

        # 1.25 is hard coded for now. TODO Fix that ?
        # All pupils in the code must have this dimension, so that the OS systems can
        #  be easily switched.
        # dim_overpad_pupil is set to an even numer and the pupil is centered in
        # between 4 pixels
        self.dim_overpad_pupil = int(self.prad * 1.25) * 2

        #Lambda over D in pixels in the focal plane
        # at the reference wavelength
        #image size and resolution in detector
        self.dimScience = modelconfig["dimScience"]
        self.Science_sampling = modelconfig["Science_sampling"]

        #pupil in meters
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]

        # Exit pupil radius
        self.exitpup_rad = self.prad
        # this is the exit pupil radius, that is used to define the L/D
        # in self.todetector function.
        # by default this is the entrance pupil rad. of course, this can be changed

        self.modelconfig = copy.copy(modelconfig)

        # wavelength
        self.Delta_wav = modelconfig["Delta_wav"]
        self.nb_wav = modelconfig["nb_wav"]
        self.wavelength_0 = modelconfig["wavelength_0"]

        if self.Delta_wav != 0:
            if (self.nb_wav % 2 == 0) or self.nb_wav < 2:
                raise Exception(
                    "please set nb_wav parameter to an odd number > 1")

            self.wav_vec = np.linspace(self.wavelength_0 - self.Delta_wav / 2,
                                       self.wavelength_0 + self.Delta_wav / 2,
                                       num=self.nb_wav,
                                       endpoint=True)
        else:
            self.wav_vec = np.array([self.wavelength_0])
            self.nb_wav = 1

        self.string_os = '_prad' + str(int(self.prad)) + '_wl' + str(
            int(self.wavelength_0 * 1e9)) + "_resFP" + str(
                round(self.Science_sampling, 2)) + "_dimFP" + str(
                    int(self.dimScience))

    #We define functions that all Optical_System object can use.
    # These can be overwritten for a subclass if need be

    def EF_through(self,
                   entrance_EF=1.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance pupil to exit pupil

        NEED TO BE DEFINED FOR ALL Optical_System

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],
                        can be complex.
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                    Electric field in the pupil plane a the entrance of the system.

        save_all_planes_to_fits: Bool, default False
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : string, default None
                            directory to save all plane in fits
                                if save_all_planes_to_fits = True
        **kwargs: 
            other parameters can be passed for Optical_System objects EF_trough functions

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system



        -------------------------------------------------- """

        # if isinstance(entrance_EF, float) or isinstance(entrance_EF, np.float):
        #     entrance_EF = np.full(
        #         (self.dim_overpad_pupil, self.dim_overpad_pupil),
        #         np.float(entrance_EF))

        if isinstance(entrance_EF, (float, np.float, np.ndarray)) == False:
            print(entrance_EF)
            raise Exception(
                "entrance_EF should be a float of a numpy array of floats")

        if save_all_planes_to_fits == True and dir_save_all_planes == None:
            raise Exception(
                "save_all_planes_to_fits = True can generate a lot of .fits files"
                +
                "please define a clear directory using dir_save_all_planes kw argument"
            )

        exit_EF = entrance_EF
        return exit_EF

    def todetector(self,
                   entrance_EF=1.,
                   wavelength=None,
                   center_on_pixel=False,
                   in_contrast=True,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system and then
        to Science focal plane.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m.
        
        in_contrast : bool, default True. normalize to 
                        np.sqrt(self.norm_monochrom[self.wav_vec.tolist().index(wavelength)]))
                        (see self.measure_normalization)

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
                This of course assume that no tip-tilt have been introduced in the entrance_EF
                or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        
        dir_save_all_planes : string efault None. 
                                directory to save all plane in fits if save_all_planes_to_fits = True

        **kwargs: 
            other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        ef_focal_plane : 2D array of size [self.dimScience, self.dimScience]
                Electric field in the focal plane.
                the lambda / D is defined such as
                self.wavelength_0 /  (2*self.exitpup_rad) = self.Science_sampling pixels


        -------------------------------------------------- """
        if center_on_pixel == True:
            Psf_offset = (0, 0)
        else:
            Psf_offset = (-0.5, -0.5)

        if wavelength is None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        exit_EF = self.EF_through(
            entrance_EF=entrance_EF,
            wavelength=wavelength,
            save_all_planes_to_fits=save_all_planes_to_fits,
            dir_save_all_planes=dir_save_all_planes,
            **kwargs)

        focal_plane_EF = prop.mft(exit_EF,
                                  self.exitpup_rad * 2,
                                  self.dimScience,
                                  self.dimScience / self.Science_sampling *
                                  lambda_ratio,
                                  X_offset_output=Psf_offset[0],
                                  Y_offset_output=Psf_offset[1],
                                  inverse=False,
                                  norm='ortho')

        if in_contrast == True:
            focal_plane_EF /= np.sqrt(
                self.norm_monochrom[self.wav_vec.tolist().index(wavelength)])

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'EF_FP_after_' + who_called_me + '_obj' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      focal_plane_EF)

        return focal_plane_EF

    def todetector_Intensity(self,
                             entrance_EF=1.,
                             wavelengths=None,
                             in_contrast=True,
                             center_on_pixel=False,
                             photon_noise=False,
                             nb_photons=1e30,
                             save_all_planes_to_fits=False,
                             dir_save_all_planes=None,
                             **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system, then
        to Science focal plane and measure intensity

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:   3D complex array of size [nb_wav,self.dim_overpad_pupil, self.dim_overpad_pupil]
                        or 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelengths : float or float array of wavelength in m.
                        Default is all wavelenthg in self.wav_vec
        
        in_contrast : bool, default True. normalize to 
                            self.norm_polychrom (see self.measure_normalization)

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
                This of course assume that no tip-tilt have been introduced in the entrance_EF
                or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : string default None. 
                                Directory to save all plane
                                in fits if save_all_planes_to_fits = True

        noise : boolean, optional
                If True, add photon noise to the image
    
        nb_photons : int, optional
                Number of photons entering the pupil

        **kwargs: 
            other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        focal_plane_Intensity : 2D array of size [self.dimScience, self.dimScience]
                    Intensity in the focal plane. The lambda / D is defined such as
                    self.wavelength_0 /  (2*self.exitpup_rad) = self.Science_sampling pixels


        -------------------------------------------------- """

        if 'wavelength' in kwargs:
            raise Exception(
                """todetector_Intensity() function is polychromatic, 
                do not use wavelength keyword.
                Use wavelengths keyword even for monochromatic intensity""")

        if wavelengths == None:
            wavelength_vec = self.wav_vec

        elif isinstance(wavelengths, (float, int)):
            wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths

        if isinstance(entrance_EF, (float, int)):
            entrance_EF = np.repeat(entrance_EF, self.nb_wav)
        elif entrance_EF.shape == self.wav_vec.shape:
            pass
        elif entrance_EF.shape == (self.dim_overpad_pupil,
                                   self.dim_overpad_pupil):
            entrance_EF = np.repeat(entrance_EF[np.newaxis, ...],
                                    self.nb_wav,
                                    axis=0)
        elif entrance_EF.shape == (self.nb_wav, self.dim_overpad_pupil,
                                   self.dim_overpad_pupil):
            pass
        else:
            raise Exception(
                """"entrance_EFs must be scalar (same for all WL), or a self.nb_wav scalars or a
                        2D array of size (self.dim_overpad_pupil, self.dim_overpad_pupil) or a 3D array of size
                        (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil)"""
            )

        focal_plane_Intensity = np.zeros((self.dimScience, self.dimScience))

        for i, wav in enumerate(wavelength_vec):
            focal_plane_Intensity += np.abs(
                self.todetector(
                    entrance_EF=entrance_EF[i],
                    wavelength=wav,
                    in_contrast=False,
                    center_on_pixel=center_on_pixel,
                    save_all_planes_to_fits=save_all_planes_to_fits,
                    dir_save_all_planes=dir_save_all_planes,
                    **kwargs))**2

        if in_contrast == True:
            if (wavelength_vec != self.wav_vec).all():
                raise Exception(
                    """carefull: contrast normalization in todetector_Intensity assumes
                     it is done in all possible BWs (wavelengths = self.wav_vec). If self.nb_wav > 1
                     and you want only one BW with the good contrast normalization, use
                     np.abs(to_detector(wavelength = wavelength))**2... If you want a specific
                     normalization for a subset of  wavelengths, use in_contrast = False and
                     measure the PSF to normalize.
                """)
            else:
                focal_plane_Intensity /= self.norm_polychrom

        if photon_noise == True:
            focal_plane_Intensity = np.random.poisson(
                focal_plane_Intensity * self.normPupto1 *
                nb_photons) / (self.normPupto1 * nb_photons)

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'Int_FP_after_' + who_called_me + '_obj'
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      focal_plane_Intensity)

        return focal_plane_Intensity

    def transmission(self, noFPM=True, **kwargs):
        """
        measure ratio of photons lost when
        crossing the system compared to a clear aperture of radius self.prad

        By default transmission is done at the reference WL, and there is
        no reason to depend heavily on the WL.

        AUTHOR : Johan Mazoyer

        Parameters
        ------
        noFPM : bool, defaut True
            if the optical transfert function EF_through has a noFPM parameter

        **kwargs: 
            other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        transimssion : float  
            ratio exit flux  / clear entrance pupil flux


        -------------------------------------------------- """
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad)

        # all parameter can be passed here, but in the case there is a coronagraph,
        # we pass noFPM = True and noentrance Field by default
        exit_EF = self.EF_through(entrance_EF=1., noFPM=noFPM, **kwargs)

        throughput = np.sum(np.abs(exit_EF)**2) / np.sum(
            np.abs(clear_entrance_pupil)**2)

        return throughput

    def measure_normalization(self):
        """ --------------------------------------------------
        Functions must me used at the end of all Optical Systems initalization

        Measure 3 differents values to normalize the data:
            - self.norm_monochrom. Array of size len(self.wav_vec)
                        the PSF per WL, use to nomrmalize to_detector
            - self.norm_polychrom. float
                        the polychromatic PSF use to nomrmalize to_detector_Intensity
            - self.normPupto1, which is used to measure the photon noise
                This is the factor that we use to measure photon noise.
                From an image in contrast, we now normalize by the total amount of 
                energy (*self.norm_polychrom / self.sum_polychrom) and then account for the energy
                lost in the process (self.transmission()). 
                Can be used as follow:
                Im_intensity_photons = Im_Intensity_contrast * self.normPupto1 * nb_photons
        
        AUTHOR : Johan Mazoyer


        -------------------------------------------------- """

        PSF_bw = np.zeros((self.dimScience, self.dimScience))
        self.norm_monochrom = np.zeros((len(self.wav_vec)))
        self.sum_monochrom = np.zeros((len(self.wav_vec)))

        for i, wav in enumerate(self.wav_vec):
            PSF_wl = np.abs(
                self.todetector(wavelength=wav,
                                noFPM=True,
                                center_on_pixel=True,
                                in_contrast=False))**2

            self.norm_monochrom[i] = np.max(PSF_wl)
            PSF_bw += PSF_wl

        self.norm_polychrom = np.max(PSF_bw)
        self.sum_polychrom = np.sum(PSF_bw)

        self.normPupto1 = self.transmission(
        ) * self.norm_polychrom / self.sum_polychrom

    def generate_phase_aberr(self,
                             SIMUconfig,
                             up_or_down='up',
                             Model_local_dir=None):
        """ --------------------------------------------------
        
        Generate and save  phase aberations
        
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        SIMUconfig : dict
                    parameter of this simualtion (describing the phase)
        
        up_or_down : string, default, 'up'
                    'up' or 'do', use to access the right parameters in the parameter file for 
                        upstream (entrance pupil) or downstream (Lyot plane) aberrations


        Model_local_dir: string, default None
                    directory to save things you can measure yourself
                    and can save to save time


        Returns
        ------
        return_phase : 2D array, real of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                phase abberation at the reference wavelength


        -------------------------------------------------- """
        if not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)
        if up_or_down == 'up':
            set_phase_abb = SIMUconfig["set_UPphase_abb"]
            set_random_phase = SIMUconfig["set_UPrandom_phase"]
            opd_rms = SIMUconfig["UPopd_rms"]
            phase_rhoc = SIMUconfig["UPphase_rhoc"]
            phase_slope = SIMUconfig["UPphase_slope"]
            phase_abb_filename = SIMUconfig["UPphase_abb_filename"]
        else:
            set_phase_abb = SIMUconfig["set_DOphase_abb"]
            set_random_phase = SIMUconfig["set_DOrandom_phase"]
            opd_rms = SIMUconfig["DOopd_rms"]
            phase_rhoc = SIMUconfig["DOphase_rhoc"]
            phase_slope = SIMUconfig["DOphase_slope"]
            phase_abb_filename = SIMUconfig["DOphase_abb_filename"]

        ## Phase map and amplitude map for the static aberrations
        if set_phase_abb == True:
            if phase_abb_filename == '':
                phase_abb_filename = up_or_down + "phase_{:d}opdrms_lam{:d}_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                    int(opd_rms * 1e9), int(self.wavelength_0 * 1e9),
                    int(phase_slope), phase_rhoc, self.prad)

            if set_random_phase is False and os.path.isfile(
                    Model_local_dir + phase_abb_filename + ".fits") == True:
                return_phase = fits.getdata(Model_local_dir +
                                            phase_abb_filename + ".fits")

            else:
                # TODO see with raphael these opd / phase issues
                phase_rms = 2 * np.pi * opd_rms / self.wavelength_0

                return_phase = phase_ampl.random_phase_map(
                    self.prad, self.dim_overpad_pupil, phase_rms, phase_rhoc,
                    phase_slope)

                fits.writeto(Model_local_dir + phase_abb_filename + ".fits",
                             return_phase,
                             overwrite=True)
            return return_phase
        else:
            return 0.

    def generate_ampl_aberr(self, SIMUconfig, Model_local_dir=None):
        """ --------------------------------------------------
        Generate and save amplitude aberations

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        SIMUconfig : dict
                    parameter of this simualtion (describing the amplitude)

        Model_local_dir: string, default None
                    directory to save things you can measure yourself
                    and can save to save time


        Returns
        ------
        return_ampl : 2D array, real of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                amplitude abberation


        -------------------------------------------------- """
        if not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)

        set_amplitude_abb = SIMUconfig["set_amplitude_abb"]
        ampl_abb_filename = SIMUconfig["ampl_abb_filename"]
        set_random_ampl = SIMUconfig["set_random_ampl"]
        ampl_rms = SIMUconfig["ampl_rms"]
        ampl_rhoc = SIMUconfig["ampl_rhoc"]
        ampl_slope = SIMUconfig["ampl_slope"]

        if set_amplitude_abb == True:
            if ampl_abb_filename != '' and os.path.isfile(
                    Model_local_dir + ampl_abb_filename +
                    ".fits") == True and set_random_ampl is False:

                return_ampl = phase_ampl.scale_amplitude_abb(
                    model_dir + ampl_abb_filename + ".fits", self.prad,
                    self.dim_overpad_pupil)
            else:
                ampl_abb_filename = "ampl_{:d}percentrms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                    int(ampl_rms), int(ampl_slope), ampl_rhoc, self.prad)

                if set_random_ampl is False and os.path.isfile(
                        Model_local_dir + ampl_abb_filename + ".fits") == True:
                    return_ampl = fits.getdata(Model_local_dir +
                                               ampl_abb_filename + ".fits")
                else:
                    return_ampl = phase_ampl.random_phase_map(
                        self.prad, self.dim_overpad_pupil, ampl_rms / 100,
                        ampl_rhoc, ampl_slope)

                    fits.writeto(Model_local_dir + ampl_abb_filename + ".fits",
                                 return_ampl,
                                 overwrite=True)

            return return_ampl
        else:
            return 0.

    def EF_from_phase_and_ampl(self,
                               phase_abb=0.,
                               ampl_abb=0.,
                               wavelengths=-1.):
        """ --------------------------------------------------
        Create an electrical field from an phase and amplitude aberrations as follows:

        EF = (1 + ample_abb)*exp(i*phase_abb * self.wavelength_0 / wavelength)
        can be monochromatic (return 2d compex array) or polychromatic (return 3d compex array)
        if no phase nor amplitude, return 1.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            phase aberration at reference wavelength self.wavelength_0. if 0, no phase aberration (default)

        ampl_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
             amplitude aberration at reference wavelength self.wavelength_0. if 0, no amplitude aberration (default)

        wavelengths : float or list of floats. 
                    Default is all the wl of the testbed self.wav_vec
                    wavelengths in m.


        Returns
        ------
        EF : scalar or numpy 2D array or numpy 3d array
            Electric field in the pupil plane a the exit of the system
            1. if no phase / amplitude
            2D array, of size phase_abb.shape if monochromatic
            or 3D array of size [self.nb_wav,phase_abb.shape] in case of polychromatic


        -------------------------------------------------- """

        if np.iscomplexobj(phase_abb) or np.iscomplexobj(ampl_abb):
            raise Exception(
                "phase_abb and ampl_abb must be real arrays or float, not complex"
            )

        if (isinstance(phase_abb,
                       (float, int))) and (phase_abb == 0.) and (isinstance(
                           ampl_abb, (float, int))) and (ampl_abb == 0.):
            return 1.

        elif isinstance(wavelengths, (float, int)):
            if wavelengths == -1:
                wavelength_vec = self.wav_vec
            else:
                wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths

        entrance_EF = list()
        for wavelength in wavelength_vec:
            entrance_EF.append(
                (1 + ampl_abb) *
                np.exp(1j * phase_abb * self.wavelength_0 / wavelength))
        entrance_EF = np.array(entrance_EF)

        if len(wavelength_vec) == 1:
            entrance_EF = entrance_EF[0]

        return entrance_EF


##############################################
##############################################
### PUPIL
class pupil(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of single pupil
    pupil is a sub class of Optical_System.

    Obviously you can define your pupil
    without that with 2d arrray multiplication (this is a fairly simple object).

    The main advantage of defining them using Optical_System is that you can
    use default Optical_System functions to obtain PSF, transmission, etc...
    and concatenate them with other elements

    AUTHOR : Johan Mazoyer

    
    -------------------------------------------------- """
    def __init__(self, modelconfig, prad=0., PupType="", filename=""):
        """ --------------------------------------------------
        Initialize a pupil object.           
        TODO: include an SCC Lyot pupil function here !
        TODO: for now pupil .fits are monochromatic but the pupil propagation EF_through
            use wavelenght as a parameter

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                    general configuration parameters (sizes and dimensions)
                        to initialize Optical_System class

        prad : int 
            Default is the pupil prad in the parameter
            radius in pixels of the round pupil.

        PupType : string (default "RoundPup") 
                Currently "RoundPup", "CleanPlane", "RomanPup", "RomanLyot")

        filename : string (default "")
            name and directory of the .fits file
            The pupil .fits files should be 2D and square([prad,prad])
            with even number of pix and centered between 4 pixels.
            if dim_fits < dim_overpad_pupil then the pupil is zero-padded
            if dim_fits > dim_overpad_pupil we raise an Exception

            This is a bit dangerous because your .fits file might must be defined
            the right way so be careful


        -------------------------------------------------- """
        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)
        if prad == 0:
            prad = self.prad

        self.exitpup_rad = prad

        self.pup = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                           1.)

        # known case, with known response
        # default case: round pupil
        if (PupType == "") or (PupType == "RoundPup"):
            self.pup = phase_ampl.roundpupil(self.dim_overpad_pupil, prad)

        # ClearPlane (in case we want to define an empty pupil plane)
        elif PupType == "ClearPlane":
            self.pup = phase_ampl.roundpupil(self.dim_overpad_pupil, prad)

        elif PupType == "RomanPup":
            self.string_os += '_' + PupType

            #Rescale to the pupil size
            pup_fits = fits.getdata(
                os.path.join(model_dir, "roman_pup_500pix_center4pixels.fits"))
            pup_fits_right_size = skimage.transform.rescale(
                pup_fits,
                2 * self.prad / pup_fits.shape[0],
                preserve_range=True,
                anti_aliasing=True,
                multichannel=False)

            self.pup = proc.crop_or_pad_image(pup_fits_right_size,
                                              self.dim_overpad_pupil)

        elif PupType == "RomanLyot":
            #Rescale to the pupil size
            pup_fits = fits.getdata(
                os.path.join(model_dir,
                             "roman_lyot_500pix_center4pixels.fits"))
            pup_fits_right_size = skimage.transform.rescale(
                pup_fits,
                2 * self.prad / pup_fits.shape[0],
                preserve_range=True,
                anti_aliasing=True,
                multichannel=False)

            self.pup = proc.crop_or_pad_image(pup_fits_right_size,
                                              self.dim_overpad_pupil)

        elif filename != "":

            # we start by a bunch of tests to check
            # that pupil has a certain acceptable form.
            # print("we load the pupil: " + filename)
            # print("we assume it is centered in its array")
            pup_fits = fits.getdata(filename)

            if len(pup_fits.shape) != 2:
                raise Exception("file " + filename + " should be a 2D array")

            if pup_fits.shape[0] != pup_fits.shape[1]:
                raise Exception("file " + filename +
                                " appears to be not square")

            # this assume that the pupil file is squared
            # and is centered in the file

            if pup_fits.shape[0] == self.prad:
                pup_fits_right_size = pup_fits
            else:
                #Rescale to the pupil size
                pup_fits_right_size = skimage.transform.rescale(
                    pup_fits,
                    2 * prad / pup_fits.shape[0],
                    preserve_range=True,
                    anti_aliasing=True,
                    multichannel=False)

            self.pup = proc.crop_or_pad_image(pup_fits_right_size,
                                              self.dim_overpad_pupil)

        else:  # no filename and no known. In this case, we can have a few
            raise Exception(
                "this is not a known 'PupType': 'RoundPup', 'ClearPlane', 'RomanPup', 'RomanLyot'"
            )

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field through the pupil
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m.

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.

        dir_save_all_planes : default None. 
                                directory to save all plane
                                in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Electric field in the pupil plane a the exit of the system


        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)
        if wavelength is None:
            wavelength = self.wavelength_0

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_pupil' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        if len(self.pup.shape) == 2:
            exit_EF = entrance_EF * self.pup

        elif len(self.pup.shape) == 3:
            if self.pup.shape != self.nb_wav:
                raise Exception(
                    "I'm confused, your pupil seem to be polychromatic" +
                    "(pup.shape=3) but the # of WL (pup.shape[0]={}) ".format(
                        self.pup.shape[0]) +
                    "is different from the system # of WL (nb_wav={})".format(
                        self.nb_wav))
            else:
                exit_EF = entrance_EF * self.pup[self.wav_vec.tolist().index(
                    wavelength)]
        else:
            raise Exception("pupil dimension are not acceptable")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_pupil' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        return exit_EF


##############################################
##############################################
### CORONAGRAPHS
class coronagraph(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a coronagraph system (from apod plane to the Lyot plane)
    coronagraph is a sub class of Optical_System.

    AUTHOR : Johan Mazoyer

    -------------------------------------------------- """
    def __init__(self, modelconfig, coroconfig):
        """ --------------------------------------------------
        Initialize a coronograph object

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)
        coroconfig : : dict
                coronagraph parameters
        

        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        #pupil and Lyot stop in m
        if coroconfig["filename_instr_lyot"] in ["RoundPup", "ClearPlane"]:
            self.diam_lyot_in_m = coroconfig["diam_lyot_in_m"]
        elif coroconfig["filename_instr_lyot"] == "RomanLyot":
            self.diam_lyot_in_m = self.diam_pup_in_m * 0.800
        elif coroconfig["filename_instr_lyot"] == "VLTLyot":
            self.diam_lyot_in_m = self.diam_pup_in_m * 0.95
        else:
            raise Exception("This is not a valid Lyot option")

        self.lyotrad = int(self.prad * self.diam_lyot_in_m /
                           self.diam_pup_in_m)

        self.exitpup_rad = self.lyotrad

        #coronagraph
        self.corona_type = coroconfig["corona_type"].lower()

        self.string_os += '_' + self.corona_type

        # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
        self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
        for i, wav in enumerate(self.wav_vec):
            self.dim_fp_fft[i] = int(
                np.ceil(self.prad * self.Science_sampling * self.wavelength_0 /
                        wav)) * 2
            # we take the ceil to be sure that we measure at least the good resolution
            # We do not need to be exact, the mft in science_focal_plane will be

        if self.corona_type == "fqpm":
            self.prop_apod2lyot = 'mft'
            self.err_fqpm = coroconfig["err_fqpm"]
            self.achrom_fqpm = coroconfig["achrom_fqpm"]
            self.FPmsk = self.FQPM()
            if self.achrom_fqpm:
                str_achrom = "achrom"
            else:
                str_achrom = "nonachrom"
            self.string_os += '_' + str_achrom
            self.perfect_coro = True

        elif self.corona_type == "classiclyot" or self.corona_type == "hlc":
            self.prop_apod2lyot = 'mft-babinet'
            self.Lyot_fpm_sampling = 30.  # hard coded for now, this is very internal cooking
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            self.string_os += '_' + "iwa" + str(round(self.rad_lyot_fpm, 2))
            self.perfect_coro = False
            if self.corona_type == "classiclyot":
                self.FPmsk = self.ClassicalLyot()
            else:
                self.transmission_fpm = coroconfig["transmission_fpm"]
                self.phase_fpm = coroconfig["phase_fpm"]
                self.string_os += '_' + "trans{:.1e}".format(
                    self.transmission_fpm) + "_pha{0}".format(
                        round(self.phase_fpm, 2))
                self.FPmsk = self.HLC()

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'mft'
            self.coro_position = coroconfig["knife_coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.string_os += '_' + self.coro_position + "_iwa" + str(
                round(self.knife_coro_offset, 2))
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'mft'
            vortex_charge = coroconfig["vortex_charge"]
            self.string_os += '_charge' + str(int(vortex_charge))
            self.FPmsk = self.Vortex(vortex_charge=vortex_charge)
            self.perfect_coro = True

        else:
            raise Exception("this coronagrpah mode does not exists yet")

        # We need a pupil only to measure the response
        # of the coronograph to a clear pupil to remove it
        # if perfect corono. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        if self.perfect_coro == True:
            self.clearpup = pupil(modelconfig, prad=self.prad)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        if coroconfig["filename_instr_apod"] == "ClearPlane" or coroconfig[
                "filename_instr_apod"] == "RoundPup":
            self.apod_pup = pupil(modelconfig,
                                  prad=self.prad,
                                  PupType=coroconfig["filename_instr_apod"])
        else:
            self.apod_pup = pupil(modelconfig,
                                  prad=self.prad,
                                  filename=coroconfig["filename_instr_apod"])

        if coroconfig["filename_instr_lyot"] in [
                "ClearPlane", "RoundPup", "RomanPup", "RomanLyot"
        ]:
            self.lyot_pup = pupil(modelconfig,
                                  prad=self.lyotrad,
                                  PupType=coroconfig["filename_instr_lyot"])
        else:
            self.string_os += coroconfig["filename_instr_lyot"]
            self.lyot_pup = pupil(modelconfig,
                                  prad=self.lyotrad,
                                  filename=coroconfig["filename_instr_lyot"])

        self.string_os += '_lrad' + str(int(self.lyotrad))

        if self.perfect_coro == True:
            # do a propagation once with self.perfect_Lyot_pupil = 0 to
            # measure the Lyot pupil that will be removed after
            self.perfect_Lyot_pupil = 0
            self.perfect_Lyot_pupil = self.EF_through(
                entrance_EF=self.clearpup.EF_through())

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   noFPM=False,
                   EF_aberrations_introduced_in_LS=1.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                        current wavelength in m.

        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
        
        EF_aberrations_introduced_in_LS: 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
                        electrical field created by the downstream aberrations introduced directly in the Lyot Stop

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        
        dir_save_all_planes : default None. 
                                directory to save all plane in
                              fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                Electric field in the pupil plane a the exit of the system
        

        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_apod' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

        if noFPM:
            FPmsk = 1.
        else:
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        lambda_ratio = wavelength / self.wavelength_0

        input_wavefront_after_apod = self.apod_pup.EF_through(
            entrance_EF=entrance_EF, wavelength=wavelength)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_apod' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      input_wavefront_after_apod)

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(
                wavelength)]
            input_wavefront_after_apod_pad = proc.crop_or_pad_image(
                input_wavefront_after_apod, dim_fp_fft_here)
            # Phase ramp to center focal plane between 4 pixels

            maskshifthalfpix = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, 0.5, 0.5)
            maskshifthalfpix_invert = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, -0.5, -0.5)

            #Apod plane to focal plane
            corono_focal_plane = np.fft.fft2(np.fft.fftshift(
                input_wavefront_after_apod_pad * maskshifthalfpix),
                                             norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)

                name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = np.fft.fftshift(
                np.fft.ifft2(corono_focal_plane * FPmsk,
                             norm='ortho')) * maskshifthalfpix_invert
            # we take the convention that when we are in pupil plane the field must be
            # "non-shifted". Because the shift in pupil plane is resolution dependent
            # which depend on the method (fft is not exactly science resolution because
            # of rounding issues, mft-babinet does not use the science resolution, etc).
            # these shift in both direction should be included in apod and pup multiplication
            # to save time

        elif self.prop_apod2lyot == "mft-babinet":
            #Apod plane to focal plane

            corono_focal_plane = prop.mft(
                input_wavefront_after_apod,
                self.dim_overpad_pupil,
                self.dim_fpm,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                X_offset_output=-0.5,
                Y_offset_output=-0.5,
                inverse=False,
                norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)

                name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

                name_plane = 'EF_FP_after_1minusFPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * (1 - FPmsk))

            # Focal plane to Lyot plane
            # Babinet's trick:
            lyotplane_before_lyot_central_part = prop.mft(
                corono_focal_plane * (1 - FPmsk),
                self.dim_fpm,
                self.dim_overpad_pupil,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                X_offset_input=-0.5,
                Y_offset_input=-0.5,
                inverse=True,
                norm='ortho')

            # Babinet's trick
            lyotplane_before_lyot = input_wavefront_after_apod - lyotplane_before_lyot_central_part

        elif self.prop_apod2lyot == "mft":
            # Apod plane to focal plane

            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          2 * self.prad,
                                          self.dimScience,
                                          self.dimScience /
                                          self.Science_sampling * lambda_ratio,
                                          X_offset_output=-0.5,
                                          Y_offset_output=-0.5,
                                          inverse=False,
                                          norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane)
                if not noFPM:
                    name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                    useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                              FPmsk)

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = proc.crop_or_pad_image(
                prop.mft(corono_focal_plane * FPmsk,
                         self.dimScience,
                         2 * self.prad,
                         self.dimScience / self.Science_sampling *
                         lambda_ratio,
                         X_offset_input=-0.5,
                         Y_offset_input=-0.5,
                         inverse=True,
                         norm='ortho'), self.dim_overpad_pupil)

        else:
            raise Exception(
                self.prop_apod2lyot +
                " is not a known prop_apod2lyot propagation mehtod")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_before_lyot)

        # we add the downstream aberrations if we need them
        lyotplane_before_lyot *= EF_aberrations_introduced_in_LS

        # crop to the dim_overpad_pupil expeted size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(
            lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(
            entrance_EF=lyotplane_before_lyot_crop, wavelength=wavelength)

        if (self.perfect_coro == True) & (noFPM == False):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        if save_all_planes_to_fits == True:
            name_plane = 'LS' + '_wl{}'.format(int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      self.lyot_pup.pup)

            name_plane = 'EF_PP_after_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_after_lyot)

        return lyotplane_after_lyot

    def FQPM(self):
        """ --------------------------------------------------
        Create a Four Quadrant Phase Mask coronagraph
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer


        Returns
        ------
        FQPM : list of len(self.wav_vec) 2D arrays   
            complex transmission of the FQPM mask at all wl
        
        
        -------------------------------------------------- """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dimScience

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
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dimScience

            phase4q = np.zeros((dim_fp, dim_fp))
            fqpm_thick_cut = proc.crop_or_pad_image(fqpm_thick, dim_fp)
            phase4q[np.where(fqpm_thick_cut != 0)] = (np.pi + self.err_fqpm)

            if self.achrom_fqpm == True:
                # if we want to do an an achromatic_fqpm, we do not include a variation
                # of the phase with the wl.
                fqpm.append(np.exp(1j * phase4q))
            else:
                # in the general case, we use the EF_from_phase_and_ampl which handle the phase
                # chromaticity.
                fqpm.append(
                    self.EF_from_phase_and_ampl(phase_abb=phase4q,
                                                wavelengths=wav))

        return fqpm

    def Vortex(self, vortex_charge=2):
        """ --------------------------------------------------
        Create a charge2 vortex.

        AUTHOR : Johan Mazoyer

        Parameters
        ------
        Charge : int, defaut 2
            charge of the vortex. can be 2, 4, 6. Defaut is charge 2

        Returns
        ------
        vortex_fpm : list of 2D numpy array
                            the FP mask at all wl


        -------------------------------------------------- """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dimScience

        xx, yy = np.meshgrid(
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2,
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2)

        phase_vortex = self.vortex_charge * np.angle(xx + 1j * yy)

        vortex = list()
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dimScience

            phasevortex_cut = proc.crop_or_pad_image(phase_vortex, dim_fp)
            vortex.append(np.exp(1j * phasevortex_cut))

        return vortex

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dimScience,dimScience)
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer

        Returns
        ------
        Knife FPM : list of len(self.wav_vec) 2D arrays 
                    gcomplex transmission of the Knife edge coronagraph mask at all wl


        -------------------------------------------------- """
        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
            if len(self.wav_vec) > 1:
                raise Exception(
                    "knife currently not coded in polychromatic fft")
        else:
            maxdimension_array_fpm = self.dimScience

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength

        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm),
                             np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(
                xx > (maxdimension_array_fpm / 2 +
                      self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "left":
            Knife[np.where(
                xx < (maxdimension_array_fpm / 2 -
                      self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(
                yy > (maxdimension_array_fpm / 2 +
                      self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "top":
            Knife[np.where(
                yy < (maxdimension_array_fpm / 2 -
                      self.knife_coro_offset * self.Science_sampling))] = 1

        knife_allwl = list()
        for i in range(len(self.wav_vec)):
            knife_allwl.append(Knife)

        return knife_allwl

    def ClassicalLyot(self):
        """ --------------------------------------------------
        Create a classical Lyot coronagraph of radius rad_LyotFP 0
        AUTHOR : Johan Mazoyer

        Returns
        ------
        classical Lyot fpm : list of 2D numpy array
                            the FP mask at all wl

        
        -------------------------------------------------- """

        rad_LyotFP_pix = self.rad_lyot_fpm * self.Lyot_fpm_sampling

        self.dim_fpm = 2 * int(2.2 * rad_LyotFP_pix / 2)
        ClassicalLyotFPM = 1. - phase_ampl.roundpupil(self.dim_fpm,
                                                      rad_LyotFP_pix)

        ClassicalLyotFPM_allwl = list()
        for wav in self.wav_vec:
            ClassicalLyotFPM_allwl.append(ClassicalLyotFPM)

        return ClassicalLyotFPM_allwl

    def HLC(self):
        """ --------------------------------------------------
        Create a HLC of radius rad_LyotFP 0
        AUTHOR : Johan Mazoyer

        Returns
        ------
        classical Lyot hlc : list of 2D numpy array
                            the FP mask at all wl
        
        
        -------------------------------------------------- """

        # we create a Classical Lyot Focal plane
        ClassicalLyotFP = self.ClassicalLyot()[0]

        whClassicalLyotstop = np.where(ClassicalLyotFP == 0.)

        # we define phase and amplitude for the HLC at the reference WL
        phase_hlc = np.zeros(ClassicalLyotFP.shape)
        phase_hlc[whClassicalLyotstop] = self.phase_fpm
        # transmission_fpm is defined in intensity and EF_from_phase_and_ampl takes amplitude
        ampl_hlc = np.zeros(ClassicalLyotFP.shape)
        ampl_hlc[whClassicalLyotstop] = np.sqrt(self.transmission_fpm) - 1

        hlc_all_wl = list()
        for wav in self.wav_vec:
            hlc_all_wl.append(
                self.EF_from_phase_and_ampl(ampl_abb=ampl_hlc,
                                            phase_abb=phase_hlc,
                                            wavelengths=wav))

        return hlc_all_wl


##############################################
##############################################
### Deformable mirrors
class deformable_mirror(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a deformable mirror
    (in pupil plane or out of pupil plane)
    coronagraph is a sub class of Optical_System.


    AUTHOR : Johan Mazoyer


    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 DMconfig,
                 Name_DM='DM3',
                 Model_local_dir=None):
        """ --------------------------------------------------
        Initialize a deformable mirror object

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)

        DMconfig : dict
                DM configuration parameters dictionary 

        Name_DM : string
                the name of the DM, which allows to find it in the parameter file
                we measure and save the pushact functions

        Model_local_dir: path
                directory to save things you can measure yourself
                    and can save to save time
        
        
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        if not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)

        self.Model_local_dir = Model_local_dir

        self.exitpup_rad = self.prad

        self.Name_DM = Name_DM
        self.z_position = DMconfig[self.Name_DM + "_z_position"]
        self.active = DMconfig[self.Name_DM + "_active"]

        self.WhichInPup_threshold = DMconfig["MinimumSurfaceRatioInThePupil"]

        # For intialization, we assume no misregistration, we introduce it after
        # estimation and correction matrices are created.
        self.misregistration = False

        if DMconfig[self.Name_DM + "_Generic"] == True:
            self.total_act = DMconfig[self.Name_DM + "_Nact1D"]**2
            self.number_act = self.total_act
            self.active_actuators = np.arange(self.number_act)
        else:
            # first thing we do is to open filename_grid_actu to check the number of
            # actuator of this DM. We need the number of act to read and load pushact .fits
            self.total_act = fits.getdata(
                model_dir +
                DMconfig[self.Name_DM + "_filename_grid_actu"]).shape[1]

            if DMconfig[self.Name_DM + "_filename_active_actu"] != "":
                self.active_actuators = fits.getdata(model_dir + DMconfig[
                    self.Name_DM + "_filename_active_actu"]).astype(int)
                self.number_act = len(self.active_actuators)

            else:
                self.number_act = self.total_act
                self.active_actuators = np.arange(self.number_act)

        self.string_os += '_' + self.Name_DM + "_z" + str(
            int(self.z_position * 100)) + "_Nact" + str(int(self.number_act))

        if DMconfig[self.Name_DM + "_Generic"] == True:
            self.string_os += "Gen"

        if self.active == False:
            print(self.Name_DM + ' is not activated')
            return

        self.DMconfig = DMconfig

        # We need a pupil in creatingpushact_inpup() and for
        # which in pup. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        self.clearpup = pupil(modelconfig, prad=self.prad)

        #define self.pradDM and check that the pupil is large enough
        # for out of pupil propagation
        if self.z_position != 0:
            dx, dxout = prop.prop_fresnel(self.dim_overpad_pupil,
                                          self.wavelength_0,
                                          self.z_position,
                                          self.diam_pup_in_m / 2,
                                          self.prad,
                                          retscale=1)
            #radius of the pupil in pixel in the DM plane
            self.pradDM = self.prad * dx / dxout

            if dx > 2 * dxout:
                print(dx, dxout)
                raise Exception(
                    "Need to enhance the pupil size in pixel for Fresnel propagation"
                )
        else:
            # radius of the pupil in pixel in the DM plane.
            # by default the size of the pupil
            self.pradDM = self.prad

        # create, save or load the DM_pushact functions
        # from the influence function

        # DM_pushact is always in the DM plane
        start_time = time.time()
        self.DM_pushact = self.creatingpushact(DMconfig)
        print("time for DM_pushact for " + self.string_os,
              time.time() - start_time)

        start_time = time.time()
        # create or load 'which actuators are in pupil'
        self.WhichInPupil = self.creatingWhichinPupil()
        print("time for WhichInPupil for " + self.string_os,
              time.time() - start_time)

        self.misregistration = DMconfig[self.Name_DM + "_misregistration"]
        # now if we relaunch self.DM_pushact, and if misregistration = True
        # it will be different due to misregistration

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   DMphase=0.,
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field through the DM.
        if z_DM = 0, then it's just a phase multiplication
        if z_DM != 0, this is where we do the fresnel

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil] or float scalar in which case entrance_EF is constant
                        default is 1.
                        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                    current wavelength in m.

        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex or float scalar in which case DM_phase is constant
                    default is 0.

                    CAREFUL !! If the DM is part of a testbed. this variable name is changed
                    to DMXXphase (DMXX: name of the DM) to avoid confusion with

        save_all_planes_to_fits: Bool, default False
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.

        dir_save_all_planes : path, default None 
                            directory to save all plane in
                                    fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system


        -------------------------------------------------- """

        # call the Optical_System super function to check
        # and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

        if isinstance(DMphase, (int, float, np.float)):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                              np.float(DMphase))

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)
            name_plane = 'phase_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane, DMphase)

        # if the DM is not active or if the surface is 0
        # we save some time : the EF is not modified
        if self.active == False or (DMphase == 0.).all():
            return entrance_EF

        if self.z_position == 0:
            EF_after_DM = entrance_EF * self.EF_from_phase_and_ampl(
                phase_abb=DMphase, wavelengths=wavelength)

        else:
            EF_after_DM = self.prop_pup_to_DM_and_back(
                entrance_EF,
                DMphase,
                wavelength,
                save_all_planes_to_fits=save_all_planes_to_fits,
                dir_save_all_planes=dir_save_all_planes)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_after_DM)

        return EF_after_DM

    def creatingpushact(self, DMconfig):
        """ --------------------------------------------------
        OPD map induced in the DM plane for each actuator.

        This large array is initialized at the beginning and will be use
        to transorm a voltage into a phase for each DM. This is saved
        in .fits to save times if the parameter have not changed

        In case of "misregistration = True" we measure it once for
        creating the interraction matrix and then once again, between
        the matrix measrueemnt and the correction with a small mismatch
        to simulate its effect.

        AUTHOR : Axel Potier

        Parameters
        ----------
        DMconfig : dict
            DM configuration parameters dictionary 

        Returns
        ------
        pushact : 3D numpy array 
                    of size [self.number_act, self.dim_overpad_pupil, self.dim_overpad_pupil]  
                    contains all the DM OPD map induced in the DM plane for each actuator.

        
        -------------------------------------------------- """

        Name_pushact_fits = "PushAct" + self.string_os

        if (self.misregistration is
                False) and (os.path.exists(self.Model_local_dir +
                                           Name_pushact_fits + '.fits')):
            pushact3d = fits.getdata(
                os.path.join(self.Model_local_dir,
                             Name_pushact_fits + '.fits'))
            return pushact3d

        if self.misregistration:
            xerror = float(DMconfig[self.Name_DM + "_xerror"])
            yerror = float(DMconfig[self.Name_DM + "_yerror"])
            angerror = float(DMconfig[self.Name_DM + "_angerror"])
            gausserror = float(DMconfig[self.Name_DM + "_gausserror"])
        else:
            xerror = 0.
            yerror = 0.
            angerror = 0.
            gausserror = 0.

        diam_pup_in_pix = 2 * self.prad
        diam_pup_in_m = self.diam_pup_in_m
        dim_array = self.dim_overpad_pupil

        pitchDM = DMconfig[self.Name_DM + "_pitch"]
        filename_actu_infl_fct = DMconfig[self.Name_DM +
                                          "_filename_actu_infl_fct"]

        if DMconfig[self.Name_DM + "_Generic"] == False:
            filename_ActuN = DMconfig[self.Name_DM + "_filename_ActuN"]
            filename_grid_actu = DMconfig[self.Name_DM + "_filename_grid_actu"]

            ActuN = DMconfig[self.Name_DM + "_ActuN"]
            y_ActuN = DMconfig[self.Name_DM + "_y_ActuN"]
            x_ActuN = DMconfig[self.Name_DM + "_x_ActuN"]
            xy_ActuN = [x_ActuN, y_ActuN]

            #Measured positions for each actuator in pixel
            measured_grid = fits.getdata(model_dir + filename_grid_actu)
            #Ratio: pupil radius in the measured position over
            # pupil radius in the numerical simulation
            sampling_simu_over_measured = diam_pup_in_pix / 2 / fits.getheader(
                model_dir + filename_grid_actu)['PRAD']
            if filename_ActuN != "":
                im_ActuN = fits.getdata(model_dir + filename_ActuN)
                im_ActuN_dim = proc.crop_or_pad_image(im_ActuN, dim_array)

                xy_ActuN = np.unravel_index(
                    np.abs(im_ActuN_dim).argmax(), im_ActuN_dim.shape)

                # shift by (0.5,0.5) pixel because the pupil is
                # centered between pixels
                xy_ActuN = xy_ActuN - 0.5

            #Position for each actuator in pixel for the numerical simulation
            simu_grid = proc.actuator_position(measured_grid, xy_ActuN, ActuN,
                                               sampling_simu_over_measured)
        else:
            # in this case we have a generic Nact1DxNact1D DM in which the pupil is centered
            Nact1D = DMconfig[self.Name_DM + "_Nact1D"]
            simu_grid = proc.generic_actuator_position(Nact1D, pitchDM,
                                                       diam_pup_in_m,
                                                       diam_pup_in_pix)

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
            actshapeinpupil[0:len(resizeactshape),
                            0:len(resizeactshape
                                  )] = resizeactshape / np.amax(resizeactshape)
            xycenttmp = len(resizeactshape) / 2
        else:
            actshapeinpupil = resizeactshape[
                0:dim_array, 0:dim_array] / np.amax(resizeactshape)
            xycenttmp = dim_array / 2

        # Fill an array with the influence functions of all actuators
        pushact3d = np.zeros((simu_grid.shape[1], dim_array, dim_array))
        for i in np.arange(pushact3d.shape[0]):
            if gausserror == 0:
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp +
                     yerror * pitch_actshape, simu_grid[0, i] + dim_array / 2 -
                     xycenttmp + xerror * pitch_actshape))

                # Add an error on the orientation of the grid
                if angerror != 0:
                    Psivector = nd.rotate(Psivector,
                                          angerror,
                                          order=5,
                                          cval=0,
                                          reshape=False)[0:dim_array,
                                                         0:dim_array]
            else:
                # Add an error on the sizes of the influence functions
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp,
                     simu_grid[0, i] + dim_array / 2 - xycenttmp))

                xy0 = np.unravel_index(Psivector.argmax(), Psivector.shape)
                x, y = np.mgrid[0:dim_array, 0:dim_array]
                xy = (x, y)
                Psivector = proc.twoD_Gaussian(xy,
                                               1,
                                               1 + gausserror,
                                               1 + gausserror,
                                               xy0[0],
                                               xy0[1],
                                               0,
                                               0,
                                               flatten=False)
            Psivector[np.where(Psivector < 1e-4)] = 0

            pushact3d[i] = Psivector

        pushact3d = pushact3d[self.active_actuators]

        if self.misregistration is False and (
                not os.path.exists(self.Model_local_dir + Name_pushact_fits +
                                   '.fits')):
            fits.writeto(self.Model_local_dir + Name_pushact_fits + '.fits',
                         pushact3d)

        return pushact3d

    def creatingWhichinPupil(self):
        """ --------------------------------------------------
        Create a vector with the index of all the actuators located in the entrance pupil
        
        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        cutinpupil: float
                    minimum surface of an actuator inside the pupil to be taken into account
                    (between 0 and 1, ratio of an actuator perfectly centered in the entrance pupil)

        Returns
        ------
        WhichInPupil: 1D array
                index of all the actuators located inside the pupil

        
        -------------------------------------------------- """

        Name_WhichInPup_fits = "WhichInPup" + self.string_os + "_thres" + str(
            self.WhichInPup_threshold)

        if os.path.exists(self.Model_local_dir + Name_WhichInPup_fits +
                          '.fits'):
            return fits.getdata(self.Model_local_dir + Name_WhichInPup_fits +
                                '.fits')

        if self.z_position != 0:
            # Propagation in DM plane out of pupil
            Pup_inDMplane, _ = prop.prop_fresnel(self.clearpup.pup,
                                                 self.wavelength_0,
                                                 self.z_position,
                                                 self.diam_pup_in_m / 2,
                                                 self.prad)
        else:
            Pup_inDMplane = self.clearpup.pup

        WhichInPupil = []
        Sum_actu_with_pup = np.zeros(self.number_act)

        for num_actu in np.arange(self.number_act):
            Sum_actu_with_pup[num_actu] = np.sum(
                np.abs(self.DM_pushact[num_actu] * Pup_inDMplane))

        Max_val = np.max(Sum_actu_with_pup)
        for num_actu in np.arange(self.number_act):
            if Sum_actu_with_pup[
                    num_actu] > Max_val * self.WhichInPup_threshold:
                WhichInPupil.append(num_actu)

        WhichInPupil = np.array(WhichInPupil)

        fits.writeto(self.Model_local_dir + Name_WhichInPup_fits + '.fits',
                     WhichInPupil,
                     overwrite=True)
        return WhichInPupil

    def prop_pup_to_DM_and_back(self,
                                entrance_EF,
                                phase_DM,
                                wavelength,
                                save_all_planes_to_fits=False,
                                dir_save_all_planes=None):
        """ --------------------------------------------------
        Propagate the field towards an out-of-pupil plane ,
        add the DM phase, and propagate to the next pupil plane
        
        AUTHOR : Raphaël Galicher, Johan Mazoyer

        REVISION HISTORY :
            Revision 1.1  2021-02-10 Raphaël Galicher (Initial revision)
            Revision 2.0 2021-02-28 Johan Mazoyer (Make it more general for all DMs, put in the struc)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
                    Wavefront in the pupil plane

        phase_DM : 2D array
                    Phase introduced by out of PP DM

        wavelength : float
                    wavelength in m

        save_all_planes_to_fits: Bool, default False
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        
        dir_save_all_planes : string default None. 
                            directory to save all plane in fits if save_all_planes_to_fits = True

        Returns
        ------
        EF_back_in_pup_plane : 2D array (complex)
                            Wavefront in the pupil plane following the DM

            

        -------------------------------------------------- """

        # Propagation in DM plane out of pupil
        EF_inDMplane, _ = prop.prop_fresnel(entrance_EF, wavelength,
                                            self.z_position,
                                            self.diam_pup_in_m / 2., self.prad)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_before_DM_in_' + self.Name_DM + 'plane_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_inDMplane)

        # Add DM phase at the right WL
        EF_inDMplane_after_DM = EF_inDMplane * self.EF_from_phase_and_ampl(
            phase_abb=phase_DM, wavelengths=wavelength)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_after_DM_in_' + self.Name_DM + 'plane_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_inDMplane)
        # and propagate to next pupil plane
        EF_back_in_pup_plane, _ = prop.prop_fresnel(EF_inDMplane_after_DM,
                                                    wavelength,
                                                    -self.z_position,
                                                    self.diam_pup_in_m / 2.,
                                                    self.prad)

        return EF_back_in_pup_plane

    def voltage_to_phase(self, actu_vect, einstein_sum=False):
        """ --------------------------------------------------
        Generate the phase applied on one DM for a give vector of actuator amplitude
        We decided to do it without matrix multiplication to save time because a
        lot of the time we have lot of zeros in it

        The phase is define at the reference wl and multiply by wl_ratio in DM.EF_through

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        actu_vect : 1D array
                    values of the amplitudes for each actuator
        einstein_sum : boolean, default false
                        Use numpy Einstein sum to sum the pushact[i]*actu_vect[i]
                        gives the same results as normal sum. Seems ot be faster for unique actuator
                        but slower for more complex phases

        Returns
        ------
            DM_phase: 2D array
                        phase map in the same unit as actu_vect * DM_pushact)
        

        -------------------------------------------------- """

        where_non_zero_voltage = np.where(actu_vect != 0)

        #opd is in nanometer
        # DM_pushact is in opd nanometer
        opd_to_phase = 2 * np.pi * 1e-9 / self.wavelength_0

        if einstein_sum == True or len(where_non_zero_voltage[0]) < 3:
            phase_on_DM = np.einsum(
                'i,ijk->jk', actu_vect[where_non_zero_voltage],
                self.DM_pushact[where_non_zero_voltage]) * opd_to_phase
        else:
            phase_on_DM = np.zeros(
                (self.dim_overpad_pupil, self.dim_overpad_pupil))
            for i in where_non_zero_voltage[0]:
                phase_on_DM += self.DM_pushact[
                    i, :, :] * actu_vect[i] * opd_to_phase

        return phase_on_DM

    def create_DM_basis(self, basis_type='actuator'):
        """ --------------------------------------------------
        Create a DM basis.
        TODO do a zernike basis ?

        AUTHOR: Johan Mazoyer

        Parameters
        ----------
        basis_type: string, default 'actuator' 
            the type of basis. 'fourier' or 'actuator'

        Returns
        ------
        basis: 2d numpy array 
            basis [Size basis, Number of active act in the DM]


        -------------------------------------------------- """

        if basis_type == 'actuator':
            active_and_in_pup = [
                value for value in self.active_actuators
                if value in self.WhichInPupil
            ]
            active_and_in_pup.sort()

            basis_size = len(active_and_in_pup)
            basis = np.zeros((basis_size, self.number_act))
            for i in range(basis_size):
                basis[i][active_and_in_pup[i]] = 1

        elif basis_type == 'fourier':
            start_time = time.time()
            activeact = [value for value in self.active_actuators]

            sqrtnbract = int(np.sqrt(self.total_act))
            Name_FourrierBasis_fits = "Fourier_basis_" + self.Name_DM + '_prad' + str(
                self.prad) + '_nact' + str(sqrtnbract) + 'x' + str(sqrtnbract)

            cossinbasis = proc.SinCosBasis(sqrtnbract)

            basis_size = cossinbasis.shape[0]
            basis = np.zeros((basis_size, self.number_act))

            for i in range(basis_size):
                vec = cossinbasis[i].flatten()[activeact]
                basis[i] = vec

            start_time = time.time()
            # This is a very time consuming part of the code.
            # from N voltage vectors with the sine and cosine value, we go N times through the
            # voltage_to_phase functions. For this reason we save the Fourrier base 2D phases on each DMs
            # in a specific .fits file
            if not os.path.exists(self.Model_local_dir +
                                  Name_FourrierBasis_fits + '.fits'):
                phasesFourrier = np.zeros((basis_size, self.dim_overpad_pupil,
                                           self.dim_overpad_pupil))
                print("Start " + Name_FourrierBasis_fits)
                for i in range(basis_size):
                    phasesFourrier[i] = self.voltage_to_phase(basis[i])
                    if i % 10:
                        useful._progress(i, basis_size, status='')
                fits.writeto(
                    self.Model_local_dir + Name_FourrierBasis_fits + '.fits',
                    phasesFourrier)
            print("time for " + Name_FourrierBasis_fits,
                  time.time() - start_time)

        else:
            raise Exception(basis_type + " is is not a valid basis_type")

        return basis


##############################################
##############################################
### Testbeds
class Testbed(Optical_System):
    """ --------------------------------------------------
    
    Initialize and describe the behavior of a testbed.
    This is a particular subclass of Optical System, because we do not know what is inside
    It can only be initialized by giving a list of Optical Systems and it will create a
    "testbed" with contains all the Optical Systems and associated EF_through functions and
    correct normlaization

    AUTHOR : Johan Mazoyer

    -------------------------------------------------- """
    def __init__(self, list_os, list_os_names):
        """ --------------------------------------------------
        This function allow you to concatenates Optical_System obsjects to create a testbed:
        parameter:
            list_os:        list of Optical_System
                            all the systems must have been defined with
                            the same modelconfig or it will send an error.
                            The list order is form the first optics system to the last in the
                            path of the light (so usually from entrance pupil to Lyot pupil)

            list_os_names:  list of string of the same size as list_os 
                            Name of the optical systems. 
                            Then can then be accessed inside the Testbed object by os_#i = Testbed.list_os_names[i]

        Returns
        ------
            testbed : an optical system which is the concatenation of all the optical systems



        -------------------------------------------------- """
        if len(list_os) != len(list_os_names):
            print("")
            raise Exception(
                "list of systems and list of names need to be of the same size"
            )

        # Initialize the Optical_System class and inherit properties
        super().__init__(list_os[0].modelconfig)

        init_string = self.string_os

        # Initialize the EF_through_function
        self.EF_through = super().EF_through

        # The exitpuprad parameter which will be used to plot the PSF in todetector functions
        # is the exitpuprad of the last one.
        self.exitpup_rad = list_os[-1].exitpup_rad

        self.number_DMs = 0
        self.number_act = 0
        self.name_of_DMs = list()

        # this is the collection of all the possible keywords that can be used in
        # practice in the final testbed.EF_through, so that can be used in
        # all the EF_through functions
        known_keywords = list()

        # we store the name of all the sub systems
        self.subsystems = list_os_names

        # we concatenate the Optical Element starting by the end
        for num_optical_sys in range(len(list_os)):

            # we first check that all variables in the list are optical systems
            # defined the same way.
            if not isinstance(list_os[num_optical_sys], Optical_System):
                raise Exception("list_os[" + str(num_optical_sys) +
                                "] is not an optical system")

            if list_os[num_optical_sys].modelconfig != self.modelconfig:
                print("")
                raise Exception(
                    "All optical systems need to be defined with the same initial modelconfig!"
                )

            # if the os is a DM we increase the number of DM counter and
            # store the number of act and its name

            for params in inspect.signature(
                    list_os[num_optical_sys].EF_through).parameters:
                known_keywords.append(params)

            if isinstance(list_os[num_optical_sys], deformable_mirror):

                #this function is to replace the DMphase variable by a XXphase variable
                # where XX is the name of the DM
                list_os[num_optical_sys].EF_through = _swap_DMphase_name(
                    list_os[num_optical_sys].EF_through,
                    list_os_names[num_optical_sys] + "phase")
                known_keywords.append(list_os_names[num_optical_sys] + "phase")

                if list_os[num_optical_sys].active == False:
                    # if the Dm is not active, we just add it to the testbed model
                    # but not to the EF_through function
                    vars(self)[list_os_names[num_optical_sys]] = list_os[
                        num_optical_sys]
                    continue

                self.number_DMs += 1
                self.number_act += list_os[num_optical_sys].number_act
                self.name_of_DMs.append(list_os_names[num_optical_sys])

            # concatenation of the EF_through functions
            self.EF_through = _concat_fun(list_os[num_optical_sys].EF_through,
                                          self.EF_through)

            # we add all systems to the Optical System so that they can be accessed
            vars(self)[
                list_os_names[num_optical_sys]] = list_os[num_optical_sys]

            self.string_os += list_os[num_optical_sys].string_os.replace(
                init_string, '')

        # in case there is no coronagraph in the system, we still add
        # noFPM so that it does not break when we run transmission and max_sum_PSFs
        # which pass this keyword by default
        known_keywords.append('noFPM')
        known_keywords.append('photon_noise')
        known_keywords.append('nb_photons')
        known_keywords.append('in_contrast')

        # we remove doubloons
        # known_keywords = list(set(known_keywords))
        known_keywords = list(dict.fromkeys(known_keywords))

        # We remove arguments we know are wrong
        if 'DMphase' in known_keywords:
            known_keywords.remove('DMphase')
            # there is at least a DM, we add voltage_vector as an authorize kw

            known_keywords.append('voltage_vector')
            self.EF_through = _control_testbed_with_voltages(
                self, self.EF_through)

        # to avoid mis-use we only use specific keywords.
        known_keywords.remove('kwargs')

        self.EF_through = _clean_EF_through(self.EF_through, known_keywords)

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def voltage_to_phases(self, actu_vect, einstein_sum=False):
        """ --------------------------------------------------
        Generate the phase applied on each DMs of the testbed from a given vector of
        actuator amplitude. I split theactu_vect and  then for each DM, it uses
        DM.voltage_to_phase (no s)

        Parameters
        ----------
        actu_vect : float or 1D array of size testbed.number_act
                    values of the amplitudes for each actuator and each DM
        einstein_sum : boolean. default false
                        Use numpy Einstein sum to sum the pushact[i]*actu_vect[i]
                        gives the same results as normal sum. Seems ot be faster for unique actuator
                        but slower for more complex phases

        Returns
        ------
            3D array of size [testbed.number_DMs, testbed.dim_overpad_pupil,testbed.dim_overpad_pupil]
            phase maps for each DMs by order of light path in the same unit as actu_vect * DM_pushact

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        DMphases = np.zeros(
            (self.number_DMs, self.dim_overpad_pupil, self.dim_overpad_pupil))
        indice_acum_number_act = 0

        if isinstance(actu_vect, (int, float)):
            return np.zeros(self.number_DMs) + float(actu_vect)

        if len(actu_vect) != self.number_act:
            raise Exception(
                "voltage vector must be 0 or array of dimension testbed.number_act,"
                + "sum of all DM.number_act")

        for i, DM_name in enumerate(self.name_of_DMs):

            DM = vars(self)[DM_name]  # type: deformable_mirror
            actu_vect_DM = actu_vect[
                indice_acum_number_act:indice_acum_number_act + DM.number_act]
            DMphases[i] = DM.voltage_to_phase(actu_vect_DM,
                                              einstein_sum=einstein_sum)

            indice_acum_number_act += DM.number_act
        # useful.quickfits(DMphases, dir="/Users/jmazoyer/Desktop/DM_phase/")

        return DMphases

    def basis_vector_to_act_vector(self, vector_basis_voltage):
        """ --------------------------------------------------
        transform a vector of voltages on the mode of a basis in a  vector of
        voltages of the actuators of the DMs of the system

        Parameters
        ----------
        vector_basis_voltage: 1D-array real : 
                        vector of voltages of size (total(basisDM sizes)) on the mode of the basis for all
                        DMs by order of the light path

        Returns
        ------
        vector_actuator_voltage: 1D-array real : 
                        vector of base coefficients for all actuators of the DMs by order of the light path
                        size (total(DM actuators))
        

        -------------------------------------------------- """

        indice_acum_basis_size = 0
        indice_acum_number_act = 0

        vector_actuator_voltage = np.zeros(self.number_act)
        for DM_name in self.name_of_DMs:

            # we access each DM object individually
            DM = vars(self)[DM_name]  # type: deformable_mirror

            # we extract the voltages for this one
            # this voltages are in the DM basis
            vector_basis_voltage_for_DM = vector_basis_voltage[
                indice_acum_basis_size:indice_acum_basis_size + DM.basis_size]

            # we change to actuator basis
            vector_actu_voltage_for_DM = np.dot(np.transpose(DM.basis),
                                                vector_basis_voltage_for_DM)

            # we recreate a voltages vector, but for each actuator
            vector_actuator_voltage[
                indice_acum_number_act:indice_acum_number_act +
                DM.number_act] = vector_actu_voltage_for_DM

            indice_acum_basis_size += DM.basis_size
            indice_acum_number_act += DM.number_act

        return vector_actuator_voltage


##############################################
##############################################
### internal functions to properly concatenate the EF_through functions
### probably not needed outside of this file


def _swap_DMphase_name(DM_EF_through_function, name_var):
    """ --------------------------------------------------
   A function to rename the DMphase parameter to another name (usually DMXXphase)
        
    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        DM_EF_through_function : function
            the function of which we want to change the params
        name_var : string 
            the name of the  new name variable

    Returns
    ------
        the_new_function: function
            with name_var as a param

    
    -------------------------------------------------- """
    def wrapper(**kwargs):

        if name_var not in kwargs.keys():
            kwargs[name_var] = 0.
        new_kwargs = copy.copy(kwargs)

        new_kwargs['DMphase'] = kwargs[name_var]

        return DM_EF_through_function(**new_kwargs)

    return wrapper


def _concat_fun(outer_EF_through_fun, inner_EF_through_fun):
    """ --------------------------------------------------
    A very small function to concatenate 2 functions
    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        outer_fun: function
                x -> outer_fun(x)
        inner_fun: function 
                x -> inner_fun(x)

    Returns
        ------
        the concatenated function: function
                x -> outer_fun(inner_fun(x))



    -------------------------------------------------- """
    def new_EF_through_fun(**kwargs):

        new_kwargs_outer = copy.copy(kwargs)
        del new_kwargs_outer['entrance_EF']

        return outer_EF_through_fun(entrance_EF=inner_EF_through_fun(**kwargs),
                                    **new_kwargs_outer)

    return new_EF_through_fun


def _clean_EF_through(testbed_EF_through, known_keywords):
    """ --------------------------------------------------
    a functions to check that we do not set unknown keyword in
    the testbed EF through function. Maybe not necessary.

    AUTHOR : Johan Mazoyer

    Parameters:
    ------
         testbed_EF_through: function
         known_keywords: list of strings of known keywords

    Returns
    ------
        cleaned_testbed_EF_through: function
            a function where only known keywords are allowed
        

    -------------------------------------------------- """
    def wrapper(**kwargs):
        for passed_arg in kwargs.keys():
            if passed_arg == 'DMphase':
                raise Exception(
                    'DMphase is an ambiguous argument if you have several DMs.'
                    + ' Please use XXphase with XX = nameDM')
            if passed_arg not in known_keywords:
                raise Exception(
                    passed_arg +
                    'is not a EF_through valid argument. Valid args are ' +
                    str(known_keywords))

        return testbed_EF_through(**kwargs)

    return wrapper


def _control_testbed_with_voltages(testbed: Testbed, testbed_EF_through):
    """ --------------------------------------------------
    A function to go from a testbed_EF_through with several DMXX_phase
    parameters (one for each DM), to a testbed_EF_through with a unique
    voltage_vector parameter of size testbed.number_act (or a single float, like 0.)

    the problem with DMXX_phase parameters is that it cannot be automated since it requires
    to know the name/number of the DMs in advance.

    DMXX_phase parameters can still be used, but are overridden by voltage_vector parameter
    if present.

    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        DM_EF_through_function : function
                the function of which we want to change the params
        name_var : string 
                the name of the  new name variable

    Returns
    ------
        the_new_function: function
                with name_var as a param


    -------------------------------------------------- """
    def wrapper(**kwargs):
        if 'voltage_vector' in kwargs:
            voltage_vector = kwargs['voltage_vector']
            DM_phase = testbed.voltage_to_phases(voltage_vector)
            for i, DM_name in enumerate(testbed.name_of_DMs):
                name_phase = DM_name + "phase"
                kwargs[name_phase] = DM_phase[i]

        return testbed_EF_through(**kwargs)

    return wrapper
