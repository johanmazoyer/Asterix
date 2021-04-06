# pylint: disable=invalid-name

import os

import inspect

import copy
import datetime
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

        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        #pupil in pixel
        self.prad = int(modelconfig["diam_pup_in_pix"] / 2)

        # 1.25 is hard coded for now. TODO Fix that.
        # All pupils in the code must have this dimensions, so that the blocks can be easily switch
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

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],
                        can be complex.
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system.

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits
                                if save_all_planes_to_fits = True


        **kwargs: other parameters can be passed for Optical_System objects EF_trough functions

        NEED TO BE DEFINED FOR ALL Optical_System

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
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
        TODO can be made polychromatic but maybe not because people
        will do bad things with it like the intensity of summed EF :-)

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m.

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF
        or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in
                                    fits if save_all_planes_to_fits = True

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        ef_focal_plane : 2D array of size [self.dimScience, self.dimScience]
        Electric field in the focal plane.
            the lambda / D is defined such as
                self.wavelength_0 /  (2*self.exitpup_rad) = self.Science_sampling pixels

        AUTHOR : Johan Mazoyer
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
                                  inverse=False)

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
                             save_all_planes_to_fits=False,
                             dir_save_all_planes=None,
                             **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system, then
        to Science focal plane and measure intensity

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system.

        wavelengths : float or float array of wavelength in m.
                        Default is all wavelenthg in self.wav_vec

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF
        or during self.EF_through

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane
                                            in fits if save_all_planes_to_fits = True

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        focal_plane_Intensity : 2D array of size [self.dimScience, self.dimScience]
        Intensity in the focal plane. The lambda / D is defined such as
                self.wavelength_0 /  (2*self.exitpup_rad) = self.Science_sampling pixels

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if wavelengths == None:
            wavelength_vec = self.wav_vec

        elif isinstance(wavelengths, float) or isinstance(
                wavelengths, np.float):
            wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths
        #TODO check if wavelengths have good format float array and raise
        # exception if not

        focal_plane_Intensity = np.zeros((self.dimScience, self.dimScience))

        for wav in wavelength_vec:
            focal_plane_Intensity += np.abs(
                self.todetector(
                    entrance_EF=entrance_EF,
                    wavelength=wav,
                    in_contrast=False,
                    center_on_pixel=center_on_pixel,
                    save_all_planes_to_fits=save_all_planes_to_fits,
                    dir_save_all_planes=dir_save_all_planes,
                    **kwargs))**2

        if in_contrast == True:
            if (wavelength_vec != self.wav_vec).all():
                # TODO to be discussed with Raphael
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

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        float is the ratio exit flux  / clear entrance pupil flux

        AUTHOR : Johan Mazoyer

        """
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad)

        # all parameter can be passed here, but in the case there is a coronagraph,
        # we pass noFPM = True and noentrance Field by default
        exit_EF = self.EF_through(entrance_EF=1., noFPM=noFPM, **kwargs)

        throughput = np.sum(np.abs(exit_EF)) / np.sum(clear_entrance_pupil)

        return throughput

    def EF_from_phase_and_ampl(self,
                               phase_abb=0.,
                               ampl_abb=0.,
                               wavelength=None):
        """ --------------------------------------------------
        Create an electrical field from an phase and amplitude aberrations

        Parameters
        ----------
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no phase aberration (default)

        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no amplitude aberration (default)


        wavelength : float. Default is the reference self.wavelength_0
             current wavelength in m.


        Returns
        ------
        EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        """
        if np.iscomplexobj(phase_abb) or np.iscomplexobj(ampl_abb):
            raise Exception(
                "phase_abb and ampl_abb must be real arrays or float, not complex"
            )

        # if isinstance(phase_abb, (int, float, np.float)):
        #     phase_abb = np.full(
        #         (self.dim_overpad_pupil, self.dim_overpad_pupil),
        #         np.float(phase_abb))

        # if isinstance(ampl_abb, (int, float, np.float)):
        #     ampl_abb = np.full(
        #         (self.dim_overpad_pupil, self.dim_overpad_pupil),
        #         np.float(ampl_abb))

        if ((phase_abb == 0.).all()) and ((ampl_abb == 0.).all()):

            return 1.

        if wavelength is None:
            wavelength = self.wavelength_0
        lambda_ratio = wavelength / self.wavelength_0

        return (1 + ampl_abb) * np.exp(1j * phase_abb / lambda_ratio)

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
                From an image in contrast, we now normalize by the total number of photons
                  (*self.norm_polychrom / self.sum_polychrom) and then account for the photons
                lost in the process (1 / self.transmission()). Can be used as follow:
                Im_ntensity_photons = Im_Intensity_contrast * self.normPupto1 * nb_photons

        AUTHOR : Johan Mazoyer
        """

        EF_PSF_bw = np.zeros((self.dimScience, self.dimScience), dtype=complex)
        self.norm_monochrom = np.zeros((len(self.wav_vec)))
        self.sum_monochrom = np.zeros((len(self.wav_vec)))

        for i, wav in enumerate(self.wav_vec):
            EF_PSF_wl = self.todetector(wavelength=wav,
                                        noFPM=True,
                                        center_on_pixel=True,
                                        in_contrast=False)
            self.norm_monochrom[i] = np.max(np.abs(EF_PSF_wl)**2)
            EF_PSF_bw += EF_PSF_wl

        self.norm_polychrom = np.max(np.abs(EF_PSF_bw)**2)
        self.sum_polychrom = np.sum(np.abs(EF_PSF_bw)**2)

        self.normPupto1 = 1 / self.transmission(
        ) * self.norm_polychrom / self.sum_polychrom


##############################################
##############################################
### PUPIL
class pupil(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of single pupil
    pupil is a sub class of Optical_System.

    Obviously you can define your pupil
    withot that with 2d arrray multiplication (this is a fairly simple object).

    The main advantage of defining them using Optical_System is that you can
    use default Optical_System functions to obtain PSF, transmission, etc...
    and concatenate them with other elements

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, modelconfig, prad=0., PupType="", filename=""):
        """ --------------------------------------------------
        Initialize a pupil object.



        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
                        to initialize Optical_System class

        prad : int Default is the pupil prad in the parameter
            radius in pixels of the round pupil.

        PupType : string (default currently "RoundPup", CleanPlane, or RomanPupman)

        filename : string (default "")
            name and directory of the .fits file
            The pupil .fits files should be 2D and square([dim_fits,dim_fits])
            with even number of pix and centered between 4 pixels.
            if dim_fits < dim_overpad_pupil then the pupil is zero-padded
            if dim_fits > dim_overpad_pupil we raise an Exception

            This is a bit dangerous because your .fits file might must be defined
            the right way so be careful

            TODO: include here function scale_amplitude_abb, shift_phase_ramp
            TODO: include an SCC Lyot pupil function here !
            TODO: for now pupil .fits are monochromatic but the pupil propagation EF_through
            use wavelenght as a parameter


        AUTHOR : Johan Mazoyer
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
            #Rescale to the pupil size
            pup_fits = fits.getdata(
                os.path.join(model_dir,
                             "roman_pup_1002pix_center4pixels.fits"))
            pup_fits_right_size = skimage.transform.rescale(
                pup_fits,
                2 * prad / pup_fits.shape[0],
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
                "this is not a known 'PupType': 'RoundPup', 'ClearPlane', 'RomanPup'"
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
        dir_save_all_planes : default None. directory to save all plane
                                            in fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_pupil' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

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

    def random_phase_map(self, phaserms, rhoc, slope):
        """ --------------------------------------------------
        Create a random phase map, whose PSD decrease in f^(-slope)
        average is null and stadard deviation is phaserms

        Parameters
        ----------
        phaserms : float
            standard deviation of aberration
        rhoc : float
            See Borde et Traub 2006
        slope : float
            Slope of the PSD
        pupil : 2D array


        Returns
        ------
        phase : 2D array
            Static random phase map (or OPD) generated
        -------------------------------------------------- """

        # create a circular pupil of the same radius of the given pupil
        # this will be the pupil over which phase rms = phaserms
        pup = phase_ampl.roundpupil(self.prad, self.dim_overpad_pupil)

        xx, yy = np.meshgrid(
            np.arange(self.dim_overpad_pupil) - self.dim_overpad_pupil / 2,
            np.arange(self.dim_overpad_pupil) - self.dim_overpad_pupil / 2)
        rho = np.hypot(yy, xx)
        PSD0 = 1
        PSD = PSD0 / (1 + (rho / rhoc)**slope)
        sqrtPSD = np.sqrt(2 * PSD)

        randomphase = np.random.randn(
            self.dim_overpad_pupil,
            self.dim_overpad_pupil) + 1j * np.random.randn(
                self.dim_overpad_pupil, self.dim_overpad_pupil)
        phase = np.real(np.fft.ifft2(np.fft.fftshift(sqrtPSD * randomphase)))
        phase = phase - np.mean(phase[np.where(pup == 1.)])
        phase = phase / np.std(phase[np.where(pup == 1.)]) * phaserms
        return phase


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

        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        coroconfig : coronagraph parameters

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        #pupil and Lyot stop in m
        self.diam_lyot_in_m = coroconfig["diam_lyot_in_m"]

        #Lyot stop in pixel
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
                np.ceil(self.prad * self.Science_sampling * self.diam_lyot_in_m
                        / self.diam_pup_in_m * self.wavelength_0 / wav)) * 2
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

        elif self.corona_type == "classiclyot":
            self.prop_apod2lyot = 'mft-babinet'
            self.Lyot_fpm_sampling = 30  # hard coded for now, this is very internal cooking
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            self.string_os += '_' + "iwa" + str(round(self.rad_lyot_fpm, 2))
            self.FPmsk = self.ClassicalLyot()
            self.perfect_coro = False

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'mft'
            self.coro_position = coroconfig["coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.string_os += '_' + self.coro_position + "_iwa" + str(
                round(self.knife_coro_offset, 2))
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'fft'
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True

        # We need a pupil only to measure the response
        # of the coronograph to a clear pupil to remove it
        # if perfect corono. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
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

        if coroconfig["filename_instr_lyot"] == "ClearPlane" or coroconfig[
                "filename_instr_lyot"] == "RoundPup":
            self.lyot_pup = pupil(modelconfig,
                                  prad=self.prad,
                                  PupType=coroconfig["filename_instr_lyot"])
        else:
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
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

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

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in
                              fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
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
            input_wavefront_pad = proc.crop_or_pad_image(
                entrance_EF, dim_fp_fft_here)
            # Phase ramp to center focal plane between 4 pixels

            maskshifthalfpix = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, 0.5, 0.5)
            maskshifthalfpix_invert = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, -0.5, -0.5)

            #Apod plane to focal plane
            corono_focal_plane = np.fft.fft2(
                np.fft.fftshift(input_wavefront_pad * maskshifthalfpix))

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
                np.fft.ifft2(
                    corono_focal_plane * FPmsk)) * maskshifthalfpix_invert
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
                inverse=False)

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
            lyotplane_before_lyot_central_part = prop.mft(
                corono_focal_plane * (1 - FPmsk),
                self.dim_fpm,
                self.dim_overpad_pupil,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                X_offset_input=-0.5,
                Y_offset_input=-0.5,
                inverse=True)

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
                                          inverse=False)

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
                         inverse=True), self.dim_overpad_pupil)

        else:
            raise Exception(
                self.prop_apod2lyot +
                " is not a known prop_apod2lyot propagation mehtod")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_before_lyot)

        # crop to the dim_overpad_pupil expeted size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(
            lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(
            entrance_EF=lyotplane_before_lyot_crop, wavelength=wavelength)

        if (self.perfect_coro == True) & (noFPM == False):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_LS' + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      lyotplane_after_lyot)

        return lyotplane_after_lyot

    def FQPM(self):
        """ --------------------------------------------------
        Create a Four Quadrant Phase Mask coronagraph


        Returns
        ------
        FQPM : list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            FQPM mask

        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
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

            phase = np.zeros((dim_fp, dim_fp))
            fqpm_thick_cut = proc.crop_or_pad_image(fqpm_thick, dim_fp)
            phase[np.where(fqpm_thick_cut != 0)] = (np.pi + self.err_fqpm)
            if self.achrom_fqpm == False:
                phase = phase * self.wavelength_0 / wav

            fqpm.append(np.exp(1j * phase))

        return fqpm

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dimScience,dimScience)

        Returns
        ------
        shift(Knife) :list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            Knife edge coronagraph mask. TODO A CHECKER YA PTET UN SOUCIS DE SHIFT

        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
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
        ld_p_0 = self.Science_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

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

        knife_allwl = list()
        for i in range(len(self.wav_vec)):
            knife_allwl.append(Knife)

        return knife_allwl

    def ClassicalLyot(self):
        """ --------------------------------------------------
        Create a classical Lyot coronagraph of radius rad_LyotFP 0

        rad_LyotFP : int, radius of the Lyot focal plane

        Returns
        ------
         classical Lyot : 2D array

        AUTHOR : Johan Mazoyer
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


##############################################
##############################################
### Deformable mirror
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
                 Model_local_dir=''):
        """ --------------------------------------------------
        Initialize a deformable mirror object
        TODO handle misregistration that is currently not working

        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        DMconfig : DM parameters
        Name_DM : the name of the DM, which allows to find it in the parameter file
        we measure and save the pushact functions

        Model_local_dir: directory to save things you can measure yourself
                    and can save to save time

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.exitpup_rad = self.prad

        self.Name_DM = Name_DM
        self.z_position = DMconfig[self.Name_DM + "_z_position"]
        self.active = DMconfig[self.Name_DM + "_active"]

        MinimumSurfaceRatioInThePupil = DMconfig[
            "MinimumSurfaceRatioInThePupil"]


        # For intialization, we assume no misregistration, we introduce it after
        # estimation and correction matrices are created.
        self.misregistration = False

        # first thing we do is to open filename_grid_actu to check the number of
        # actuator of this DM
        self.number_act = fits.getdata(
            model_dir +
            DMconfig[self.Name_DM + "_filename_grid_actu"]).shape[1]
        self.string_os += '_' + self.Name_DM + "_Nact" + str(
            int(self.number_act)) + "_z" + str(int(self.z_position * 100))

        if self.active == False:
            print(self.Name_DM + ' is not activated')
            return

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

        # create, save or load the DM_pushact and DM_pushact_inpup functions
        # from the influence function

        # DM_pushact is always in the DM plane
        self.DM_pushact = self.creatingpushact(DMconfig,
                                               Model_local_dir=Model_local_dir)

        if self.z_position != 0:
            # DM_pushact_inpup is always in the pupil plane
            self.DM_pushact_inpup = self.creatingpushact_inpup(
                Model_local_dir=Model_local_dir)
        else:
            # if the DM plane IS the pupil plane
            self.DM_pushact_inpup = self.DM_pushact
            # This is a duplicate of the same file but coherent and
            # allows you to easily concatenate wherever are the DMs
            # this is not taking memory this is just 2 names for the
            # same object in memory:
            # print(id(self.DM_pushact_inpup))
            # print(id(self.DM_pushact))

        # create or load 'which actuators are in pupil'
        self.WhichInPupil = self.creatingWhichinPupil(
            MinimumSurfaceRatioInThePupil,
            Model_local_dir=Model_local_dir)

        self.misregistration = DMconfig[self.Name_DM + "_misregistration"]
        # now if we relaunch self.DM_pushact, it will be different due to misregistration

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

        Parameters
        ----------
        entrance_EF:    2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system.

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m.

        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.
                    Can also be a float scalar in which case DM_phase is constant
                    default is 0.

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in
                                    fits if save_all_planes_to_fits = True

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
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

        lambda_ratio = wavelength / self.wavelength_0

        if self.z_position == 0:
            EF_after_DM = entrance_EF * np.exp(1j * DMphase / lambda_ratio)

        else:
            EF_after_DM = self.prop_pup_to_DM_and_back(
                entrance_EF,
                DMphase / lambda_ratio,
                wavelength,
                save_all_planes_to_fits=save_all_planes_to_fits,
                dir_save_all_planes=dir_save_all_planes)

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_after_' + self.Name_DM + '_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      entrance_EF)

        return EF_after_DM

    def creatingpushact(self, DMconfig, Model_local_dir=''):
        """ --------------------------------------------------
        OPD map induced in the DM plane for each actuator

        Parameters
        ----------
        DMconfig : structure with all information on DMs
        Model_local_dir: directory to save things you can measure yourself
                    and can save to save time

        Error on the model of the DM

        This function can to cleaned (names and comment).

        Returns
        ------
        pushact :
        -------------------------------------------------- """

        Name_pushact_fits = "PushAct" + self.string_os

        if (self.misregistration is False) and (
                os.path.exists(Model_local_dir + Name_pushact_fits + '.fits')):
            return fits.getdata(
                os.path.join(Model_local_dir, Name_pushact_fits + '.fits'))

        diam_pup_in_pix = 2 * self.prad
        diam_pup_in_m = self.diam_pup_in_m
        dim_array = self.dim_overpad_pupil

        pitchDM = DMconfig[self.Name_DM + "_pitch"]
        filename_ActuN = DMconfig[self.Name_DM + "_filename_ActuN"]
        filename_grid_actu = DMconfig[self.Name_DM + "_filename_grid_actu"]
        filename_actu_infl_fct = DMconfig[self.Name_DM +
                                          "_filename_actu_infl_fct"]
        ActuN = DMconfig[self.Name_DM + "_ActuN"]
        y_ActuN = DMconfig[self.Name_DM + "_y_ActuN"]
        x_ActuN = DMconfig[self.Name_DM + "_x_ActuN"]
        xy_ActuN = [x_ActuN, y_ActuN]

        if self.misregistration == True:
            xerror = DMconfig[self.Name_DM + "_xerror"]
            yerror = DMconfig[self.Name_DM + "_yerror"]
            angerror = DMconfig[self.Name_DM + "_angerror"]
            gausserror = DMconfig[self.Name_DM + "_gausserror"]
        else:
            xerror = 0.
            yerror = 0.
            angerror = 0.
            gausserror = 0.

        #Measured positions for each actuator in pixel
        measured_grid = fits.getdata(model_dir + filename_grid_actu)
        #Ratio: pupil radius in the measured position over
        # pupil radius in the numerical simulation
        sampling_simu_over_measured = diam_pup_in_pix / 2 / fits.getheader(
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
        simu_grid = proc.actuator_position(measured_grid, xy_ActuN, ActuN,
                                           sampling_simu_over_measured)
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
                                          reshape=False)[0:dim_array,
                                                         0:dim_array]
            else:
                # Add an error on the sizes of the influence functions
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp,
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

        if self.misregistration == False and (
                not os.path.exists(Model_local_dir + Name_pushact_fits +
                                   '.fits')):
            fits.writeto(Model_local_dir + Name_pushact_fits + '.fits',
                         pushact)

        return pushact

    def creatingpushact_inpup(self, Model_local_dir=''):
        """ --------------------------------------------------
        OPD map induced by out-of-pupil DM in the pupil plane for each actuator
        ## Create the influence functions of an out-of-pupil DM
        ## in the pupil plane

        Parameters
        ----------
        Model_local_dir: directory to save things you can measure yourself
                    and can save to save time
        Returns
        ------
        pushact_inpup : Map of the complex phase induced in pupil plane
        -------------------------------------------------- """

        Name_pushact_inpup_fits = "PushActInPup" + self.string_os

        if (os.path.exists(Model_local_dir + Name_pushact_inpup_fits +
                           '_RE.fits')
            ) and (os.path.exists(Model_local_dir + Name_pushact_inpup_fits +
                                  '_IM.fits')):
            DM_pushact_inpup_real = fits.getdata(
                os.path.join(Model_local_dir,
                             Name_pushact_inpup_fits + '_RE.fits'))
            DM_pushact_inpup_imag = fits.getdata(
                os.path.join(Model_local_dir,
                             Name_pushact_inpup_fits + '_IM.fits'))

            return DM_pushact_inpup_real + 1j * DM_pushact_inpup_imag

        # TODO do we really need a pupil here ?!? seems to me it would be more general
        # with all the actuators ? It seems to me that it reduce the generality of
        # this function which is: what is the influence of the influce of DM1 in the
        # pupil plane ? WE also need to find a way to measure which in pup for DM1
        # because the way it is done currently, all actuators are selected !
        # I think we really need a pupil, so we need the real pupil not a generic
        # round pupil
        EF_inDMplane, _ = prop.prop_fresnel(self.clearpup.pup,
                                            self.wavelength_0, self.z_position,
                                            self.diam_pup_in_m / 2, self.prad)
        pushact_inpup = np.zeros(
            (self.number_act, self.dim_overpad_pupil, self.dim_overpad_pupil),
            dtype=complex)

        for i in np.arange(self.number_act):
            EF_back_in_pup_plane, _ = prop.prop_fresnel(
                EF_inDMplane * self.DM_pushact[i], self.wavelength_0,
                -self.z_position, self.diam_pup_in_m / 2, self.prad)
            pushact_inpup[i] = EF_back_in_pup_plane

            if not ((os.path.exists(Model_local_dir + Name_pushact_inpup_fits +
                                    '_RE.fits')) and
                    (os.path.exists(Model_local_dir + Name_pushact_inpup_fits +
                                    '_IM.fits'))):
                fits.writeto(Model_local_dir + Name_pushact_inpup_fits +
                             '_RE.fits',
                             np.real(pushact_inpup),
                             overwrite=True)
                fits.writeto(Model_local_dir + Name_pushact_inpup_fits +
                             '_IM.fits',
                             np.imag(pushact_inpup),
                             overwrite=True)

        return pushact_inpup

    def creatingWhichinPupil(self,
                             cutinpupil,
                             Model_local_dir=''):
        """ --------------------------------------------------
        Create a vector with the index of all the actuators located in the entrance pupil

        Parameters:
        ----------
        cutinpupil: float, minimum surface of an actuator inside the pupil to be taken into account
                    (between 0 and 1, ratio of an actuator perfectly centered in the entrance pupil)
        Model_local_dir: directory to save things you can measure yourself
                        and can save to save time
        Return:
        ------
        WhichInPupil: 1D array, index of all the actuators located inside the pupil
        -------------------------------------------------- """

        Name_WhichInPup_fits = "ActinPup" + self.string_os

        if os.path.exists(Model_local_dir + Name_WhichInPup_fits + '.fits'):
            return useful.check_and_load_fits(Model_local_dir,
                                              Name_WhichInPup_fits)

        WhichInPupil = []
        # TODO I don't think this is working for DM1 ! All actuators are selected
        # because for DM1 wecheck how much actu = fresenl(pushactu*fresnel(Pup)),
        # has energy inside Pup. but it has almost no impact out of pup.
        # so pup*actu/actu always ~1
        #  We can do it in pupil plane (we check energy
        # of fresenl(pushactu) inside Pup or in DM plane (we check energy of
        # pushactu inside of fresnel(Pup).
        for i in np.arange(self.DM_pushact_inpup.shape[0]):
            actu = self.DM_pushact_inpup[i]
            # cut = cutinpupil * np.sum(np.abs(actu))
            if np.sum(np.abs(actu * self.clearpup.pup)) / np.sum(
                    np.abs(actu)) > cutinpupil:
                WhichInPupil.append(i)

        WhichInPupil = np.array(WhichInPupil)

        if not os.path.exists(Model_local_dir + Name_WhichInPup_fits +
                                   '.fits'):
            fits.writeto(Model_local_dir + Name_WhichInPup_fits + '.fits',
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
        Propagate the field towards an out-of-pupil plane (DM1 plane),
        add the DM1 phase, and propagate to the next pupil plane (DM3 plane)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
                    Wavefront in the pupil plane

        phase_DM1 : 2D array
                    Phase introduced by DM1

        wavelength : float
                    wavelength in m

        save_all_planes_to_fits: Bool, default False.
                if True, save all planes to fits for debugging purposes to dir_save_all_planes
                This can generate a lot of fits especially if in a loop so the code force you
                to define a repository.
        dir_save_all_planes : default None. directory to save all plane in fits
                                            if save_all_planes_to_fits = True

        Returns
        ------
        UDM3 : 2D array (complex)
                Wavefront in the pupil plane after DM1
                (corresponds to DM3 plane on THD2 bench)

        AUTHOR : Raphaël Galicher

        REVISION HISTORY :
        Revision 1.1  2021-02-10 Raphaël Galicher
            Initial revision
        Revision 2.0 2021-02-28 Johan Mazoyer
            Make it more general for all DMs, put in the struc
        -------------------------------------------------- """

        # Propagation in DM plane out of pupil
        EF_inDMplane, _ = prop.prop_fresnel(entrance_EF, wavelength,
                                            self.z_position,
                                            self.diam_pup_in_m / 2., self.prad)

        # Add DM phase
        if save_all_planes_to_fits == True:
            name_plane = 'EF_before_DM_in_' + self.Name_DM + 'plane_wl{}'.format(
                int(wavelength * 1e9))
            useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                      EF_inDMplane)

        EF_inDMplane_after_DM = EF_inDMplane * np.exp(1j * phase_DM)

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

    def voltage_to_phase(self, actu_vect, wavelength=None):
        """ --------------------------------------------------
        Generate the phase applied on one DM for a give vector of actuator amplitude

        Parameters:
        ----------
        actu_vect : 1D array
                    values of the amplitudes for each actuator

        Return:
        ------
            2D array
            phase map in the same unit as actu_vect times DM_pushact)
        -------------------------------------------------- """

        if wavelength is None:
            wavelength = self.wavelength_0

        surface_reshaped = np.dot(
            actu_vect,
            self.DM_pushact.reshape(
                self.DM_pushact.shape[0],
                self.DM_pushact.shape[1] * self.DM_pushact.shape[2]))

        phase_on_DM = surface_reshaped.reshape(
            self.DM_pushact.shape[1],
            self.DM_pushact.shape[2]) * 2 * np.pi * 1e-9 / wavelength

        return phase_on_DM


##############################################
##############################################
### Testbed


class Testbed(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a testbed.
    This is a particular subclass of Optical System, because we do not know what is inside
    It can only be initialized by giving a list of Optical Systems and it will create a
    "testbed" which contains all the Optical System in a


    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, list_os, list_os_names):
        """ --------------------------------------------------
        This function allow you to concatenates Optical_System obsjects to create a testbed:
        parameter:
            list_os:        a list of optical systems. all the systems must have been defined with
                                the same modelconfig. or it will send an error.
                            The list order is form the first optics system to the last in the
                            path of the light (so usually from entrance pupil to Lyot pupil)

            list_os_names: a list of string of the same size as list_os to define
                            the names of the optical systems. Then can then be accessed
                            inside the Testbed object by os_#i = Testbed.list_os_names[i]

        Returns
            ------
            testbed : an optical system which is the concatenation of all the optical systems

        AUTHOR : Johan Mazoyer
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
        self.WhichInPupil = []

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

                # elsetestbed.DM3.number_act + testbed.DM1.number_act
                self.number_DMs += 1
                self.WhichInPupil = np.concatenate(
                    (self.WhichInPupil,
                     self.number_act + list_os[num_optical_sys].WhichInPupil))
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
        # we remove doubloons
        # known_keywords = list(set(known_keywords))
        known_keywords = list(dict.fromkeys(known_keywords))

        # we add no
        # We remove arguments we know are wrong
        if 'DMphase' in known_keywords:
            known_keywords.remove('DMphase')
        known_keywords.remove('kwargs')

        self.EF_through = _clean_EF_through(self.EF_through, known_keywords)

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()


##############################################
##############################################
### internal functions to properly concatenate the EF_through functions
### probably not needed outside of this file


def _swap_DMphase_name(DM_EF_through_function, name_var):
    """ --------------------------------------------------
   A function to rename the DM phase

    parameter:
        DM_EF_through_function : the function of which we want to change the params
        name_var : string the name of the  new name variable

    Returns
        ------
        the_new_function: with name_var as a param


    AUTHOR : Johan Mazoyer
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

    parameter: 2 functions
         outer_fun: x -> outer_fun(x)
         inner_fun: x -> inner_fun(x)

    Returns
        ------
        the concatenated function:
        concat_fun: x -> outer_fun(inner_fun(x))


    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def new_EF_through_fun(**kwargs):

        new_kwargs_outer = copy.copy(kwargs)
        del new_kwargs_outer['entrance_EF']

        return outer_EF_through_fun(entrance_EF=inner_EF_through_fun(**kwargs),
                                    **new_kwargs_outer)

    return new_EF_through_fun


def _clean_EF_through(testbed_EF_through, known_keywords):
    """ --------------------------------------------------
    a functions to finally check that we do not set unknown keyword in
    the testbed EF through function

    parameter: 2 functions
         testbed_EF_through function
         inner_fun: x -> inner_fun(x)

    Returns
        ------
        the concatenated function:
        concat_fun: x -> outer_fun(inner_fun(x))


    AUTHOR : Johan Mazoyer
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
