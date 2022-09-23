# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace
import os
import copy
import numpy as np
from astropy.io import fits

from Asterix import model_dir
import Asterix.optics.propagation_functions as prop
import Asterix.optics.phase_amplitude_functions as phase_ampl
from Asterix.utils import save_plane_in_fits, ft_subpixel_shift, ft_zoom_out, crop_or_pad_image


class OpticalSystem:
    """ --------------------------------------------------
    Super class OpticalSystem allows passing parameters to all subclasses.
    We can then creat blocks inside this super class. An OpticalSystem start and
    end in the pupil plane.
    The entrance and exit pupil plane must always of the same size (dim_overpad_pupil)
    With these conventions, they can be easily assemble to create complex optical systems.


    AUTHOR : Johan Mazoyer

        -------------------------------------------------- """

    def __init__(self, modelconfig):
        """ --------------------------------------------------
        Initialize OpticalSystem objects
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)

        -------------------------------------------------- """

        #pupil in pixel
        self.prad = modelconfig["diam_pup_in_pix"] / 2

        # All pupils in the code must have this dimension, so that the OS systems can
        # be easily switched.
        # dim_overpad_pupil is set to an even numer and the pupil is centered in
        # between 4 pixels
        self.dim_overpad_pupil = int(
            round(self.prad * float(modelconfig["overpadding_pupilplane_factor"])) * 2)

        #Lambda over D in pixels in the focal plane
        # at the reference wavelength
        #image size and resolution in detector
        self.dimScience = modelconfig["dimScience"]
        self.Science_sampling = modelconfig["Science_sampling"]

        #pupil in meters
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]

        self.modelconfig = copy.copy(modelconfig)

        # wavelength
        self.Delta_wav = modelconfig["Delta_wav"]
        self.nb_wav = modelconfig["nb_wav"]
        self.wavelength_0 = modelconfig["wavelength_0"]

        if self.Delta_wav != 0:
            if (self.nb_wav % 2 == 0) or self.nb_wav < 2:
                raise Exception("please set nb_wav parameter to an odd number > 1")

            self.wav_vec = np.linspace(self.wavelength_0 - self.Delta_wav / 2,
                                       self.wavelength_0 + self.Delta_wav / 2,
                                       num=self.nb_wav,
                                       endpoint=True)
        else:
            self.wav_vec = np.array([self.wavelength_0])
            self.nb_wav = 1

        self.string_os = '_dimPP' + str(int(self.dim_overpad_pupil)) + '_wl' + str(
            int(self.wavelength_0 * 1e9)) + "_resFP" + str(round(self.Science_sampling, 2)) + "_dimFP" + str(
                int(self.dimScience))

    #We define functions that all OpticalSystem object can use.
    # These can be overwritten for a subclass if need be

    def EF_through(self, entrance_EF=1., **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance pupil to exit pupil

        NEED TO BE DEFINED FOR ALL OpticalSystem subclasses

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
            other parameters can be passed for OpticalSystem objects EF_trough functions

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system


        -------------------------------------------------- """

        if isinstance(entrance_EF, (float, np.float, np.ndarray)) == False:
            print(entrance_EF)
            raise Exception("entrance_EF should be a float of a numpy array of floats")

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
                the lambda / D is defined with the entrance pupil diameter, such as:
                self.wavelength_0 /  (2*self.prad) = self.Science_sampling pixels


        -------------------------------------------------- """
        if center_on_pixel == True:
            Psf_offset = (0.5, 0.5)
        else:
            Psf_offset = (0, 0)

        if wavelength is None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        exit_EF = self.EF_through(entrance_EF=entrance_EF,
                                  wavelength=wavelength,
                                  save_all_planes_to_fits=save_all_planes_to_fits,
                                  dir_save_all_planes=dir_save_all_planes,
                                  **kwargs)

        focal_plane_EF = prop.mft(exit_EF,
                                  int(self.prad * 2),
                                  self.dimScience,
                                  self.dimScience / self.Science_sampling * lambda_ratio,
                                  X_offset_output=Psf_offset[0],
                                  Y_offset_output=Psf_offset[1],
                                  inverse=False,
                                  norm='ortho')

        if in_contrast == True:
            focal_plane_EF /= np.sqrt(self.norm_monochrom[self.wav_vec.tolist().index(wavelength)])

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'EF_FP_after_' + who_called_me + '_obj' + '_wl{}'.format(int(wavelength * 1e9))
            save_plane_in_fits(dir_save_all_planes, name_plane, focal_plane_EF)

        return focal_plane_EF

    def todetector_intensity(self,
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
                    Intensity in the focal plane. the lambda / D is defined with 
                    the entrance pupil diameter, such as:
                    self.wavelength_0 /  (2*self.prad) = self.Science_sampling pixels


        -------------------------------------------------- """

        if 'wavelength' in kwargs:
            raise Exception("""todetector_intensity() function is polychromatic, 
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
        elif entrance_EF.shape == (self.dim_overpad_pupil, self.dim_overpad_pupil):
            entrance_EF = np.repeat(entrance_EF[np.newaxis, ...], self.nb_wav, axis=0)
        elif entrance_EF.shape == (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil):
            pass
        else:
            raise Exception(""""entrance_EFs must be scalar (same for all WL), or a self.nb_wav scalars or a
                        2D array of size (self.dim_overpad_pupil, self.dim_overpad_pupil) or a 3D array of size
                        (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil)""")

        focal_plane_Intensity = np.zeros((self.dimScience, self.dimScience))

        for i, wav in enumerate(wavelength_vec):
            focal_plane_Intensity += np.abs(
                self.todetector(entrance_EF=entrance_EF[i],
                                wavelength=wav,
                                in_contrast=False,
                                center_on_pixel=center_on_pixel,
                                save_all_planes_to_fits=save_all_planes_to_fits,
                                dir_save_all_planes=dir_save_all_planes,
                                **kwargs))**2

        if in_contrast == True:
            if (wavelength_vec != self.wav_vec).all():
                raise Exception("""Careful: contrast normalization in todetector_intensity assumes
                     it is done in all possible BWs (wavelengths = self.wav_vec). If self.nb_wav > 1
                     and you want only one BW with the good contrast normalization, use
                     np.abs(to_detector(wavelength = wavelength))**2... If you want a specific
                     normalization for a subset of  wavelengths, use in_contrast=False and
                     measure the PSF to normalize.
                """)
            
            focal_plane_Intensity /= self.norm_polychrom

        if photon_noise == True:
            focal_plane_Intensity = np.random.poisson(
                focal_plane_Intensity * self.normPupto1 * nb_photons) / (self.normPupto1 * nb_photons)

        if save_all_planes_to_fits == True:
            who_called_me = self.__class__.__name__
            name_plane = 'Int_FP_after_' + who_called_me + '_obj'
            save_plane_in_fits(dir_save_all_planes, name_plane, focal_plane_Intensity)

        return focal_plane_Intensity

    def transmission(self, noFPM=True, **kwargs):
        """
        measure ratio of photons lost when
        crossing the system compared to a clear round aperture of radius self.prad

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
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil, self.prad)

        # all parameter can be passed here, but in the case there is a coronagraph,
        # we pass noFPM = True and noentrance Field by default
        exit_EF = self.EF_through(entrance_EF=1., noFPM=noFPM, **kwargs)

        throughput = np.sum(np.abs(exit_EF)**2) / np.sum(np.abs(clear_entrance_pupil)**2)

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
                self.todetector(wavelength=wav, noFPM=True, center_on_pixel=True, in_contrast=False))**2

            self.norm_monochrom[i] = np.max(PSF_wl)
            PSF_bw += PSF_wl

        self.norm_polychrom = np.max(PSF_bw)
        self.sum_polychrom = np.sum(PSF_bw)

        self.normPupto1 = self.transmission() * self.norm_polychrom / self.sum_polychrom

    def generate_phase_aberr(self, SIMUconfig, up_or_down='up', Model_local_dir=None):
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
                    In this case the phase aberrations is saved if Model_local_dir is not None


        Returns
        ------
        return_phase : 2D array, real of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                phase abberation at the reference wavelength


        -------------------------------------------------- """
        if Model_local_dir is None:
            pass
        elif not os.path.exists(Model_local_dir):
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

        if set_phase_abb is False:
            return 0.

        if phase_abb_filename == '':
            phase_abb_filename = up_or_down + "phase_{:d}opdrms_lam{:d}_spd{:d}_rhoc{:.1f}_rad{:.1f}".format(
                int(opd_rms * 1e9), int(self.wavelength_0 * 1e9), int(phase_slope), phase_rhoc, self.prad)

        if set_random_phase is False and Model_local_dir is not None and os.path.isfile(Model_local_dir +
                                                                                        phase_abb_filename +
                                                                                        ".fits") == True:
            return_phase = fits.getdata(Model_local_dir + phase_abb_filename + ".fits")

        else:
            phase_rms = 2 * np.pi * opd_rms / self.wavelength_0

            return_phase = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil, phase_rms,
                                                       phase_rhoc, phase_slope)
            if Model_local_dir is not None:
                fits.writeto(Model_local_dir + phase_abb_filename + ".fits", return_phase, overwrite=True)
        return return_phase

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
            if set_random_ampl is False:

                if ampl_abb_filename == '':
                    # in this case, the user does not want a random amplitude map but did not specified a name
                    # we will create an amplitude map (if it does not exist) for these parameters, save it as .fits
                    # and always use the same one in the future

                    ampl_abb_filename = "ampl_{:d}percentrms_spd{:d}_rhoc{:.1f}_rad{:d}.fits".format(
                        int(ampl_rms), int(ampl_slope), ampl_rhoc, self.prad)

                    if os.path.isfile(Model_local_dir + ampl_abb_filename) == True:
                        return fits.getdata(Model_local_dir + ampl_abb_filename)

                    else:
                        return_ampl = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil,
                                                                  ampl_rms / 100, ampl_rhoc, ampl_slope)

                        fits.writeto(Model_local_dir + ampl_abb_filename, return_ampl, overwrite=True)

                        return return_ampl

                else:
                    # in this case, the user wants the THD2 amplitude aberration
                    if ampl_abb_filename == 'Amplitude_THD2':
                        testbedampl = fits.getdata(model_dir + ampl_abb_filename + '.fits')
                        testbedampl_header = fits.getheader(model_dir + ampl_abb_filename + '.fits')
                        # in this case we know it's already well centered

                    else:
                        # in this case, the user wants his own amplitude aberrations
                        # the fits must be squared, with an even number of pixel and
                        #  have centerX, centerY and RESPUP keyword in header.

                        if not os.path.exists(ampl_abb_filename):
                            # check existence
                            print("Specified amplitude file {0} does not exist.".format(ampl_abb_filename))
                            print("")
                            print("")
                            raise

                        print("Opening {0} file for testbed aberrations.".format(ampl_abb_filename))
                        print("This file should have centerX, centerY and RESPUP keyword in header")

                        testbedampl = fits.getdata(ampl_abb_filename)
                        testbedampl_header = fits.getheader(ampl_abb_filename)
                        centerX = testbedampl_header["CENTERX"]
                        centerY = testbedampl_header["CENTERX"]
                        size_ampl = testbedampl.shape[0]

                        # recenter
                        if centerX != size_ampl // 2 - 1 / 2 or centerY != size_ampl // 2 - 1 / 2:
                            testbedampl = ft_subpixel_shift(testbedampl,
                                                                 xshift=size_ampl // 2 - 1 / 2 - centerX,
                                                                 yshift=size_ampl // 2 - 1 / 2 - centerY)

                    # reshape at the good size
                    # TODO we may have to check the centering is ok
                    res_pup = testbedampl_header["RESPUP"]  #Pup resolution meter/pixel
                    testbedampl = crop_or_pad_image(
                        ft_zoom_out(testbedampl, res_pup / (self.diam_pup_in_m / (2 * self.prad))),
                        self.dim_overpad_pupil)

                    #Set the average to 0 inside entrancepupil
                    pup_here = phase_ampl.roundpupil(self.dim_overpad_pupil, self.prad)
                    testbedampl = (testbedampl - np.mean(testbedampl[np.where(pup_here != 0)])) * pup_here
                    testbedampl = testbedampl / np.std(testbedampl[np.where(pup_here == 1.)]) * 0.1
                    return testbedampl

            else:
                ampl_abb_filename = "ampl_{:d}percentrms_spd{:d}_rhoc{:.1f}_rad{:d}.fits".format(
                    int(ampl_rms), int(ampl_slope), ampl_rhoc, self.prad)

                return_ampl = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil, ampl_rms / 100,
                                                          ampl_rhoc, ampl_slope)

                fits.writeto(Model_local_dir + ampl_abb_filename, return_ampl, overwrite=True)

                return return_ampl

        else:
            return 0.

    def EF_from_phase_and_ampl(self, phase_abb=0., ampl_abb=0., wavelengths=-1.):
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
            raise Exception("phase_abb and ampl_abb must be real arrays or float, not complex")

        if (isinstance(phase_abb, (float, int))) and (phase_abb == 0.) and (isinstance(
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
            entrance_EF.append((1 + ampl_abb) * np.exp(1j * phase_abb * self.wavelength_0 / wavelength))
        entrance_EF = np.array(entrance_EF)

        if len(wavelength_vec) == 1:
            entrance_EF = entrance_EF[0]

        return entrance_EF