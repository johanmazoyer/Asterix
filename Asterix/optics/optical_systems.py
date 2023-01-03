import errno
import os
import copy
import numpy as np
from astropy.io import fits

from Asterix import model_dir
import Asterix.optics.propagation_functions as prop
import Asterix.optics.phase_amplitude_functions as phase_ampl
from Asterix.utils import save_plane_in_fits, ft_subpixel_shift, ft_zoom_out, crop_or_pad_image


class OpticalSystem:
    """Super class OpticalSystem allows passing parameters to all subclasses.

    We can then creat blocks inside this super class. An OpticalSystem start
    and end in the pupil plane. The entrance and exit pupil plane must always
    of the same size (dim_overpad_pupil) With these conventions, they can be
    easily assemble to create complex optical systems.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self, modelconfig):
        """
        Initialize OpticalSystem objects
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)

        """

        if modelconfig["diam_pup_in_pix"] % 2 == 1:
            raise ValueError("Please set diam_pup_in_pix parameter to an even number.")

        # pupil in pixel
        self.prad = modelconfig["diam_pup_in_pix"] / 2

        # All pupils in the code must have this dimension, so that the OS systems can
        # be easily switched.
        # dim_overpad_pupil is set to an even number and the pupil is centered in
        # between 4 pixels
        self.dim_overpad_pupil = int(round(self.prad * float(modelconfig["overpadding_pupilplane_factor"])) * 2)

        # Lambda over D in pixels in the focal plane
        # at the reference wavelength
        # image size and resolution in detector
        self.dimScience = modelconfig["dimScience"]
        self.Science_sampling = modelconfig["Science_sampling"]

        # pupil in meters
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]

        self.modelconfig = copy.copy(modelconfig)

        # wavelength
        self.Delta_wav = modelconfig["Delta_wav"]
        self.nb_wav = modelconfig["nb_wav"]
        self.wavelength_0 = modelconfig["wavelength_0"]

        if self.Delta_wav != 0:
            if (self.nb_wav % 2 == 0) or self.nb_wav < 2:
                raise ValueError("please set nb_wav parameter to an odd number > 1")

            delta_wav_interval = self.Delta_wav / self.nb_wav
            self.wav_vec = self.wavelength_0 + (np.arange(self.nb_wav) - self.nb_wav // 2) * delta_wav_interval

        else:
            self.wav_vec = np.array([self.wavelength_0])
            self.nb_wav = 1

        self.string_os = '_dimPP' + str(int(self.dim_overpad_pupil)) + "_resFP" + str(round(
            self.Science_sampling, 2)) + "_dimFP" + str(int(self.dimScience))

        grey_pup_bool = modelconfig["grey_pupils"]
        if grey_pup_bool:
            self.grey_pup_bin_factor = 10  # this is hardcoded here, as it is internal cooking
        else:
            self.grey_pup_bin_factor = 1

        # we measure the AA and BB matrix and norm0 for all MFTs used to go to final focal plane
        # TODO in practice, those will be remeasured each time we initialize an OpticalSystem
        # I am not sure this is a problem because we only do it a few times (~10 in thd2)
        # Maybe there is a possibility to do it only once ?
        # we can maybe pass AA,BB and norm0 as parameter in case of very internal definition
        # when we just create a clear pupil we need for example

        self.AA_direct_final = []
        self.BB_direct_final = []
        self.norm0_direct_final = []

        for wave_i in self.wav_vec:

            lambda_ratio = wave_i / self.wavelength_0

            a, b, c = prop.mft(np.zeros((self.dim_overpad_pupil, self.dim_overpad_pupil)),
                               real_dim_input=int(2 * self.prad),
                               dim_output=self.dimScience,
                               nbres=self.dimScience / self.Science_sampling / lambda_ratio,
                               inverse=False,
                               norm='ortho',
                               returnAABB=True)

            self.AA_direct_final.append(a)
            self.BB_direct_final.append(b)
            self.norm0_direct_final.append(c)

    # We define functions that all OpticalSystem object can use.
    # These can be overwritten for a subclass if need be

    def EF_through(self, entrance_EF=1., **kwargs):
        """Propagate the electric field from entrance pupil to exit pupil.

        NEED TO BE DEFINED FOR ALL OpticalSystem subclasses

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil], or complex/float scalar (entrance_EF is constant), default is 1.
            Electric field in the pupil plane a the entrance of the system.
        dir_save_all_planes : string or None, default None
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.
        **kwargs :
            other parameters can be passed for OpticalSystem objects EF_trough functions

        Returns
        --------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane a the exit of the system
        """

        if not isinstance(entrance_EF, (float, np.ndarray)):
            print(entrance_EF)
            raise TypeError("entrance_EF should be a float of a numpy array of floats")

        exit_EF = entrance_EF
        return exit_EF

    def todetector(self,
                   entrance_EF=1.,
                   wavelength=None,
                   center_on_pixel=False,
                   in_contrast=True,
                   dir_save_all_planes=None,
                   **kwargs):
        """Propagate the electric field from entrance plane through the system
        and then to Science focal plane.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF : 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil], or complex/float scalar (entrance_EF is constant), default is 1. Electric field in the pupil plane a the entrance of the system.
        wavelength : float. Default is self.wavelength_0 the reference wavelength
            Current wavelength in m.
        in_contrast : bool, default True
            Normalize to np.sqrt(self.norm_monochrom[self.wav_vec.tolist().index(wavelength)]))
            (see self.measure_normalization)
        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
            This of course assume that no tip-tilt have been introduced
            in the entrance_EFor during self.EF_through
        dir_save_all_planes : string or None, default None
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.
        **kwargs :
            other kw parameters can be passed direclty to self.EF_through function

        Returns
        --------
        ef_focal_plane : 2D array of size [self.dimScience, self.dimScience]
            Electric field in the focal plane.
            the lambda / D is defined with the entrance pupil diameter, such as:
            self.wavelength_0 /  (2*self.prad) = self.Science_sampling pixels
        """

        if wavelength is None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        exit_EF = self.EF_through(entrance_EF=entrance_EF,
                                  wavelength=wavelength,
                                  dir_save_all_planes=dir_save_all_planes,
                                  **kwargs)

        if center_on_pixel:
            # if we need center on pixel, lets remeasure the whole mft but this is quite rare
            Psf_offset = (0.5, 0.5)
            focal_plane_EF = prop.mft(exit_EF,
                                      real_dim_input=int(self.prad * 2),
                                      dim_output=self.dimScience,
                                      nbres=self.dimScience / self.Science_sampling / lambda_ratio,
                                      X_offset_output=Psf_offset[0],
                                      Y_offset_output=Psf_offset[1],
                                      inverse=False,
                                      norm='ortho')
        else:
            # most often we center in between 4 pixels. In this case we used the AA and BB already measured
            # and only do the multiplication matrix
            focal_plane_EF = prop.mft(exit_EF,
                                      AA=self.AA_direct_final[self.wav_vec.tolist().index(wavelength)],
                                      BB=self.BB_direct_final[self.wav_vec.tolist().index(wavelength)],
                                      norm0=self.norm0_direct_final[self.wav_vec.tolist().index(wavelength)],
                                      only_mat_mult=True)

        if in_contrast:
            focal_plane_EF /= np.sqrt(self.norm_monochrom[self.wav_vec.tolist().index(wavelength)])

        if dir_save_all_planes is not None:
            who_called_me = self.__class__.__name__
            name_plane = 'EF_FP_after_' + who_called_me + '_obj' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, focal_plane_EF)

        return focal_plane_EF

    def todetector_intensity(self,
                             entrance_EF=1.,
                             wavelengths=None,
                             in_contrast=True,
                             center_on_pixel=False,
                             nb_photons=0,
                             dir_save_all_planes=None,
                             **kwargs):
        """Propagate the electric field from entrance plane through the system,
        then to Science focal plane and measure intensity.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        entrance_EF : 3D complex array of size [nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil]
                        or 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
                        or complex/float scalar (entrance_EF is constant), default is 1.
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
        dir_save_all_planes : string or None, default None
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.
        nb_photons : float, optional, default 0
            Number of photons entering the pupil. If 0, no photon noise.
        **kwargs :
            Other kw parameters can be passed direclty to self.EF_through function

        Returns
        --------
        focal_plane_intensity : 2D array of size [self.dimScience, self.dimScience]
            Intensity in the focal plane. the lambda / D is defined with
            the entrance pupil diameter, such as:
            self.wavelength_0 /  (2*self.prad) = self.Science_sampling pixels
        """

        if 'wavelength' in kwargs:
            raise ValueError(("todetector_intensity() function is polychromatic, "
                              "do not use wavelength keyword. "
                              "Use wavelengths keyword even for monochromatic intensity"))

        if wavelengths is None:
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
            raise TypeError(("entrance_EFs must be scalar (same for all WL), or a self.nb_wav scalars "
                             "or a2D array of size (self.dim_overpad_pupil, self.dim_overpad_pupil) "
                             "or a 3D array of size (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil)"))

        focal_plane_intensity = np.zeros((self.dimScience, self.dimScience))

        for i, wav in enumerate(wavelength_vec):
            focal_plane_intensity += np.abs(
                self.todetector(entrance_EF=entrance_EF[i],
                                wavelength=wav,
                                in_contrast=False,
                                center_on_pixel=center_on_pixel,
                                dir_save_all_planes=dir_save_all_planes,
                                **kwargs))**2

        if in_contrast:
            if (wavelength_vec != self.wav_vec).all():
                raise ValueError(("Careful: contrast normalization in todetector_intensity assumes "
                                  "it is done in all possible BWs (wavelengths = self.wav_vec). If "
                                  "you want a specific normalization for a subset of  wavelengths, "
                                  "use in_contrast=False and measure the PSF to normalize."))

            focal_plane_intensity /= self.norm_polychrom

        if nb_photons > 0:
            focal_plane_intensity = self.add_photon_noise(focal_plane_intensity, nb_photons, in_contrast=in_contrast)

        if dir_save_all_planes is not None:
            who_called_me = self.__class__.__name__
            name_plane = 'Int_FP_after_' + who_called_me + '_obj'
            save_plane_in_fits(dir_save_all_planes, name_plane, focal_plane_intensity)

        return focal_plane_intensity

    def add_photon_noise(self, focal_plane_intensity, nb_photons, in_contrast=True):
        """Add photon noise to an image in contrast.

        This is only applied to images for which the normalization
        factors have been measured (for wavelength in self.wave_vec). You need to have measured the normalization
        previously (running self.measure_normalization). Making it separate allow us to run the propagation only
        once in cases where we want both the image with and without photon noise.

        AUTHOR : Johan Mazoyer

        Parameters
        --------
        focal_plane_intensity : numpy array of shape (self.dimScience,self.dimScience)
            the focal plane intensity, normalized in contrast
        nb_photons : float
            Number of photons entering the pupil.
        in_contrast : bool, default True.
            If True, the data are normalized in contrast

        Returns
        --------
        focal_plane_intensity : numpy array of shape (self.dimScience,self.dimScience)
            the focal plane intensity, with photon noise, normalized in contrast
        """
        if nb_photons > 0:
            if in_contrast:
                return np.random.poisson(
                    focal_plane_intensity * self.normPupto1 * nb_photons) / (self.normPupto1 * nb_photons)
            else:
                return np.random.poisson(focal_plane_intensity * self.normPupto1_nocontrast *
                                         nb_photons) / (self.normPupto1_nocontrast * nb_photons)
        else:
            return focal_plane_intensity

    def transmission(self, noFPM=True, **kwargs):
        """Measure ratio of photons lost when crossing the system compared to a
        clear round aperture of radius self.prad.

        By default, transmission is done at the reference WL, and there is
        no reason to depend heavily on the WL.

        AUTHOR : Johan Mazoyer

        Parameters
        --------
        noFPM : bool, defaut True
            if the optical transfert function EF_through has a noFPM parameter
        **kwargs :
            other kw parameters can be passed direclty to self.EF_through function

        Returns
        --------
        transmission : float
            ratio exit flux  / clear entrance pupil flux
        """
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad,
                                                     grey_pup_bin_factor=self.grey_pup_bin_factor)

        # all parameter can be passed here, but in the case there is a coronagraph,
        # we pass noFPM = True and noentrance Field by default
        exit_EF = self.EF_through(entrance_EF=1., noFPM=noFPM, **kwargs)

        transmission = np.sum(np.abs(exit_EF)**2) / np.sum(np.abs(clear_entrance_pupil)**2)

        return transmission

    def measure_normalization(self):
        """Measure several values to normalize the data.

        Function must be used at the end of all Optical Systems initalization.

        Measure 3 values:
            - self.norm_monochrom. Array of size len(self.wav_vec)
                        the PSF per WL, use to normalize to_detector
            - self.norm_polychrom. float
                        the polychromatic PSF used to normalize to_detector_Intensity
            - self.normPupto1, which is used to measure the photon noise
                This is the factor that we use to measure photon noise.
                From an image in contrast, we now normalize by the total amount of
                energy (*self.norm_polychrom / self.sum_polychrom) and then account for the energy
                lost in the process (self.transmission()).
                Can be used as follow:
                Im_intensity_photons = Im_Intensity_contrast * self.normPupto1 * nb_photons


        AUTHOR : Johan Mazoyer
        """

        self.norm_polychrom, sum_polychrom, self.norm_monochrom, _ = self.individual_normalizations(self.wav_vec)

        self.normPupto1 = self.transmission() * self.norm_polychrom / sum_polychrom
        self.normPupto1_nocontrast = self.normPupto1 / self.norm_polychrom

    def individual_normalizations(self, wavelengths=None):
        """
        For a given wavelength list, this function is used to measure the maximum
        and total energy for each wavelength and then for the whole bandwidth.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        wavelengths : float or list of floats.
            Default is all the wl of the testbed self.wav_vec
            wavelengths in m.

        Returns
        -------
        norm_polychrom : float
            Maximum value of the PSF in polychrom light. Used to normalize to_detector_intensity().
        sum_polychrom : float
            Sum of the PSF in polychrom light. Used to normalize for photon noise.
        norm_monochrom : numpy array of the same length as wavelengths
            Norm of PSFs at each wavelength. Can be used to individually normalize
            PSFs in monochromatic light.
        sum_monochrom : numpy array of the same length as wavelengths
            Sum of PSFs at each wavelength. Not sure if useful at all but comes for free here.
        """

        if wavelengths is None:
            wavelength_vec = self.wav_vec
        elif isinstance(wavelengths, (float, int)):
            wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths

        PSF_bw = np.zeros((self.dimScience, self.dimScience))
        norm_monochrom = np.zeros((len(wavelength_vec)))
        sum_monochrom = np.zeros((len(wavelength_vec)))

        for i, wav in enumerate(wavelength_vec):
            PSF_wl = np.abs(self.todetector(wavelength=wav, noFPM=True, center_on_pixel=True, in_contrast=False))**2

            norm_monochrom[i] = np.max(PSF_wl)
            sum_monochrom[i] = np.sum(PSF_wl)
            PSF_bw += PSF_wl

        norm_polychrom = np.max(PSF_bw)
        sum_polychrom = np.sum(PSF_bw)

        return norm_polychrom, sum_polychrom, norm_monochrom, sum_monochrom

    def generate_phase_aberr(self, SIMUconfig, up_or_down='up', Model_local_dir=None):
        """Generate and save  phase aberrations.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        SIMUconfig : dict
            parameter of this simualtion (describing the phase)
        up_or_down : string, default, 'up'
            'up' or 'do', use to access the right parameters in the parameter file for
            upstream (entrance pupil) or downstream (Lyot plane) aberrations
        Model_local_dir : string or None, default None
            Directory output path for model-related files created on the file for later reuse.
            In this case the phase aberrations is saved if Model_local_dir is not None

        Returns
        --------
        return_phase : 2D array, real of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            phase aberration at the reference wavelength
        """
        if Model_local_dir is None:
            pass
        elif not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir)
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

        if not set_phase_abb:
            return 0.

        if phase_abb_filename == '':
            phase_abb_filename = up_or_down + f"phase_{int(opd_rms * 1e9):d}opdrms_lam{int(self.wavelength_0 * 1e9):d}_spd{int(phase_slope):d}_rhoc{phase_rhoc:.1f}_rad{self.prad:.1f}"

        if (not set_random_phase) and Model_local_dir is not None and os.path.isfile(
                os.path.join(Model_local_dir, phase_abb_filename + ".fits")):
            return_phase = fits.getdata(os.path.join(Model_local_dir, phase_abb_filename + ".fits"))

        else:
            phase_rms = 2 * np.pi * opd_rms / self.wavelength_0

            return_phase = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil, phase_rms, phase_rhoc,
                                                       phase_slope)
            if Model_local_dir is not None:
                fits.writeto(os.path.join(Model_local_dir, phase_abb_filename + ".fits"), return_phase, overwrite=True)
        return return_phase

    def generate_ampl_aberr(self, SIMUconfig, Model_local_dir=None):
        """Generate and save amplitude aberations.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        SIMUconfig : dict
            Parameter of this simualtion (describing the amplitude)
        Model_local_dir : string or None, default None
            Directory output path for model-related files created on the file for later reuse.

        Returns
        --------
        return_ampl : 2D array, real of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Amplitude abberation
        """
        if not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir)
            os.makedirs(Model_local_dir)

        set_amplitude_abb = SIMUconfig["set_amplitude_abb"]
        ampl_abb_filename = SIMUconfig["ampl_abb_filename"]
        set_random_ampl = SIMUconfig["set_random_ampl"]
        ampl_rms = SIMUconfig["ampl_rms"]
        ampl_rhoc = SIMUconfig["ampl_rhoc"]
        ampl_slope = SIMUconfig["ampl_slope"]

        if set_amplitude_abb:
            if not set_random_ampl:

                if ampl_abb_filename == '':
                    # in this case, the user does not want a random amplitude map but did not specified a name
                    # we will create an amplitude map (if it does not exist) for these parameters, save it as .fits
                    # and always use the same one in the future

                    ampl_abb_filename = f"ampl_{int(ampl_rms):d}percentrms_spd{int(ampl_slope):d}_rhoc{ampl_rhoc:.1f}_rad{self.prad:d}.fits"

                    if os.path.isfile(os.path.join(Model_local_dir, ampl_abb_filename)):
                        return fits.getdata(os.path.join(Model_local_dir, ampl_abb_filename))

                    else:
                        return_ampl = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil, ampl_rms / 100,
                                                                  ampl_rhoc, ampl_slope)

                        fits.writeto(os.path.join(Model_local_dir, ampl_abb_filename), return_ampl, overwrite=True)

                        return return_ampl

                else:
                    # in this case, the user wants the THD2 amplitude aberration
                    if ampl_abb_filename == 'Amplitude_THD2':
                        testbedampl = fits.getdata(os.path.join(model_dir, ampl_abb_filename + '.fits'))
                        testbedampl_header = fits.getheader(os.path.join(model_dir, ampl_abb_filename + '.fits'))
                        # in this case we know it's already well centered

                    else:
                        # in this case, the user wants his own amplitude aberrations
                        # the fits must be squared, with an even number of pixel and
                        #  have centerX, centerY and RESPUP keyword in header.

                        if not os.path.exists(ampl_abb_filename):
                            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ampl_abb_filename)

                        print(f"Opening {ampl_abb_filename} file for testbed aberrations.")
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
                    res_pup = testbedampl_header["RESPUP"]  # Pup resolution meter/pixel
                    testbedampl = crop_or_pad_image(
                        ft_zoom_out(testbedampl, res_pup / (self.diam_pup_in_m / (2 * self.prad))),
                        self.dim_overpad_pupil)

                    # Set the average to 0 inside entrancepupil
                    pup_here = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad,
                                                     grey_pup_bin_factor=self.grey_pup_bin_factor)
                    testbedampl = (testbedampl - np.mean(testbedampl[np.where(pup_here != 0)])) * pup_here
                    testbedampl = testbedampl / np.std(testbedampl[np.where(pup_here == 1.)]) * 0.1
                    return testbedampl

            else:
                ampl_abb_filename = f"ampl_{int(ampl_rms):d}percentrms_spd{int(ampl_slope):d}_rhoc{ampl_rhoc:.1f}_rad{self.prad:d}.fits"

                return_ampl = phase_ampl.random_phase_map(self.prad, self.dim_overpad_pupil, ampl_rms / 100, ampl_rhoc,
                                                          ampl_slope)

                fits.writeto(os.path.join(Model_local_dir, ampl_abb_filename), return_ampl, overwrite=True)

                return return_ampl

        else:
            return 0.

    def EF_from_phase_and_ampl(self, phase_abb=0., ampl_abb=0., wavelengths=-1.):
        """Create an electrical field from an phase and amplitude aberrations
        as follows:

        EF = (1 + ample_abb)*exp(i*phase_abb * self.wavelength_0 / wavelength)
        can be monochromatic (return 2d compex array) or polychromatic (return 3d compex array)
        if no phase nor amplitude, return 1.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            Phase aberration at reference wavelength self.wavelength_0. if 0, no phase aberration (default)
        ampl_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
             Amplitude aberration at reference wavelength self.wavelength_0. if 0, no amplitude aberration (default)
        wavelengths : float or list of floats
            Default is all the wl of the testbed self.wav_vec
            wavelengths in m.

        Returns
        --------
        EF : scalar or numpy 2D array or numpy 3d array
            Electric field in the pupil plane a the exit of the system:
                1. if no phase / amplitude
                2D array, of size phase_abb.shape if monochromatic
                or 3D array of size [self.nb_wav,phase_abb.shape] in case of polychromatic
        """

        if np.iscomplexobj(phase_abb) or np.iscomplexobj(ampl_abb):
            raise TypeError("phase_abb and ampl_abb must be real arrays or float, not complex")

        if (isinstance(phase_abb,
                       (float, int))) and (phase_abb == 0.) and (isinstance(ampl_abb,
                                                                            (float, int))) and (ampl_abb == 0.):
            return 1.

        elif isinstance(wavelengths, (float, int)):
            if wavelengths == -1:
                wavelength_vec = self.wav_vec
            else:
                wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths

        entrance_EF = []
        for wavelength in wavelength_vec:
            entrance_EF.append((1 + ampl_abb) * np.exp(1j * phase_abb * self.wavelength_0 / wavelength))
        entrance_EF = np.array(entrance_EF)

        if len(wavelength_vec) == 1:
            entrance_EF = entrance_EF[0]

        return entrance_EF
