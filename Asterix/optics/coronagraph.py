import os
import numpy as np

from Asterix.utils import save_plane_in_fits, crop_or_pad_image
import Asterix.optics.optical_systems as optsy
import Asterix.optics.pupil as pupil
import Asterix.optics.propagation_functions as prop
import Asterix.optics.phase_amplitude_functions as phase_ampl


class Coronagraph(optsy.OpticalSystem):
    """Initialize and describe the behavior of a coronagraph system (from apod
    plane to the Lyot plane).

    AUTHOR : Johan Mazoyer
    """

    def __init__(self, modelconfig, coroconfig, Model_local_dir=None):
        """Initialize a coronagraph object.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
            general configuration parameters (sizes and dimensions)
        coroconfig : dict
            coronagraph parameters
        Model_local_dir : string or None, default None
            Directory output path for model-related files created on the file for later reuse.
        """

        # Initialize the OpticalSystem class and inherit properties
        super().__init__(modelconfig)

        if (Model_local_dir is not None) and not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir)
            os.makedirs(Model_local_dir)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        self.apod_pup = pupil.Pupil(modelconfig,
                                    prad=self.prad,
                                    PupType=coroconfig["filename_instr_apod"],
                                    angle_rotation=coroconfig['apod_pup_rotation'],
                                    Model_local_dir=Model_local_dir)

        self.string_os += '_Apod' + self.apod_pup.string_os

        # Define coronagraph focal plane mask type
        self.corona_type = coroconfig["corona_type"].lower()
        self.string_os += '_' + self.corona_type

        # define the types of propagation first for each coronagraph type
        # this is also used to hard code some of the internal propagation of the
        # coronagraph like the size of the focal plane
        if self.corona_type in ("classiclyot", "hlc"):
            self.prop_apod2lyot = 'mft-babinet'

            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]

            # we oversample the center in babinet's mode because we can
            # hard coded for now, this is very internal cooking
            self.Lyot_fpm_sampling = 20.  # self.Science_sampling
            rad_LyotFP_pix = self.rad_lyot_fpm * self.Lyot_fpm_sampling
            self.dim_fpm = 2 * int(2.2 * rad_LyotFP_pix / 2)

        elif self.corona_type in ("vortex", "wrapped_vortex"):
            self.prop_apod2lyot = 'regional-sampling'
            if self.prad < 100:
                raise ValueError(f"In regional-sampling mode, 'diam_pup_in_pix' must be > 200 to be most accurate")

            # with 'diam_pup_in_pix' = 256 these parameter give ~3 10-9 for the wrapped vortex
            # it is possible to do better but would require higher pupil and fpm size
            # Ideally try to find the 2 best 'nbres' so that we can get better than 10-10
            self.dim_fpm = 256
            self.nbrs_res_list = [4, 32]

            # Following lines ~7 10-10 for 'diam_pup_in_pix' = 512. This is very slow though
            # self.dim_fpm = 512
            # self.nbrs_res_list = [6, 42]

        elif self.corona_type in ("fqpm", "knife"):
            self.prop_apod2lyot = 'mft'

        else:
            raise ValueError(f"The requested coronagraph mode '{self.corona_type}' does not exists.")

        # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
        self.corono_fpm_sampling = self.Science_sampling
        self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=int)
        for i, wav in enumerate(self.wav_vec):
            self.dim_fp_fft[i] = int(np.ceil(self.prad * self.corono_fpm_sampling * self.wavelength_0 / wav)) * 2
            # we take the ceil to be sure that we measure at least the good resolution
            # We do not need to be exact, the mft in science_focal_plane will be

        self.achrom_phase_coro = coroconfig["achrom_phase_coro"]

        if self.corona_type == "fqpm":
            self.err_fqpm = coroconfig["err_fqpm"]
            self.FPmsk = self.FQPM()
            if self.achrom_phase_coro:
                self.string_os += '_' + "achrom"
            self.perfect_coro = True

        elif self.corona_type in ("classiclyot", "hlc"):

            self.string_os += '_' + "iwa" + str(round(self.rad_lyot_fpm, 2))
            self.perfect_coro = False
            if self.corona_type == "classiclyot":
                self.FPmsk = self.ClassicalLyot()
            else:
                self.transmission_fpm = coroconfig["transmission_fpm"]
                self.phase_fpm = coroconfig["phase_fpm"]
                self.string_os += '_' + f"trans{self.transmission_fpm:.1e}_pha{round(self.phase_fpm, 2)}"
                self.FPmsk = self.HLC()
                if self.achrom_phase_coro:
                    self.string_os += '_' + "achrom"

        elif self.corona_type == "knife":
            self.coro_position = coroconfig["knife_coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.string_os += '_' + self.coro_position + "_iwa" + str(round(self.knife_coro_offset, 2))
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            vortex_charge = coroconfig["vortex_charge"]
            self.string_os += '_charge' + str(int(vortex_charge))
            self.FPmsk = self.Vortex(vortex_charge=vortex_charge)
            self.perfect_coro = False
            if self.achrom_phase_coro:
                self.string_os += '_' + "achrom"

        elif self.corona_type == "wrapped_vortex":
            self.string_os += '2020'
            self.FPmsk = self.WrappedVortex()
            self.perfect_coro = False
            if self.achrom_phase_coro:
                self.string_os += '_' + "achrom"

        self.lyot_pup = pupil.Pupil(modelconfig,
                                    prad=self.prad * coroconfig["diam_lyot_in_m"] / self.diam_pup_in_m,
                                    PupType=coroconfig["filename_instr_lyot"],
                                    angle_rotation=coroconfig['lyot_pup_rotation'],
                                    Model_local_dir=Model_local_dir)

        self.string_os += '_LS' + self.lyot_pup.string_os

        if self.prop_apod2lyot in ['mft', 'mft-babinet']:

            # we measure the AA and BB matrix and norm0 for all MFTs used in coronagraphy
            if self.prop_apod2lyot == 'mft':
                dim_science_here = self.dimScience
                fpm_sampling_here = self.Science_sampling
            if self.prop_apod2lyot == 'mft-babinet':
                dim_science_here = self.dim_fpm
                fpm_sampling_here = self.Lyot_fpm_sampling

            self.AA_direct = []
            self.BB_direct = []
            self.norm0_direct = []

            self.AA_inverse = []
            self.BB_inverse = []
            self.norm0_inverse = []

            for i, wave_i in enumerate(self.wav_vec):

                lambda_ratio = wave_i / self.wavelength_0

                if self.prop_apod2lyot == 'mft':
                    # in practice in MFT mode, the final MFT is identical to the
                    # one in the corono so we save a bit of time / memory here
                    self.AA_direct.append(self.AA_direct_final[i])
                    self.BB_direct.append(self.BB_direct_final[i])
                    self.norm0_direct.append(self.norm0_direct_final[i])

                else:
                    a, b, c = prop.mft(np.zeros((self.dim_overpad_pupil, self.dim_overpad_pupil)),
                                       real_dim_input=int(2 * self.prad),
                                       dim_output=dim_science_here,
                                       nbres=dim_science_here / fpm_sampling_here / lambda_ratio,
                                       inverse=False,
                                       norm='ortho',
                                       returnAABB=True)
                    self.AA_direct.append(a)
                    self.BB_direct.append(b)
                    self.norm0_direct.append(c)

                a, b, c = prop.mft(np.zeros((dim_science_here, dim_science_here)),
                                   real_dim_input=dim_science_here,
                                   dim_output=int(2 * self.prad),
                                   nbres=dim_science_here / fpm_sampling_here / lambda_ratio,
                                   inverse=True,
                                   norm='ortho',
                                   returnAABB=True)

                self.AA_inverse.append(a)
                self.BB_inverse.append(b)
                self.norm0_inverse.append(c)

        if "bool_overwrite_perfect_coro" in coroconfig:
            if coroconfig["bool_overwrite_perfect_coro"]:
                self.perfect_coro = True
            else:
                self.perfect_coro = False

        if self.perfect_coro:

            if coroconfig["filename_instr_apod"] == "Clear":
                # We need a round pupil only to measure the response
                # of the coronagraph to a round pupil to remove it
                # THIS IS NOT THE ENTRANCE PUPIL,
                # this is a round pupil of the same size
                pup_for_perfect_coro = pupil.Pupil(modelconfig, prad=self.prad)

                # do a propagation once with self.perfect_Lyot_pupil = 0 to
                # measure the Lyot pupil that will be removed after
                self.perfect_Lyot_pupil = [0] * self.nb_wav
                for i, wave_here in enumerate(self.wav_vec):
                    self.perfect_Lyot_pupil[i] = self.EF_through(
                        entrance_EF=pup_for_perfect_coro.EF_through(wavelength=wave_here), wavelength=wave_here)
            else:
                # In this case we have a coronagraph entrance pupil.
                # Do a propagation once with self.perfect_Lyot_pupil = 0 to
                # measure the Lyot pupil that will be removed after.
                self.perfect_Lyot_pupil = [0] * self.nb_wav
                for i, wave_here in enumerate(self.wav_vec):
                    self.perfect_Lyot_pupil[i] = self.EF_through(wavelength=wave_here)

        # Initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   noFPM=False,
                   EF_aberrations_introduced_in_LS=1.,
                   dir_save_all_planes=None,
                   **kwargs):
        """Propagate the electric field from the apodizer plane before the
        apodizer pupil to the Lyot plane after the Lyot pupil.

        AUTHOR : Johan Mazoyer

        03/22 : Correction in the babinet propagation

        Parameters
        ----------
        entrance_EF : 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]; or float
            Can also be a float scalar in which case entrance_EF is constant; default=1.
            Electric field in the pupil plane at the entrance of the system.
        wavelength : float
            Current wavelength in m.
            Default is self.wavelength_0, the reference wavelength.
        noFPM : bool
            If True, remove the FPM if one want to measure an un-obstructed PSF; default False.
        EF_aberrations_introduced_in_LS : 2D complex array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electrical field created by the downstream aberrations introduced directly in the Lyot Stop.
            Can also be a float scalar in which case entrance_EF is constant; default=1.
        dir_save_all_planes : string or None, default None
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.

        Returns
        --------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
            Electric field in the pupil plane at the exit of the system
        """

        # call the OpticalSystem super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0

        if dir_save_all_planes is not None:
            name_plane = 'EF_PP_before_apod' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, entrance_EF)

        if noFPM:
            FPmsk = 1.
        else:
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        input_wavefront_after_apod = self.apod_pup.EF_through(entrance_EF=entrance_EF, wavelength=wavelength)

        if dir_save_all_planes is not None:
            name_plane = 'apod' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, self.apod_pup.pup)

            name_plane = 'EF_PP_after_apod' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, input_wavefront_after_apod)

        # we take the convention that for all propagation methods, the PSF must be
        # "in between 4 pixels" in the focal plane.

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(wavelength)]
            input_wavefront_after_apod_pad = crop_or_pad_image(input_wavefront_after_apod, dim_fp_fft_here)

            corono_focal_plane = prop.fft_choosecenter(input_wavefront_after_apod_pad,
                                                       inverse=False,
                                                       center_pos='bb',
                                                       norm='ortho')

            if dir_save_all_planes is not None:
                name_plane = 'EF_FP_before_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, np.fft.fftshift(corono_focal_plane))

                name_plane = 'PSF EF_FP_before_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, np.fft.fftshift(np.abs(corono_focal_plane)**2))
                if not noFPM:
                    name_plane = 'FPM' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, np.fft.fftshift(corono_focal_plane * FPmsk))

            # Focal plane to Lyot plane
            lyotplane_before_lyot = prop.fft_choosecenter(corono_focal_plane * FPmsk,
                                                          inverse=True,
                                                          center_pos='bb',
                                                          norm='ortho')

        elif self.prop_apod2lyot == "mft-babinet":
            # Apod plane to focal plane
            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          AA=self.AA_direct[self.wav_vec.tolist().index(wavelength)],
                                          BB=self.BB_direct[self.wav_vec.tolist().index(wavelength)],
                                          norm0=self.norm0_direct[self.wav_vec.tolist().index(wavelength)],
                                          only_mat_mult=True)

            if dir_save_all_planes is not None:
                name_plane = 'EF_FP_before_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane)
                if not noFPM:
                    name_plane = 'FPM' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * FPmsk)

                name_plane = 'EF_FP_after_1minusFPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * (1 - FPmsk))

            # Focal plane to Lyot plane
            # Babinet's trick:
            lyotplane_before_lyot_central_part = crop_or_pad_image(
                prop.mft(corono_focal_plane * (1 - FPmsk),
                         AA=self.AA_inverse[self.wav_vec.tolist().index(wavelength)],
                         BB=self.BB_inverse[self.wav_vec.tolist().index(wavelength)],
                         norm0=self.norm0_inverse[self.wav_vec.tolist().index(wavelength)],
                         only_mat_mult=True), self.dim_overpad_pupil)

            lyotplane_before_lyot = input_wavefront_after_apod - lyotplane_before_lyot_central_part

        elif self.prop_apod2lyot == "mft":
            # Apod plane to focal plane
            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          AA=self.AA_direct[self.wav_vec.tolist().index(wavelength)],
                                          BB=self.BB_direct[self.wav_vec.tolist().index(wavelength)],
                                          norm0=self.norm0_direct[self.wav_vec.tolist().index(wavelength)],
                                          only_mat_mult=True)

            if dir_save_all_planes is not None:
                name_plane = 'EF_FP_before_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane)
                if not noFPM:
                    name_plane = 'FPM' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + f'_wl{int(wavelength * 1e9)}'
                save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = crop_or_pad_image(
                prop.mft(corono_focal_plane * FPmsk,
                         AA=self.AA_inverse[self.wav_vec.tolist().index(wavelength)],
                         BB=self.BB_inverse[self.wav_vec.tolist().index(wavelength)],
                         norm0=self.norm0_inverse[self.wav_vec.tolist().index(wavelength)],
                         only_mat_mult=True), self.dim_overpad_pupil)

        elif self.prop_apod2lyot == "regional-sampling":
            # Apod plane to Lyot plane
            if noFPM:
                FPmsk = np.ones((self.dim_fpm, self.dim_fpm))

            lyotplane_before_lyot = prop_fpm_regional_sampling(input_wavefront_after_apod,
                                                               FPmsk,
                                                               self.nbrs_res_list,
                                                               real_dim_input=int(2 * self.prad),
                                                               dir_save_all_planes=dir_save_all_planes,
                                                               shift=(0, 0))

        else:
            raise ValueError(f"{self.prop_apod2lyot} is not a known `prop_apod2lyot` propagation method")

        if dir_save_all_planes is not None:
            name_plane = 'EF_PP_before_LS' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, lyotplane_before_lyot)

        # we add the downstream aberrations if we need them
        lyotplane_before_lyot *= EF_aberrations_introduced_in_LS

        # crop to the dim_overpad_pupil expected size
        # TODO I do not like this line. A random crop or pad is the best way to mask an error in coding
        # if we made an error in dimenensions. We should make sure that our arrays are of the right dimensions at all times
        lyotplane_before_lyot_crop = crop_or_pad_image(lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(entrance_EF=lyotplane_before_lyot_crop, wavelength=wavelength)

        if self.perfect_coro & (not noFPM):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil[self.wav_vec.tolist().index(
                wavelength)]

        if dir_save_all_planes is not None:
            name_plane = 'LS' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, self.lyot_pup.pup)

            name_plane = 'EF_PP_after_LS' + f'_wl{int(wavelength * 1e9)}'
            save_plane_in_fits(dir_save_all_planes, name_plane, lyotplane_after_lyot)

        return lyotplane_after_lyot

    def FQPM(self):
        """Create a Four Quadrant Phase Mask coronagraph.

        AUTHOR : Axel Potier
        Modified by Johan Mazoyer

        Returns
        --------
        fqpm : list of len(self.wav_vec) 2D arrays
            Complex transmission of the FQPM mask at all wavelengths.
        """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dimScience

        phase_fqpm = fqpm_mask(maxdimension_array_fpm)

        fqpm = []
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dimScience

            phase4q = np.zeros((dim_fp, dim_fp))
            fqpm_thick_cut = crop_or_pad_image(phase_fqpm, dim_fp)
            phase4q[np.where(fqpm_thick_cut != 0)] = np.pi + self.err_fqpm

            if self.achrom_phase_coro:
                # If we want to do an achromatic fqpm, we do not include a variation
                # of the phase with the wl.
                fqpm.append(np.exp(1j * phase4q))
            else:
                # In the general case, we use the EF_from_phase_and_ampl which handle the phase chromaticity.
                fqpm.append(self.EF_from_phase_and_ampl(phase_abb=phase4q, wavelengths=wav))

        return fqpm

    def Vortex(self, vortex_charge=2):
        """Create a vortex coronagraph with charge 'vortex_charge'.

        AUTHOR : Johan Mazoyer

        Parameters
        --------
        vortex_charge : int, default 2
            Charge of the vortex. Usually a positive, even number (2, 4, 6). Default is charge 2.

        Returns
        --------
        vortex : list of 2D numpy array
            The complex FP mask at all wavelengths.
        """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dim_fpm

        xx, yy = np.meshgrid(
            np.arange(maxdimension_array_fpm) - maxdimension_array_fpm / 2 + 1 / 2,
            np.arange(maxdimension_array_fpm) - maxdimension_array_fpm / 2 + 1 / 2)

        phase_vortex = vortex_charge * np.angle(xx + 1j * yy)

        vortex = []
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dim_fpm

            phasevortex_cut = crop_or_pad_image(phase_vortex, dim_fp)  # *phase_ampl.roundpupil(dim_fp, dim_fp/2)

            if self.achrom_phase_coro:
                # If we want to do an achromatic vortex, we do not include a variation
                # of the phase with the wl.
                vortex.append(np.exp(1j * phasevortex_cut))
            else:
                # In the general case, we use the EF_from_phase_and_ampl which handle the phase chromaticity.
                vortex.append(self.EF_from_phase_and_ampl(phase_abb=phasevortex_cut, wavelengths=wav))

        return vortex

    def WrappedVortex(self, offset=0, cen_shift=(0, 0), inclination_x=0, inclination_y=0):
        """Create a wrapped vortex coronagraph.

        Parameters
        ----------
        offset : float
            General offset to the whole ramp; default 0.
        cen_shift : tuple of floats
            x- and y-shift of the center of the mask with respect to the center of the array in pixels, which is
            between pixels as long as 'dim' is even; default (0,0).
        inclination_x : float, default 0
            Inclination of the phase mask around the x-axis in degrees.
        inclination_y : float, default 0
            Inclination of the phase mask around the y-axis in degrees.

        Returns
        --------
        wrapped_vortex : list of 2D numpy array
            The complex FP masks at all wavelengths.
        """
        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dim_fpm

        # TODO: The below should not be hard-coded, but ok until we actually want to be able to use different values.
        thval = np.array([0, 3, 4, 5, 8]) * np.pi / 8
        phval = np.array([3, 0, 1, 2, 1]) * np.pi
        jump = np.array([2, 2, 2, 2]) * np.pi
        _, phase_wrapped_vortex = create_wrapped_vortex_mask(dim=maxdimension_array_fpm,
                                                             thval=thval,
                                                             phval=phval,
                                                             jump=jump,
                                                             offset=offset,
                                                             cen_shift=cen_shift,
                                                             inclination_x=inclination_x,
                                                             inclination_y=inclination_y)

        # Match the orientation of the phase mask to THD2
        phase_wrapped_vortex = np.rot90(phase_wrapped_vortex)

        wrapped_vortex = list()
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dim_fpm

            phasevortex_cut = crop_or_pad_image(phase_wrapped_vortex, dim_fp)  # *phase_ampl.roundpupil(dim_fp, dim_fp/2)

            if self.achrom_phase_coro:
                # If we want to do an achromatic vortex, we do not include a variation
                # of the phase with the wl.
                wrapped_vortex.append(np.exp(1j * phasevortex_cut))
            else:
                # In the general case, we use the EF_from_phase_and_ampl which handle the phase chromaticity.
                wrapped_vortex.append(self.EF_from_phase_and_ampl(phase_abb=phasevortex_cut, wavelengths=wav))

        return wrapped_vortex

    def KnifeEdgeCoro(self):
        """Create a Knife edge coronagraph of size (dimScience,dimScience).

        AUTHOR : Axel Potier
        Modified by Johan Mazoyer

        Returns
        --------
        knife_allwl : list of len(self.wav_vec) 2D arrays
            Complex transmission of the knife-edge coronagraph mask at all wavelengths.
        """
        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
            if len(self.wav_vec) > 1:
                raise NotImplementedError("knife currently not coded in polychromatic fft")
        else:
            maxdimension_array_fpm = self.dimScience

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength
        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm), np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(xx > (maxdimension_array_fpm / 2 + self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "left":
            Knife[np.where(xx < (maxdimension_array_fpm / 2 - self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(yy > (maxdimension_array_fpm / 2 + self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "top":
            Knife[np.where(yy < (maxdimension_array_fpm / 2 - self.knife_coro_offset * self.Science_sampling))] = 1

        knife_allwl = []
        for i in range(len(self.wav_vec)):
            knife_allwl.append(Knife)

        return knife_allwl

    def ClassicalLyot(self):
        """Create a classical Lyot coronagraph of radius rad_LyotFP.

        AUTHOR : Johan Mazoyer

        Returns
        --------
        ClassicalLyotFPM_allwl : list of 2D numpy arrays
            The FP masks at all wavelengths.
        """

        rad_LyotFP_pix = self.rad_lyot_fpm * self.Lyot_fpm_sampling
        ClassicalLyotFPM = 1. - phase_ampl.roundpupil(self.dim_fpm, rad_LyotFP_pix, grey_pup_bin_factor=1)

        ClassicalLyotFPM_allwl = []
        for wav in self.wav_vec:
            ClassicalLyotFPM_allwl.append(ClassicalLyotFPM)

        return ClassicalLyotFPM_allwl

    def HLC(self):
        """Create an HLC of radius rad_LyotFP.

        AUTHOR : Johan Mazoyer

        Returns
        --------
        hlc_all_wl : list of 2D numpy array
            The FP masks at all wavelengths.
        """

        # we create a Classical Lyot Focal plane
        ClassicalLyotFP = self.ClassicalLyot()[0]
        whClassicalLyotstop = np.where(ClassicalLyotFP == 0.)

        # we define phase and amplitude for the HLC at the reference WL
        phase_hlc = np.zeros(ClassicalLyotFP.shape)
        phase_hlc[whClassicalLyotstop] = self.phase_fpm
        # transmission_fpm is defined in intensity and EF_from_phase_and_ampl takes amplitude
        ampl_hlc = np.zeros(ClassicalLyotFP.shape)
        ampl_hlc[whClassicalLyotstop] = np.sqrt(self.transmission_fpm) - 1

        hlc_all_wl = []
        for wav in self.wav_vec:

            if self.achrom_phase_coro:
                # If we want to do an achromatic hlc, we do not include a variation
                # of the phase with the wl.
                hlc_all_wl.append((1 + ampl_hlc) * np.exp(1j * phase_hlc))
            else:
                # In the general case, we use the EF_from_phase_and_ampl which handle the phase chromaticity.
                hlc_all_wl.append(self.EF_from_phase_and_ampl(ampl_abb=ampl_hlc, phase_abb=phase_hlc, wavelengths=wav))

        return hlc_all_wl


def fqpm_mask(dim):
    """Create a FQPM phase mask.

    AUTHOR: Axel Potier

    Parameters
    ----------
    dim : int
        Number of pixels for the resulting phase mask.

    Returns
    -------
    fqpm_thick : array
        Array holding the phase mask, in radians.
    """
    xx, yy = np.meshgrid(np.arange(dim) - dim / 2, np.arange(dim) - dim / 2)

    fqpm_thick_vert = np.zeros((dim, dim))
    fqpm_thick_vert[np.where(xx < 0)] = 1
    fqpm_thick_hor = np.zeros((dim, dim))
    fqpm_thick_hor[np.where(yy >= 0)] = 1
    fqpm_thick = fqpm_thick_vert - fqpm_thick_hor
    fqpm_thick[np.where(fqpm_thick != 0)] = np.pi

    return fqpm_thick


def create_wrapped_vortex_mask(dim,
                               thval,
                               phval,
                               jump,
                               return_1d=False,
                               piperiodic=True,
                               offset=0,
                               cen_shift=(0, 0),
                               inclination_x=0,
                               inclination_y=0):
    """Create a wrapped vortex phase mask.

    Analytical calculation of this phase mask coronagraph see [Galicher2020]_.

    AUTHOR: Raphaël Galicher (in IDL)
            ILa (to Python)

    .. [Galicher2020] Galicher et al. 2020, "A family of phase masks for broadband coronagraphy example of the wrapped
            vortex phase mask theory and laboratory demonstration "

    Parameters
    ----------
    dim : int
       Number of pixels for the resulting phase mask, needs to be even to keep center between pixels.
    thval : array
        Angle values at location of phase jumps. First is 0, last is pi, completing a half of a unit circle.
    phval : array
        Phase values corresponding to the angle values thval just after the phase jump happened.
    jump : array
        Phase jump between phase segments on the unit circle. Has one less element than thval and phval.
    return_1d : bool
        If True, return a 1D phase profile, otherwise a 2D phase mask; default False.
    piperiodic: bool
        If True, assume max(thval) is pi and return a pi-periodic mask. Needs to be True to get full 2D phase mask.
    offset : float
        General offset to the whole ramp; default 0.
    cen_shift : tuple of floats
        x- and y-shift of the center of the mask in pixels with respect to the center of the array, which is between
        pixels as long as 'dim' is even; default (0,0).
    inclination_x : float, default 0
        Inclination of the phase mask around the x-axis in degrees.
    inclination_y : float, default 0
        Inclination of the phase mask around the y-axis in degrees.

    Returns
    -------
    angles : array
        Array holding angle values in radians. 1D array or 2D array depending on return_1d.
    phase_mask : array
        Array holding the phase mask, in radians. 1D array or 2D array depending on return_1d.

    Examples
    --------
    ### Galicher et al 2020 mask (orientation for THD2 on sep-2022)
    # Input parameters
    thval = np.array([0, 3, 4, 5, 8]) * np.pi / 8
    phval = np.array([3, 0, 1, 2, 1]) * np.pi
    jump = np.array([2, 2, 2, 2]) * np.pi

    # Create an dplot focal plane mask in 1D
    angles_1d, phase_1d = create_wrapped_vortex_mask(dim=128, thval=thval, phval=phval, jump=jump, return_1d=True)
    plt.figure(figsize=(16,8))
    plt.plot(angles_1d, phase_1d)
    plt.xlabel("Angle (rad)")
    plt.ylabel("Phase (rad)")
    plt.show()

    # Create and plot focal plane mask in 2D
    angles_2d, phase_2d = create_wrapped_vortex_mask(dim=128, thval=thval, phval=phval, jump=jump, return_1d=False)
    plt.figure(figsize=(7,7))
    plt.imshow(phase_2d, cmap="Reds", origin="lower")
    plt.colorbar(label="Phase (rad)")
    plt.show()
    """
    if phval.shape != thval.shape:
        raise ValueError("The arrays 'phval' and 'thval' need to have the same shape.")

    if return_1d:
        # Create a continuous 1D phase ramp from 0 to pi, including an offset.
        theta = (np.arange(dim) / (dim - 1) * (np.max(thval) - np.min(thval)) + np.min(thval) + offset) % np.pi
    else:
        # Define the 2D theta array
        ty = (np.arange(dim) - dim / 2 - cen_shift[0] + 0.5)
        tx = (np.arange(dim) - dim / 2 - cen_shift[1] + 0.5)

        if np.abs(inclination_x > 90) and np.abs(inclination_y > 90):
            raise ValueError("Inclination angles should be between -90 and 90 degrees.")
        if inclination_x != 0:
            tx /= np.cos(np.deg2rad(inclination_x))
        if inclination_y != 0:
            ty /= np.cos(np.deg2rad(inclination_y))

        xx, yy = np.meshgrid(ty, tx)
        theta = (-(np.arctan2(yy, xx) - np.pi) + offset) % (2 * np.pi)

    # Create empty phase mask.
    phase = np.zeros_like(theta)

    # Find the angles between thval[k] and thval[k+1].
    for k in range(thval.shape[0] - 1):
        section = np.where((theta >= thval[k]) & (theta <= thval[k + 1]))

        # If such angles exist then:
        if section[0].shape[0] != 0:

            # 1st step (k=0): Create phase mask section going from phval[k] to phval[k+1].
            if k == 0:
                phase[section] = phval[k] + (theta[section] - thval[k]) / (thval[k + 1] - thval[k]) * (phval[k + 1] -
                                                                                                       phval[k])
            # All other steps, do the same thing but add the phase shift jump[k-1] first.
            else:
                phase[section] = phval[k] + jump[k - 1] + (theta[section] - thval[k]) / (thval[k + 1] - thval[k]) * (
                    phval[k + 1] - phval[k] - jump[k - 1])

    if return_1d:
        # Define the angle in radians.
        theta = np.arange(dim) / (dim - 1) * (np.max(thval) - np.min(thval)) + np.min(thval)

        # If piperiodic is True, then assume max(thval) is pi and make phase pi-periodic.
        if piperiodic:
            angles = np.concatenate((theta, theta + np.pi))
            phase_mask = np.concatenate((phase, phase))
        else:
            angles = theta
            phase_mask = phase
    else:
        # If piperiodic is True, then assume max(thval) is pi and make phase pi-periodic.
        if piperiodic:
            # Find the angles between thval(k)+pi and thval(k+1)+pi.
            for k in range(thval.shape[0] - 1):
                section = np.where((theta >= (thval[k] + np.pi)) & (theta <= (thval[k + 1] + np.pi)))

                # If such elements exist then:
                if section[0].shape[0] != 0:

                    # 1st step [k=1]: Create phase mask section going from phval[k] to phval[k+1].
                    if k == 0:
                        phase[section] = phval[k] + (theta[section] - thval[k] -
                                                     np.pi) / (thval[k + 1] - thval[k]) * (phval[k + 1] - phval[k])
                    # All other steps, do the same thing but add the phase shift jump[k-1] first.
                    else:
                        phase[section] = phval[k] + jump[k - 1] + (theta[section] - thval[k] - np.pi) / (
                            thval[k + 1] - thval[k]) * (phval[k + 1] - phval[k] - jump[k - 1])

        angles = theta
        phase_mask = phase

    return angles, phase_mask


def prop_fpm_regional_sampling(pup,
                               fpm,
                               nbres,
                               real_dim_input=None,
                               shift=(0, 0),
                               filter_order=15,
                               alpha=1.5,
                               dir_save_all_planes=None):
    """
    Calculate the coronagraphic electric field in the Lyot plane by using varying sampling in different parts of the FPM.

    Starting from the electric field in an input pupil plane, propagate through a phase-mask coronagraph with
    different sampling values in different rings around the FPM center. Each propagation goes from the pupil to the FPM
    and then to the Lyot plane (without Lyot stop applied), using MFTs, while a (combination of) Butterworth filters
    marks annular regions around the FPM center that are considered in the propagation in each step. The electric field
    contribution from each step is then summed up to yield the total electric field in the Lyot plane (before the LS).

    AUTHOR: R. Galicher (in IDL)
            ILa (to Python)
            J. Mazoyer : small modifs to adapt to Asterix Dec 22

    Parameters
    ----------
    pup : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]
        Input mage array containing the wavefront at the entrance pupil of the optical system.
    fpm : 2D array
        Complex electric field in the focal plane of the focal-plane mask.
    nbres : list
        List of the number of resolution elements across the total image plane for all propagation layers.
        As a general rule, it is probably safest to choose these numbers such that there is not sampling
        shift right in the middle of the DH. This would ensure that the frequencies inside the DH are all
        calculated with the same resolution in the EFC matrix. So if we correct the DH between two radii
        IWA and OWA (in lambda/D), nbres should not have any elements between 2*IWA and 2*OWA.
    real_dim_input : int or None, default None
        Diameter of the support in pup (can differ from pup.shape). If None, real_dim_input = pup.shape
        If the pupil fills its array out until the edges, then real_dim_input = pup.shape; if the pupil
        is padded, then real_dim_input is the size of the pupil in pixels.
    shift : tuple, default (0, 0)
        Shift of FPM with respect to optical axis in units of lambda/D.
    filter_order : int, default 15
        Order of the Butterworth filter.
    alpha : float, default 1.5
        Scale factor for the filter size. The larger this number, the smaller the filter size with respect to the
        input array.
    dir_save_all_planes : string or None, default None
        If not None, absolute directory to save all planes in fits for debugging purposes.
        This can generate a lot of fits especially if in a loop, use with caution.


    Returns
    -------
    EF_before_LS : 2D array (complex)
        E-field before the Lyot stop.
    """

    if not isinstance(nbres, list):
        raise TypeError(f"'nbres' parameter needs to be of type list. Currently type = {type(nbres)}")

    nbres = np.array(nbres)

    dim_overpad_pupil = pup.shape[0]
    dim_fpm = fpm.shape[0]

    if real_dim_input is None:
        real_dim_input = dim_overpad_pupil
    samplings = dim_fpm / nbres

    if not np.all(np.diff(nbres) >= 0):
        raise ValueError(f"'nbres' parameter needs to be sorted from the highest to lowest number of elements."
                         f"Currently 'nbres' = {nbres}.")

    if np.min(samplings) < 2:
        raise ValueError(f"The outer sampling in prop_fpm_regional_sampling is hardcoded to 2. We need the samplings"
                         f"defined by the 'nbres' parameter (real_dim_input/nbres) to be always >= 2. Currently, with"
                         f"real_dim_input = {real_dim_input}, samplings are {samplings}.")

    if np.min(samplings) != 2:
        # If the smaller sampling defined by parameter 'nbres' is not 2, we append it to the list. This is a
        # way to force the last sampling to be harcoded at 2.
        nbres = np.append(nbres, dim_fpm / 2)
        samplings = np.append(samplings, 2)

    if max(shift) >= nbres[0] / 2:
        raise ValueError(f"shift {shift} is larger than the minimum number of elements of resolution {nbres[0]}: the "
                         f"tip/tilt is out of the array! Increase min(nbres) or decrease shift.")

    ef_pp_before_ls_tot = np.zeros((real_dim_input, real_dim_input), dtype='complex128')
    const_but = phase_ampl.butterworth_circle(dim_fpm, dim_fpm / alpha, filter_order, xshift=-0.5, yshift=-0.5)

    # From inner to outer part of FPM
    for k in range(nbres.shape[0]):

        if k == 0:
            # Innermost part of the focal plane
            but_here = np.copy(const_but)
        elif k < nbres.shape[0] - 1:
            # Butterworth filter in each layer
            sizebut_here = dim_fpm / alpha * nbres[k - 1] / nbres[k]
            but_here = (1 - phase_ampl.butterworth_circle(dim_fpm, sizebut_here, filter_order, xshift=-0.5,
                                                          yshift=-0.5)) * const_but
        else:
            # Outer part of the FPM
            sizebut_here = dim_fpm / alpha * nbres[-2] / nbres[-1]
            but_here = 1 - phase_ampl.butterworth_circle(dim_fpm, sizebut_here, filter_order, xshift=-0.5, yshift=-0.5)

        ef_fp_before_fpm = prop.mft(
            pup,
            real_dim_input=real_dim_input,
            dim_output=dim_fpm,
            nbres=nbres[k],
            norm='ortho',
            X_offset_output=shift[0] * samplings[k],
            Y_offset_output=shift[1] * samplings[k],
        )
        ef_pp_before_ls_reg = prop.mft(ef_fp_before_fpm * fpm * but_here,
                                       real_dim_input=dim_fpm,
                                       dim_output=real_dim_input,
                                       nbres=nbres[k],
                                       X_offset_input=shift[0] * samplings[k],
                                       Y_offset_input=shift[1] * samplings[k],
                                       norm='ortho',
                                       inverse=True)

        if dir_save_all_planes is not None:
            name_plane = f'FPbeforeFPM_nbr{int(nbres[k])}_sampling{int(samplings[k])}'
            save_plane_in_fits(dir_save_all_planes, name_plane, ef_fp_before_fpm)
            name_plane = f'FPafterButandFPM_nbr{int(nbres[k])}_sampling{int(samplings[k])}'
            save_plane_in_fits(dir_save_all_planes, name_plane, ef_fp_before_fpm * fpm * but_here)
            name_plane = f'PPbeforeLyot_nbr{int(nbres[k])}_sampling{int(samplings[k])}'
            save_plane_in_fits(dir_save_all_planes, name_plane, ef_pp_before_ls_reg)

        # Sum up E-field contributions before the LS
        ef_pp_before_ls_tot += ef_pp_before_ls_reg

    return crop_or_pad_image(ef_pp_before_ls_tot, dim_overpad_pupil)
