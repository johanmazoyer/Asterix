# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import numpy as np
from astropy.io import fits

from Asterix.optical_systems import OpticalSystem, model_dir, Pupil
import Asterix.processing_functions as proc
import Asterix.propagation_functions as prop
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.save_and_read as saveread


class Coronagraph(OpticalSystem):
    """ --------------------------------------------------
    initialize and describe the behavior of a coronagraph system (from apod plane to the Lyot plane)
    coronagraph is a sub class of OpticalSystem.

    AUTHOR : Johan Mazoyer

    -------------------------------------------------- """

    def __init__(self, modelconfig, coroconfig, Model_local_dir=None):
        """ --------------------------------------------------
        Initialize a coronograph object

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        modelconfig : dict
                general configuration parameters (sizes and dimensions)
        coroconfig : : dict
                coronagraph parameters

        Model_local_dir: string, default None
                    directory to save things you can measure yourself
                    and can save to save times      
        

        -------------------------------------------------- """

        # Initialize the OpticalSystem class and inherit properties
        super().__init__(modelconfig)

        if (Model_local_dir is not None) and not os.path.exists(Model_local_dir):
            print("Creating directory " + Model_local_dir + " ...")
            os.makedirs(Model_local_dir)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        self.apod_pup = Pupil(modelconfig,
                              prad=self.prad,
                              PupType=coroconfig["filename_instr_apod"],
                              angle_rotation=coroconfig['apod_pup_rotation'],
                              Model_local_dir=Model_local_dir)

        self.string_os += '_Apod' + self.apod_pup.string_os

        #coronagraph focal plane mask type
        self.corona_type = coroconfig["corona_type"].lower()

        self.string_os += '_' + self.corona_type

        # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
        self.corono_fpm_sampling = self.Science_sampling
        self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
        for i, wav in enumerate(self.wav_vec):
            self.dim_fp_fft[i] = int(np.ceil(
                self.prad * self.corono_fpm_sampling * self.wavelength_0 / wav)) * 2
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

        elif self.corona_type in ("classiclyot", "hlc"):
            self.prop_apod2lyot = 'mft-babinet'
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            # we oversample the center in babinet's mode because we can
            # hard coded for now, this is very internal cooking

            self.Lyot_fpm_sampling = 20.  #self.Science_sampling
            rad_LyotFP_pix = self.rad_lyot_fpm * self.Lyot_fpm_sampling
            self.dim_fpm = 2 * int(2.2 * rad_LyotFP_pix / 2)

            self.string_os += '_' + "iwa" + str(round(self.rad_lyot_fpm, 2))
            self.perfect_coro = False
            if self.corona_type == "classiclyot":
                self.FPmsk = self.ClassicalLyot()
            else:
                self.transmission_fpm = coroconfig["transmission_fpm"]
                self.phase_fpm = coroconfig["phase_fpm"]
                self.string_os += '_' + "trans{:.1e}".format(self.transmission_fpm) + "_pha{0}".format(
                    round(self.phase_fpm, 2))
                self.FPmsk = self.HLC()

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'mft'
            self.coro_position = coroconfig["knife_coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.string_os += '_' + self.coro_position + "_iwa" + str(round(self.knife_coro_offset, 2))
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'mft'
            vortex_charge = coroconfig["vortex_charge"]
            self.string_os += '_charge' + str(int(vortex_charge))
            self.FPmsk = self.Vortex(vortex_charge=vortex_charge)
            self.perfect_coro = True

        elif self.corona_type == "wrapped_vortex":
            self.prop_apod2lyot = 'mft'
            self.string_os += '2020'
            self.FPmsk = list([
                self.EF_from_phase_and_ampl(phase_abb=proc.crop_or_pad_image(
                    fits.getdata(model_dir + coroconfig["wrapped_vortex_fits_file"]), self.dimScience))
            ])
            self.perfect_coro = True

        else:
            raise Exception(f"The requested coronagraph mode '{self.corona_type}' does not exists.")

        self.lyot_pup = Pupil(modelconfig,
                              prad=self.prad * coroconfig["diam_lyot_in_m"] / self.diam_pup_in_m,
                              PupType=coroconfig["filename_instr_lyot"],
                              angle_rotation=coroconfig['lyot_pup_rotation'],
                              Model_local_dir=Model_local_dir)

        self.string_os += '_LS' + self.lyot_pup.string_os

        if "bool_overwrite_perfect_coro" in coroconfig:
            if coroconfig["bool_overwrite_perfect_coro"] is True:
                self.perfect_coro = True
            if coroconfig["bool_overwrite_perfect_coro"] is False:
                self.perfect_coro = False

        if self.perfect_coro is True:

            if coroconfig["filename_instr_apod"] == "Clear":
                # We need a round pupil only to measure the response
                # of the coronograph to a round pupil to remove it
                # THIS IS NOT THE ENTRANCE PUPIL,
                # this is a round pupil of the same size
                pup_for_perfect_coro = Pupil(modelconfig, prad=self.prad)

                # do a propagation once with self.perfect_Lyot_pupil = 0 to
                # measure the Lyot pupil that will be removed after
                self.perfect_Lyot_pupil = [0] * self.nb_wav
                for i, wave_here in enumerate(self.wav_vec):
                    self.perfect_Lyot_pupil[i] = self.EF_through(
                        entrance_EF=pup_for_perfect_coro.EF_through(wavelength=wave_here),
                        wavelength=wave_here)
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
                   save_all_planes_to_fits=False,
                   dir_save_all_planes=None,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

        AUTHOR : Johan Mazoyer

        03/22 : Correction in the babinet propagation

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

        # call the OpticalSystem super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength is None:
            wavelength = self.wavelength_0
        
        if save_all_planes_to_fits == True and dir_save_all_planes == None:
            raise Exception("save_all_planes_to_fits = True can generate a lot of .fits files" +
                            "please define a clear directory using dir_save_all_planes kw argument")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_apod' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, entrance_EF)

        if noFPM:
            FPmsk = 1.
        else:
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        lambda_ratio = wavelength / self.wavelength_0

        input_wavefront_after_apod = self.apod_pup.EF_through(entrance_EF=entrance_EF, wavelength=wavelength)

        if save_all_planes_to_fits == True:
            name_plane = 'apod' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, self.apod_pup.pup)

            name_plane = 'EF_PP_after_apod' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, input_wavefront_after_apod)

        # we take the convention that for all propation methods, the PSF must be
        # "in between 4 pixels" in the focal plane.

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(wavelength)]
            input_wavefront_after_apod_pad = proc.crop_or_pad_image(input_wavefront_after_apod,
                                                                    dim_fp_fft_here)

            corono_focal_plane = prop.fft_choosecenter(input_wavefront_after_apod_pad,
                                                       inverse=False,
                                                       center_pos='bb',
                                                       norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          np.fft.fftshift(corono_focal_plane))

                name_plane = 'PSF EF_FP_before_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          np.fft.fftshift(np.abs(corono_focal_plane)**2))
                if not noFPM:
                    name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane,
                                          np.fft.fftshift(corono_focal_plane * FPmsk))

            # Focal plane to Lyot plane
            lyotplane_before_lyot = prop.fft_choosecenter(corono_focal_plane * FPmsk,
                                                          inverse=True,
                                                          center_pos='bb',
                                                          norm='ortho')

        elif self.prop_apod2lyot == "mft-babinet":
            #Apod plane to focal plane

            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          int(2 * self.prad),
                                          self.dim_fpm,
                                          self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                                          inverse=False,
                                          norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane)
                if not noFPM:
                    name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * FPmsk)

                name_plane = 'EF_FP_after_1minusFPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * (1 - FPmsk))

            # Focal plane to Lyot plane
            # Babinet's trick:
            lyotplane_before_lyot_central_part = proc.crop_or_pad_image(
                prop.mft(corono_focal_plane * (1 - FPmsk),
                         self.dim_fpm,
                         int(2 * self.prad),
                         self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                         inverse=True,
                         norm='ortho'), self.dim_overpad_pupil)

            # Babinet's trick
            lyotplane_before_lyot = input_wavefront_after_apod - lyotplane_before_lyot_central_part

        elif self.prop_apod2lyot == "mft":
            # Apod plane to focal plane

            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          int(2 * self.prad),
                                          self.dimScience,
                                          self.dimScience / self.Science_sampling * lambda_ratio,
                                          inverse=False,
                                          norm='ortho')

            if save_all_planes_to_fits == True:
                name_plane = 'EF_FP_before_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane)
                if not noFPM:
                    name_plane = 'FPM' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, FPmsk)

                    name_plane = 'FPMphase' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.angle(FPmsk))

                    name_plane = 'FPMmod' + '_wl{}'.format(int(wavelength * 1e9))
                    saveread.save_plane_in_fits(dir_save_all_planes, name_plane, np.abs(FPmsk))

                name_plane = 'EF_FP_after_FPM' + '_wl{}'.format(int(wavelength * 1e9))
                saveread.save_plane_in_fits(dir_save_all_planes, name_plane, corono_focal_plane * FPmsk)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = proc.crop_or_pad_image(
                prop.mft(corono_focal_plane * FPmsk,
                         self.dimScience,
                         int(2 * self.prad),
                         self.dimScience / self.Science_sampling * lambda_ratio,
                         inverse=True,
                         norm='ortho'), self.dim_overpad_pupil)

        else:
            raise Exception(self.prop_apod2lyot + " is not a known prop_apod2lyot propagation mehtod")

        if save_all_planes_to_fits == True:
            name_plane = 'EF_PP_before_LS' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, lyotplane_before_lyot)

        # we add the downstream aberrations if we need them
        lyotplane_before_lyot *= EF_aberrations_introduced_in_LS

        # crop to the dim_overpad_pupil expeted size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(entrance_EF=lyotplane_before_lyot_crop,
                                                        wavelength=wavelength)

        if (self.perfect_coro == True) & (noFPM == False):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil[self.wav_vec.tolist().index(
                wavelength)]

        if save_all_planes_to_fits == True:
            name_plane = 'LS' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, self.lyot_pup.pup)

            name_plane = 'EF_PP_after_LS' + '_wl{}'.format(int(wavelength * 1e9))
            saveread.save_plane_in_fits(dir_save_all_planes, name_plane, lyotplane_after_lyot)

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

        fqpm_thick_vert = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_vert[np.where(xx < 0)] = 1
        fqpm_thick_hor = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
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
                fqpm.append(self.EF_from_phase_and_ampl(phase_abb=phase4q, wavelengths=wav))

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
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2 + 1 / 2,
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2 + 1 / 2)

        phase_vortex = vortex_charge * np.angle(xx + 1j * yy)

        vortex = list()
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dimScience

            phasevortex_cut = proc.crop_or_pad_image(phase_vortex,
                                                     dim_fp)  #*phase_ampl.roundpupil(dim_fp, dim_fp/2)
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
                raise Exception("knife currently not coded in polychromatic fft")
        else:
            maxdimension_array_fpm = self.dimScience

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength

        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm), np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(
                xx > (maxdimension_array_fpm / 2 + self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "left":
            Knife[np.where(
                xx < (maxdimension_array_fpm / 2 - self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(
                yy > (maxdimension_array_fpm / 2 + self.knife_coro_offset * self.Science_sampling))] = 1
        if self.coro_position == "top":
            Knife[np.where(
                yy < (maxdimension_array_fpm / 2 - self.knife_coro_offset * self.Science_sampling))] = 1

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

        ClassicalLyotFPM = 1. - phase_ampl.roundpupil(self.dim_fpm, rad_LyotFP_pix)

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
                self.EF_from_phase_and_ampl(ampl_abb=ampl_hlc, phase_abb=phase_hlc, wavelengths=wav))

        return hlc_all_wl
