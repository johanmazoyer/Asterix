import os
import numpy as np
from astropy.io import fits

from Asterix.utils import resizing, save_plane_in_fits
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed

import Asterix.wfsc.wf_sensing_functions as wfs


class Estimator:
    """Estimator Class allows you to define a WF estimator.

        It must contains 2 functions at least:
            - an initialization (e.g. PW matrix) Estimator.__init__(), with parameters:
                    - the testbed structure
                    - the estimation parameters
                    - saving directories
                The estimator initialization requires previous initialization of the testbed.

            - an estimation function Estimator.estimate(), with parameters:
                    - the entrance EF
                    - DM voltages
                    - the wavelength
                It returns the estimation as a 2D array. In all generality, it can be pupil or focal plane,
                complex or real with keywords (Estim.is_focal_plane = True, Estim.is_complex = True)
                to explain the form of the output and potentially prevent wrongfull combination of
                estim + correc.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self,
                 Estimationconfig,
                 testbed: Testbed,
                 matrix_dir='',
                 save_for_bench=False,
                 realtestbed_dir='',
                 silence=False):
        """Initialize the estimator. This is where you define the pw matrix,
        the modified Lyot stop or the COFFEE gradiant...

        For all large files you should use a method of "save to fits" if
        it does not exist "load from fits" if it does, in matrix_dir

        Store in the structure only what you need for estimation. Everything not
        used in self.estimate shoud not be stored

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        Estimationconfig : dict
                general estimation parameters
        testbed : OpticalSystem.Testbed
                Testbed object which describe your testbed
        matrix_dir : string
            path to directory. save all the matrix files here
        save_for_bench : bool, default false
                should we save for the real testbed in realtestbed_dir
        realtestbed_dir : string
            path to directory to save all the files the real thd2 testbed need to run your code
        silence : boolean, default False.
            Whether to silence print outputs.
        """
        if not os.path.exists(matrix_dir):
            if not silence:
                print("Creating directory " + matrix_dir)
            os.makedirs(matrix_dir)

        if not isinstance(testbed, OpticalSystem):
            raise TypeError("testbed must be an OpticalSystem object")

        self.technique = Estimationconfig["estimation"].lower()
        self.polychrom = Estimationconfig["polychromatic"].lower()

        # For now estimation central wl and simu central wl are the same
        # wavelength_0_estim = testbed.wavelength_0

        if self.polychrom == "centralwl":
            print("")
            print(("DEPRECATED PARAMETER VALUE: use 'singlelw' instead of 'centralwl' "
                   "in [Estimationconfig]['polychromatic'] parameter."
                   "'centralwl' value for this parameter will not ne supported in the near future"))
            print("")
            self.polychrom = "singlewl"

        if len(testbed.wav_vec) == 1:
            # monochromatic simulation, polychromatic correction keywords are ignored
            self.polychrom = "singlewl"
            self.wav_vec_estim = np.array([testbed.wavelength_0])
            self.nb_wav_estim = 1

        elif self.polychrom == 'multiwl':
            # For now estimation BW and testbed BW are the same and can be easily changed.
            self.delta_wave_estim = testbed.Delta_wav

            self.delta_wav_estim_individual = Estimationconfig["delta_wav_estim_individual"]

            estimation_wls = [float(x) for x in Estimationconfig["estimation_wls"]]

            if estimation_wls != []:
                self.wav_vec_estim = np.asarray(estimation_wls)
                self.nb_wav_estim = len(self.wav_vec_estim)

            # we measure the WL for each individual monochromatic channel.
            else:
                self.nb_wav_estim = Estimationconfig["nb_wav_estim"]

                if (self.nb_wav_estim % 2 == 0) or self.nb_wav_estim < 2:
                    raise ValueError(("If [Estimationconfig]['polychromatic'] = 'multiwl' estimation mode "
                                      "either hand pick wavelengths using [Estimationconfig]['estimation_wls'] "
                                      "parameter, or set [Estimationconfig]['nb_wav_estim'] parameter to an odd "
                                      "number > 1"))

                delta_wav_estim_interval = self.delta_wave_estim / self.nb_wav_estim
                self.wav_vec_estim = testbed.wavelength_0 + (np.arange(self.nb_wav_estim) -
                                                             self.nb_wav_estim // 2) * delta_wav_estim_interval
        elif self.polychrom == 'singlewl':
            # polychromatic simulation with 'singlewl' correction
            estimation_wls = [float(x) for x in Estimationconfig["estimation_wls"]]
            if estimation_wls != []:
                if len(estimation_wls) > 1:
                    raise ValueError(("If [Estimationconfig]['polychromatic'] = 'singlewl' estimation mode, "
                                      "[Estimationconfig]['estimation_wls'] list must be either an empty list "
                                      "(in this case the estimtion will be done at the central wavelength "
                                      "{testbed.wavelength_0} only) or have no more than a single element"))
                self.wav_vec_estim = np.array(estimation_wls)
            else:
                self.wav_vec_estim = np.array([testbed.wavelength_0])
            self.nb_wav_estim = 1
        elif self.polychrom == 'broadband_pwprobes':
            self.wav_vec_estim = np.array([testbed.wavelength_0])
            self.nb_wav_estim = 1
        else:
            raise ValueError(self.polychrom + "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")

        for wavei in self.wav_vec_estim:
            if wavei not in testbed.wav_vec:
                raise ValueError((f"Wavelength {wavei} is in estimator.wav_vec_estim but not in "
                                  "testbed.wav_vec. If you added one or several estimation wavelengths"
                                  "([Estimationconfig]['estimation_wls'] parameter) manually, also add them in as "
                                  "[modelconfig]['mandatory_wls'] parameter. If you did not used "
                                  "[Estimationconfig]['estimation_wls'] parameter then make sure "
                                  "([Estimationconfig]['nb_wav_estim'] parameter is equal to, "
                                  "or a divisor, of [modelconfig]['nb_wav'] parameter and both must be odd."))

        # we measure the estimation sampling for the central wavelength wavelength_0.
        self.Estim_sampling = testbed.Science_sampling / Estimationconfig["Estim_bin_factor"]

        for wavei in self.wav_vec_estim:
            if self.Estim_sampling / testbed.wavelength_0 * wavei < 2.5:
                raise ValueError(f"Estimator sampling must be >= 2.5 at all estimation wavelengths. "
                                 f"For [Estimationconfig]['Estim_bin_factor'] = {Estimationconfig['Estim_bin_factor']}, "
                                 f"the estimator sampling is {self.Estim_sampling} at wavelength {wavei*1e9} nm. "
                                 "Please decrease [Estimationconfig]['Estim_bin_factor'] parameter.")

        # image size after binning. This is the size of the estimation !
        # We round and make it so we're always even sized and slightly smaller than the ideal size.
        self.dimEstim = int(np.floor(self.Estim_sampling / testbed.Science_sampling * testbed.dimScience / 2) * 2)

        if self.technique == "perfect":
            self.is_focal_plane = True
            self.is_complex = True
            if self.polychrom == 'broadband_pwprobes':
                raise ValueError("Cannot use [Estimationconfig]['polychromatic']='broadband_pwprobes' in perfect mode.")

        elif self.technique in ["pairwise", "pw"]:
            self.is_focal_plane = True
            self.is_complex = True

            self.amplitudePW = Estimationconfig["amplitudePW"]
            self.posprobes = list(Estimationconfig["posprobes"])
            cutsvdPW = Estimationconfig["cut"]

            testbed.name_DM_to_probe_in_PW = find_DM_to_probe(testbed)

            self.PWMatrix = wfs.create_pw_matrix(testbed,
                                                 self.amplitudePW,
                                                 self.posprobes,
                                                 self.dimEstim,
                                                 cutsvdPW,
                                                 matrix_dir,
                                                 self.polychrom,
                                                 self.wav_vec_estim,
                                                 silence=silence)

            # Saving PW matrix in Labview directory
            if save_for_bench:

                if not os.path.exists(realtestbed_dir):
                    if not silence:
                        print("Creating directory: " + realtestbed_dir)
                    os.makedirs(realtestbed_dir)

                if self.polychrom in ['broadband_pwprobes']:
                    wl_in_pw_matrix = [testbed.wavelength_0]
                else:
                    wl_in_pw_matrix = self.wav_vec_estim

                for k, wave_k in enumerate(wl_in_pw_matrix):
                    probes = np.zeros((len(self.posprobes), testbed.DM3.number_act), dtype=np.float32)
                    vectorPW = np.zeros((2, self.dimEstim * self.dimEstim * len(self.posprobes)), dtype=np.float32)

                    for i in np.arange(len(self.posprobes)):
                        probe = fits.getdata(os.path.join('/Users/ilaginja/Documents/LESIA/THD/Roman_probes/sinc_probes', f'2024-11-14_12_cgi-like/probe_{i}_sinc-freq1-12_sinc-freq2-24_sine-freq6_cgi-like.fits')).ravel()

                        # TODO WTH is the hardcoded 17. @Raphael @Axel
                        probes[i] = probe * self.amplitudePW / 17
                        vectorPW[0, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                                 self.dimEstim] = self.PWMatrix[k][:, 0, i].flatten()
                        vectorPW[1, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                                 self.dimEstim] = self.PWMatrix[k][:, 1, i].flatten()

                    # Because the exact laser WLs can change a bit quite often and labview reads a given
                    # fits name, we name the PW matrix laser1, 2, 3 for a range of WL.

                    string_laser = ''
                    if 625 < wave_k * 1e9 < 645:
                        string_laser = "_laser1"  # ~635nm laser source
                    if 695 < wave_k * 1e9 < 715:
                        string_laser = "_laser2"  # ~705nm laser source
                    if 775 < wave_k * 1e9 < 795:
                        string_laser = "_laser3"  # ~785nm laser source

                    namepwmatrix = '_PW_' + testbed.name_DM_to_probe_in_PW + string_laser
                    fits.writeto(os.path.join(realtestbed_dir, "Probes" + namepwmatrix + ".fits"),
                                 probes,
                                 overwrite=True)
                    fits.writeto(os.path.join(realtestbed_dir, "Matr_mult_estim" + namepwmatrix + ".fits"),
                                 vectorPW,
                                 overwrite=True)

        elif self.technique == 'coffee':
            pass

        else:
            raise NotImplementedError("This estimation algorithm is not yet implemented")

    def estimate(self, testbed: Testbed, entrance_EF=1., voltage_vector=0., perfect_estimation=False, **kwargs):
        """Run an estimation from a testbed, with a given input wavefront and a
        state of the DMs.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        testbed : OpticalSystem.Testbed
                Testbed object which describe your testbed
        entrance_EF : complex float or 2D array, default 1.
            initial EF field
        voltage_vector : 1D float array
            vector of voltages vectors for each DMs
        perfect_estimation : bool, default = False
            if true This is equivalent to have self.technique = "perfect"
            but even if we are using another technique, we sometimes
            need a perfect estimation and it avoid re-initialization of
            the estimation.

        Returns
        --------
        estimation : list of 2D array
            list is the number of wl in the estimation, usually 1 or testbed.nb_wav
            Each arrays are of size of sixe [dimEstim, dimEstim].
            estimation of the Electrical field
        """

        if 'wavelength' in kwargs:
            raise ValueError(("estimate() function is polychromatic, "
                              "do not use wavelength keyword. "
                              "Use 'wavelengths' keyword even for monochromatic intensity"))

        if isinstance(entrance_EF, (float, int)):
            entrance_EF = np.repeat(entrance_EF, testbed.nb_wav)
        elif entrance_EF.shape == testbed.wav_vec.shape:
            pass
        elif entrance_EF.shape == (testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
            entrance_EF = np.repeat(entrance_EF[np.newaxis, ...], testbed.nb_wav, axis=0)
        elif entrance_EF.shape == (testbed.nb_wav, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
            pass
        else:
            raise TypeError(
                ("entrance_EFs must be scalar (same for all WLs), or a testbed.nb_wav scalars "
                 "or a2D array of size (testbed.dim_overpad_pupil, testbed.dim_overpad_pupil) "
                 "or a 3D array of size(testbed.nb_wav, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil)"))

        if (self.technique == "perfect") or (perfect_estimation):
            # If polychromatic, assume a perfect estimation at one wavelength

            result_estim = []

            if self.polychrom == 'multiwl':
                for i, wavei in enumerate(self.wav_vec_estim):
                    resultatestimation = testbed.todetector(
                        entrance_EF=entrance_EF[testbed.wav_vec.tolist().index(wavei)],
                        voltage_vector=voltage_vector,
                        wavelength=wavei)
                    result_estim.append(resizing(resultatestimation, self.dimEstim))

                    if 'dir_save_all_planes' in kwargs.keys():
                        if kwargs['dir_save_all_planes'] is not None:
                            name_plane = 'Perfect_estimate_multiwl' + f'_wl{int(wavei * 1e9)}'
                            save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            elif self.polychrom == 'singlewl':
                resultatestimation = testbed.todetector(entrance_EF=entrance_EF[testbed.wav_vec.tolist().index(
                    self.wav_vec_estim[0])],
                                                        voltage_vector=voltage_vector,
                                                        wavelength=self.wav_vec_estim[0])
                result_estim.append(resizing(resultatestimation, self.dimEstim))

                if 'dir_save_all_planes' in kwargs.keys():
                    if kwargs['dir_save_all_planes'] is not None:
                        name_plane = 'Perfect_estimate_singlewl' + f'_wl{int(self.wav_vec_estim[0] * 1e9)}'
                        save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            elif self.polychrom == 'broadband_pwprobes':
                raise ValueError("cannot use [Estimationconfig]['polychromatic']='broadband_pwprobes' in perfect mode")
            else:
                raise ValueError(self.polychrom +
                                 "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")
            return result_estim

        elif self.technique in ["pairwise", "pw"]:

            result_estim = []

            # nb_photons parameter is normally for the whole bandwidth (testbed.Delta_wav). For this
            # case, we reduce it to self.delta_wav_estim_individual bandwidth
            if self.polychrom == 'multiwl':
                if 'nb_photons' in kwargs.keys():
                    if kwargs['nb_photons'] > 0:
                        kwargs['nb_photons'] = kwargs['nb_photons'] / testbed.Delta_wav * self.delta_wav_estim_individual

                for i, wavei in enumerate(self.wav_vec_estim):
                    Difference = wfs.simulate_pw_difference(entrance_EF[testbed.wav_vec.tolist().index(wavei)],
                                                            testbed,
                                                            self.posprobes,
                                                            self.dimEstim,
                                                            self.amplitudePW,
                                                            voltage_vector=voltage_vector,
                                                            wavelengths=wavei,
                                                            **kwargs)

                    result_estim.append(
                        wfs.calculate_pw_estimate(Difference, self.PWMatrix[i], dtype_complex=testbed.dtype_complex))

                    if 'dir_save_all_planes' in kwargs.keys():
                        if kwargs['dir_save_all_planes'] is not None:
                            name_plane = 'PW_estimate_multiwl' + f'_wl{int(wavei * 1e9)}'
                            save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            elif self.polychrom == 'singlewl':
                Difference = wfs.simulate_pw_difference(entrance_EF[testbed.wav_vec.tolist().index(
                    self.wav_vec_estim[0])],
                                                        testbed,
                                                        self.posprobes,
                                                        self.dimEstim,
                                                        self.amplitudePW,
                                                        voltage_vector=voltage_vector,
                                                        wavelengths=self.wav_vec_estim[0],
                                                        **kwargs)

                result_estim.append(
                    wfs.calculate_pw_estimate(Difference, self.PWMatrix[0], dtype_complex=testbed.dtype_complex))
                if 'dir_save_all_planes' in kwargs.keys():
                    if kwargs['dir_save_all_planes'] is not None:
                        name_plane = 'PW_estimate_singlewl' + f'_wl{int(self.wav_vec_estim[0] * 1e9)}'
                        save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            elif self.polychrom == 'broadband_pwprobes':
                Difference = wfs.simulate_pw_difference(entrance_EF,
                                                        testbed,
                                                        self.posprobes,
                                                        self.dimEstim,
                                                        self.amplitudePW,
                                                        voltage_vector=voltage_vector,
                                                        wavelengths=testbed.wav_vec,
                                                        **kwargs)

                result_estim.append(
                    wfs.calculate_pw_estimate(Difference, self.PWMatrix[0], dtype_complex=testbed.dtype_complex))
                if 'dir_save_all_planes' in kwargs.keys():
                    if kwargs['dir_save_all_planes'] is not None:
                        name_plane = 'PW_estimate_broadband_pwprobes' + f'_wl{int(testbed.wavelength_0 * 1e9)}_bw{testbed.Delta_wav * 1e9}'
                        save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            else:
                raise ValueError(self.polychrom +
                                 "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")
            return result_estim

        else:
            raise NotImplementedError("This estimation algorithm is not yet implemented "
                                      "([Estimationconfig]['estimation'] parameter)")


def find_DM_to_probe(testbed: Testbed):
    """Find which DM to use for the PW probes.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    testbed : OpticalSystem.Testbed
        Testbed object which describe your testbed

    Returns
    ------------
    name_DM_to_probe_in_PW : string
        name of the DM to probe in PW
    """

    # we chose it already. We only check its existence
    if hasattr(testbed, 'name_DM_to_probe_in_PW'):
        if testbed.name_DM_to_probe_in_PW not in testbed.name_of_DMs:
            raise ValueError(f"Testbed has no DM '{testbed.name_DM_to_probe_in_PW}', choose another DM name "
                             "for PW, using using 'testbed.name_DM_to_probe_in_PW'.")
        return testbed.name_DM_to_probe_in_PW

    # If name_DM_to_probe_in_PW is not already set,
    # automatically check which DM to use to probe in this case
    # this is only done once.
    if len(testbed.name_of_DMs) == 0:
        raise ValueError("You need at least one activated DM to do PW.")
    # If only one DM, we use this one, independenlty of its position
    elif len(testbed.name_of_DMs) == 1:
        name_DM_to_probe_in_PW = testbed.name_of_DMs[0]
    else:
        # If several DMs we check if there is at least one in PP
        number_DMs_in_PP = 0
        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]
            if DM.z_position == 0.:
                number_DMs_in_PP += 1
                name_DM_to_probe_in_PW = DM_name

        # If there are several DMs in PP, error, you need to set name_DM_to_probe_in_PW
        if number_DMs_in_PP > 1:
            raise ValueError("You have several DM in PP, choose manually one for "
                             "the PW probes using 'testbed.name_DM_to_probe_in_PW'.")
        # Several DMS, none in PP, error, you need to set name_DM_to_probe_in_PW
        if number_DMs_in_PP == 0:
            raise ValueError("You have several DMs, none in PP, choose manually one for "
                             "the PW probes using 'testbed.name_DM_to_probe_in_PW'.")
    return name_DM_to_probe_in_PW
