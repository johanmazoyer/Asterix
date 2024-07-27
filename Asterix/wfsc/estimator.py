import os
from datetime import datetime
import numpy as np
from astropy.io import fits

from Asterix.utils import resizing, save_plane_in_fits, from_param_to_header
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed

import Asterix.wfsc.wf_sensing_functions as wfs


class Estimator:
    """Estimator Class allows you to define a WF estimator.

        It must contains 2 functions at least:
            - an initialization (e.g. PWP matrix) Estimator.__init__(), with parameters:
                    - the testbed structure
                    - the estimation parameters
                    - saving directories
                The estimator initialization requires previous initialization of the testbed.

            - an probe function Estimator.probe(), with parameters:
                    - the entrance EF
                    - DM voltages
                    - the estimation wavelengths
                It returns the probed images as a list (of length nb_wav_estim) of
                3d arrays (nprobes,dimEstim,dimEstim).

            - an estimation function Estimator.estimate(), with parameters:
                - the probed images
                It returns the estimation as a list (of length nb_wav_estim) of 2D arrays (dimEstim,dimEstim).
                In all generality, it can be pupil or focal plane, complex or real with keywords
                (Estim.is_focal_plane = True, Estim.is_complex = True) to explain the form of the output and
                potentially prevent wrongfull combination of estim + correc.

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
                                 f"the estimator sampling is {self.Estim_sampling} at wavelength {wavei * 1e9} nm. "
                                 "Please decrease [Estimationconfig]['Estim_bin_factor'] parameter.")

        # image size after binning. This is the size of the estimation !
        # We round and make it so we're always even sized and slightly smaller than the ideal size.
        self.dimEstim = int(np.floor(self.Estim_sampling / testbed.Science_sampling * testbed.dimScience / 2) * 2)

        if self.technique == "perfect":
            self.is_focal_plane = True
            self.is_complex = True
            if self.polychrom == 'broadband_pwprobes':
                raise ValueError("Cannot use [Estimationconfig]['polychromatic']='broadband_pwprobes' in perfect mode.")

        elif self.technique in ["pairwise", "pw", "pwp", "btp"]:
            self.is_focal_plane = True
            self.is_complex = True

            amplitudePW = Estimationconfig["amplitudePW"]
            posprobes = list(Estimationconfig["posprobes"])
            cutsvdPW = Estimationconfig["cut"]

            name_DM_to_probe_in_PW = Estimationconfig["name_DM_to_probe_in_PW"]

            self.voltage_probe = wfs.generate_actu_probe_voltages(testbed, posprobes, amplitudePW,
                                                                  name_DM_to_probe_in_PW)

            self.PWMatrix = wfs.create_pw_matrix(testbed,
                                                 self.voltage_probe,
                                                 self.dimEstim,
                                                 cutsvdPW,
                                                 matrix_dir,
                                                 self.wav_vec_estim,
                                                 silence=silence)

            # Saving PWP matrix in Labview directory
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
                    probes = np.zeros((len(posprobes), testbed.DM3.number_act), dtype=np.float32)
                    vectorPW = np.zeros((2, self.dimEstim * self.dimEstim * len(posprobes)), dtype=np.float32)

                    for i in np.arange(len(posprobes)):
                        # TODO WTH is the hardcoded 17. @Raphael @Axel
                        probes[i, posprobes[i]] = amplitudePW / 17
                        vectorPW[0, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                                 self.dimEstim] = self.PWMatrix[k][:, 0, i].flatten()
                        vectorPW[1, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                                 self.dimEstim] = self.PWMatrix[k][:, 1, i].flatten()

                    # Because the exact laser WLs can change a bit quite often and labview reads a given
                    # fits name, we name the PWP matrix laser1, 2, 3 for a range of WL.

                    string_laser = ''
                    if 625 < wave_k * 1e9 < 645:
                        string_laser = "_laser1"  # ~635nm laser source
                    if 695 < wave_k * 1e9 < 715:
                        string_laser = "_laser2"  # ~705nm laser source
                    if 775 < wave_k * 1e9 < 795:
                        string_laser = "_laser3"  # ~785nm laser source

                    header = fits.Header()
                    if hasattr(testbed, 'config_file'):
                        header = from_param_to_header(testbed.config_file, header)
                    header.insert(0, ('date_mat', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "matrix creation date"))

                    namepwmatrix = '_PW_' + name_DM_to_probe_in_PW + string_laser
                    fits.writeto(os.path.join(realtestbed_dir, "Probes" + namepwmatrix + ".fits"),
                                 probes,
                                 header,
                                 overwrite=True)
                    fits.writeto(os.path.join(realtestbed_dir, "Matr_mult_estim" + namepwmatrix + ".fits"),
                                 vectorPW,
                                 header,
                                 overwrite=True)

        elif self.technique == 'coffee':
            pass

        else:
            raise NotImplementedError("This estimation algorithm is not yet implemented")

    def probe(self, testbed: Testbed, entrance_EF=1., voltage_vector=0., perfect_estimation=False, **kwargs):
        """Use the DM or the testbed to probe the aberrations with a given input wavefront and a
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
        probed_fp_images : list of 3D arrays
            Probed images for each wavelengths.
            Each arrays are of size of sixe [2*nprobes, dimEstim, dimEstim] if PWP
            or [1 + nprobes, dimEstim, dimEstim] if BTP.

        """

        if 'wavelength' in kwargs:
            raise ValueError(("probe() function is polychromatic, "
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

        probed_fp_images = []

        if (self.technique == "perfect") or (perfect_estimation):
            # in the perfect estimation case, we return the focal plane electric field as "probed" images

            if self.polychrom == 'multiwl':
                for i, wavei in enumerate(self.wav_vec_estim):
                    resultatestimation = testbed.todetector(
                        entrance_EF=entrance_EF[testbed.wav_vec.tolist().index(wavei)],
                        voltage_vector=voltage_vector,
                        wavelength=wavei)
                    probed_fp_images.append(resizing(resultatestimation, self.dimEstim))

                    if 'dir_save_all_planes' in kwargs.keys():
                        if kwargs['dir_save_all_planes'] is not None:
                            name_plane = 'Perfect_estimate_multiwl' + f'_wl{int(wavei * 1e9)}'
                            save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, probed_fp_images[-1])

            elif self.polychrom == 'singlewl':
                resultatestimation = testbed.todetector(entrance_EF=entrance_EF[testbed.wav_vec.tolist().index(
                    self.wav_vec_estim[0])],
                                                        voltage_vector=voltage_vector,
                                                        wavelength=self.wav_vec_estim[0])
                probed_fp_images.append(resizing(resultatestimation, self.dimEstim))

                if 'dir_save_all_planes' in kwargs.keys():
                    if kwargs['dir_save_all_planes'] is not None:
                        name_plane = 'Perfect_estimate_singlewl' + f'_wl{int(self.wav_vec_estim[0] * 1e9)}'
                        save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, probed_fp_images[-1])

            elif self.polychrom == 'broadband_pwprobes':
                raise ValueError("cannot use [Estimationconfig]['polychromatic']='broadband_pwprobes' in perfect mode")
            else:
                raise ValueError(self.polychrom +
                                 "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")

        elif self.technique in ["pairwise", "pw", "pwp", "btp"]:

            # nb_photons parameter is normally for the whole bandwidth (testbed.Delta_wav). For this
            # case, we reduce it to self.delta_wav_estim_individual bandwidth
            if self.polychrom == 'multiwl':
                if 'nb_photons' in kwargs.keys():
                    if kwargs['nb_photons'] > 0:
                        kwargs['nb_photons'] = kwargs['nb_photons'] / testbed.Delta_wav * self.delta_wav_estim_individual

                for i, wavei in enumerate(self.wav_vec_estim):
                    probed_fp_images_i = wfs.simulate_pw_probes(entrance_EF[testbed.wav_vec.tolist().index(wavei)],
                                                                testbed,
                                                                self.voltage_probe,
                                                                voltage_vector=voltage_vector,
                                                                wavelengths=wavei,
                                                                pwp_or_btp=self.technique,
                                                                **kwargs)
                    probed_fp_images.append(probed_fp_images_i)

            elif self.polychrom == 'singlewl':
                probed_fp_images_i = wfs.simulate_pw_probes(entrance_EF[testbed.wav_vec.tolist().index(
                    self.wav_vec_estim[0])],
                                                            testbed,
                                                            self.voltage_probe,
                                                            voltage_vector=voltage_vector,
                                                            wavelengths=self.wav_vec_estim[0],
                                                            pwp_or_btp=self.technique,
                                                            **kwargs)
                probed_fp_images.append(probed_fp_images_i)

            elif self.polychrom == 'broadband_pwprobes':
                probed_fp_images_i = wfs.simulate_pw_probes(entrance_EF,
                                                            testbed,
                                                            self.voltage_probe,
                                                            voltage_vector=voltage_vector,
                                                            wavelengths=self.wav_vec_estim[0],
                                                            pwp_or_btp=self.technique,
                                                            **kwargs)
                probed_fp_images.append(probed_fp_images_i)

        return probed_fp_images

    def estimate(self, probed_fp_images, perfect_estimation=False, dtype_complex='complex128', testbed=None, **kwargs):
        """Run an estimation from a testbed, with a given input wavefront and a
        state of the DMs.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        probed_fp_images : list of 3D arrays
            Probed images for each probes.
            Each arrays are of size of sixe [2*nprobes, dimEstim, dimEstim] if PWP
            or [1 + nprobes, dimEstim, dimEstim] if BTP.
        perfect_estimation : bool, default = False
            if true This is equivalent to have self.technique = "perfect"
            but even if we are using another technique, we sometimes
            need a perfect estimation and it avoid re-initialization of
            the estimation.
        dtype_complex : string, default 'complex128'
            bit number for the complex arrays in the PW matrices.
            Can be 'complex128' or 'complex64'. The latter increases the speed of the mft but at the
            cost of lower precision.

        Returns
        --------
        estimation : list of 2D array
            list is the number of wl in the estimation, usually 1 or testbed.nb_wav
            Each arrays are of size of sixe [dimEstim, dimEstim].
            estimation of the Electrical field
        """

        if (self.technique == "perfect") or (perfect_estimation):
            return probed_fp_images

        elif self.technique in ["pairwise", "pw", "pwp", "btp"]:
            result_estim = []

            if self.polychrom == 'multiwl':

                for i, wavei in enumerate(self.wav_vec_estim):
                    if self.technique in ["btp"]:
                        differences = wfs.btp_difference(probed_fp_images[i], testbed, self.probes_vector, wavei)
                    else:
                        differences = wfs.pw_difference(probed_fp_images[i])

                    result_estim.append(
                        wfs.calculate_pw_estimate(differences,
                                                  self.PWMatrix[i],
                                                  self.dimEstim,
                                                  dtype_complex=dtype_complex))

                    if 'dir_save_all_planes' in kwargs.keys():
                        if kwargs['dir_save_all_planes'] is not None:
                            name_plane = 'PW_estimate_multiwl' + f'_wl{int(wavei * 1e9)}'
                            save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            elif self.polychrom in ['singlewl', 'broadband_pwprobes']:
                if self.technique in ["btp"]:
                    differences = wfs.btp_difference(probed_fp_images, testbed, self.probes_vector,
                                                     self.wav_vec_estim[0])
                else:
                    differences = wfs.pw_difference(probed_fp_images[0])

                result_estim.append(
                    wfs.calculate_pw_estimate(differences, self.PWMatrix[0], self.dimEstim, dtype_complex=dtype_complex))

                if 'dir_save_all_planes' in kwargs.keys():
                    if kwargs['dir_save_all_planes'] is not None:
                        name_plane = 'PW_estimate_singlewl' + f'_wl{int(self.wav_vec_estim[0] * 1e9)}'
                        save_plane_in_fits(kwargs['dir_save_all_planes'], name_plane, result_estim[-1])

            else:
                raise ValueError(self.polychrom +
                                 "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")
            return result_estim

        else:
            raise NotImplementedError("This estimation algorithm is not yet implemented "
                                      "([Estimationconfig]['estimation'] parameter)")
