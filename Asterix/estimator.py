# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import copy
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits

import Asterix.fits_functions as useful
import Asterix.processing_functions as proc
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.Optical_System_functions as OptSy
import Asterix.WSC_functions as wsc
from Asterix.MaskDH import MaskDH


class Estimator:
    """ --------------------------------------------------
    Estimator Class allows you to define a WF estimator.

        It must contains 2 functions at least:
        - an initialization (e.g. PW matrix) Estimator.__init__()
        The initialization will require previous initialization of
        the testbed
        it takes as parameter:
                - the testbed structure
                - the estimation parameters
                -saving dirs

        - an estimatation function itself with parameters
                - the entrance EF
                - DM voltages
                - the wavelength
        Estimation = Estimator.estimate(entrance EF, DM voltage, WL)

        It returns the estimation as a 2D array. In all generality, it can be pupil or focal plane,
        complex or real with keywords (Estim.is_focal_plane = True, Estim.is_complex = True)
        to explain the form of the output and potentially prevent wrongfull combination of
        estim + correc.

    AUTHOR : Johan Mazoyer
    Version with SCC Added on 22 Sept 2021


    -------------------------------------------------- """
    def __init__(self,
                 Estimationconfig,
                 testbed: OptSy.Testbed,
                 mask_dh: MaskDH,
                 matrix_dir='',
                 save_for_bench=False,
                 realtestbed_dir=''):
        """ --------------------------------------------------
        Initialize the estimator.
        This is where you define the pw matrix, the modified Lyot stop
        or the COFFEE gradiant...

        For all large files you should use a method of "save to fits" if
        it does not exist "load from fits" if it does, in matrix_dir

        Store in the structure only what you need for estimation. Everything not
        used in self.estimate shoud not be stored

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        Estimationconfig : dict
                general estimation parameters

        testbed :  Optical_System.Testbed 
                Testbed object which describe your testbed

        matrix_dir: path. 
            save all the matrices files here

        save_for_bench. bool default: false
                should we save for the real testbed in realtestbed_dir

        realtestbed_dir: path 
            save all the files the real thd2 testbed need to run your code


        -------------------------------------------------- """
        if not os.path.exists(matrix_dir):
            print("Creating directory " + matrix_dir + " ...")
            os.makedirs(matrix_dir)

        if isinstance(testbed, OptSy.Optical_System) == False:
            raise Exception("testbed must be an Optical_System objet")

        self.technique = Estimationconfig["estimation"].lower()

        self.Estim_sampling = Estimationconfig["Estim_sampling"]

        #image size after binning. This is the size of the estimation !
        self.dimEstim = int(self.Estim_sampling / testbed.Science_sampling *
                            testbed.dimScience / 2) * 2

        if self.technique == "perfect":
            self.is_focal_plane = True
            self.is_complex = True

        elif self.technique in ["pairwise", "pw"]:
            self.is_focal_plane = True
            self.is_complex = True

            self.amplitudePW = Estimationconfig["amplitudePW"]
            self.posprobes = [int(i) for i in Estimationconfig["posprobes"]]
            cutsvdPW = Estimationconfig["cut"]

            if hasattr(testbed, 'name_DM_to_probe_in_PW'):
                if testbed.name_DM_to_probe_in_PW not in testbed.name_of_DMs:
                    raise Exception(
                        "Cannot use this DM for PW, this testbed has no DM named "
                        + testbed.name_DM_to_probe_in_PW)
            else:
                # If name_DM_to_probe_in_PW is not set,
                # automatically check which DM to use to probe in this case
                # this is only done once.
                if len(testbed.name_of_DMs) == 0:
                    raise Exception(
                        "you need at least one activated DM to do PW")
                #If only one DM, we use this one, independenlty of its position
                elif len(testbed.name_of_DMs) == 1:
                    testbed.name_DM_to_probe_in_PW = testbed.name_of_DMs[0]
                else:
                    #If several DMs we check if there is at least one in PP
                    number_DMs_in_PP = 0
                    for DM_name in testbed.name_of_DMs:
                        DM = vars(testbed)[
                            DM_name]  # type: OptSy.deformable_mirror
                        if DM.z_position == 0.:
                            number_DMs_in_PP += 1
                            testbed.name_DM_to_probe_in_PW = DM_name

                    #If there are several DMs in PP, error, you need to set name_DM_to_probe_in_PW
                    if number_DMs_in_PP > 1:
                        raise Exception(
                            """You have several DM in PP, choose one for the 
                                    PW probes using testbed.name_DM_to_probe_in_PW"""
                        )
                    #Several DMS, none in PP, error, you need to set name_DM_to_probe_in_PW
                    if number_DMs_in_PP == 0:
                        raise Exception(
                            """You have several DMs none in PP, choose one for the 
                                    PW probes using testbed.name_DM_to_probe_in_PW"""
                        )

            string_dims_PWMatrix = "actProb_" + "_".join(
                map(str, self.posprobes)
            ) + "with" + testbed.name_DM_to_probe_in_PW + "_PWampl" + str(
                int(self.amplitudePW)) + "_cut" + str(int(
                    cutsvdPW // 1000)) + "k_dimEstim" + str(
                        self.dimEstim) + testbed.string_os

            ####Calculating and Saving PW matrix
            filePW = "MatrixPW_" + string_dims_PWMatrix
            if os.path.exists(matrix_dir + filePW + ".fits") == True:
                print("The matrix " + filePW + " already exists")
                self.PWMatrix = fits.getdata(matrix_dir + filePW + ".fits")
            else:
                print("Saving " + filePW + " ...")
                self.PWMatrix, showSVD = wsc.createPWmatrix(
                    testbed, self.amplitudePW, self.posprobes, self.dimEstim,
                    cutsvdPW, testbed.wavelength_0)
                fits.writeto(matrix_dir + filePW + ".fits",
                             np.array(self.PWMatrix))
                visuPWMap = "EigenValPW_" + string_dims_PWMatrix
                fits.writeto(matrix_dir + visuPWMap + ".fits",
                             np.array(showSVD[1]))

            # Saving PW matrix in Labview directory
            if save_for_bench == True:
                if not os.path.exists(realtestbed_dir):
                    print("Creating directory " + realtestbed_dir + " ...")
                    os.makedirs(realtestbed_dir)

                probes = np.zeros(
                    (len(self.posprobes), testbed.DM3.number_act),
                    dtype=np.float32)
                vectorPW = np.zeros(
                    (2, self.dimEstim * self.dimEstim * len(self.posprobes)),
                    dtype=np.float32)

                for i in np.arange(len(self.posprobes)):
                    # TODO WTH is the hardcoded 17. @Raphael @Axel
                    probes[i, self.posprobes[i]] = self.amplitudePW / 17
                    vectorPW[0, i * self.dimEstim * self.dimEstim:(i + 1) *
                             self.dimEstim *
                             self.dimEstim] = self.PWMatrix[:, 0, i].flatten()
                    vectorPW[1, i * self.dimEstim * self.dimEstim:(i + 1) *
                             self.dimEstim *
                             self.dimEstim] = self.PWMatrix[:, 1, i].flatten()
                namepwmatrix = '_PW_' + testbed.name_DM_to_probe_in_PW
                fits.writeto(realtestbed_dir + "Probes" + namepwmatrix +
                             ".fits",
                             probes,
                             overwrite=True)
                fits.writeto(realtestbed_dir + "Matr_mult_estim" +
                             namepwmatrix + ".fits",
                             vectorPW,
                             overwrite=True)
        elif self.technique == 'coffee':
            pass

        elif self.technique == 'scc':
            self.is_focal_plane = True
            self.is_complex = True

            self.ref_SCC_radiuspix = Estimationconfig[
                "ref_SCC_diam"] / 2 / testbed.diam_pup_in_m * (testbed.prad *
                                                               2)
            self.ref_SCC_distancepix = Estimationconfig[
                "ref_SCC_distance"] / testbed.diam_pup_in_m * (testbed.prad *
                                                               2)
            self.ref_SCC_angle = Estimationconfig["ref_SCC_angle"]

            self.ref_xpos = self.ref_SCC_distancepix * np.cos(
                np.radians(self.ref_SCC_angle))
            self.ref_ypos = self.ref_SCC_distancepix * np.sin(
                np.radians(self.ref_SCC_angle))

            # newsize = 2*np.floor((1.1*(np.max([self.ref_xpos, self.ref_ypos])+
            # self.ref_SCC_radiuspix))).astype('int')
            # TODO DO THAT A BIT BETTER. newsize must be dependent on ref dist and angle
            newsize = 8 * testbed.prad

            scc_ref = np.roll(phase_ampl.roundpupil(newsize,
                                                    self.ref_SCC_radiuspix),
                              (np.around(self.ref_xpos).astype('int'),
                               np.around(self.ref_ypos).astype('int')),
                              axis=(0, 1))

            # creating a copy of the testbed with everything is identical 
            # except the coronagraph Lyot.Only the coronagraph is really duplicated to save 
            # memory space, the rest of the items are just the same object.

            list_os = []
            list_os_names = []
            for osname in testbed.subsystems:
                if isinstance(vars(testbed)[osname], OptSy.coronagraph):
                    the_corono = copy.deepcopy(vars(testbed)[osname])
                    the_corono.oversizelyot = True
                    the_corono.lyot_pup.pup = proc.crop_or_pad_image(
                        the_corono.lyot_pup.pup, newsize) + scc_ref
                    self.lyotrad = the_corono.lyotrad
                    if the_corono.perfect_coro == True:
                        # do a propagation once with self.perfect_Lyot_pupil = 0 to
                        # measure the Lyot pupil that will be removed after
                        the_corono.perfect_Lyot_pupil = 0
                        the_corono.perfect_Lyot_pupil = the_corono.EF_through(
                            entrance_EF=the_corono.clearpup.EF_through())

                    the_corono.string_os += "SCC"

                    list_os.append(the_corono)
                    list_os_names.append(osname + "SCC")
                else:
                    list_os.append(vars(testbed)[osname])
                    list_os_names.append(osname)

            self.testbed_sccmode = OptSy.Testbed(list_os, list_os_names)

            # we measure and save all the quantities we need
            # to exctract the I- peak in the FFT of the SCC FP
            self.posx_I_peak = self.ref_xpos / testbed.Science_sampling * testbed.prad / self.lyotrad
            self.posy_I_peak = self.ref_ypos / testbed.Science_sampling * testbed.prad / self.lyotrad
            self.ray_I_peak = np.round(
                (self.ref_SCC_radiuspix + self.lyotrad) /
                testbed.Science_sampling * testbed.prad /
                self.lyotrad).astype("int")

            self.I_peak_mask = phase_ampl.roundpupil(2 * self.ray_I_peak,
                                                     self.ray_I_peak)
            self.mask_dh_scc = gaussian_filter(
                mask_dh.creatingMaskDH(testbed.dimScience,
                                       testbed.Science_sampling), 1)

            self.string_mask = mask_dh.string_mask

        else:
            raise Exception("This estimation algorithm is not yet implemented")

    def estimate(self,
                 testbed: OptSy.Testbed,
                 entrance_EF=1.,
                 voltage_vector=0.,
                 wavelength=None,
                 perfect_estimation=False,
                 save_all_planes_to_fits=False,
                 dir_save_all_planes=None,
                 **kwargs):
        """ --------------------------------------------------
        Run an estimation from a testbed, with a given input wavefront
        and a state of the DMs
        
        AUTHOR : Johan Mazoyer
        Version with SCC Added on 22 Sept 2021


        Parameters
        ----------
        testbed :  Optical_System.Testbed 
                Testbed object which describe your testbed

        entrance_EF :    complex float or 2D array, default 1.
            initial EF field

        voltage_vector : 1D float array
            vector of voltages vectors for each DMs

        wavelength  :  float default None,
            wavelength of the estimation in m
        
        perfect_estimation: bool, default = False. 
                    if true This is equivalent to have self.technique = "perfect" 
                    but even if we are using another technique, we sometimes 
                    need a perfect estimation and it avoid re-initialization of 
                    the estimation.
        
        save_all_planes_to_fits: Bool, default False.
            if True, save all planes to fits for debugging purposes to dir_save_all_planes
            This can generate a lot of fits especially if in a loop so the code force you
            to define a repository.

        dir_save_all_planes : path, default None. 
                            directory to save all plane
                            in fits if save_all_planes_to_fits = True
                                            
        Returns
        ------
        estimation : 2D array 
                array of size [self.dimEstim,self.dimEstim]
                estimation of the Electrical field



        -------------------------------------------------- """

        if isinstance(entrance_EF, (float, int)):
            pass
        elif entrance_EF.shape == testbed.wav_vec.shape:
            entrance_EF = entrance_EF[testbed.wav_vec.tolist().index(
                wavelength)]
        elif entrance_EF.shape == (testbed.dim_overpad_pupil,
                                   testbed.dim_overpad_pupil):
            pass
        elif entrance_EF.shape == (testbed.nb_wav, testbed.dim_overpad_pupil,
                                   testbed.dim_overpad_pupil):
            entrance_EF = entrance_EF[testbed.wav_vec.tolist().index(
                wavelength)]
        else:
            raise Exception(
                """"entrance_EFs must be scalar (same for all WL), or a self.nb_wav scalars or a
                        2D array of size (self.dim_overpad_pupil, self.dim_overpad_pupil) 
                        or a 3D array of size
                        (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil)"""
            )

        if (self.technique == "perfect") or (perfect_estimation is True):
            # If polychromatic, assume a perfect estimation at one wavelength

            resultatestimation = testbed.todetector(
                entrance_EF=entrance_EF,
                voltage_vector=voltage_vector,
                wavelength=wavelength,
                save_all_planes_to_fits=save_all_planes_to_fits,
                dir_save_all_planes=dir_save_all_planes,
                **kwargs)
            
            if save_all_planes_to_fits == True:
                # save PP plane before this subsystem
                name_plane = 'Estimation_+'+self.technique + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                            proc.resampling(resultatestimation, self.dimEstim))

            return proc.resampling(resultatestimation, self.dimEstim)

        elif self.technique in ["pairwise", "pw"]:
            Difference = wsc.createdifference(entrance_EF,
                                              testbed,
                                              self.posprobes,
                                              self.dimEstim,
                                              self.amplitudePW,
                                              voltage_vector=voltage_vector,
                                              wavelength=wavelength,
                                              save_all_planes_to_fits=save_all_planes_to_fits,
                                              dir_save_all_planes=dir_save_all_planes,
                                              **kwargs)
            
            if save_all_planes_to_fits == True:
                # save PP plane before this subsystem
                name_plane = 'Estimation_+'+self.technique + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                            proc.resampling(wsc.FP_PWestimate(Difference, self.PWMatrix), self.dimEstim))

            return wsc.FP_PWestimate(Difference, self.PWMatrix)

        elif self.technique == 'coffee':
            return np.zeros((self.dimEstim, self.dimEstim))

        elif self.technique == 'scc':

            # fp_nonscc = testbed.todetector_Intensity(
            #     entrance_EF=entrance_EF,
            #     wavelengths=wavelength,
            #     voltage_vector=voltage_vector,
            #     save_all_planes_to_fits=True,
            #     dir_save_all_planes='/Users/jmazoyer/Desktop/toto/',
            #     **kwargs)

            fp_scc = self.testbed_sccmode.todetector_Intensity(
                entrance_EF=entrance_EF,
                wavelengths=wavelength,
                voltage_vector=voltage_vector,
                save_all_planes_to_fits=save_all_planes_to_fits,
                dir_save_all_planes=dir_save_all_planes,
                **kwargs)
            
            if save_all_planes_to_fits == True:
                # save PP plane before this subsystem
                name_plane = 'Estimation_+'+self.technique + '_wl{}'.format(
                    int(wavelength * 1e9))
                useful.save_plane_in_fits(dir_save_all_planes, name_plane,
                                            wsc.extractI_peak(fp_scc, self))

            return wsc.extractI_peak(fp_scc, self)

        else:
            raise Exception("This estimation algorithm is not yet implemented")
