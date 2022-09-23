# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import numpy as np
from astropy.io import fits

from Asterix.utils import resizing
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed
import Asterix.wfsc.wf_sensing_functions as wfs


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


    -------------------------------------------------- """

    def __init__(self,
                 Estimationconfig,
                 testbed: Testbed,
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

        testbed :  OpticalSystem.Testbed
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

        if isinstance(testbed, OpticalSystem) == False:
            raise Exception("testbed must be an OpticalSystem object")

        self.technique = Estimationconfig["estimation"].lower()

        self.Estim_sampling = testbed.Science_sampling / Estimationconfig["Estim_bin_factor"]

        if self.Estim_sampling < 3:
            raise Exception("Estim_sampling must be > 3, please decrease Estim_bin_factor parameter")

        #image size after binning. This is the size of the estimation !
        # we round and make it so we're always even size and slightly smaller than the ideal size
        self.dimEstim = int(
            np.floor(self.Estim_sampling / testbed.Science_sampling * testbed.dimScience / 2) * 2)

        if self.technique == "perfect":
            self.is_focal_plane = True
            self.is_complex = True

        elif self.technique in ["pairwise", "pw"]:
            self.is_focal_plane = True
            self.is_complex = True

            self.amplitudePW = Estimationconfig["amplitudePW"]
            self.posprobes = list(Estimationconfig["posprobes"])
            cutsvdPW = Estimationconfig["cut"]

            if hasattr(testbed, 'name_DM_to_probe_in_PW'):
                if testbed.name_DM_to_probe_in_PW not in testbed.name_of_DMs:
                    raise Exception("Cannot use this DM for PW, this testbed has no DM named " +
                                    testbed.name_DM_to_probe_in_PW)
            # If name_DM_to_probe_in_PW is not set,
            # automatically check which DM to use to probe in this case
            # this is only done once.
            if len(testbed.name_of_DMs) == 0:
                raise Exception("you need at least one activated DM to do PW")
            #If only one DM, we use this one, independenlty of its position
            elif len(testbed.name_of_DMs) == 1:
                testbed.name_DM_to_probe_in_PW = testbed.name_of_DMs[0]
            else:
                #If several DMs we check if there is at least one in PP
                number_DMs_in_PP = 0
                for DM_name in testbed.name_of_DMs:
                    DM = vars(testbed)[DM_name]  # type: DeformableMirror
                    if DM.z_position == 0.:
                        number_DMs_in_PP += 1
                        testbed.name_DM_to_probe_in_PW = DM_name

                #If there are several DMs in PP, error, you need to set name_DM_to_probe_in_PW
                if number_DMs_in_PP > 1:
                    raise Exception(
                        "You have several DM in PP, choose one for the PW probes using testbed.name_DM_to_probe_in_PW"
                    )
                #Several DMS, none in PP, error, you need to set name_DM_to_probe_in_PW
                if number_DMs_in_PP == 0:
                    raise Exception(
                        "You have several DMs none in PP, choose one for the PW probes using testbed.name_DM_to_probe_in_PW"
                    )

            string_dims_PWMatrix = testbed.name_DM_to_probe_in_PW + "Prob" + "_".join(map(
                str, self.posprobes)) + "_PWampl" + str(int(self.amplitudePW)) + "_cut" + str(
                    int(cutsvdPW // 1000)) + "k_dimEstim" + str(self.dimEstim) + testbed.string_os

            ####Calculating and Saving PW matrix
            filePW = "MatPW_" + string_dims_PWMatrix
            if os.path.exists(matrix_dir + filePW + ".fits") == True:
                print("The matrix " + filePW + " already exists")
                self.PWMatrix = fits.getdata(matrix_dir + filePW + ".fits")
            else:
                print("Saving " + filePW + " ...")
                self.PWMatrix, showSVD = wfs.create_pw_matrix(testbed, self.amplitudePW, self.posprobes,
                                                              self.dimEstim, cutsvdPW, testbed.wavelength_0)
                fits.writeto(matrix_dir + filePW + ".fits", np.array(self.PWMatrix))
                visuPWMap = "EigenPW_" + string_dims_PWMatrix
                fits.writeto(matrix_dir + visuPWMap + ".fits", np.array(showSVD[1]))

            # Saving PW matrix in Labview directory
            if save_for_bench == True:
                if not os.path.exists(realtestbed_dir):
                    print("Creating directory " + realtestbed_dir + " ...")
                    os.makedirs(realtestbed_dir)

                probes = np.zeros((len(self.posprobes), testbed.DM3.number_act), dtype=np.float32)
                vectorPW = np.zeros((2, self.dimEstim * self.dimEstim * len(self.posprobes)),
                                    dtype=np.float32)

                for i in np.arange(len(self.posprobes)):
                    # TODO WTH is the hardcoded 17. @Raphael @Axel
                    probes[i, self.posprobes[i]] = self.amplitudePW / 17
                    vectorPW[0, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                             self.dimEstim] = self.PWMatrix[:, 0, i].flatten()
                    vectorPW[1, i * self.dimEstim * self.dimEstim:(i + 1) * self.dimEstim *
                             self.dimEstim] = self.PWMatrix[:, 1, i].flatten()
                namepwmatrix = '_PW_' + testbed.name_DM_to_probe_in_PW
                fits.writeto(realtestbed_dir + "Probes" + namepwmatrix + ".fits", probes, overwrite=True)
                fits.writeto(realtestbed_dir + "Matr_mult_estim" + namepwmatrix + ".fits",
                             vectorPW,
                             overwrite=True)
        
        elif self.technique == 'coffee':
            pass

        else:
            raise Exception("This estimation algorithm is not yet implemented")

    def estimate(self,
                 testbed: Testbed,
                 entrance_EF=1.,
                 voltage_vector=0.,
                 wavelength=None,
                 perfect_estimation=False,
                 **kwargs):
        """ --------------------------------------------------
        Run an estimation from a testbed, with a given input wavefront
        and a state of the DMs
        
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        testbed :  OpticalSystem.Testbed
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
                                            
        Returns
        ------
        estimation : 2D array 
                array of size [self.dimEstim,self.dimEstim]
                estimation of the Electrical field



        -------------------------------------------------- """

        if isinstance(entrance_EF, (float, int)):
            pass
        elif entrance_EF.shape == testbed.wav_vec.shape:
            entrance_EF = entrance_EF[testbed.wav_vec.tolist().index(wavelength)]
        elif entrance_EF.shape == (testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
            pass
        elif entrance_EF.shape == (testbed.nb_wav, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
            entrance_EF = entrance_EF[testbed.wav_vec.tolist().index(wavelength)]
        else:
            raise Exception(""""entrance_EFs must be scalar (same for all WL), or a self.nb_wav scalars or a
                        2D array of size (self.dim_overpad_pupil, self.dim_overpad_pupil) or a 3D array of size
                        (self.nb_wav, self.dim_overpad_pupil, self.dim_overpad_pupil)""")

        if (self.technique == "perfect") or (perfect_estimation is True):
            # If polychromatic, assume a perfect estimation at one wavelength

            resultatestimation = testbed.todetector(entrance_EF=entrance_EF,
                                                    voltage_vector=voltage_vector,
                                                    wavelength=wavelength,
                                                    **kwargs)

            return resizing(resultatestimation, self.dimEstim)

        elif self.technique in ["pairwise", "pw"]:
            Difference = wfs.simulate_pw_difference(entrance_EF,
                                                    testbed,
                                                    self.posprobes,
                                                    self.dimEstim,
                                                    self.amplitudePW,
                                                    voltage_vector=voltage_vector,
                                                    wavelength=wavelength,
                                                    **kwargs)

            return wfs.calculate_pw_estimate(Difference, self.PWMatrix)

        elif self.technique == 'coffee':
            return np.zeros((self.dimEstim, self.dimEstim))

        else:
            raise Exception("This estimation algorithm is not yet implemented")