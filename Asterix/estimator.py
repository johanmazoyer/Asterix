# pylint: disable=invalid-name

import os
import numpy as np
from astropy.io import fits

import Asterix.processing_functions as proc
import Asterix.Optical_System_functions as OptSy

import Asterix.WSC_functions as wsc


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
                 testbed,
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

        Parameters
        ----------
        Estimationconfig : general estimation parameters

        testbed : an Optical_System object which describe your testbed


        matrix_dir: path. save all the difficult to measure files here

        save_for_bench. bool default: false
                should we save for the real testbed in realtestbed_dir

        realtestbed_dir: path save all the files the real testbed need to
                            run your code


        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

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

            string_dims_PWMatrix = "actProb[" + "_".join(
                map(str, self.posprobes)) + "]PWampl" + str(
                    int(self.amplitudePW)) + "_cut" + str(int(
                        cutsvdPW // 1000)) + "k_dimEstim" + str(
                            self.dimEstim) + testbed.string_os

            ####Calculating and Saving PW matrix
            filePW = "MatrixPW_" + string_dims_PWMatrix
            if os.path.exists(matrix_dir + filePW + ".fits") == True:
                print("The matrix " + filePW + " already exists")
                self.PWVectorprobes = fits.getdata(matrix_dir + filePW +
                                                   ".fits")
            else:
                print("Saving " + filePW + " ...")
                self.PWVectorprobes, showsvd = wsc.createvectorprobes(
                    testbed, self.amplitudePW, self.posprobes, self.dimEstim,
                    cutsvdPW, testbed.wavelength_0)
                fits.writeto(matrix_dir + filePW + ".fits",
                             self.PWVectorprobes)

            visuPWMap = "EigenValPW_" + string_dims_PWMatrix

            if os.path.exists(matrix_dir + visuPWMap + ".fits") is False:
                print("Saving " + visuPWMap + " ...")
                fits.writeto(matrix_dir + visuPWMap + ".fits", showsvd[1])

            # Saving PW matrices in Labview directory
            if save_for_bench == True:
                probes = np.zeros(
                    (len(self.posprobes), testbed.DM3.number_act),
                    dtype=np.float32)
                vectorPW = np.zeros(
                    (2, self.dimEstim * self.dimEstim * len(self.posprobes)),
                    dtype=np.float32)

                for i in np.arange(len(self.posprobes)):
                    probes[i, self.posprobes[i]] = self.amplitudePW / 17
                    vectorPW[0, i * self.dimEstim * self.dimEstim:(i + 1) *
                             self.dimEstim *
                             self.dimEstim] = self.PWVectorprobes[:, 0,
                                                                  i].flatten()
                    vectorPW[1, i * self.dimEstim * self.dimEstim:(i + 1) *
                             self.dimEstim *
                             self.dimEstim] = self.PWVectorprobes[:, 1,
                                                                  i].flatten()
                fits.writeto(realtestbed_dir + "Probes_EFC_default.fits",
                             probes,
                             overwrite=True)
                fits.writeto(realtestbed_dir + "Matr_mult_estim_PW.fits",
                             vectorPW,
                             overwrite=True)
        elif self.technique == 'coffee':
            pass

        else:
            raise Exception("This estimation algorithm is not yet implemented")

    def estimate(self,
                 testbed,
                 entrance_EF=1.,
                 voltage_vector=0.,
                 photon_noise=False,
                 nb_photons=1e30,
                 perfect_estimation=False,
                 **kwargs):
        """ --------------------------------------------------
        Run an estimation from a testbed, with a given input wavefront
        and a state of the DMs


        Parameters
        ----------
        testbed:        a testbed element
        entrance_EF     default 0., float or 2D array can be complex, initial EF field
        DM1phase        default 0., float or 2D real array, phase on DM1
        DM3phase        default 0., float or 2D real array, phase on DM1
        wavelength      default None, float, wavelenght of the estimation
        photon_noise    default False, boolean,  If True, add photon noise.
        nb_photons      default 1e30, int Number of photons entering the pupil
        perfect_estimation default = False. if true This is equivalent to
                                            have self.technique = "perfect" but even
                                            if we are using another technique, we
                                            sometimes need a perfect estimation
                                            especially in EFC. if perfect_estimation

        Returns
        ------
        estimation : 2D array od size [self.dimEstim,self.dimEstim]
                    estimation of the Electrical field

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if (self.technique == "perfect") or (perfect_estimation is True):
            # If polychromatic, assume a perfect estimation at one wavelength

            resultatestimation = testbed.todetector(
                entrance_EF=entrance_EF,
                voltage_vector=voltage_vector,
                **kwargs)

            if photon_noise == True:
                resultatestimation = np.random.poisson(
                    resultatestimation * testbed.normPupto1 *
                    nb_photons) / (testbed.normPupto1 * nb_photons)

            return proc.resampling(resultatestimation, self.dimEstim)

        elif self.technique in ["pairwise", "pw"]:
            Difference = wsc.createdifference(
                entrance_EF,
                testbed,
                self.posprobes,
                self.dimEstim,
                self.amplitudePW,
                voltage_vector=voltage_vector,
                photon_noise=photon_noise,
                nb_photons=nb_photons,
                **kwargs)

            return wsc.FP_PWestimate(Difference, self.PWVectorprobes)

        elif self.technique == 'coffee':
            return np.zeros((self.dimEstim, self.dimEstim))

        else:
            raise Exception("This estimation algorithm is not yet implemented")
