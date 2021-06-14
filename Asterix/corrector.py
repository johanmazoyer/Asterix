# pylint: disable=invalid-name

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time

import Asterix.fits_functions as useful
import Asterix.processing_functions as proc
import Asterix.Optical_System_functions as OptSy

import Asterix.WSC_functions as wsc


class Corrector:
    """ --------------------------------------------------
    Corrector Class allows you to define a corrector for one of 2 DM with
    different algorithms.

    Corrector is a class which takes as parameter:
        - the testbed structure
        - the correction parameters

        It must contains 2 functions at least:
        - an initialization (e.g. Jacobian matrix) Corrector.__init__
        The initialization will probaby require previous initialization of
        the testbed and of the estimator

        - an correction function itself with parameters
                - the estimation as a 2D array, potentially 3D for polychromatic correction
        DMVoltage = Corrector.toDM_voltage(estimation)
        It returns the DM Voltage. In all generality, it can one or 2 DMs. Depending on the testbed



    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 Correctionconfig,
                 testbed,
                 MaskDH,
                 estimator,
                 initial_DM_voltage = 0.,
                 matrix_dir='',
                 save_for_bench=False,
                 realtestbed_dir=''):
        """ --------------------------------------------------
        Initialize the corrector.
        This is where you define the EFC matrix
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
        if not os.path.exists(matrix_dir):
            print("Creating directory " + matrix_dir + " ...")
            os.makedirs(matrix_dir)

        if isinstance(testbed, OptSy.Optical_System) == False:
            raise Exception("testbed must be an Optical_System objet")

        basis_type = Correctionconfig["DM_basis"].lower()
        self.total_number_modes = 0

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]
            DM.basis = DM.create_DM_basis(basis_type=basis_type)
            DM.basis_size = DM.basis.shape[0]
            self.total_number_modes += DM.basis_size
            DM.basis_type = basis_type

        self.correction_algorithm = Correctionconfig[
            "correction_algorithm"].lower()

        self.amplitudeEFC = Correctionconfig["amplitudeEFC"]
        self.regularization = Correctionconfig["regularization"]
        self.MaskEstim = MaskDH.creatingMaskDH(estimator.dimEstim,
                                                   estimator.Estim_sampling)

        self.matrix_dir = matrix_dir

        self.update_matrices(testbed, estimator)

        if self.correction_algorithm == "efc" and save_for_bench == True:
            if not os.path.exists(realtestbed_dir):
                print("Creating directory " + realtestbed_dir + " ...")
                os.makedirs(realtestbed_dir)

            Nbmodes = Correctionconfig["Nbmodes_OnTestbed"]
            _, _, invertGDH = wsc.invertSVD(self.Gmatrix,
                                            Nbmodes,
                                            goal="c",
                                            regul=self.regularization,
                                            visu=True,
                                            filename_visu=realtestbed_dir +
                                            "SVD_Modes" + str(Nbmodes) +
                                            ".png")

            if testbed.DM1.active:
                invertGDH_DM1 = invertGDH[:testbed.DM1.basis_size]

                EFCmatrix_DM1 = np.transpose(
                    np.dot(np.transpose(testbed.DM1.basis), invertGDH_DM1))
                fits.writeto(realtestbed_dir +
                                "Matrix_control_EFC_DM1.fits",
                                EFCmatrix_DM1.astype(np.float32),
                                overwrite=True)
                if testbed.DM3.active:
                    invertGDH_DM3 = invertGDH[testbed.DM1.basis_size:]
                    EFCmatrix_DM3 = np.transpose(
                        np.dot(np.transpose(testbed.DM3.basis),
                                invertGDH_DM3))
                    fits.writeto(realtestbed_dir +
                                    "Matrix_control_EFC_DM3.fits",
                                    EFCmatrix_DM3.astype(np.float32),
                                    overwrite=True)
            elif testbed.DM3.active:
                invertGDH_DM3 = invertGDH
                EFCmatrix_DM3 = np.transpose(
                    np.dot(np.transpose(testbed.DM3.basis), invertGDH_DM3))
                fits.writeto(realtestbed_dir +
                                "Matrix_control_EFC_DM3.fits",
                                EFCmatrix_DM3.astype(np.float32),
                                overwrite=True)
            else:
                raise Exception("No active DMs")

            fits.writeto(realtestbed_dir + "DH_mask.fits",
                            self.MaskEstim.astype(np.float32),
                            overwrite=True)
            fits.writeto(realtestbed_dir + "DH_mask_where_x_y.fits",
                            np.array(np.where(self.MaskEstim == 1)).astype(
                                np.float32),
                            overwrite=True)



        ## Adding error on the DM model. Now that the matrix is measured, we can
        # introduce a small movememnt on one DM or the other. By changeing DM_pushact
        # we are changeing the position of the actuator and therre the phase of the
        # DM for a given voltage when using DM.voltage_to_phase

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]
            if DM.misregistration:
                print(DM_name + " Misregistration!")
                DM.DM_pushact = DM.creatingpushact(DM.DMconfig)

        ######################
        # Preparation of the correction loop
        ######################
        # in the initialization we have not inverted the matrix just yet so
        self.previousmode = np.nan

    def update_matrices(self, testbed, estimator, initial_DM_voltage = 0.,input_wavefront = 1.):
        if self.correction_algorithm in ["efc", "em", "steepest", "sm"]:

            self.G = 0
            self.Gmatrix = 0
            self.M0 = 0

            start_time = time.time()
            interMat = wsc.creatingInterractionmatrix(testbed,
                                                      estimator.dimEstim,
                                                      self.amplitudeEFC,
                                                      self.matrix_dir, initial_DM_voltage = initial_DM_voltage)

            print("time for direct matrix " + testbed.string_os,
                  time.time() - start_time)
            print("")

            self.Gmatrix = wsc.cropDHInterractionMatrix(
                interMat, self.MaskEstim)

            if self.correction_algorithm in ["em", "steepest", "sm"]:

                self.G = np.zeros(
                    (int(np.sum(self.MaskEstim)), self.Gmatrix.shape[1]),
                    dtype=complex)
                self.G = (
                    self.Gmatrix[0:int(self.Gmatrix.shape[0] / 2), :] +
                    1j * self.Gmatrix[int(self.Gmatrix.shape[0] / 2):, :])
                transposecomplexG = np.transpose(np.conjugate(self.G))
                self.M0 = np.real(np.dot(transposecomplexG, self.G))
                self.Gmatrix = 0.
        else:
            raise Exception("This correction algorithm is not yet implemented")

    def toDM_voltage(self,
                     testbed,
                     estimate,
                     mode,
                     ActualCurrentContrast=1,
                     gain=0.,
                     **kwargs):
        """ --------------------------------------------------
        Run an correction from a testbed, and return the DM voltage for one or 2 DMS

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if self.correction_algorithm == "efc":
            if mode != self.previousmode:
                self.previousmode = mode
                _, _, self.invertGDH = wsc.invertSVD(self.Gmatrix,
                                                     mode,
                                                     goal="c",
                                                     visu=False,
                                                     regul=self.regularization)

            return -gain * self.amplitudeEFC * wsc.solutionEFC(
                self.MaskEstim, estimate, self.invertGDH, testbed)

        if self.correction_algorithm == "sm":
            # see Mazoyer et al 2018 ACAD-OSM I paper to understand algorithm
            if np.isnan(self.previousmode):
                # This is the first time
                self.previousmode = 0
                self.last_best_alpha = 0.1
                self.expected_gain_in_contrast = 0.4
                self.last_best_contrast = ActualCurrentContrast
                self.times_we_lowered_gain = 0
                self.count_since_last_best = 0

            if self.last_best_contrast < ActualCurrentContrast:
                # problem: the algorithm did not actully improved contrast last iteration
                # it's ok if it's only once, but we
                self.count_since_last_best += 1
            else:
                self.count_since_last_best = 0
                self.last_best_contrast = ActualCurrentContrast

            if self.count_since_last_best > 5 or ActualCurrentContrast > 2 * self.last_best_contrast:
                self.times_we_lowered_gain += 1
                self.expected_gain_in_contrast = 1 - (1 - self.expected_gain_in_contrast)/3
                self.count_since_last_best = 0
                print("we do not improve contrast anymore, we change the gain to {:f}".format(self.expected_gain_in_contrast))



            if self.times_we_lowered_gain == 3:
                #it's been too long we have not increased
                # or we're so far off linearity that SM is actually heavily degrading contrast
                # It's time to stop !
                return np.nan

            DesiredContrast = self.expected_gain_in_contrast * ActualCurrentContrast

            solutionSM, self.last_best_alpha = wsc.solutionSM(
                self.MaskEstim, testbed, estimate, self.M0, self.G,
                DesiredContrast, self.last_best_alpha)

            return -self.amplitudeEFC * solutionSM

        if self.correction_algorithm == "em":

            if mode != self.previousmode:
                self.previousmode = mode
                _, _, self.invertM0 = wsc.invertSVD(self.M0,
                                                    mode,
                                                    goal="c",
                                                    visu=False,
                                                    regul=self.regularization)

            return -gain * self.amplitudeEFC * wsc.solutionEM(
                self.MaskEstim, estimate, self.invertM0, self.G, testbed)

        if self.correction_algorithm == "steepest":

            return -gain * self.amplitudeEFC * wsc.solutionSteepest(
                self.MaskEstim, estimate, self.M0, self.G, testbed)
        else:
            raise Exception("This correction algorithm is not yet implemented")


