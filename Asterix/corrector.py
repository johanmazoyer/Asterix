# pylint: disable=invalid-name

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time

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

        if isinstance(testbed, OptSy.Optical_System) == False:
            raise Exception("testbed must be an Optical_System objet")

        self.correction_algorithm = Correctionconfig[
            "correction_algorithm"].lower()

        if self.correction_algorithm == "efc" or self.correction_algorithm == "em" or self.correction_algorithm == "steepest":

            self.amplitudeEFC = Correctionconfig["amplitudeEFC"]
            self.regularization = Correctionconfig["regularization"]

            self.MaskEstim = MaskDH.creatingMaskDH(estimator.dimEstim,
                                              estimator.Estim_sampling)


            # I think currently the name of the actuator inside the pupil is
            # used as the basis, which is not ideal at all, these are 2 different things.
            self.DM1_otherbasis = Correctionconfig["DM1_otherbasis"]
            self.DM3_otherbasis = Correctionconfig["DM3_otherbasis"]
            # DM3
            if self.DM3_otherbasis == True:
                testbed.DM1.WhichInPupil = np.arange(testbed.DM3.number_act)
                self.DM3_basis = fits.getdata(realtestbed_dir +
                                              "Map_modes_DM3_foc.fits")
            else:
                self.DM3_basis = 0



            fileDirectMatrix = "DirectMatrix_EFCampl" + str(
                self.amplitudeEFC)+ MaskDH.string_mask + testbed.string_os

            if os.path.exists(matrix_dir + fileDirectMatrix +
                                                        ".fits"):
                print("The matrix " + fileDirectMatrix + " already exists")
                self.Gmatrix = fits.getdata(matrix_dir + fileDirectMatrix +
                                            ".fits")

            else:
                # Creating EFC Interaction Matrix if does not exist

                print("Saving " + fileDirectMatrix + " ...")
                start_time = time.time()
                if testbed.DM1.active == True:
                    DM_pushact_inpup = np.concatenate(
                        (testbed.DM3.DM_pushact_inpup, testbed.DM1.DM_pushact_inpup))
                    DM_WhichInPupil = np.concatenate(
                        (testbed.DM3.WhichInPupil,
                         testbed.DM3.number_act + testbed.DM1.WhichInPupil))
                else:
                    DM_pushact_inpup = testbed.DM3.DM_pushact_inpup
                    DM_WhichInPupil = testbed.DM3.WhichInPupil
                print("time for concat for "+testbed.string_os, time.time() - start_time)

                start_time = time.time()
                self.Gmatrix = wsc.creatingCorrectionmatrix(
                    testbed.entrancepupil.pup,
                    testbed,
                    estimator.dimEstim,
                    DM_pushact_inpup * self.amplitudeEFC * 2 * np.pi * 1e-9 /
                    testbed.wavelength_0,
                    self.MaskEstim,
                    DM_WhichInPupil,
                    otherbasis=self.DM3_otherbasis,
                    basisDM3=self.DM3_basis,
                )

                fits.writeto(matrix_dir + fileDirectMatrix + ".fits",
                             self.Gmatrix)
                print("time for direct matrix "+testbed.string_os, time.time() - start_time)



                if self.correction_algorithm == "em" or self.correction_algorithm == "steepest":

                    self.G = np.zeros(
                        (int(np.sum(self.MaskEstim)), self.Gmatrix.shape[1]),
                        dtype=complex)
                    self.G = (
                        self.Gmatrix[0:int(self.Gmatrix.shape[0] / 2), :] +
                        1j * self.Gmatrix[int(self.Gmatrix.shape[0] / 2):, :])
                    transposecomplexG = np.transpose(np.conjugate(self.G))
                    self.M0 = np.real(np.dot(transposecomplexG, self.G))

                if save_for_bench == True:

                    #### Not sure what it does... Is this still useful ?
                    # I modified it with the new mask parameters
                    # TODO talk with raphael
                    if MaskDH.DH_shape == "square":
                        print(
                            "TO SET ON LABVIEW: ",
                            str(estimator.dimEstim / 2 + np.array(
                                np.fft.fftshift(MaskDH.corner_pos *
                                                estimator.Estim_sampling))))

                    Nbmodes = Correctionconfig["Nbmodes"]
                    SVD, _ , invertGDH = wsc.invertSVD(
                        self.Gmatrix,
                        Nbmodes,
                        goal="c",
                        regul=self.regularization,
                        otherbasis=self.DM3_otherbasis,
                        basisDM3=self.DM3_basis)


                    plt.clf()
                    plt.plot(SVD, "r.")
                    plt.yscale("log")

                    figSVDEFC = matrix_dir + "SVD_Modes"+str(Nbmodes)+'_' + fileDirectMatrix + ".png"

                    plt.savefig(figSVDEFC)

                    EFCmatrix_DM3 = np.zeros(
                        (invertGDH.shape[1], testbed.DM3.number_act),
                        dtype=np.float32)
                    for i in np.arange(len(testbed.DM3.WhichInPupil)):
                        EFCmatrix_DM3[:,
                                      testbed.DM3.WhichInPupil[i]] = invertGDH[
                                          i, :]
                    fits.writeto(realtestbed_dir +
                                 "Matrix_control_EFC_DM3_default.fits",
                                 EFCmatrix_DM3,
                                 overwrite=True)
                    if testbed.DM1.active:
                        EFCmatrix_DM1 = np.zeros(
                            (invertGDH.shape[1], testbed.DM1.number_act),
                            dtype=np.float32)
                        for i in np.arange(len(testbed.DM1.WhichInPupil)):
                            EFCmatrix_DM1[:, testbed.DM1.
                                          WhichInPupil[i]] = invertGDH[
                                              i +
                                              len(testbed.DM3.WhichInPupil), :]
                        fits.writeto(realtestbed_dir +
                                     "Matrix_control_EFC_DM1_default.fits",
                                     EFCmatrix_DM1,
                                     overwrite=True)

        else:
            raise Exception("This correction algorithm is not yet implemented")


        ######################
        # Preparation of the correction loop
        ######################
        # in the initialization we have not inverted the matrix just yet so
        self.previousmode = np.nan

    def toDM_voltage(self, testbed, estimate, mode, **kwargs):
        """ --------------------------------------------------
        Run an correction from a testbed, and return the DM voltage for one or 2 DMS

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if self.correction_algorithm == "efc":
            _, _, invertGDH = wsc.invertSVD(
                    self.Gmatrix,
                    mode,
                    goal="c",
                    visu=False,
                    regul=self.regularization,
                    otherbasis=self.DM3_otherbasis,
                    basisDM3=self.DM3_basis,
                )

            if testbed.DM1.active == True:
                return wsc.solutionEFC(
                    self.MaskEstim, estimate, invertGDH,
                    np.concatenate(
                        (testbed.DM3.WhichInPupil,
                         testbed.DM3.number_act + testbed.DM1.WhichInPupil)),
                    testbed.DM3.number_act + testbed.DM1.number_act)
                # TODO Concatenate should be done in the THD2 structure
            else:
                return wsc.solutionEFC(self.MaskEstim, estimate,
                                            invertGDH, testbed.DM3.WhichInPupil,
                                            testbed.DM3.number_act)


        if self.correction_algorithm == "em":
            if mode != self.previousmode:
                self.previousmode = mode
                _, _, invertM0 = wsc.invertSVD(
                    self.M0,
                    mode,
                    goal="c",
                    visu=False,
                    regul=self.regularization,
                    otherbasis=self.DM3_otherbasis,
                    basisDM3=self.DM3_basis)

                # TODO Concatenate should be done in the THD2 structure
            if testbed.DM1.active == True:
                return wsc.solutionEM(
                    self.MaskEstim, estimate, invertM0, self.G,
                    np.concatenate(
                        (testbed.DM3.WhichInPupil,
                         testbed.DM3.number_act + testbed.DM1.WhichInPupil)),
                    testbed.DM3.number_act + testbed.DM1.number_act)
                # TODO Concatenate should be done in the THD2 structure
            else:
                return wsc.solutionEM(self.MaskEstim, estimate,
                                           invertM0, self.G, testbed.DM3.WhichInPupil,
                                           testbed.DM3.number_act)

        if self.correction_algorithm == "steepest":
            if testbed.DM1.active == True:
                return wsc.solutionSteepest(
                    self.MaskEstim, estimate, self.M0, self.G,
                    np.concatenate(
                        (testbed.DM3.WhichInPupil,
                         testbed.DM3.number_act + testbed.DM1.WhichInPupil)),
                    testbed.DM3.number_act + testbed.DM1.number_act)
                # Concatenate should be done in the THD2 structure
            else:
                return wsc.solutionSteepest(self.MaskEstim, estimate,
                                                 self.M0, self.G, testbed.DM3.WhichInPupil,
                                                 testbed.DM3.number_act)
        else:
            raise Exception("This correction algorithm is not yet implemented")


    def cropDHInterractionMatrix(self, FullInterractionMatrix):
        """ --------------------------------------------------
        Crop the  Interraction Matrix. to the mask size


        Parameters:
        ----------
        FullInterractionMatrix: Interraction matrix over the full focal plane


        Return: DHInterractionMatrix: matrix only inside the DH
        ------

        -------------------------------------------------- """
        size_full_matrix = FullInterractionMatrix.shape[0]

        size_DH_matrix = 2 * int(np.sum(self.MaskEstim))
        where_mask_flatten = np.where(self.MaskEstim.flatten() == 1.)
        DHInterractionMatrix = np.zeros(
            (size_DH_matrix, FullInterractionMatrix.shape[1]), dtype=float)

        for i in range(FullInterractionMatrix.shape[1]):
            DHInterractionMatrix[:int(
                size_DH_matrix /
                2), i] = FullInterractionMatrix[:int(size_full_matrix / 2),
                                                i][where_mask_flatten]
            DHInterractionMatrix[int(size_DH_matrix / 2):,
                                 i] = FullInterractionMatrix[
                                     int(size_full_matrix / 2):,
                                     i][where_mask_flatten]

        return DHInterractionMatrix
