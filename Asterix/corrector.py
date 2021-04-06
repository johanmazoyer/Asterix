# pylint: disable=invalid-name

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
            self.Nbmodes = Correctionconfig["Nbmodes"]
            self.regularization = Correctionconfig["regularization"]

            self.DM1_otherbasis = Correctionconfig["DM1_otherbasis"]
            self.DM3_otherbasis = Correctionconfig["DM3_otherbasis"]

            MaskEstim = MaskDH.creatingMaskDH(estimator.dimEstim,
                                              estimator.Estim_sampling)
            string_dhshape = MaskDH.tostring()
            # I think currently the name of the actuator inside the pupil is
            # used as the basis, which is not ideal at all, these are 2 different things.

            # DM1
            if self.DM1_otherbasis == True:
                testbed.DM1.WhichInPupil = np.arange(testbed.DM1.number_act)

            # DM3
            if self.DM3_otherbasis == True:
                testbed.DM1.WhichInPupil = np.arange(testbed.DM3.number_act)

            #useful string
            string_dims_EFCMatrix = '_EFCampl' + str(
                self.amplitudeEFC) + "_modes" + str(
                    self.Nbmodes) + testbed.string_os

            # Creating EFC control matrix
            fileEFCMatrix = "MatrixEFC" + string_dhshape + string_dims_EFCMatrix

            if os.path.exists(matrix_dir + fileEFCMatrix + ".fits") == True:
                print("The matrix " + fileEFCMatrix + " already exists")
                invertGDH = fits.getdata(matrix_dir + fileEFCMatrix + ".fits")
            else:

                # Actuator basis or another one?
                # TODO lot of cleaning to do on the "other basis ??"
                if self.DM3_otherbasis == True:
                    self.DM3_basis = fits.getdata(realtestbed_dir +
                                                  "Map_modes_DM3_foc.fits")
                else:
                    self.DM3_basis = 0

                # Creating EFC Interaction matrix
                fileDirectMatrix = "DirectMatrix" + string_dhshape + string_dims_EFCMatrix

                if os.path.exists(matrix_dir + fileDirectMatrix +
                                  ".fits") == True:
                    print("The matrix " + fileDirectMatrix + " already exists")
                    Gmatrix = fits.getdata(matrix_dir + fileDirectMatrix +
                                           ".fits")
                else:

                    # Creating EFC Interaction Matrix if does not exist
                    print("Saving " + fileDirectMatrix + " ...")

                    if testbed.DM1.active == True:
                        DM_pushact = np.concatenate(
                            (testbed.DM3.DM_pushact,
                             testbed.DM1.DM_pushact_inpup))
                        DM_WhichInPupil = np.concatenate(
                            (testbed.DM3.WhichInPupil, testbed.DM3.number_act +
                             testbed.DM1.WhichInPupil))
                    else:
                        DM_pushact = testbed.DM3.DM_pushact
                        DM_WhichInPupil = testbed.DM3.WhichInPupil

                    self.Gmatrix = wsc.creatingCorrectionmatrix(
                        testbed.entrancepupil.pup,
                        testbed,
                        estimator.dimEstim,
                        DM_pushact * self.amplitudeEFC * 2 * np.pi * 1e-9 /
                        testbed.wavelength_0,
                        MaskEstim,
                        DM_WhichInPupil,
                        otherbasis=self.DM3_otherbasis,
                        basisDM3=self.DM3_basis,
                    )

                    fits.writeto(matrix_dir + fileDirectMatrix + ".fits",
                                 self.Gmatrix)

                # Calculating and saving EFC Control Matrix
                print("Saving " + fileEFCMatrix + " ...")
                SVD, SVD_trunc, invertGDH = wsc.invertSVD(
                    Gmatrix,
                    self.Nbmodes,
                    goal="c",
                    regul=self.regularization,
                    otherbasis=self.DM3_otherbasis,
                    basisDM3=self.DM3_basis)
                fits.writeto(matrix_dir + fileEFCMatrix + ".fits", invertGDH)

                plt.clf()
                plt.plot(SVD, "r.")
                plt.yscale("log")

                figSVDEFC = matrix_dir + "invertSVDEFC_square" + string_dhshape + string_dims_EFCMatrix + ".png"

                plt.savefig(figSVDEFC)

                if self.correction_algorithm == "em" or self.correction_algorithm == "steepest":

                    self.G = np.zeros(
                        (int(np.sum(MaskEstim)), Gmatrix.shape[1]),
                        dtype=complex)
                    self.G = (Gmatrix[0:int(Gmatrix.shape[0] / 2), :] +
                              1j * Gmatrix[int(Gmatrix.shape[0] / 2):, :])
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

    def toDM_voltage(self, testbed, estimate, **kwargs):
        """ --------------------------------------------------
        Run an correction from a testbed, and return the DM voltage for one or 2 DMS

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if self.correction_algorithm == "efc":
            pass
        else:
            raise Exception("This correction algorithm is not yet implemented")
