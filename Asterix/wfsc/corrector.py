# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import time

import numpy as np
from astropy.io import fits

from Asterix.utils import invert_svd
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed
from Asterix.wfsc.estimator import Estimator
from Asterix.wfsc.thd_quick_invert import THD_quick_invert
import Asterix.wfsc.wf_control_functions as wfc


class Corrector:
    """ --------------------------------------------------
    Corrector Class allows you to define a corrector with
    different algorithms.

    Corrector is a class which takes as parameter:
        - the testbed structure
        - the correction parameters
        - the estimator

        It must contains 2 functions at least:
        - an initialization (e.g. Jacobian matrix) Corrector.__init__
        The initialization requires previous initialization of
        the testbed and of the estimator

        - an correction function itself with parameters
                - the estimation as a 2D array, potentially 3D for polychromatic correction
        DMVoltage = Corrector.toDM_voltage(estimation)
        It returns the DM Voltage. In all generality, it can one or 2 DMs. Depending on the testbed

    AUTHOR : Johan Mazoyer

    -------------------------------------------------- """

    def __init__(self,
                 Correctionconfig,
                 testbed: Testbed,
                 MaskDH,
                 estimator: Estimator,
                 matrix_dir=None,
                 save_for_bench=False,
                 realtestbed_dir=''):
        """ --------------------------------------------------
        Initialize the corrector.
        This is where you define the EFC matrix
        For all large files you should use a method of "save to fits" if
        it does not exist "load from fits" if it does, in matrix_dir

        Store in the structure only what you need for estimation. Everything not
        used in self.estimate shoud not be stored
        
        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        Correctionconfig : dict
                general correction parameters

        testbed :  OpticalSystem.Testbed
                Testbed object which describe your testbed

        MaskDH: 2d numpy array
                binary array of size [dimEstim, dimEstim] : dark hole mask

        estimator: Estimator
                an estimator object. This contains all information about the estimation

        matrix_dir: path default: None
                save all the difficult to measure files here

        save_for_bench: bool default: false
                should we save for the real testbed in realtestbed_dir

        realtestbed_dir: path 
                save all the files the real testbed need to run your code


        -------------------------------------------------- """
        if not os.path.exists(matrix_dir):
            print("Creating directory " + matrix_dir + " ...")
            os.makedirs(matrix_dir)

        if isinstance(testbed, OpticalSystem) == False:
            raise Exception("testbed must be an OpticalSystem object")

        basis_type = Correctionconfig["DM_basis"].lower()
        self.total_number_modes = 0

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]  # type: DeformableMirror
            DM.basis = DM.create_DM_basis(basis_type=basis_type)
            DM.basis_size = DM.basis.shape[0]
            self.total_number_modes += DM.basis_size
            DM.basis_type = basis_type

        self.correction_algorithm = Correctionconfig["correction_algorithm"].lower()
        self.MatrixType = Correctionconfig["MatrixType"].lower()

        if basis_type == 'actuator':
            self.amplitudeEFC = Correctionconfig["amplitudeEFC"]
        else:
            self.amplitudeEFC = 1.

        if self.correction_algorithm == "sm":
            self.expected_gain_in_contrast = 0.1

        self.regularization = Correctionconfig["regularization"]

        self.MaskEstim = MaskDH.creatingMaskDH(estimator.dimEstim, estimator.Estim_sampling)

        self.matrix_dir = matrix_dir

        self.update_matrices(testbed, estimator)

        if self.correction_algorithm == "efc" and save_for_bench == True:
            if not os.path.exists(realtestbed_dir):
                print("Creating directory " + realtestbed_dir + " ...")
                os.makedirs(realtestbed_dir)

            if testbed.DM1.active & testbed.DM3.active:
                fits.writeto(os.path.join(realtestbed_dir, "Direct_Matrix_2DM.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM1.fits"),
                             testbed.DM1.basis,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM3.fits"),
                             testbed.DM3.basis,
                             overwrite=True)
                number_Active_testbeds = 13

            elif testbed.DM1.active:
                fits.writeto(os.path.join(realtestbed_dir, "Direct_Matrix_DM1only.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM1.fits"),
                             testbed.DM1.basis,
                             overwrite=True)
                number_Active_testbeds = 1
            elif testbed.DM3.active:
                fits.writeto(os.path.join(realtestbed_dir, "Direct_Matrix_DM3only.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM3.fits"),
                             testbed.DM3.basis,
                             overwrite=True)
                number_Active_testbeds = 3
            else:
                raise Exception("No active DMs")

            if testbed.DM1.active & testbed.DM3.active:
                if Correctionconfig["Nbmodes_OnTestbed"] < 500:
                    raise Exception(
                        "Nbmodes_OnTestbed ({0}) in inversion for THD is probably too low for 2DM".format(
                            Correctionconfig["Nbmodes_OnTestbed"]))
            if not testbed.DM1.active & testbed.DM3.active:
                if Correctionconfig["Nbmodes_OnTestbed"] > 500:
                    raise Exception(
                        "Nbmodes_OnTestbed ({0}) in inversion for THD is probably too high for 1DM".format(
                            Correctionconfig["Nbmodes_OnTestbed"]))

            THD_quick_invert(Correctionconfig["Nbmodes_OnTestbed"], number_Active_testbeds, realtestbed_dir,
                             self.regularization)

            fits.writeto(realtestbed_dir + "DH_mask.fits", self.MaskEstim.astype(np.float32), overwrite=True)
            fits.writeto(realtestbed_dir + "DH_mask_where_x_y.fits",
                         np.array(np.where(self.MaskEstim == 1)).astype(np.float32),
                         overwrite=True)

        # Adding error on the DM model. Now that the matrix is measured, we can
        # introduce a small movememnt on one DM or the other. By changeing DM_pushact
        # we are changeing the position of the actuator and therfore the phase of the
        # DM for a given voltage when using DM.voltage_to_phase

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]  # type: DeformableMirror
            if DM.misregistration:
                print(DM_name + " Misregistration!")
                DM.DM_pushact = DM.creatingpushact(DM.DMconfig)

        ######################
        # Preparation of the correction loop
        ######################
        # in the initialization we have not inverted the matrix just yet so
        self.previousmode = np.nan

    def update_matrices(self,
                        testbed: Testbed,
                        estimator: Estimator,
                        initial_DM_voltage=0.,
                        input_wavefront=1.):
        """ --------------------------------------------------
        Measure the interaction matrices needed for the correction
        Is launch once in the Correction initialization and then once each time we update the matrix

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
       
        testbed :  OpticalSystem.Testbed
                Testbed object which describe your testbed

        estimator: Estimator
                an estimator object. This contains all information about the estimation

        initial_DM_voltage : float or 1d numpy array, default 0.
                initial DM voltages to measure the Matrix

        input_wavefront : float or 2d numpy array or 3d numpy array, default 1.
            initial wavefront to measure the Matrix


        -------------------------------------------------- """

        if self.correction_algorithm in ["efc", "em", "steepest", "sm"]:

            self.G = 0
            self.Gmatrix = 0
            self.M0 = 0

            self.FirstIterNewMat = True

            start_time = time.time()
            interMat = wfc.create_interaction_matrix(testbed,
                                                     estimator.dimEstim,
                                                     self.amplitudeEFC,
                                                     self.matrix_dir,
                                                     initial_DM_voltage=initial_DM_voltage,
                                                     input_wavefront=input_wavefront,
                                                     MatrixType=self.MatrixType,
                                                     save_all_planes_to_fits=False,
                                                     dir_save_all_planes="/Users/jmazoyer/Desktop/g0_all/")

            print("time for direct matrix " + testbed.string_os + " (s):", round(time.time() - start_time))
            print("")

            self.Gmatrix = wfc.crop_interaction_matrix_to_dh(interMat, self.MaskEstim)

            if self.correction_algorithm in ["em", "steepest", "sm"]:

                self.G = np.zeros((int(np.sum(self.MaskEstim)), self.Gmatrix.shape[1]), dtype=complex)
                self.G = (self.Gmatrix[0:int(self.Gmatrix.shape[0] / 2), :] +
                          1j * self.Gmatrix[int(self.Gmatrix.shape[0] / 2):, :])
                transposecomplexG = np.transpose(np.conjugate(self.G))
                self.M0 = np.real(np.dot(transposecomplexG, self.G))
                self.Gmatrix = 0.
        else:
            raise Exception("This correction algorithm is not yet implemented")

    def toDM_voltage(self, testbed: Testbed, estimate, mode=1, ActualCurrentContrast=1., **kwargs):
        """ --------------------------------------------------
        Run a correction from a estimate, and return the DM voltage compatible with the testbed

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        testbed :  OpticalSystem.Testbed
                Testbed object which describe your testbed

        estimate: 2D complex array 
                    Array of size of sixe [dimEstim, dimEstim]. 
                    This is the result of Estimator.estimate, from which this function 
                    send a command to the DM
        
        mode: int, defaut 1
                Use in EFC, EM, and Steepest, this is the mode we use in the SVD inversion
                if the mode is the same than the previous iteration, we store the inverted 
                matrix to avoid inverted it again
        
        
        ActualCurrentContrast: float defaut 1. 
                Use in StrokeMin to find a target contrast
                Contrast at the current iteration of the loop 

        
        Return
        ----------
        solution: 1d numpy real float array
            a voltage vector to be applied to the testbed


        -------------------------------------------------- """

        if self.correction_algorithm == "efc":
            if mode != self.previousmode:
                self.previousmode = mode
                # we only re-invert the matrix if it is different from last time
                _, _, self.invertGDH = invert_svd(self.Gmatrix,
                                                      mode,
                                                      goal="c",
                                                      visu=False,
                                                      regul=self.regularization)

            solutionefc = wfc.calc_efc_solution(self.MaskEstim, estimate, self.invertGDH, testbed)

            # # gain_individual_DM = [1.,1.]
            # # gain_individual_DM = [0.5,1.]
            # gain_individual_DM = [1.,0.9]

            # indice_acum_number_act = 0
            # for num_DM, DM_name in enumerate(testbed.name_of_DMs):

            #     # we access each DM object individually
            #     DM = vars(testbed)[DM_name]  # type: DeformableMirror

            #     # we multpily each DM by a specific DM gain
            #     solutionefc[
            #         indice_acum_number_act:indice_acum_number_act +
            #         DM.number_act] *= gain_individual_DM[num_DM]

            #     indice_acum_number_act += DM.number_act

            return -self.amplitudeEFC * solutionefc

        if self.correction_algorithm == "sm":
            # see Mazoyer et al 2018 ACAD-OSM I paper to understand algorithm
            if self.FirstIterNewMat:
                # This is the first time
                self.last_best_alpha = 1
                self.last_best_contrast = ActualCurrentContrast
                self.times_we_lowered_gain = 0
                self.count_since_last_best = 0
                self.FirstIterNewMat = False

            if self.last_best_contrast < ActualCurrentContrast:
                # problem: the algorithm did not actually improved contrast at the last last iteration
                # it's ok if it's only once, but we increase the count_since_last_best counter to stop
                # if we go several iteration wihtout improvement (a few lines below)
                self.count_since_last_best += 1
            else:
                self.count_since_last_best = 0
                self.last_best_contrast = ActualCurrentContrast

            if self.times_we_lowered_gain == 3:
                #it's been too long we have not increased
                # or we're so far off linearity that SM is actually heavily degrading contrast
                # It's time to stop !
                return "StopTheLoop"

            DesiredContrast = self.expected_gain_in_contrast * ActualCurrentContrast

            solutionSM, self.last_best_alpha = wfc.calc_strokemin_solution(self.MaskEstim, estimate, self.M0, self.G,
                                                                           DesiredContrast, self.last_best_alpha, testbed)

            if self.count_since_last_best > 5 or ActualCurrentContrast > 2 * self.last_best_contrast or (
                    isinstance(solutionSM, str) and solutionSM == "SMFailedTooManyTime"):
                self.times_we_lowered_gain += 1
                self.expected_gain_in_contrast = 1 - (1 - self.expected_gain_in_contrast) / 3
                self.count_since_last_best = 0
                self.last_best_alpha *= 20
                print(
                    "we do not improve contrast anymore, we go back to last best and change the gain to {:f}".
                    format(self.expected_gain_in_contrast))
                return "RebootTheLoop"

            # # gain_individual_DM = [1.,1.]
            # # gain_individual_DM = [0.5,1.]
            # gain_individual_DM = [1.,0.9]

            # indice_acum_number_act = 0
            # for num_DM, DM_name in enumerate(testbed.name_of_DMs):

            #     # we access each DM object individually
            #     DM = vars(testbed)[DM_name]  # type: DeformableMirror

            #     # we multpily each DM by a specific DM gain
            #     solutionSM[
            #         indice_acum_number_act:indice_acum_number_act +
            #         DM.number_act] *= gain_individual_DM[num_DM]

            #     indice_acum_number_act += DM.number_act

            return -self.amplitudeEFC * solutionSM

        if self.correction_algorithm == "em":

            if mode != self.previousmode:
                self.previousmode = mode
                _, _, self.invertM0 = invert_svd(self.M0,
                                                     mode,
                                                     goal="c",
                                                     visu=False,
                                                     regul=self.regularization)

            return -self.amplitudeEFC * wfc.calc_em_solution(self.MaskEstim, estimate, self.invertM0, self.G,
                                                       testbed)

        if self.correction_algorithm == "steepest":

            return -self.amplitudeEFC * wfc.calc_steepest_solution(self.MaskEstim, estimate, self.M0, self.G,
                                                                   testbed)
        else:
            raise Exception("This correction algorithm is not yet implemented")