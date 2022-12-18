import os
import numpy as np
from astropy.io import fits

from Asterix.utils import invert_svd
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed

import Asterix.wfsc.estimator as estimator_mod
import Asterix.wfsc.thd_quick_invert as thd_quick_invert
import Asterix.wfsc.wf_control_functions as wfc


class Corrector:
    """Corrector Class allows you to define a corrector with different algorithms.

    Corrector is a class which takes as parameter:
        - the testbed structure
        - the correction parameters
        - the estimator

    It must contains 2 functions at least:
        - an initialization (e.g. Jacobian matrix) Corrector.__init__
            The initialization requires previous initialization of
            the testbed and of the estimator.

        - a correction function Corrector.toDM_voltage(estimation), which returns the DM Voltage vector
            using as parameter the estimation (2D array or 3D for polychromatic correction).
            It can one DM or more, depending on the testbed.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self,
                 Correctionconfig,
                 testbed: Testbed,
                 MaskDH,
                 estimator: estimator_mod.Estimator,
                 matrix_dir=None,
                 save_for_bench=False,
                 realtestbed_dir=''):
        """Initialize the corrector. This is where you define the EFC matrix
        For all large files you should use a method of "save to fits" if it
        does not exist "load from fits" if it does, in matrix_dir.

        Store in the structure only what you need for estimation. Everything not
        used in self.estimate should not be stored

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        Correctionconfig : dict
            general correction parameters
        testbed : OpticalSystem.Testbed
            Testbed object which describe your testbed
        MaskDH : 2d numpy array
            binary array of size [dimEstim, dimEstim] : dark hole mask
        estimator : Estimator
            an estimator object. This contains all information about the estimation
        matrix_dir : string, default: None
            path to directory to save interraction matrices
        save_for_bench : bool default: false
            should we save for the real testbed in realtestbed_dir
        realtestbed_dir : string
            path to directory to save all the files the real testbed need
        """
        if not os.path.exists(matrix_dir):
            print("Creating directory " + matrix_dir)
            os.makedirs(matrix_dir)

        if not isinstance(testbed, OpticalSystem):
            raise TypeError("testbed must be an OpticalSystem object")

        basis_type = Correctionconfig["DM_basis"].lower()
        self.total_number_modes = 0

        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]
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

        if self.correction_algorithm == "efc" and save_for_bench:
            if not os.path.exists(realtestbed_dir):
                print("Creating directory " + realtestbed_dir)
                os.makedirs(realtestbed_dir)

            if estimator.polychrom in ['centralwl', 'broadband_pwprobes']:
                number_wl_in_matrix = 1
            else:
                number_wl_in_matrix = estimator.nb_wav_estim

            if testbed.DM1.active & testbed.DM3.active:
                fits.writeto(os.path.join(realtestbed_dir, f"Direct_Matrix_2DM_wl{number_wl_in_matrix}.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM1.fits"), testbed.DM1.basis, overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM3.fits"), testbed.DM3.basis, overwrite=True)
                number_Active_testbeds = 13

            elif testbed.DM1.active:
                fits.writeto(os.path.join(realtestbed_dir, f"Direct_Matrix_DM1only_wl{number_wl_in_matrix}.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM1.fits"), testbed.DM1.basis, overwrite=True)
                number_Active_testbeds = 1
            elif testbed.DM3.active:
                fits.writeto(os.path.join(realtestbed_dir, f"Direct_Matrix_DM3only_wl{number_wl_in_matrix}.fits"),
                             self.Gmatrix,
                             overwrite=True)
                fits.writeto(os.path.join(realtestbed_dir, "Base_Matrix_DM3.fits"), testbed.DM3.basis, overwrite=True)
                number_Active_testbeds = 3
            else:
                raise ValueError("No active DMs")

            if testbed.DM1.active & testbed.DM3.active:
                if Correctionconfig["Nbmodes_OnTestbed"] < 500:
                    input(f"Nbmodes_OnTestbed ({Correctionconfig['Nbmodes_OnTestbed']})" +
                          " in inversion for THD is probably too low for 2DM. " +
                          "This is just a warning, if you kown what your are doing, " + "press any key to continue")
            if not testbed.DM1.active & testbed.DM3.active:
                if Correctionconfig["Nbmodes_OnTestbed"] > 500:
                    input(f"Nbmodes_OnTestbed ({Correctionconfig['Nbmodes_OnTestbed']})" +
                          " in inversion for THD is probably too high for 1DM. " +
                          "This is just a warning, if you kown what your are doing, " + "press any key to continue")

            thd_quick_invert.THD_quick_invert(Correctionconfig["Nbmodes_OnTestbed"],
                                              number_Active_testbeds,
                                              realtestbed_dir,
                                              self.regularization,
                                              number_wl_in_matrix=number_wl_in_matrix)

            fits.writeto(os.path.join(realtestbed_dir, "DH_mask.fits"),
                         self.MaskEstim.astype(np.float32),
                         overwrite=True)
            fits.writeto(os.path.join(realtestbed_dir, "DH_mask_where_x_y.fits"),
                         np.array(np.where(self.MaskEstim == 1)).astype(np.float32),
                         overwrite=True)

        # Adding error on the DM model. Now that the matrix is measured, we can
        # introduce a small movememnt on one DM or the other. By changeing DM_pushact
        # we are changeing the position of the actuator and therfore the phase of the
        # DM for a given voltage when using DM.voltage_to_phase

        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]
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
                        estimator: estimator_mod.Estimator,
                        initial_DM_voltage=0.,
                        input_wavefront=1.):
        """Measure the interaction matrices needed for the correction Is launch
        once in the Correction initialization and then once each time we update
        the matrix.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        testbed : OpticalSystem.Testbed
            Testbed object which describe your testbed
        estimator : Estimator
            an estimator object. This contains all information about the estimation
        initial_DM_voltage : float or 1d numpy array, default 0.
            initial DM voltages to measure the Matrix
        input_wavefront : float or 2d numpy array or 3d numpy array, default 1.
            initial wavefront to measure the Matrix
        """

        if self.correction_algorithm in ["efc", "em", "steepest", "sm"]:

            self.G = 0
            self.Gmatrix = 0
            self.M0 = 0

            self.FirstIterNewMat = True

            interMat = wfc.create_interaction_matrix(testbed,
                                                     estimator.dimEstim,
                                                     self.amplitudeEFC,
                                                     self.matrix_dir,
                                                     initial_DM_voltage=initial_DM_voltage,
                                                     input_wavefront=input_wavefront,
                                                     MatrixType=self.MatrixType,
                                                     polychrom=estimator.polychrom,
                                                     wav_vec_estim=estimator.wav_vec_estim,
                                                     dir_save_all_planes=None,
                                                     visu=False)

            self.Gmatrix = wfc.crop_interaction_matrix_to_dh(interMat, self.MaskEstim)

            if self.correction_algorithm in ["em", "steepest", "sm"]:
                pixel_in_mask = int(np.sum(self.MaskEstim))
                number_wl_matrix = self.Gmatrix.shape[0] // (2 * pixel_in_mask)

                self.G = np.zeros((number_wl_matrix * pixel_in_mask, self.Gmatrix.shape[1]), dtype=complex)

                for i in range(number_wl_matrix):
                    self.G[i * pixel_in_mask:(i + 1) * pixel_in_mask, :] = (
                        self.Gmatrix[2 * i * pixel_in_mask:(2 * i + 1) * pixel_in_mask, :] +
                        1j * self.Gmatrix[(2 * i + 1) * pixel_in_mask:(2 * i + 2) * pixel_in_mask, :])

                transposecomplexG = np.transpose(np.conjugate(self.G))
                self.M0 = np.real(np.dot(transposecomplexG, self.G))
                self.Gmatrix = 0.
        else:
            raise NotImplementedError("This correction algorithm is not yet implemented")

    def toDM_voltage(self, testbed: Testbed, estimate, mode=1, ActualCurrentContrast=1., **kwargs):
        """Run a correction from a estimate, and return the DM voltage
        compatible with the testbed.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        testbed : OpticalSystem.Testbed
            Testbed object which describe your testbed
        estimate : list of 2D complex array
            list is the number of wl in the estimation, usually 1 or testbed.nb_wav
            Each arrays are of size of sixe [dimEstim, dimEstim].
            This is the result of Estimator.estimate, from which this function
            send a command to the DM
        mode : int, defaut 1
            Use in EFC, EM, and Steepest, this is the mode we use in the SVD inversion
            if the mode is the same than the previous iteration, we store the inverted
            matrix to avoid inverted it again
        ActualCurrentContrast : float, defau,t 1.
            Use in StrokeMin to find a target contrast
            Contrast at the current iteration of the loop

        Return
        ----------
        solution : 1d numpy real float array
            a voltage vector to be applied to the testbed
        """

        if self.correction_algorithm == "efc":
            if mode != self.previousmode:
                self.previousmode = mode
                # we only re-invert the matrix if it is different from last time
                _, _, self.invertGDH = invert_svd(self.Gmatrix, mode, goal="c", visu=False, regul=self.regularization)

            solutionefc = wfc.calc_efc_solution(self.MaskEstim, estimate, self.invertGDH, testbed)

            # # gain_individual_DM = [1.,1.]
            # # gain_individual_DM = [0.5,1.]
            # gain_individual_DM = [1.,0.9]

            # indice_acum_number_act = 0
            # for num_DM, DM_name in enumerate(testbed.name_of_DMs):

            #     # we access each DM object individually
            #     DM: DeformableMirror = vars(testbed)[DM_name]

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
                # it's been too long we have not increased
                # or we're so far off linearity that SM is actually heavily degrading contrast
                # It's time to stop !
                return "StopTheLoop"

            DesiredContrast = self.expected_gain_in_contrast * ActualCurrentContrast

            solutionSM, self.last_best_alpha = wfc.calc_strokemin_solution(self.MaskEstim, estimate, self.M0, self.G,
                                                                           DesiredContrast, self.last_best_alpha,
                                                                           testbed)

            if self.count_since_last_best > 5 or ActualCurrentContrast > 2 * self.last_best_contrast or (isinstance(
                    solutionSM, str) and solutionSM == "SMFailedTooManyTime"):
                self.times_we_lowered_gain += 1
                self.expected_gain_in_contrast = 1 - (1 - self.expected_gain_in_contrast) / 3
                self.count_since_last_best = 0
                self.last_best_alpha *= 20
                print("we do not improve contrast anymore, we go back to last " +
                      f"best and change the gain to {self.expected_gain_in_contrast:f}")
                return "RebootTheLoop"

            # # gain_individual_DM = [1.,1.]
            # # gain_individual_DM = [0.5,1.]
            # gain_individual_DM = [1.,0.9]

            # indice_acum_number_act = 0
            # for num_DM, DM_name in enumerate(testbed.name_of_DMs):

            #     # we access each DM object individually
            #     DM: DeformableMirror = vars(testbed)[DM_name]

            #     # we multpily each DM by a specific DM gain
            #     solutionSM[
            #         indice_acum_number_act:indice_acum_number_act +
            #         DM.number_act] *= gain_individual_DM[num_DM]

            #     indice_acum_number_act += DM.number_act

            return -self.amplitudeEFC * solutionSM

        if self.correction_algorithm == "em":

            if mode != self.previousmode:
                self.previousmode = mode
                _, _, self.invertM0 = invert_svd(self.M0, mode, goal="c", visu=False, regul=self.regularization)

            return -self.amplitudeEFC * wfc.calc_em_solution(self.MaskEstim, estimate, self.invertM0, self.G, testbed)

        if self.correction_algorithm == "steepest":

            return -self.amplitudeEFC * wfc.calc_steepest_solution(self.MaskEstim, estimate, self.M0, self.G, testbed)
        else:
            raise NotImplementedError("This correction algorithm is not yet implemented")
