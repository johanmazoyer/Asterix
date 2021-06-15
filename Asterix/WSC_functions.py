# pylint: disable=invalid-name
__author__ = "Axel Potier"

from functools import total_ordering
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time

import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc
import Asterix.fits_functions as useful

#################################################################################
### Correction functions
#################################################################################


def invertSVD(matrix_to_invert,
              cut,
              goal="e",
              regul="truncation",
              visu=False,
              filename_visu=None):
    """ --------------------------------------------------
    Invert a matrix after a Singular Value Decomposition. The inversion can be regularized.

    Parameters:
    ----------
    matrix_to_invert: numpy array. The matrix

    cut:    int (see below)

    goal:   string, can be 'e' or 'c'
            if 'e': the cut set the inverse singular value not to exceed
            if 'c': the cut set the number of modes to take into account
                            (keep the lowest inverse singular values)

    regul:  string, can be 'truncation' or 'tikhonov'
            if 'truncation': when goal is set to 'c', the modes with the highest inverse
                            singular values are truncated
            if 'tikhonov': when goal is set to 'c', the modes with the highest inverse
                            singular values are smoothed (low pass filter)

    visu:   boolean, if True, plot and save the crescent inverse singular values,
                            before regularization

    Return:
    ------
    np.diag(InvS): Inverse eigenvalues of the input matrix
    np.diag(InvS_truncated): Inverse eigenvalues of the input matrix after regularization
    pseudoinverse: Regularized inverse of the input matrix

    AUTHOR : Axel Potier

    -------------------------------------------------- """
    U, s, V = np.linalg.svd(matrix_to_invert, full_matrices=False)
    #print(np.max(np.abs(U @ np.diag(s) @ V)))

    S = np.diag(s)
    InvS = np.linalg.inv(S)
    InvS_truncated = np.linalg.inv(S)
    # print(InvS)
    if visu == True:
        plt.ion()
        plt.figure()
        plt.clf
        plt.plot(np.diag(InvS), "r.")
        plt.yscale("log")
        plt.savefig(filename_visu)
        plt.close()

    if goal == "e":
        InvS_truncated[np.where(InvS_truncated > cut)] = 0
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated),
                               np.transpose(U))

    if goal == "c":
        if regul == "truncation":
            InvS_truncated[cut:] = 0
        if regul == "tikhonov":
            InvS_truncated = np.diag(s / (s**2 + s[cut]**2))
            if visu == True:
                plt.plot(np.diag(InvS_truncated), "b.")
                plt.yscale("log")
                plt.show()
                plt.close()
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated),
                               np.transpose(U))

    return [np.diag(InvS), np.diag(InvS_truncated), pseudoinverse]


def creatingInterractionmatrix( testbed, dimEstim,
                               amplitudeEFC, matrix_dir, initial_DM_voltage = 0., input_wavefront = 1., perfect_mat = False):
    """ --------------------------------------------------
    Create the jacobian matrix for Electric Field Conjugation

    Parameters:
    ----------

    testbed: Optical_element
        testbed structure
    dimEstim: int
        size of the output image in teh estimator
    amplitudeEFC: float, amplitude of the EFC probe on the DM
    matrix_dir : path. save all the difficult to measure files here
    input_wavefront: 1D-array real
        initial DM voltage
    input_wavefront: 2D-array complex
        input wavefront in pupil plane


    Return:
    ------
    InterMat: 2D array, jacobian matrix for Electric Field Conjugation

    AUTHOR : Axel Potier and Johan Mazoyer
    -------------------------------------------------- """
    if isinstance(initial_DM_voltage, (int, float)):
            initial_DM_voltage = np.zeros(testbed.number_act) + float(initial_DM_voltage)


    total_number_basis_modes = 0
    string_testbed_without_DMS = testbed.string_os

    for DM_name in testbed.name_of_DMs:
        DM = vars(testbed)[DM_name]
        total_number_basis_modes += DM.basis_size
        DM_small_str = "_" + "_".join(DM.string_os.split("_")[5:])
        string_testbed_without_DMS = string_testbed_without_DMS.replace(
            DM_small_str, '')


    G0 = proc.resampling(
                    testbed.todetector(voltage_vector=initial_DM_voltage), dimEstim)

    print("Start Interraction Matrix")
    InterMat = np.zeros((2 * int(dimEstim**2), total_number_basis_modes))

    pos_in_matrix = 0
    indice_acum_number_act = 0

    for DM_name in testbed.name_of_DMs:

        DM = vars(testbed)[DM_name]
        DM_small_str = "_" + "_".join(DM.string_os.split("_")[5:])

        basis_str = DM_small_str + "_" + DM.basis_type + "Basis" + str(
            DM.basis_size)


        if perfect_mat:
            headfile = "DirectMatrixPerf_EFCampl"
        else:
            headfile = "DirectMatrix_EFCampl"


        fileDirectMatrix = headfile+ str(
            amplitudeEFC) + basis_str + string_testbed_without_DMS
        
    
        if os.path.exists(matrix_dir + fileDirectMatrix + ".fits") and (initial_DM_voltage == 0.).all():
            print("The matrix " + fileDirectMatrix + " already exists")

            InterMat[:, pos_in_matrix:pos_in_matrix +
                     DM.basis_size] = fits.getdata(matrix_dir +
                                                   fileDirectMatrix + ".fits")

            pos_in_matrix += DM.basis_size

        else:
            
            if DM.basis_type == 'fourier':
                print("Load Fourier Basis Phases for " + DM_name)
                Name_FourrierBasis_fits = "Fourier_basis" + DM.string_os
                phasesBasis = fits.getdata(matrix_dir + Name_FourrierBasis_fits + '.fits')

            else:
                phasesBasis = np.zeros((DM.basis_size, DM.dim_overpad_pupil, DM.dim_overpad_pupil))
                for i in range(DM.basis_size):
                    DM.voltage_to_phase(DM.basis[i])* amplitudeEFC
                    DM.voltage_to_phase(DM.basis[i])


            actu_vect_DM = initial_DM_voltage[
            indice_acum_number_act:indice_acum_number_act + DM.number_act]
            DMphase_init = DM.voltage_to_phase(actu_vect_DM)

            indice_acum_number_act += DM.number_act

            # Creating Interaction Matrix for thd DM if does not exist
            init_pos_in_matrix = pos_in_matrix
            print("")
            print(fileDirectMatrix + " does not exists:")
            print("Start " + DM_name)

            if DM.z_position != 0:

                #TODO make something smarter to check automatically the name of the entrance pup
                Pup_inDMplane, _ = prop.prop_fresnel(testbed.entrancepupil.pup,
                                                     DM.wavelength_0,
                                                     DM.z_position,
                                                     DM.diam_pup_in_m / 2,
                                                     DM.prad)

            for i in range(DM.basis_size):

                if i % 10:
                    useful.progress(i, DM.basis_size, status='')


                if perfect_mat:
                    wavefront_phaseDM = testbed.EF_from_phase_and_ampl(phase_abb= phasesBasis[i]+ DMphase_init)

                    if DM.z_position != 0:
                        wavefront_phaseDM, _ = prop.prop_fresnel(Pup_inDMplane * wavefront_phaseDM,
                                                    DM.wavelength_0,
                                                    -DM.z_position,
                                                    DM.diam_pup_in_m / 2,
                                                    DM.prad)

                else:
                    
                    wavefront_phaseDM = 1j * phasesBasis[i] * testbed.EF_from_phase_and_ampl(phase_abb= DMphase_init)


                    if DM.z_position != 0:
                        wavefront_phaseDM, _ = prop.prop_fresnel(Pup_inDMplane * wavefront_phaseDM,
                                                    DM.wavelength_0,
                                                    -DM.z_position,
                                                    DM.diam_pup_in_m / 2,
                                                    DM.prad)

                Gvector = proc.resampling(
                    testbed.todetector(entrance_EF=wavefront_phaseDM), dimEstim) #- G0

                InterMat[:dimEstim**2,
                         pos_in_matrix] = np.real(Gvector).flatten()
                InterMat[dimEstim**2:,
                         pos_in_matrix] = np.imag(Gvector).flatten()

                pos_in_matrix += 1

            if (initial_DM_voltage == 0.).all():
                fits.writeto(
                    matrix_dir + fileDirectMatrix + ".fits",
                    InterMat[:, init_pos_in_matrix:init_pos_in_matrix +
                            DM.basis_size])

    print("")
    print("End Interraction Matrix")
    return InterMat


def cropDHInterractionMatrix(FullInterractionMatrix, mask):
    """ --------------------------------------------------
    Crop the  Interraction Matrix. to the mask size

    Parameters:
    ----------
    FullInterractionMatrix: Interraction matrix over the full focal plane

    mask : a binary mask to delimitate the DH

    Return: DHInterractionMatrix: matrix only inside the DH
    ------

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    size_full_matrix = FullInterractionMatrix.shape[0]

    size_DH_matrix = 2 * int(np.sum(mask))
    where_mask_flatten = np.where(mask.flatten() == 1.)
    DHInterractionMatrix = np.zeros(
        (size_DH_matrix, FullInterractionMatrix.shape[1]), dtype=float)

    for i in range(FullInterractionMatrix.shape[1]):
        DHInterractionMatrix[:int(size_DH_matrix / 2),
                             i] = FullInterractionMatrix[:int(
                                 size_full_matrix / 2), i][where_mask_flatten]
        DHInterractionMatrix[int(size_DH_matrix / 2):,
                             i] = FullInterractionMatrix[int(size_full_matrix /
                                                             2):,
                                                         i][where_mask_flatten]

    return DHInterractionMatrix


def basis_voltage_to_act_voltage(vector_basis_voltage, testbed):
    """ --------------------------------------------------
    transform a vector of voltages on the mode of a basis in a  vector of
    voltages of the actuators of the DMs of the system

    Parameters:
    ----------
    vector_basis_voltage: 1D-array real : dim (total(basisDM sizes))
                     vector of voltages on the mode of the basis for all
                     DMs by order of the light path
    testbed: Optical_element
        testbed structure (with DMs)


    Return:
    ------
    vector_actuator_voltage: 1D-array real : dim (total(DM actuators))
                     vector of base coefficients for all actuators of the DMs by order of the light path

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """

    indice_acum_basis_size = 0
    indice_acum_number_act = 0

    vector_actuator_voltage = np.zeros(testbed.number_act)
    for DM_name in testbed.name_of_DMs:
        DM = vars(testbed)[DM_name]
        vector_basis_voltage_for_DM = vector_basis_voltage[
            indice_acum_basis_size:indice_acum_basis_size + DM.basis_size]

        vector_actu_voltage_for_DM = np.dot(np.transpose(DM.basis),
                                            vector_basis_voltage_for_DM)

        vector_actuator_voltage[indice_acum_number_act:indice_acum_number_act +
                                DM.number_act] = vector_actu_voltage_for_DM

        indice_acum_basis_size += DM.basis_size
        indice_acum_number_act += DM.number_act

    return vector_actuator_voltage


def solutionEFC(mask, Result_Estimate, inversed_jacobian, testbed):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle
        intensity in the dark hole region

    Parameters:
    ----------
    mask:               2D Binary mask corresponding to the dark hole region

    Result_Estimate:    2D array can be complex, focal plane electric field

    inversed_jacobian:  2D array, inverse of the jacobian matrix linking the
                                    estimation to the basis coefficient

    testbed: a testbed with one or more DM

    Return:
    ------
    solution:           1D array, voltage to apply on each deformable
                        mirror actuator

    AUTHOR : Axel Potier
    -------------------------------------------------- """

    EF_vector = np.zeros(2 * int(np.sum(mask)))
    Resultat_cropdh = Result_Estimate[np.where(mask == 1)]
    EF_vector[0:int(np.sum(mask))] = np.real(Resultat_cropdh).flatten()
    EF_vector[int(np.sum(mask)):] = np.imag(Resultat_cropdh).flatten()
    produit_mat = np.dot(inversed_jacobian, EF_vector)

    return basis_voltage_to_act_voltage(produit_mat, testbed)


def solutionEM(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle
    intensity in the dark hole region

    Parameters:
    ----------
    mask:               Binary mask corresponding to the dark hole region

    Result_Estimate:    2D array can be complex, focal plane electric field

    Hessian_Matrix:     2D array , Hessian matrix of the DH energy

    Jacobian:           2D array, jacobian matrix created linking the
                                    estimation to the basis coefficient

    testbed: a testbed with one or more DM

    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator

    AUTHOR : Axel Potier
    -------------------------------------------------- """

    # With notations from Potier PhD eq 4.74 p78:
    Eab = Result_Estimate[np.where(mask == 1)]
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)),
                            Eab)).flatten()
    produit_mat = np.dot(Hessian_Matrix, realb0)

    return basis_voltage_to_act_voltage(produit_mat, testbed)


def solutionSM(mask, testbed, Result_Estimate, Jacob_trans_Jacob, Jacobian,
               DesiredContrast, last_best_alpha, ):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle
    intensity in the dark hole region in the stroke min solution
    See Axel Potier Phd for notation and Mazoyer et al. 2018a for alpha search improvement

    Parameters:
    ----------
    mask:               Binary mask corresponding to the dark hole region

    testbed: a testbed with one or more DM

    Result_Estimate:    2D array can be complex, focal plane electric field

    Jacob_trans_Jacob:     2D array , Jabobian.Transpose(Jabobian) matrix

    Jacobian:           2D array, jacobian matrix created linking the
                                    estimation to the basis coefficient

    DesiredContrast : float : the contrast value we wish to achieve

    last_best_alpha : This avoid to recalculate the best alpha at each iteration
                            since it's often a very close value to the last one working

    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    lasbestalpha : scalar. This avoid to recalculate the best alpha at each iteration
                            since it's often a very close value

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """

    pixel_in_mask = np.sum(mask)

    # With notations from Potier PhD eq 4.74 p78:
    Eab = Result_Estimate[np.where(mask == 1)]

    d0 = np.sum(np.abs(Eab)**2) / pixel_in_mask
    M0 = Jacob_trans_Jacob / pixel_in_mask
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)),
                            Eab)).flatten() / pixel_in_mask

    Identity_M0size = np.identity(M0.shape[0])


    # we put this keyword to True to do at least 1 try
    TestSMfailed = True
    number_time_failed =0

    while TestSMfailed:

        step_alpha = 1.3  #hard coded but can maybe be changed
        alpha = last_best_alpha * step_alpha**2

        #eq 4.79 Potier Phd
        DMSurfaceCoeff = np.dot(np.linalg.inv(M0 + alpha * Identity_M0size),
                                realb0)

        ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(
            M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0
        CurrentContrast = ResidualEnergy
        iteralpha = 0

        while CurrentContrast > DesiredContrast and alpha > 1e-12:

            # if this counter is not even incremented once, it means that our initial
            # alpha is probably too big
            iteralpha += 1

            alpha = alpha / step_alpha
            LastDMSurfaceCoeff = DMSurfaceCoeff

            DMSurfaceCoeff = np.dot(
                np.linalg.inv(M0 + alpha * Identity_M0size), realb0)
            ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(
                M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0

            LastCurrentContrast = CurrentContrast
            CurrentContrast = ResidualEnergy

            if (CurrentContrast > 3 * LastCurrentContrast) or (number_time_failed>5):
                # this step is to check if the SM is divergeing too quickly
                return np.nan,np.nan

            # print(
            #     "For alpha={:f}, Current Contrast:{:f}, Last Contrast:{:f}, Desired Contrast: {:f}"
            #     .format(np.log10(alpha), np.log10(CurrentContrast),
            #             np.log10(LastCurrentContrast), np.log10(DesiredContrast)))

        if iteralpha == 0:
            # we must do at least 1 iteration (the SM found a solution that dig the contrast)
            # or we fail !
            TestSMfailed = True
            number_time_failed +=1
            last_best_alpha *= 10
            print("SM failed, we increase alpha 10 times")
            if number_time_failed > 10: 
                return np.nan,np.nan
        else:
            TestSMfailed = False

    print(
        "Number of iteration in this stroke min (number of tested alpha): {:d}"
        .format(iteralpha))
    return basis_voltage_to_act_voltage(DMSurfaceCoeff, testbed), alpha

    # return basis_voltage_to_act_voltage(produit_mat, testbed)


def solutionSteepest(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize
    the speckle intensity in the dark hole region

    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    Hessian_Matrix: 2D array , Hessian matrix of the DH energy
    Jacobian: 2D array, inverse of the jacobian matrix linking the
                                    estimation to the basis coefficient
    testbed: a testbed with one or more DM

    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- """

    Eab = np.zeros(int(np.sum(mask)))
    Resultat_cropdh = Result_Estimate[np.where(mask == 1)]
    Eab = np.real(np.dot(np.transpose(np.conjugate(Jacobian)),
                         Resultat_cropdh)).flatten()
    pas = 2e3
    solution = pas * 2 * Eab

    return basis_voltage_to_act_voltage(solution, testbed)


#################################################################################
### PWD functions
#################################################################################


def createPWmastrix(testbed, amplitude, posprobes, dimEstim, cutsvd,
                    wavelength):
    """ --------------------------------------------------
    Build the interaction matrix for pair-wise probing.

    Parameters:
    ----------
    testbed:    testbed structure
    amplitude:  float, amplitude of the actuator pokes for pair(wise probing in nm
    posprobes:  1D-array, index of the actuators to push and pull for pair-wise probing
    dimEstim:  int, size of the output image after resampling in pixels
    cutsvd:     float, value not to exceed for the inverse eigeinvalues at each pixels
    wavelength: float, wavelength of the incoming flux in meter


    Return:
    ------
    PWVector:   2D array, vector probe to be multiplied by the image difference
                matrix in order to retrieve the focal plane electric field

    SVD:        2D array, map of the inverse singular values for each pixels and
                before regularization
    -------------------------------------------------- """
    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe, dimEstim, dimEstim), dtype=complex)
    probephase = np.zeros(
        (numprobe, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil))
    matrix = np.zeros((numprobe, 2))
    PWMatrix = np.zeros((dimEstim**2, 2, numprobe))
    SVD = np.zeros((2, dimEstim, dimEstim))

    DM_probe = vars(testbed)[testbed.name_DM_to_probe_in_PW]

    k = 0

    for i in posprobes:

        Voltage_probe = np.zeros(DM_probe.number_act)
        Voltage_probe[i] = amplitude
        probephase[k] = DM_probe.voltage_to_phase(Voltage_probe)

        # for PW the probes are not sent in the DM but at the entrance of the testbed.
        # with an hypothesis of small phase.
        # I tried to remove "1+"". It breaks the code
        # (coronagraph does not "remove the 1 exactly")

        deltapsik[k] = proc.resampling(
            testbed.todetector(entrance_EF=1 + 1j * probephase[k]), dimEstim)
        k = k + 1

    l = 0
    for i in np.arange(dimEstim):
        for j in np.arange(dimEstim):
            matrix[:, 0] = np.real(deltapsik[:, i, j])
            matrix[:, 1] = np.imag(deltapsik[:, i, j])

            try:
                inversion = invertSVD(matrix, cutsvd, visu=False)
                SVD[:, i, j] = inversion[0]
                PWMatrix[l] = inversion[2]
            except:
                print("Careful: Error in invertSVD! for l=" + str(l))
                SVD[:, i, j] = np.zeros(2)
                PWMatrix[l] = np.zeros((2, numprobe))
            l = l + 1
    return [PWMatrix, SVD]


def FP_PWestimate(Difference, Vectorprobes):
    """ --------------------------------------------------
    Calculate the focal plane electric field from the prone image
    differences and the modeled probe matrix

    Parameters:
    ----------
    Difference: 3D array, cube with image difference for each probes
    Vectorprobes: 2D array, model probe matrix for the same probe as for difference

    Return:
    ------
    Difference: 3D array, cube with image difference for each probes.
                Used for pair-wise probing
    -------------------------------------------------- """

    dimimages = len(Difference[0])
    numprobe = len(Vectorprobes[0, 0])
    Differenceij = np.zeros((numprobe))
    Resultat = np.zeros((dimimages, dimimages), dtype=complex)
    l = 0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:] = Difference[:, i, j]
            Resultatbis = np.dot(Vectorprobes[l], Differenceij)
            Resultat[i, j] = Resultatbis[0] + 1j * Resultatbis[1]

            l = l + 1

    return Resultat / 4.


def createdifference(input_wavefront,
                     testbed,
                     posprobes,
                     dimimages,
                     amplitudePW,
                     voltage_vector=0.,
                     wavelength=None,
                     photon_noise=False,
                     nb_photons=1e30,
                     **kwargs):
    """ --------------------------------------------------
    Simulate the acquisition of probe images using Pair-wise
    and calculate the difference of images [I(+probe) - I(-probe)]

    Parameters
    ----------
    input_wavefront : 2D-array (complex)
        Input wavefront in pupil plane
    posprobes : 1D-array
        Index of the actuators to push and pull for pair-wise probing
    pushact : 3D-array
        OPD created by the pokes of all actuators in the DM
        Unit = phase with the amplitude of the wished probe
    testbed: Optical system structure
    dimimages : int
        Size of the output image after resampling in pixels
    perfect_coro : bool, optional
        Set if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    noise : boolean, optional
        If True, add photon noise.
    numphot : int, optional
        Number of photons entering the pupil

    Returns
    ------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing
    -------------------------------------------------- """

    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    for count, num_probe in enumerate(posprobes):

        Voltage_probe = np.zeros(testbed.number_act)
        indice_acum_number_act = 0

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]

            if DM_name == testbed.name_DM_to_probe_in_PW:
                Voltage_probeDMprobe = np.zeros(DM.number_act)
                Voltage_probeDMprobe[num_probe] = amplitudePW
                Voltage_probe[indice_acum_number_act:indice_acum_number_act +
                              DM.number_act] = Voltage_probeDMprobe

            indice_acum_number_act += DM.number_act

        # TODO Can these be replaced by todetector_intensity ?
        Ikmoins = np.abs(
            testbed.todetector(entrance_EF=input_wavefront,
                               voltage_vector=voltage_vector - Voltage_probe,
                               wavelength=wavelength,
                               **kwargs))**2

        Ikplus = np.abs(
            testbed.todetector(entrance_EF=input_wavefront,
                               voltage_vector=voltage_vector + Voltage_probe,
                               wavelength=wavelength,
                               **kwargs))**2

        if photon_noise == True:
            Ikplus = np.random.poisson(
                Ikplus * testbed.normPupto1 *
                nb_photons) / (testbed.normPupto1 * nb_photons)
            Ikmoins = np.random.poisson(
                Ikmoins * testbed.normPupto1 *
                nb_photons) / (testbed.normPupto1 * nb_photons)

        Difference[count] = proc.resampling(Ikplus - Ikmoins, dimimages)

    return Difference
