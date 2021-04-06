# pylint: disable=invalid-name
__author__ = "Axel Potier"

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import Asterix.processing_functions as proc
import Asterix.fits_functions as useful


def invertSVD(matrix_to_invert,
              cut,
              goal="e",
              regul="truncation",
              visu=False,
              otherbasis=False,
              basisDM3=0):
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

    otherbasis:     boolean,
    basisDM3:       goes with other basis

    Return:
    ------
    np.diag(InvS): Inverse eigenvalues of the input matrix
    np.diag(InvS_truncated): Inverse eigenvalues of the input matrix after regularization
    pseudoinverse: Regularized inverse of the input matrix
    -------------------------------------------------- """
    U, s, V = np.linalg.svd(matrix_to_invert, full_matrices=False)
    # print(s)
    S = np.diag(s)
    InvS = np.linalg.inv(S)
    InvS_truncated = np.linalg.inv(S)
    # print(InvS)
    if visu == True:
        plt.plot(np.diag(InvS), "r.")
        plt.yscale("log")
        plt.savefig('invertSVDEFC.png')
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

    if otherbasis == True:
        pseudoinverse = np.dot(np.transpose(basisDM3), pseudoinverse)

    return [np.diag(InvS), np.diag(InvS_truncated), pseudoinverse]


def creatingCorrectionmatrix(input_wavefront,
                             testbed,
                             dimimages,
                             pushact,
                             mask,
                             Whichact,
                             otherbasis=False,
                             basisDM3=0):
    """ --------------------------------------------------
    Create the jacobian matrix for Electric Field Conjugation

    Parameters:
    ----------
    input_wavefront: 2D-array complex
        input wavefront in pupil plane
    testbed: Optical_element
        testbed structure
    dimimages: int
        size of the output image after resampling in pixels
    pushact: 3D-array
        phase created by the pokes of each actuator of DM with the wished amplitude
    mask: 2D array
        binary mask whose pixel=1 will be taken into account
    Whichact: 1D array
        index of the actuators taken into account to create the jacobian matrix
    otherbasis:
    basisDM3:

    Return:
    ------
    Gmatrixbis: 2D array, jacobian matrix for Electric Field Conjugation
    -------------------------------------------------- """
    # TODO this is not super clear to me, I need to clean it with Raphael,
    # with available tools.
    #
    # We can save tones of ram here !! This is why computer are crashing !
    # We duplicate pushact 2 times:  probephase, bas_fct !!!!

    # other basis need to be cleared basisDM3 need to be defined in all cases
    # and it should be the same function for each basis, just with a different
    # basis

    # change basis if needed
    if otherbasis == True:
        nb_fct = basisDM3.shape[0]  # number of functions in the basis
        tmp = pushact.reshape(pushact.shape[0],
                              pushact.shape[1] * pushact.shape[2])
        bas_fct = basisDM3 @ tmp.reshape(nb_fct, pushact.shape[1],
                                         pushact.shape[2])
    else:
        probephase = np.zeros((pushact.shape[0], testbed.dim_overpad_pupil,
                               testbed.dim_overpad_pupil),
                              dtype=complex)
        for k in range(pushact.shape[0]):
            probephase[k] = pushact[k]

        bas_fct = np.array([probephase[ind] for ind in Whichact])
        nb_fct = len(Whichact)
    print("Start EFC")
    Gmatrixbis = np.zeros((2 * int(np.sum(mask)), nb_fct))
    k = 0
    for i in range(nb_fct):
        if i % 100 == 0:
            print(i)
        Psivector = bas_fct[i]

        # TODO we shoudl replace by perfect estimation. This is equivalent but it would be more coherent
        # also i and k are the same indice I think :-)
        # the names here are confusing

        Gvector = proc.resampling(
            testbed.todetector(entrance_EF=input_wavefront * 1j * Psivector),
            dimimages)
        Gmatrixbis[0:int(np.sum(mask)),
                   k] = np.real(Gvector[np.where(mask == 1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,
                   k] = np.imag(Gvector[np.where(mask == 1)]).flatten()
        k = k + 1
    print("End EFC")
    return Gmatrixbis


def solutionEFC(mask, Result_Estimate, inversed_jacobian, WhichInPupil,
                nbDMactu):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle
        intensity in the dark hole region

    Parameters:
    ----------
    mask:               2D Binary mask corresponding to the dark hole region

    Result_Estimate:    2D array can be complex, focal plane electric field

    inversed_jacobian:  2D array, inverse of the jacobian matrix created
                                with all the actuators in WhichInPupil

    WhichInPupil:       1D array, index of the actuators taken into account
                            to create the jacobian matrix

    nbDMactu:           int, number of DM actuators

    Return:
    ------
    solution:           1D array, voltage to apply on each deformable
                        mirror actuator
    -------------------------------------------------- """

    Eab = np.zeros(2 * int(np.sum(mask)))
    Resultatbis = Result_Estimate[np.where(mask == 1)]
    Eab[0:int(np.sum(mask))] = np.real(Resultatbis).flatten()
    Eab[int(np.sum(mask)):] = np.imag(Resultatbis).flatten()
    cool = np.dot(inversed_jacobian, Eab)

    solution = np.zeros(nbDMactu)
    solution[WhichInPupil] = cool
    return solution


def solutionEM(mask, Result_Estimate, Hessian_Matrix, Jacobian, WhichInPupil,
               nbDMactu):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle
    intensity in the dark hole region

    Parameters:
    ----------
    mask:               Binary mask corresponding to the dark hole region

    Result_Estimate:    2D array can be complex, focal plane electric field

    Hessian_Matrix:     2D array , Hessian matrix of the DH energy

    Jacobian:           2D array, inverse of the jacobian matrix created with all the actuators
                        in WhichInPupil

    WhichInPupil:       1D array, index of the actuators taken into account
                        to create the jacobian matrix

    nbDMactu:           number of DM actuators

    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- """

    Eab = np.zeros(int(np.sum(mask)))
    Resultatbis = Result_Estimate[np.where(mask == 1)]
    Eab = np.real(np.dot(np.transpose(np.conjugate(Jacobian)),
                         Resultatbis)).flatten()
    cool = np.dot(Hessian_Matrix, Eab)

    solution = np.zeros(nbDMactu)
    solution[WhichInPupil] = cool
    return solution


def solutionSteepest(mask, Result_Estimate, Hessian_Matrix, Jacobian,
                     WhichInPupil, nbDMactu):
    """ --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize
    the speckle intensity in the dark hole region

    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    Hessian_Matrix: 2D array , Hessian matrix of the DH energy
    Jacobian: 2D array, inverse of the jacobian matrix created with all
                the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account
                to create the jacobian matrix
    nbDMactu:number of DM actuators
    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- """

    Eab = np.zeros(int(np.sum(mask)))
    Resultatbis = Result_Estimate[np.where(mask == 1)]
    Eab = np.real(np.dot(np.transpose(np.conjugate(Jacobian)),
                         Resultatbis)).flatten()
    pas = 2e3
    cool = pas * 2 * Eab

    solution = np.zeros(nbDMactu)
    solution[WhichInPupil] = cool
    return solution


#################################################################################
### PWD functions
#################################################################################


def createvectorprobes(testbed, amplitude, posprobes, dimEstim, cutsvd,
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
    PWVector = np.zeros((dimEstim**2, 2, numprobe))
    SVD = np.zeros((2, dimEstim, dimEstim))

    k = 0

    for i in posprobes:

        # TODO: we shoudl maybe put a which_DM_to_do_probes parameter
        Voltage_probe = np.zeros(testbed.DM3.number_act)
        Voltage_probe[i] = amplitude
        probephase[k] = testbed.DM3.voltage_to_phase(Voltage_probe,
                                                     wavelength=wavelength)

        # for PW the probes are not sent in the DM but at the entrance of the testbed.
        # with an hypothesis of small phase.
        #  not sure the pupil multiplication is necessary
        # inputwavefront = testbed.entrancepupil.EF_through(entrance_EF=1 +
        #                                                   1j * probephase[k])

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
                PWVector[l] = inversion[2]
            except:
                print("Careful: Error in invertSVD! for l=" + str(l))
                SVD[:, i, j] = np.zeros(2)
                PWVector[l] = np.zeros((2, numprobe))
            l = l + 1
    return [PWVector, SVD]


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
    return Resultat / 4


def createdifference(input_wavefront,
                     testbed,
                     posprobes,
                     dimimages,
                     amplitudePW,
                     DM1phase=0,
                     DM3phase=0,
                     photon_noise=False,
                     nb_photons=1e30,
                     wavelength=None):
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
    if wavelength == None:
        wavelength = testbed.wavelength_0

    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    #TODO if the DM1 is active we can measure once the EFthoughDM1 ans store it in entrance_EF
    #to save time. To check
    # if testbed.DM1.active is True:
    #     input_wavefront = testbed.DM1.EF_through(entrance_EF=input_wavefront, DM1phase = DM1phase,wavelength=wavelength)

    for count, num_probe in enumerate(posprobes):

        Voltage_probe = np.zeros(testbed.DM3.number_act)
        Voltage_probe[num_probe] = amplitudePW
        probephase = testbed.DM3.voltage_to_phase(Voltage_probe,
                                                  wavelength=wavelength)

        # Not 100% sure about wavelength here, so I prefeer to use
        # todetector to keep it monochromatic instead of todetector_Intensity
        # which is large band
        Ikmoins = np.abs(
            testbed.todetector(entrance_EF=input_wavefront,
                               DM1phase=DM1phase,
                               DM3phase=DM3phase - probephase,
                               wavelength=wavelength))**2

        Ikplus = np.abs(
            testbed.todetector(entrance_EF=input_wavefront,
                               DM1phase=DM1phase,
                               DM3phase=DM3phase + probephase,
                               wavelength=wavelength))**2

        if photon_noise == True:
            Ikplus = np.random.poisson(
                Ikplus * testbed.normPupto1 *
                nb_photons) / (testbed.normPupto1 * nb_photons)
            Ikmoins = np.random.poisson(
                Ikmoins * testbed.normPupto1 *
                nb_photons) / (testbed.normPupto1 * nb_photons)

        Difference[count] = proc.resampling(Ikplus - Ikmoins, dimimages)

    return Difference
