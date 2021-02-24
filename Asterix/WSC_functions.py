__author__ = "Axel Potier"

import numpy as np
import matplotlib.pyplot as plt

import Asterix.InstrumentSimu_functions as instr
import Asterix.processing_functions as proc


def invertSVD(matrix_to_invert,
              cut,
              goal="e",
              regul="truncation",
              visu=True,
              otherbasis=False,
              basisDM3=0,
              intermatrix_dir="./"):
    """ --------------------------------------------------
    Invert a matrix after a Singular Value Decomposition. The inversion can be regularized.
    
    Parameters:
    ----------
    matrix_to_invert: 
    cut: 
    goal: string, can be 'e' or 'c'
          if 'e': the cut set the inverse singular value not to exceed
          if 'c': the cut set the number of modes to take into account (keep the lowest inverse singular values)
    regul: string, can be 'truncation' or 'tikhonov'
          if 'truncation': when goal is set to 'c', the modes with the highest inverse singular values are truncated
          if 'tikhonov': when goal is set to 'c', the modes with the highest inverse singular values are smooth (low pass filter)
    visu: boolean, if True, plot and save the crescent inverse singular values , before regularization
    otherbasis: boolean, 
    basisDM3: goes with other basis
    
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
    #     plt.savefig(intermatrix_dir+'invertSVDEFC_'+ '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_.png')

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
                # plt.show()
        pseudoinverse = np.dot(np.dot(np.transpose(V), InvS_truncated),
                               np.transpose(U))

    if otherbasis == True:
        pseudoinverse = np.dot(np.transpose(basisDM3), pseudoinverse)

    return [np.diag(InvS), np.diag(InvS_truncated), pseudoinverse]


def createvectorprobes(wavelength, corona_struct, amplitude, posprobes,
                       pushact, dimimages, cutsvd):
    """ --------------------------------------------------
    Build the interaction matrix for pair-wise probing.
    
    Parameters:
    ----------
    wavelength: float, wavelength of the  incoming flux in meter
    corona_struct: coronagraph structure
    amplitude: float, amplitude of the actuator pokes for pair(wise probing in nm
    posprobes: 1D-array, index of the actuators to push and pull for pair-wise probing
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    dimimages: int, size of the output image after resampling in pixels
    cutsvd: float, value not to exceed for the inverse eigeinvalues at each pixels
    
    
    Return:
    ------
    PWVector: 2D array, vector probe to be multiplied by the image difference matrix in order to retrieve the focal plane electric field
    SVD: 2D array, map of the inverse singular values for each pixels and before regularization
    -------------------------------------------------- """
    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe, dimimages, dimimages), dtype=complex)
    probephase = np.zeros((numprobe, corona_struct.entrancepupil.shape[1],
                           corona_struct.entrancepupil.shape[1]))
    matrix = np.zeros((numprobe, 2))
    PWVector = np.zeros((dimimages**2, 2, numprobe))
    SVD = np.zeros((2, dimimages, dimimages))

    k = 0
    for i in posprobes:
        tmp = proc.crop_or_pad_image(pushact[i],
                                     corona_struct.entrancepupil.shape[1])
        probephase[k] = tmp * amplitude * 1e-9 * 2 * np.pi / wavelength

        inputwavefront = corona_struct.entrancepupil * (1 + 1j * probephase[k])
        deltapsikbis = (corona_struct.apodtodetector(inputwavefront) /
                        np.sqrt(corona_struct.maxPSF))
        deltapsik[k] = proc.resampling(deltapsikbis, dimimages)
        k = k + 1

    l = 0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
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


def creatingWhichinPupil(pushact, entrancepupil, cutinpupil):
    """ --------------------------------------------------
    Create a vector with the index of all the actuators located in the entrance pupil
    
    Parameters:
    ----------
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    entrancepupil: 2D-array, entrance pupil shape
    cutinpupil: float, minimum surface of an actuator inside the pupil to be taken into account (between 0 and 1, in ratio of an actuator perfectly centered in the entrance pupil)
    
    Return:
    ------
    WhichInPupil: 1D array, index of all the actuators located inside the pupil
    -------------------------------------------------- """
    WhichInPupil = []
    tmp_entrancepupil = proc.crop_or_pad_image(entrancepupil, pushact.shape[2])

    for i in np.arange(pushact.shape[0]):
        Psivector = pushact[i]
        cut = cutinpupil * np.sum(np.abs(Psivector))

        if np.sum(Psivector * tmp_entrancepupil) > cut:
            WhichInPupil.append(i)

    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil


def creatingMaskDH(dimimages,
                   shape,
                   choosepixDH=[8, 35, -35, 35],
                   circ_rad=[8, 10],
                   circ_side="full",
                   circ_offset=8,
                   circ_angle=0):
    """ --------------------------------------------------
    Create a binary mask.
    
    Parameters:
    ----------
    dimimages: int, size of the output squared mask
    shape: string, can be 'square' or 'circle' , define the shape of the binary mask.
    choosepixDH: 1D array, if shape is 'square', define the edges of the binary mask in pixels.
    circ_rad: 1D array, if shape is 'circle', define the inner and outer edge of the binary mask
    circ_side: string, if shape is 'circle', can define to keep only one side of the circle
    circ_offset : float, remove pixels that are closer than circ_offset if circ_side is set
    circ_angle : float, if circ_side is set, remove pixels within a cone of angle circ_angle

    Return:
    ------
    maskDH: 2D array, binary mask
    -------------------------------------------------- """
    xx, yy = np.meshgrid(
        np.arange(dimimages) - (dimimages) / 2,
        np.arange(dimimages) - (dimimages) / 2)
    rr = np.hypot(yy, xx)
    if shape == "square":
        maskDH = np.ones((dimimages, dimimages))
        maskDH[xx < choosepixDH[0]] = 0
        maskDH[xx > choosepixDH[1]] = 0
        maskDH[yy < choosepixDH[2]] = 0
        maskDH[yy > choosepixDH[3]] = 0
    if shape == "circle":
        maskDH = np.ones((dimimages, dimimages))
        maskDH[rr >= circ_rad[1]] = 0
        maskDH[rr < circ_rad[0]] = 0
        if circ_side == "right":
            maskDH[xx < np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx / np.tan(circ_angle * np.pi / 180) > 0] = 0
                maskDH[yy + xx / np.tan(circ_angle * np.pi / 180) < 0] = 0
        if circ_side == "left":
            maskDH[xx > -np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx / np.tan(circ_angle * np.pi / 180) < 0] = 0
                maskDH[yy + xx / np.tan(circ_angle * np.pi / 180) > 0] = 0
        if circ_side == "bottom":
            maskDH[yy < np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx * np.tan(circ_angle * np.pi / 180) < 0] = 0
                maskDH[yy + xx * np.tan(circ_angle * np.pi / 180) < 0] = 0
        if circ_side == "top":
            maskDH[yy > -np.abs(circ_offset)] = 0
            if circ_angle != 0:
                maskDH[yy - xx * np.tan(circ_angle * np.pi / 180) > 0] = 0
                maskDH[yy + xx * np.tan(circ_angle * np.pi / 180) > 0] = 0
    return maskDH


def creatingCorrectionmatrix(input_wavefront,
                             corona_struct,
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
    corona_struct: structure
        coronagraph structure
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
    # change basis if needed
    if otherbasis == True:
        nb_fct = basisDM3.shape[0]  # number of functions in the basis
        tmp = pushact.reshape(pushact.shape[0],
                              pushact.shape[1] * pushact.shape[2])
        bas_fct = basisDM3 @ tmp.reshape(nb_fct, pushact.shape[1],
                                         pushact.shape[2])
    else:
        probephase = np.zeros(
            (pushact.shape[0], corona_struct.entrancepupil.shape[1],
             corona_struct.entrancepupil.shape[1]),
            dtype=complex)
        for k in range(pushact.shape[0]):
            probephase[k] = proc.crop_or_pad_image(
                pushact[k], corona_struct.entrancepupil.shape[1])
        bas_fct = np.array([probephase[ind] for ind in Whichact])
        nb_fct = len(Whichact)
    print("Start EFC")
    Gmatrixbis = np.zeros((2 * int(np.sum(mask)), nb_fct))
    k = 0
    for i in range(nb_fct):
        if i % 100 == 0:
            print(i)
        Psivector = bas_fct[i]
        Gvector = (
            corona_struct.apodtodetector(input_wavefront * 1j * Psivector) /
            np.sqrt(corona_struct.maxPSF))
        Gvector = proc.resampling(Gvector, dimimages)
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
    Voltage to apply on the deformable mirror in order to minimize the speckle intensity in the dark hole region
    
    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    inversed_jacobian: 2D array, inverse of the jacobian matrix created with all the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account to create the jacobian matrix
    nbDMactu:number of DM actuators

    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
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
    Voltage to apply on the deformable mirror in order to minimize the speckle intensity in the dark hole region
    
    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    Hessian_Matrix: 2D array , Hessian matrix of the DH energy
    Jacobian: 2D array, inverse of the jacobian matrix created with all the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account to create the jacobian matrix
    nbDMactu:number of DM actuators

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


def apply_on_DM(actu_vect, DM_pushact):
    """ --------------------------------------------------
    Generate the phase applied on one DM for a give vector of actuator amplitude
    
    Parameters:
    ----------
    actu_vect : 1D array
                values of the amplitudes for each actuator
    DM_pushact : 2D array
                array of the DM actuator functions
    Return:
    ------
        2D array
        phase map in the same unit as actu_vect times DM_pushact)
    -------------------------------------------------- """
    return np.dot(
        actu_vect,
        DM_pushact.reshape(DM_pushact.shape[0],
                           DM_pushact.shape[1] * DM_pushact.shape[1])).reshape(
                               DM_pushact.shape[1], DM_pushact.shape[1])
