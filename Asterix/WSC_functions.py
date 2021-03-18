__author__ = "Axel Potier"

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import Asterix.InstrumentSimu_functions as instr
import Asterix.processing_functions as proc
import Asterix.fits_functions as useful


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


def createvectorprobes(wavelength, testbed, amplitude, posprobes, pushact,
                       dimimages, cutsvd):
    """ --------------------------------------------------
    Build the interaction matrix for pair-wise probing.
    
    Parameters:
    ----------
    wavelength: float, wavelength of the  incoming flux in meter
    testbed: testbed structure
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
    probephase = np.zeros(
        (numprobe, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil))
    matrix = np.zeros((numprobe, 2))
    PWVector = np.zeros((dimimages**2, 2, numprobe))
    SVD = np.zeros((2, dimimages, dimimages))

    k = 0
    for i in posprobes:

        #
        # lines inputwavefront = and deltapsikbis = need to be replace by
        # testbed.todetector(DM3phase = probephase[k])/ np.sqrt(testbed.maxPSF)
        # but chaque chose en son temps donc laissons comme ca now

        tmp = proc.crop_or_pad_image(pushact[i], testbed.dim_overpad_pupil)
        probephase[k] = tmp * amplitude * 1e-9 * 2 * np.pi / wavelength

        inputwavefront = testbed.entrancepupil.pup * (1 + 1j * probephase[k])
        deltapsikbis = (testbed.corono.todetector(entrance_EF=inputwavefront) /
                        np.sqrt(testbed.maxPSF))

        deltapsik[k] = proc.resampling(deltapsikbis, dimimages)
        k = k + 1
        # useful.quickfits(np.abs(deltapsikbis),dir="/home/rgalicher/tt/")
        # azs

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


def load_or_save_maskDH(intermatrix_dir, EFCconfig, dim_sampl, DH_sampling,
                        dim_im, science_sampling):
    """ --------------------------------------------------
        define at a single place the complicated file name of the mask and do the saving
        and loading depending in existence
        THIS IS BAD, THE DH SHOULD BE CODED IN l/D and not in pixel in the DH_sampling sampling.
        ONCE CORRECTED THIS FUNCTION CAN BE SIMPLIFIED A LOT and loaded twice, once for each dimension
        
        Parameters:
        ----------
        intermatrix_dir: Directory where to save the fits
        EFCconfig: all the EFC parameters containing shape and size of the DH.
        dim_sampl: dimension of the re-sampled focal plane
        DH_sampling : sampling of the re-sampled DH
        dim_im: dimension of the FP in the detector focal plane
        science_sampling : sampling of the FP in the detector focal plane
        
        Return:
        ------
        the 2 dark hole mask in each dimensions and the string name
    -------------------------------------------------- """

    DHshape = EFCconfig["DHshape"]
    choosepix = EFCconfig["choosepix"]
    choosepix = [int(i) for i in choosepix]
    circ_rad = EFCconfig["circ_rad"]
    circ_rad = [int(i) for i in circ_rad]
    circ_side = EFCconfig["circ_side"].lower()
    circ_offset = EFCconfig["circ_offset"]
    circ_angle = EFCconfig["circ_angle"]

    if DHshape == "square":
        stringdh = "_square_" + "_".join(map(str, choosepix)) + "pix_"
    else:
        stringdh = "_circle_" + "_".join(map(
            str, circ_rad)) + "pix_" + str(circ_side) + '_'
        if circ_side != 'full':
            stringdh = stringdh + str(circ_offset) + 'pix_' + str(
                circ_angle) + 'deg_'

    fileMaskDH = "MaskDH" + stringdh

    fileMaskDH_sampl = fileMaskDH + 'dim' + str(
        dim_sampl) + 'res{:.1f}'.format(DH_sampling)

    if os.path.exists(intermatrix_dir + fileMaskDH_sampl + ".fits") == True:
        print("Mask of DH " + fileMaskDH + " already exist")
        maskDH = fits.getdata(intermatrix_dir + fileMaskDH_sampl + ".fits")
    else:
        print("We measure and save " + fileMaskDH_sampl)
        maskDH = creatingMaskDH(dim_sampl,
                                DHshape,
                                choosepixDH=choosepix,
                                circ_rad=circ_rad,
                                circ_side=circ_side,
                                circ_offset=circ_offset,
                                circ_angle=circ_angle)
        fits.writeto(intermatrix_dir + fileMaskDH_sampl + ".fits", maskDH)

    fileMaskDH_detect = fileMaskDH + 'dim' + str(dim_im) + 'res{:.1f}'.format(
        science_sampling)

    if os.path.exists(intermatrix_dir + fileMaskDH_detect + ".fits") == True:
        print("Mask of DH " + fileMaskDH_detect + " already exist")
        maskDHcontrast = fits.getdata(intermatrix_dir + fileMaskDH_detect +
                                      ".fits")
    else:
        print("We measure and save " + fileMaskDH_detect)
        maskDHcontrast = creatingMaskDH(
            dim_im,
            DHshape,
            choosepixDH=[
                element * dim_im / dim_sampl for element in choosepix
            ],
            circ_rad=[element * dim_im / dim_sampl for element in circ_rad],
            circ_side=circ_side,
            circ_offset=circ_offset * dim_im / dim_sampl,
            circ_angle=circ_angle)

        fits.writeto(intermatrix_dir + fileMaskDH_detect + ".fits",
                     maskDHcontrast)
    return maskDH, maskDHcontrast, stringdh


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
            probephase[k] = proc.crop_or_pad_image(pushact[k],
                                                   testbed.dim_overpad_pupil)
        bas_fct = np.array([probephase[ind] for ind in Whichact])
        nb_fct = len(Whichact)
    print("Start EFC")
    Gmatrixbis = np.zeros((2 * int(np.sum(mask)), nb_fct))
    k = 0
    for i in range(nb_fct):
        if i % 100 == 0:
            print(i)
        Psivector = bas_fct[i]

        # TODO for now we only keep the corono structure,
        # we should use testbed.todetector(entrance_EF= input_wavefront, DMXX = Psivector)
        # directly
        #
        # also i and k are the same indice I think :-)
        Gvector = (testbed.corono.todetector(entrance_EF=input_wavefront * 1j *
                                             Psivector) /
                   np.sqrt(testbed.maxPSF))
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


def createdifference(input_wavefront,
                     posprobes,
                     pushact,
                     testbed,
                     dimimages,
                     DM1phase=0,
                     DM3phase=0,
                     noise=False,
                     numphot=1e30):
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
    Ikmoins = np.zeros((testbed.dim_im, testbed.dim_im))
    Ikplus = np.zeros((testbed.dim_im, testbed.dim_im))
    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    ## To convert in photon flux
    # This will be replaced by transmission!

    contrast_to_photons = (np.sum(testbed.entrancepupil.pup) /
                           np.sum(testbed.corono.lyot_pup.pup) * numphot *
                           testbed.maxPSF / testbed.sumPSF)

    dim_pup = testbed.dim_overpad_pupil
    input_wavefront *= testbed.entrancepupil.pup

    k = 0
    for i in posprobes:
        probephase = proc.crop_or_pad_image(pushact[i], dim_pup)

        Ikmoins = testbed.todetector_Intensity(
            entrance_EF=input_wavefront,
            DM1phase=DM1phase,
            DM3phase=DM3phase - probephase) / testbed.maxPSF
        #Ikmoins = np.abs(testbed.corono.todetector(entrance_EF=
        #        input_wavefront * np.exp(-1j * probephase)))**2 / testbed.maxPSF
        # Ikmoins = np.abs(corona_struct.apodtodetector(input_wavefront * np.exp(
        #         -1j * probephase)))**2 / corona_struct.maxPSF

        Ikplus = testbed.todetector_Intensity(
            entrance_EF=input_wavefront,
            DM1phase=DM1phase,
            DM3phase=DM3phase + probephase) / testbed.maxPSF
        #Ikplus = np.abs(testbed.corono.todetector(entrance_EF=
        #        input_wavefront * np.exp(1j * probephase)))**2 / testbed.maxPSF
        # Ikplus = np.abs(
        #     corona_struct.apodtodetector(input_wavefront * np.exp(
        #         1j * probephase)))**2 / corona_struct.maxPSF

        if noise == True:
            Ikplus = (np.random.poisson(Ikplus * contrast_to_photons) /
                      contrast_to_photons)
            Ikmoins = (np.random.poisson(Ikmoins * contrast_to_photons) /
                       contrast_to_photons)

        Ikplus = np.abs(proc.resampling(Ikplus, dimimages))
        Ikmoins = np.abs(proc.resampling(Ikmoins, dimimages))

        Difference[k] = Ikplus - Ikmoins
        k = k + 1
    # useful.quickfits(np.abs(Difference),dir="/home/rgalicher/tt/")
    # azs

    return Difference
