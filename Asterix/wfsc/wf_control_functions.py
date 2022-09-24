# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits

from Asterix.utils import resizing, crop_or_pad_image, save_plane_in_fits, progress
import Asterix.optics.propagation_functions as prop
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed


def create_interaction_matrix(testbed: Testbed,
                              dimEstim,
                              amplitudeEFC,
                              matrix_dir,
                              initial_DM_voltage=0.,
                              input_wavefront=1.,
                              MatrixType='',
                              save_all_planes_to_fits=False,
                              dir_save_all_planes=None,
                              visu=False):
    """
    Create the jacobian matrix for Electric Field Conjugation. The Matrix is not
    limited to the DH size but to the whole FP [dimEstim, dimEstim]. 
    First half is real part, second half is imag part.

    The matrix size is therefore [total(DM.basis_size), 2*dimEstim^2]

    We save the matrix in .fits independently for each DMs, only if the initial 
    wavefront and DMs are flat.

    This code works for all testbeds without prior assumption (if we have at 
    least 1 DM of course). We have optimized the code to only do once the optical 
    elements before the DM movement and repeat only what is after the DMS
    
    AUTHOR : Axel Potier and Johan Mazoyer

    Parameters
    ----------

    testbed: Testbed Optical_element
        testbed structure with at least 1 DM
    
    dimEstim: int
        size of the output image in teh estimator
    
    amplitudeEFC: float
        amplitude of the EFC probe on the DM
    
    matrix_dir : path. 
        save all the matrices here
    
    MatrixType: string
            'smallphase' (when applying modes on the DMs we, do a small phase assumption : exp(i phi) = 1+ i.phi) 
            or 'perfect' (we keep exp(i phi)).
            in both case, if the DMs are not initially flat (non zero initial_DM_voltage), 
                    we do not make the small phase assumption for initial DM phase

    input_wavefront: 1D-array real
        initial DM voltage for all DMs
    
    input_wavefront: 2D-array complex
        input wavefront in pupil plane
    
    save_all_planes_to_fits: Bool, default False.
            if True, save all planes to fits for debugging purposes to dir_save_all_planes
            This can generate a lot of fits especially if in a loop so the code force you
            to define a repository.

    dir_save_all_planes : path, default None. 
                            directory to save all plane
                            in fits if save_all_planes_to_fits = True
    
    visu : bool default false
            if true show the focal plane intensity in 2D for each mode

    Returns
    ------
    InterMat: 2D array of size [total(DM.basis_size), 2*dimEstim^2]
        jacobian matrix for Electric Field Conjugation.

    """
    if isinstance(initial_DM_voltage, (int, float)):
        initial_DM_voltage = np.zeros(testbed.number_act) + float(initial_DM_voltage)

    wavelength = testbed.wavelength_0
    normalisation_testbed_EF_contrast = np.sqrt(
        testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelength)])

    # This is for the case we take a non zero DM vectors already, we need to calculate
    # the initial phase for each DM
    DM_phase_init = testbed.voltage_to_phases(initial_DM_voltage)

    # First run throught the DMs to :
    #   - string matrix to create a name for the matrix
    #   - check total size of basis
    #   - attach the initial phase to each DM
    total_number_basis_modes = 0
    string_testbed_without_DMS = testbed.string_os

    for i, DM_name in enumerate(testbed.name_of_DMs):

        DM = vars(testbed)[DM_name]  # type: DeformableMirror
        total_number_basis_modes += DM.basis_size
        DM_small_str = "_" + "_".join(DM.string_os.split("_")[5:])
        string_testbed_without_DMS = string_testbed_without_DMS.replace(DM_small_str, '')
        #attach the initial phase for each DM
        DM.phase_init = DM_phase_init[i]

    # remove to save memory
    del DM_phase_init

    # Some string manips to name the matrix if we save it
    if MatrixType == 'perfect':
        headfile = "DirectMatrixPerf"
    elif MatrixType == 'smallphase':
        headfile = "DirectMatrixSP"
    else:
        raise Exception("This Matrix type does not exist")

    if DM.basis_type == 'fourier':
        pass
    elif DM.basis_type == 'actuator':
        headfile += "_EFCampl" + str(amplitudeEFC)
    else:
        raise Exception("This Basis type does not exist")
    print("")
    print("Start Interaction Matrix")

    InterMat = np.zeros((2 * int(dimEstim**2), total_number_basis_modes))
    pos_in_matrix = 0

    for DM_name in testbed.name_of_DMs:

        DM = vars(testbed)[DM_name]  # type: DeformableMirror
        DM_small_str = "_" + "_".join(DM.string_os.split("_")[5:])

        basis_str = DM_small_str + "_" + DM.basis_type + "Basis" + str(DM.basis_size)

        fileDirectMatrix = headfile + basis_str + '_dimEstim' + str(dimEstim) + string_testbed_without_DMS

        # We only save the 'first' matrix meaning the one with no initial DM voltages
        # Matrix is saved/loaded for each DM independetly which allow quick switch
        # For 1DM test / 2DM test
        # Matrix is saved/loaded for all the FP and then crop at the good size later

        if os.path.exists(matrix_dir + fileDirectMatrix + ".fits") and (initial_DM_voltage == 0.).all():
            print("The matrix " + fileDirectMatrix + " already exists")

            InterMat[:, pos_in_matrix:pos_in_matrix +
                     DM.basis_size] = fits.getdata(matrix_dir + fileDirectMatrix + ".fits")

            pos_in_matrix += DM.basis_size

        else:
            # We measure the initial Focal plane that will be removed at the end.
            # Be careful because todetector automatically normalized to contrast with the full testbed
            # We checked that this is the same normalization as in Gvector
            G0 = resizing(
                testbed.todetector(entrance_EF=input_wavefront,
                                   voltage_vector=initial_DM_voltage,
                                   save_all_planes_to_fits=save_all_planes_to_fits,
                                   dir_save_all_planes=dir_save_all_planes), dimEstim)

            if (initial_DM_voltage == 0.).all():
                print("")
                print("The matrix " + fileDirectMatrix + " does not exists")

            print("Start " + DM_name)

            # we measure the phases of the Basis we will apply on the DM.
            # In the case of the Fourier Basis, this is a bit long so we load an existing .fits file
            if DM.basis_type == 'fourier':
                sqrtnbract = int(np.sqrt(DM.total_act))
                Name_FourrierBasis_fits = "Fourier_basis_" + DM.Name_DM + '_prad' + str(
                    DM.prad) + '_nact' + str(sqrtnbract) + 'x' + str(sqrtnbract)
                phasesBasis = fits.getdata(DM.Model_local_dir + Name_FourrierBasis_fits + '.fits')

            else:
                phasesBasis = np.zeros((DM.basis_size, DM.dim_overpad_pupil, DM.dim_overpad_pupil))
                for i in range(DM.basis_size):
                    phasesBasis[i] = DM.voltage_to_phase(DM.basis[i]) * amplitudeEFC

            if save_all_planes_to_fits == True:
                # save the basis phase to check what is happening
                name_plane = DM_name + '_' + DM.basis_type + '_basis_Phase'
                save_plane_in_fits(dir_save_all_planes, name_plane, phasesBasis)

            # to be applicable to all Testbed configurations and save time we separate the testbed in 3 parts:
            # - The optics before the DM we want to actuate (these can be propagated through only once)
            # - The DM we want to actuate (if not in PP, the first Fresnel transform can be calculated only once)
            # - The optics after the DM we want to actuate (these have to be propagated through for each phase of the basis)

            positioonDMintestbed = testbed.subsystems.index(DM_name)
            OpticSysNameBefore = testbed.subsystems[:positioonDMintestbed]
            OpticSysNameAfter = testbed.subsystems[positioonDMintestbed + 1:]

            # we go through all subsystems of the testbed

            # First before the DM we want to actuate (aperture, other DMs, etc).
            # This ones, we only do once !
            wavefrontupstream = input_wavefront

            for osname in OpticSysNameBefore:
                OpticSysbefore = vars(testbed)[osname]  # type: OpticalSystem

                if save_all_planes_to_fits == True:
                    # save PP plane before this subsystem
                    name_plane = 'EF_PP_before_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefrontupstream)
                if isinstance(OpticSysbefore, DeformableMirror) and OpticSysbefore.active:
                    # this subsystem is an active DM but not the one we actuate now (located before the one we actuate)

                    if OpticSysbefore.z_position == 0:
                        wavefrontupstream = wavefrontupstream * OpticSysbefore.EF_from_phase_and_ampl(
                            phase_abb=OpticSysbefore.phase_init, wavelengths=wavelength)
                    else:
                        wavefrontupstream = OpticSysbefore.prop_pup_to_DM_and_back(
                            wavefrontupstream, OpticSysbefore.phase_init, wavelength)

                    if save_all_planes_to_fits == True:
                        # save phase on this DM
                        name_plane = 'Phase_init_on_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                        save_plane_in_fits(dir_save_all_planes, name_plane, OpticSysbefore.phase_init)

                else:
                    wavefrontupstream = OpticSysbefore.EF_through(entrance_EF=wavefrontupstream)

                if save_all_planes_to_fits == True:
                    # save PP plane after this subsystem
                    name_plane = 'EF_PP_after_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefrontupstream)

            # then the DM we want to actuate !
            #
            # if the DM is not in pupil plane, we can measure the first Fresnel transf only once
            if DM.z_position != 0:

                wavefrontupstreaminDM = crop_or_pad_image(
                    prop.prop_angular_spectrum(wavefrontupstream, DM.wavelength_0, DM.z_position,
                                               DM.diam_pup_in_m / 2, DM.prad), DM.dim_overpad_pupil)

            if visu:
                plt.ion()
                plt.figure()

            # now we go throught the DM basis
            init_pos_in_matrix = pos_in_matrix  # where we store the next vect in the matrix

            for i in range(DM.basis_size):

                if i % 10:
                    progress(i, DM.basis_size, status='')

                if MatrixType == 'perfect':
                    if DM.z_position == 0:

                        wavefront = wavefrontupstream * DM.EF_from_phase_and_ampl(phase_abb=phasesBasis[i] +
                                                                                  DM.phase_init)

                        if save_all_planes_to_fits == True:
                            name_plane = 'Phase_on_' + DM_name + '_wl{}'.format(int(wavelength * 1e9))
                            save_plane_in_fits(dir_save_all_planes, name_plane, OpticSysbefore.phase_init)

                    else:

                        wavefront = crop_or_pad_image(
                            prop.prop_angular_spectrum(
                                wavefrontupstreaminDM *
                                DM.EF_from_phase_and_ampl(phase_abb=phasesBasis[i] + DM.phase_init),
                                DM.wavelength_0, -DM.z_position, DM.diam_pup_in_m / 2, DM.prad),
                            DM.dim_overpad_pupil)

                if MatrixType == 'smallphase':
                    # TODO we added a 1+ which was initially in Axel's code and that was
                    # removed. Need to be tested with and without on the testbed
                    if DM.z_position == 0:
                        wavefront = (1 + 1j * phasesBasis[i]) * wavefrontupstream * DM.EF_from_phase_and_ampl(
                            phase_abb=DM.phase_init)
                    else:

                        wavefront = crop_or_pad_image(
                            prop.prop_angular_spectrum(
                                wavefrontupstreaminDM * (1 + 1j * phasesBasis[i]) *
                                DM.EF_from_phase_and_ampl(phase_abb=DM.phase_init), DM.wavelength_0,
                                -DM.z_position, DM.diam_pup_in_m / 2, DM.prad), DM.dim_overpad_pupil)

                if save_all_planes_to_fits == True:
                    name_plane = 'EF_PP_after_' + DM_name + '_wl{}'.format(int(wavelength * 1e9))
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefront)
                # and finally we go through the subsystems after the DMs we want to actuate
                # (other DMs, coronagraph, etc). These ones we have to go through for each phase of the Basis
                for osname in OpticSysNameAfter:
                    OpticSysAfter = vars(testbed)[osname]  # type: OpticalSystem
                    if osname != OpticSysNameAfter[-1]:

                        if isinstance(OpticSysAfter, DeformableMirror) and OpticSysAfter.active:

                            # this subsystem is an active DM but not the one we actuate now (located after the one we actuate)
                            if OpticSysAfter.z_position == 0:
                                wavefront = wavefront * OpticSysAfter.EF_from_phase_and_ampl(
                                    phase_abb=OpticSysAfter.phase_init, wavelengths=wavelength)
                            else:
                                wavefront = OpticSysAfter.prop_pup_to_DM_and_back(
                                    wavefront, OpticSysAfter.phase_init, wavelength)

                            if save_all_planes_to_fits == True:
                                name_plane = 'Phase_init_on_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                                save_plane_in_fits(dir_save_all_planes, name_plane, OpticSysAfter.phase_init)

                        else:
                            wavefront = OpticSysAfter.EF_through(entrance_EF=wavefront)

                        if save_all_planes_to_fits == True:
                            name_plane = 'EF_PP_after_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                            save_plane_in_fits(dir_save_all_planes, name_plane, wavefront)
                    else:
                        # this is the last one ! so we propagate to FP and resample to estimation size
                        # we have to be careful with the normalization, by default this is the
                        # normalization of the last optical system (probably the coronograph)
                        # not of the full system (because we went through each optics one by one, not
                        # through the whole system at once). For this reason, we do not use the defaut
                        # automatic normalization (in_contrast=False) but normalize "by hand" using
                        # normalisation_testbed_EF_contrast which is the  max value of the PSF at this
                        # wavelength for the whole testbed. This is the same normalization as G0.

                        Gvector = resizing(
                            OpticSysAfter.todetector(entrance_EF=wavefront, in_contrast=False) /
                            normalisation_testbed_EF_contrast, dimEstim)

                        if save_all_planes_to_fits == True:
                            name_plane = 'FPAfterTestbed_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                            save_plane_in_fits(dir_save_all_planes, name_plane, Gvector)

                # TODO Should we remove the intial FP field G0 in all casese ? For ideal
                # corono and flat DMs, this is 0, but it's not for non ideal coronagraph
                # or if we have a strong initial DM voltages. This needs
                # to be investigated, in simulation and on the testbed
                Gvector = Gvector - G0

                if save_all_planes_to_fits == True:
                    name_plane = 'Gvector_in_matrix_' + osname + '_wl{}'.format(int(wavelength * 1e9))
                    save_plane_in_fits(dir_save_all_planes, name_plane, Gvector)

                if visu:
                    plt.clf()
                    plt.imshow(np.log10(np.abs(Gvector)**2), vmin=-10, vmax=-6)
                    print("Max contrast", np.log10(np.max(np.abs(Gvector)**2)))
                    plt.gca().invert_yaxis()
                    plt.colorbar()
                    plt.pause(0.01)
                # We fill the interaction matrix:
                InterMat[:dimEstim**2, pos_in_matrix] = np.real(Gvector).flatten()
                InterMat[dimEstim**2:, pos_in_matrix] = np.imag(Gvector).flatten()
                # Note that we do not crop to DH. This is done after so that we can change DH more easily
                # without changeing the matrix

                pos_in_matrix += 1

            if visu:
                plt.close()
                plt.ioff()
            # We save the interaction matrix:
            if (initial_DM_voltage == 0.).all():
                fits.writeto(matrix_dir + fileDirectMatrix + ".fits",
                             InterMat[:, init_pos_in_matrix:init_pos_in_matrix + DM.basis_size])

    # clean to save memory
    for i, DM_name in enumerate(testbed.name_of_DMs):
        DM = vars(testbed)[DM_name]  # type: DeformableMirror
        DM.phase_init = 0

    print("")
    print("End Interaction Matrix")
    return InterMat


def crop_interaction_matrix_to_dh(FullInteractionMatrix: np.ndarray, mask: np.ndarray):
    """
    Crop the  Interaction Matrix to the DH mask size.
    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    FullInteractionMatrix: Interaction matrix over the full focal plane

    mask : a binary mask to delimitate the DH

    Returns 
    ------
    DHInteractionMatrix: 2D numpy array
        matrix only inside the DH. first half is real part, second half is imag part

    """
    size_full_matrix = FullInteractionMatrix.shape[0]

    size_DH_matrix = 2 * int(np.sum(mask))
    where_mask_flatten = np.where(mask.flatten() == 1.)
    DHInteractionMatrix = np.zeros((size_DH_matrix, FullInteractionMatrix.shape[1]), dtype=float)

    for i in range(FullInteractionMatrix.shape[1]):
        DHInteractionMatrix[:int(size_DH_matrix / 2), i] = FullInteractionMatrix[:int(size_full_matrix / 2),
                                                                                 i][where_mask_flatten]
        DHInteractionMatrix[int(size_DH_matrix / 2):, i] = FullInteractionMatrix[int(size_full_matrix / 2):,
                                                                                 i][where_mask_flatten]

    return DHInteractionMatrix


def calc_efc_solution(mask, Result_Estimate, inversed_jacobian, testbed: Testbed):
    """
    Voltages to apply on the deformable mirrors in order to minimize the speckle
    intensity in the dark hole region
    
    AUTHOR : Axel Potier

    Parameters
    ----------
    mask:               2D Binary mask 
        corresponding to the dark hole region

    Result_Estimate:    2D array 
        can be complex, focal plane electric field

    inversed_jacobian:  2D array
        inverse of the jacobian matrix linking the
                                    estimation to the basis coefficient

    testbed: Testbed Optical_element
        a testbed with one or more DM

    Returns
    ------
    solution:   1D array
                    voltage to apply on each deformable mirror actuator

    """

    EF_vector = np.zeros(2 * int(np.sum(mask)))
    Resultat_cropdh = Result_Estimate[np.where(mask == 1)]
    EF_vector[0:int(np.sum(mask))] = np.real(Resultat_cropdh).flatten()
    EF_vector[int(np.sum(mask)):] = np.imag(Resultat_cropdh).flatten()
    produit_mat = np.dot(inversed_jacobian, EF_vector)

    return testbed.basis_vector_to_act_vector(produit_mat)


def calc_em_solution(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed: Testbed):
    """
    Voltage to apply on the deformable mirror in order to minimize the speckle
    intensity in the dark hole region
    
    AUTHOR : Axel Potier

    Parameters
    ----------
    mask:               Binary mask
             corresponding to the dark hole region

    Result_Estimate:    2D array
             can be complex, focal plane electric field

    Hessian_Matrix:     2D array 
            Hessian matrix of the DH energy

    Jacobian:           2D array 
            jacobian matrix created linking the
                                    estimation to the basis coefficient

    testbed: Testbed Optical_element
        a testbed with one or more DM

    Returns
    ------
    solution: 1D array
        voltage to apply on each deformable mirror actuator

    """

    # With notations from Potier PhD eq 4.74 p78:
    Eab = Result_Estimate[np.where(mask == 1)]
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Eab)).flatten()
    produit_mat = np.dot(Hessian_Matrix, realb0)

    return testbed.basis_vector_to_act_vector(produit_mat)


def calc_strokemin_solution(mask, Result_Estimate, Jacob_trans_Jacob, Jacobian, DesiredContrast,
                            last_best_alpha, testbed: Testbed):
    """
    Voltage to apply on the deformable mirror in order to minimize the speckle
    intensity in the dark hole region in the stroke min solution
    See Axel Potier Phd for notation and Mazoyer et al. 2018a for alpha search improvement

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    mask:               Binary mask
             corresponding to the dark hole region

    Result_Estimate:    2D array
             can be complex, focal plane electric field

    Jacob_trans_Jacob:     2D array 
            Jabobian.Transpose(Jabobian) matrix

    Jacobian: 2D array
                    jacobian matrix created linking the
                    estimation to the basis coefficient

    DesiredContrast : float
        the contrast value we wish to achieve

    last_best_alpha : float
        starting point for alpha

    testbed: Testbed Optical_element
            a testbed with one or more DM

    Returns
    ------
    solution: 1D array
        voltage to apply on each deformable mirror actuator
    lasbestalpha : float
            we return the last best alpha. This avoid to recalculate the best alpha from scratch
                        at each iteration since it's often a very close value

    """

    pixel_in_mask = np.sum(mask)

    # With notations from Potier PhD eq 4.74 p78:
    Eab = Result_Estimate[np.where(mask == 1)]

    d0 = np.sum(np.abs(Eab)**2) / pixel_in_mask
    M0 = Jacob_trans_Jacob / pixel_in_mask
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Eab)).flatten() / pixel_in_mask

    Identity_M0size = np.identity(M0.shape[0])

    # we put this keyword to True to do at least 1 try
    TestSMfailed = True
    number_time_failed = 0

    while TestSMfailed:

        step_alpha = 1.3  #hard coded but can maybe be changed
        alpha = last_best_alpha * step_alpha**2

        #eq 4.79 Potier Phd
        DMSurfaceCoeff = np.dot(np.linalg.inv(M0 + alpha * Identity_M0size), realb0)

        ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(
            M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0
        CurrentContrast = ResidualEnergy
        iteralpha = 0

        while CurrentContrast > DesiredContrast and alpha > 1e-12:

            # if this counter is not even incremented once, it means that our initial
            # alpha is probably too big
            iteralpha += 1

            alpha = alpha / step_alpha
            # LastDMSurfaceCoeff = DMSurfaceCoeff

            DMSurfaceCoeff = np.dot(np.linalg.inv(M0 + alpha * Identity_M0size), realb0)
            ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(
                M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0

            LastCurrentContrast = CurrentContrast
            CurrentContrast = ResidualEnergy

            if (CurrentContrast > 3 * LastCurrentContrast):
                # this step is to check if the SM is divergeing too quickly
                return "SMFailedTooManyTime", alpha

            print("For alpha={:f}, Current Contrast:{:f}, Last Contrast:{:f}, Desired Contrast: {:f}".format(
                np.log10(alpha), np.log10(CurrentContrast), np.log10(LastCurrentContrast),
                np.log10(DesiredContrast)))

        if iteralpha == 0:
            # we must do at least 1 iteration (the SM found a solution that dig the contrast)
            # or we fail !
            TestSMfailed = True
            number_time_failed += 1
            last_best_alpha *= 10
            print("SM failed, we increase alpha 10 times")
            if number_time_failed > 20:
                return "SMFailedTooManyTime", alpha
        else:
            TestSMfailed = False

    print("Number of iteration in this stroke min (number of tested alpha): {:d}".format(iteralpha))
    return testbed.basis_vector_to_act_vector(DMSurfaceCoeff), alpha


def calc_steepest_solution(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed: Testbed):
    """
    Voltage to apply on the deformable mirror in order to minimize
    the speckle intensity in the dark hole region

    AUTHOR : Axel Potier

    Parameters
    ----------
    mask: Binary mask 
        corresponding to the dark hole region
    Result_Estimate: 2D array 
        can be complex, focal plane electric field
    Hessian_Matrix: 2D array 
         Hessian matrix of the DH energy
    
    Jacobian: 2D array
        inverse of the jacobian matrix linking the
                                    estimation to the basis coefficient
    testbed: Testbed Optical_element
            a testbed with one or more DM

    Returns
    ------
    solution: 1D array
         voltage to apply on each deformable mirror actuator
   
    """

    Eab = np.zeros(int(np.sum(mask)))
    Resultat_cropdh = Result_Estimate[np.where(mask == 1)]
    Eab = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Resultat_cropdh)).flatten()
    pas = 2e3
    solution = pas * 2 * Eab

    return testbed.basis_vector_to_act_vector(solution)
