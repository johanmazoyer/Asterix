import os
import time
from datetime import datetime
import numpy as np
import matplotlib
from IPython import get_ipython
if get_ipython() is None:  # this matplotlib option is just in non-notebook case
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits

from Asterix.utils import resizing, crop_or_pad_image, save_plane_in_fits, progress, from_param_to_header
import Asterix.optics.propagation_functions as prop
from Asterix.optics import OpticalSystem, DeformableMirror, Testbed


def create_interaction_matrix(testbed: Testbed,
                              dimEstim,
                              amplitudeEFC,
                              matrix_dir,
                              initial_DM_voltage=0.,
                              input_wavefront=1.,
                              MatrixType='',
                              polychrom='singlewl',
                              wav_vec_estim=None,
                              dir_save_all_planes=None,
                              silence=False,
                              visu=False):
    """Create the jacobian matrix for Electric Field Conjugation. The Matrix is
    not limited to the DH size but to the whole FP [dimEstim, dimEstim]. First
    half is real part, second half is imag part.

    The matrix size is therefore [total(DM.basis_size), 2*dimEstim^2]

    We save the matrix in .fits independently for each DMs, only if the initial
    wavefront and DMs are flat.

    This code works for all testbeds without prior assumption (if we have at
    least 1 DM of course). We have optimized the code to only do once the optical
    elements before the DM movement and repeat only what is after the DMS

    AUTHOR : Johan Mazoyer

    Sept 2022 : created

    Parameters
    ----------
    testbed : Testbed Optical_element
        testbed structure with at least 1 DM
    dimEstim : int
        size of the output image in teh estimator
    amplitudeEFC : float
        amplitude of the EFC probe on the DM
    matrix_dir : string
        path to directory to save all the matrices here
    MatrixType : string
        'smallphase' (when applying modes on the DMs we, do a small phase assumption : exp(i phi) = 1+ i.phi)
        or 'perfect' (we keep exp(i phi)).
        in both case, if the DMs are not initially flat (non zero initial_DM_voltage),
        we do not make the small phase assumption for initial DM phase
    polychrom : string
        For polychromatic estimation and correction:
        - 'singlewl': only a single wavelength is used for estimation / correction. 1 Interation Matrix
        - 'broadband_pwprobes': probes images PWP are broadband but Matrices are at central wavelength: 1 PWP Matrix and 1 Interation Matrix
        - 'multiwl': nb_wav images are used for estimation and there are nb_wav matrices of estimation and nb_wav matrices for correction
    initial_DM_voltage : 1D-array real
        initial DM voltage for all DMs
    input_wavefront : complex scalar or 2d complex array or 3d complex array. Default is 1 (flat WF)
        Input wavefront in pupil plane
    dir_save_all_planes : string or None, default None
        If not None, absolute directory to save all planes in fits for debugging purposes.
        This can generate a lot of fits especially if in a loop, use with caution.
    silence : boolean, default False.
        Whether to silence print outputs.
    visu : bool default false
        if true show the focal plane intensity in 2D for each mode

    Returns
    --------
    InterMat : 2D array of size [total(DM.basis_size), 2*dimEstim^2]
        jacobian matrix for Electric Field Conjugation.
    """

    ## careful here if we do not want the matrix done exactly at the same wl as the testbed
    if isinstance(input_wavefront, (float, int)):
        input_wavefront = np.repeat(input_wavefront, testbed.nb_wav)
    elif input_wavefront.shape == testbed.wav_vec.shape:
        pass
    elif input_wavefront.shape == (testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
        input_wavefront = np.repeat(input_wavefront[np.newaxis, ...], testbed.nb_wav, axis=0)
    elif input_wavefront.shape == (testbed.nb_wav, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil):
        pass
    else:
        raise TypeError(("input_wavefront must be scalar (same for all WL), or a nb_wav scalars or a "
                         "2D array of size (dim_overpad_pupil, dim_overpad_pupil) or a 3D array of size "
                         "(nb_wav, dim_overpad_pupil, dim_overpad_pupil)"))

    if wav_vec_estim is None:
        wav_vec_estim = testbed.wav_vec

    return_matrix = []

    if polychrom == 'singlewl':

        return_matrix.append(
            create_singlewl_interaction_matrix(testbed,
                                               dimEstim,
                                               amplitudeEFC,
                                               wav_vec_estim[0],
                                               matrix_dir,
                                               initial_DM_voltage=initial_DM_voltage,
                                               input_wavefront=input_wavefront[testbed.wav_vec.tolist().index(
                                                   wav_vec_estim[0])],
                                               MatrixType=MatrixType,
                                               dir_save_all_planes=dir_save_all_planes,
                                               silence=silence,
                                               visu=visu))
    elif polychrom == 'broadband_pwprobes':

        return_matrix.append(
            create_singlewl_interaction_matrix(testbed,
                                               dimEstim,
                                               amplitudeEFC,
                                               testbed.wavelength_0,
                                               matrix_dir,
                                               initial_DM_voltage=initial_DM_voltage,
                                               input_wavefront=input_wavefront[testbed.wav_vec.tolist().index(
                                                   testbed.wavelength_0)],
                                               MatrixType=MatrixType,
                                               dir_save_all_planes=dir_save_all_planes,
                                               silence=silence,
                                               visu=visu))

    elif polychrom == 'multiwl':

        for i, wave_i in enumerate(wav_vec_estim):
            return_matrix.append(
                create_singlewl_interaction_matrix(testbed,
                                                   dimEstim,
                                                   amplitudeEFC,
                                                   wave_i,
                                                   matrix_dir,
                                                   initial_DM_voltage=initial_DM_voltage,
                                                   input_wavefront=input_wavefront[i],
                                                   MatrixType=MatrixType,
                                                   dir_save_all_planes=dir_save_all_planes,
                                                   silence=silence,
                                                   visu=visu))
    else:
        raise ValueError(polychrom + "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")

    return return_matrix


def create_singlewl_interaction_matrix(testbed: Testbed,
                                       dimEstim,
                                       amplitudeEFC,
                                       wavelength,
                                       matrix_dir,
                                       initial_DM_voltage=0.,
                                       input_wavefront=1.,
                                       MatrixType='',
                                       dir_save_all_planes=None,
                                       silence=False,
                                       visu=False):
    """Create the jacobian matrix for Electric Field Conjugation for one
    wavelength. The Matrix is not limited to the DH size but to the whole FP
    [dimEstim, dimEstim]. First half is real part, second half is imag part.

    The matrix size is therefore [total(DM.basis_size), 2*dimEstim^2]

    We save the matrix in .fits independently for each DMs, only if the initial
    wavefront and DMs are flat.

    This code works for all testbeds without prior assumption (if we have at
    least 1 DM of course). We have optimized the code to only do once the optical
    elements before the DM movement and repeat only what is after the DMS

    AUTHOR : Axel Potier and Johan Mazoyer

    Parameters
    ----------
    testbed : Testbed Optical_element
        testbed structure with at least 1 DM
    dimEstim: int
        size of the output image in teh estimator
    amplitudeEFC: float
        amplitude of the EFC probe on the DM
    wavelength : float
        wavelength in m.
    matrix_dir : string
        path to directory to save all the matrices here
    MatrixType: string
        'smallphase' (when applying modes on the DMs we, do a small phase assumption : exp(i phi) = 1+ i.phi)
        or 'perfect' (we keep exp(i phi)).
        in both case, if the DMs are not initially flat (non zero initial_DM_voltage),
        we do not make the small phase assumption for initial DM phase
    initial_DM_voltage : 1D-array real
        initial DM voltage for all DMs
    input_wavefront : 2D complex array or complex scalar. Default is 1 (flat WF)
        Input wavefront in pupil plane
    dir_save_all_planes : string, default None
        If not None, path to directory to save all planes in fits for debugging purposes.
        This can generate a lot of fits especially if in a loop, use with caution.
    silence : boolean, default False.
        Whether to silence print outputs.
    visu : bool default false
        if true, show the focal plane intensity in 2D for each mode.

    Returns
    --------
    InterMat : 2D array of size [total(DM.basis_size), 2*dimEstim^2]
        jacobian matrix for Electric Field Conjugation.
    """
    if isinstance(initial_DM_voltage, (int, float)):
        initial_DM_voltage = np.zeros(testbed.number_act) + float(initial_DM_voltage)

    # wavelength = testbed.wavelength_0
    normalisation_testbed_EF_contrast = np.sqrt(testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelength)])

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

        DM: DeformableMirror = vars(testbed)[DM_name]
        if DM.active:
            total_number_basis_modes += DM.basis_size
        DM_small_str = "_" + "_".join(DM.string_os.split("_")[3:])
        string_testbed_without_DMS = string_testbed_without_DMS.replace(DM_small_str, '')

    testbed.string_testbed_without_DMS = string_testbed_without_DMS

    InterMat = np.zeros((2 * int(dimEstim**2), total_number_basis_modes))
    pos_in_matrix = 0

    for DM_name in testbed.name_of_DMs:
        DM: DeformableMirror = vars(testbed)[DM_name]
        if not DM.active:
            continue

        fileDirectMatrix, header_expected, bool_already_existing_matrix = name_header_efc_matrix(
            testbed, DM, amplitudeEFC, MatrixType, dimEstim, wavelength, matrix_dir)

        DM.fnameDirectMatrix = os.path.join(matrix_dir, fileDirectMatrix + ".fits")

        # We only save the 'first' matrix meaning the one with no initial DM voltages
        # Matrix is saved/loaded for each DM independetly which allow quick switch
        # For 1DM test / 2DM test
        # Matrix is saved/loaded for all the FP and then crop at the good size later

        if bool_already_existing_matrix and (initial_DM_voltage == 0.).all():
            if not silence:
                print("The matrix " + fileDirectMatrix + " already exists")

            InterMat[:, pos_in_matrix:pos_in_matrix + DM.basis_size] = fits.getdata(
                os.path.join(matrix_dir, fileDirectMatrix + ".fits"))

        else:
            start_time = time.time()

            # We measure the initial Focal plane that will be removed at the end.
            # Be careful because todetector automatically normalized to contrast with the full testbed
            # We checked that this is the same normalization as in Gvector
            G0 = resizing(
                testbed.todetector(entrance_EF=input_wavefront,
                                   voltage_vector=initial_DM_voltage,
                                   dir_save_all_planes=dir_save_all_planes), dimEstim)
            if not silence:
                if (initial_DM_voltage == 0.).all():
                    print("")
                    print("The matrix " + fileDirectMatrix + " does not exists")

                print("Start interaction Matrix " + DM_name + ' at ' + str(int(wavelength * 1e9)) +
                      'nm (wait a few 10s of seconds)')

            # we measure the phases of the Basis we will apply on the DM.
            # In the case of the Fourier Basis, this is a bit long so we load an existing .fits file
            if DM.basis_type == 'fourier':
                sqrtnbract = int(np.sqrt(DM.total_act))
                Name_FourrierBasis_fits = "Fourier_basis_" + DM.Name_DM + '_prad' + str(
                    DM.prad) + '_nact' + str(sqrtnbract) + 'x' + str(sqrtnbract)
                phasesBasis = fits.getdata(os.path.join(DM.Model_local_dir, Name_FourrierBasis_fits + '.fits'))

            else:
                phasesBasis = np.zeros((DM.basis_size, DM.dim_overpad_pupil, DM.dim_overpad_pupil))
                for i in range(DM.basis_size):
                    phasesBasis[i] = DM.voltage_to_phase(DM.basis[i]) * amplitudeEFC

            if dir_save_all_planes is not None:
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

            # I tried something different to be able to parralellize the function, but it did not
            # allow multi matrix (which requires non flat DM initially) and it did not work because
            # functions have to be defined outside of the function to be pickable. However, this is a
            # very compact way to do that so I think this is interresting sop I keep it here for now.

            # OpticSysbefore = []
            # for osname in OpticSysNameBefore:
            #     OpticSysbefore.append(vars(testbed)[osname])
            # testbed_upstream = Testbed(OpticSysbefore, OpticSysNameBefore)

            # OpticSysafter = []
            # for osname in OpticSysNameAfter:
            #     OpticSysafter.append(vars(testbed)[osname])
            # testbed_downstream = Testbed(OpticSysafter, OpticSysNameAfter)

            # # we go through all subsystems of the testbed

            # # First before the DM we want to actuate (aperture, other DMs, etc).
            # # This ones, we only do once !
            # wavefrontupstream = input_wavefront

            # wavefrontupstream = testbed_upstream.EF_through(entrance_EF=wavefrontupstream, wavelength=wavelength)

            # if DM.z_position != 0:

            #     wavefrontupstreaminDM = crop_or_pad_image(
            #         prop.prop_angular_spectrum(wavefrontupstream, wavelength, DM.z_position, DM.diam_pup_in_m / 2,
            #                                    DM.prad), DM.dim_overpad_pupil)

            # if MatrixType == 'perfect':
            #     if DM.z_position == 0:

            #         InterMat[:, pos_in_matrix + DM.basis_size] = np.array(
            #             map(
            #                 concat_flat_real_imag(
            #                     resizing(
            #                         testbed_downstream.todetector(
            #                             entrance_EF=wavefrontupstream * DM.EF_from_phase_and_ampl(
            #                                 phase_abb=phase_in_basis + DM_phase_init[testbed.name_of_DMs.index(DM_name)],
            #                                 wavelengths=wavelength),
            #                             wavelength=wavelength,
            #                             in_contrast=False) / normalisation_testbed_EF_contrast, dimEstim) - G0),
            #                 [phase_in_basis for phase_in_basis in phasesBasis]))
            #     else:
            #         InterMat[:, pos_in_matrix + DM.basis_size] = np.array(
            #             map(
            #                 concat_flat_real_imag(
            #                     resizing(
            #                         testbed_downstream.todetector(entrance_EF=crop_or_pad_image(
            #                             prop.prop_angular_spectrum(
            #                                 wavefrontupstreaminDM *
            #                                 DM.EF_from_phase_and_ampl(phase_abb=phase_in_basis +
            #                                                           DM_phase_init[testbed.name_of_DMs.index(DM_name)],
            #                                                           wavelengths=wavelength), wavelength,
            #                                 -DM.z_position, DM.diam_pup_in_m / 2, DM.prad), DM.dim_overpad_pupil),
            #                                                       wavelength=wavelength,
            #                                                       in_contrast=False) /
            #                         normalisation_testbed_EF_contrast, dimEstim) - G0),
            #                 [phase_in_basis for phase_in_basis in phasesBasis]))

            # if MatrixType == 'smallphase':
            #     if DM.z_position == 0:
            #         InterMat[:, pos_in_matrix + DM.basis_size] = np.array(
            #             map(
            #                 concat_flat_real_imag(
            #                     resizing(
            #                         testbed_downstream.todetector(entrance_EF=(
            #                             1 + 1j * phase_in_basis) * wavefrontupstream * DM.EF_from_phase_and_ampl(
            #                                 phase_abb=DM_phase_init[testbed.name_of_DMs.index(DM_name)],
            #                                 wavelengths=wavelength),
            #                                                       wavelength=wavelength,
            #                                                       in_contrast=False) /
            #                         normalisation_testbed_EF_contrast, dimEstim) - G0),
            #                 [phase_in_basis for phase_in_basis in phasesBasis]))

            #     else:
            #         InterMat[:, pos_in_matrix + DM.basis_size] = np.array(
            #             map(
            #                 concat_flat_real_imag(
            #                     resizing(
            #                         testbed_downstream.todetector(entrance_EF=crop_or_pad_image(
            #                             prop.prop_angular_spectrum(
            #                                 wavefrontupstreaminDM *
            #                                 (1 + 1j * phase_in_basis) * DM.EF_from_phase_and_ampl(
            #                                     phase_abb=DM_phase_init[testbed.name_of_DMs.index(DM_name)],
            #                                     wavelengths=wavelength), wavelength, -DM.z_position, DM.diam_pup_in_m /
            #                                 2, DM.prad), DM.dim_overpad_pupil),
            #                                                       wavelength=wavelength,
            #                                                       in_contrast=False) /
            #                         normalisation_testbed_EF_contrast, dimEstim) - G0),
            #                 [phase_in_basis for phase_in_basis in phasesBasis]))

            # we go through all subsystems of the testbed

            # First before the DM we want to actuate (aperture, other DMs, etc).
            # This ones, we only do once !
            wavefrontupstream = input_wavefront
            for osname in OpticSysNameBefore:
                OpticSysbefore: OpticalSystem = vars(testbed)[osname]

                if dir_save_all_planes is not None:
                    # save PP plane before this subsystem
                    name_plane = 'EF_PP_before_' + osname + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefrontupstream)

                if isinstance(OpticSysbefore, DeformableMirror) and OpticSysbefore.active:
                    # this subsystem is an active DM but not the one we actuate now (located before the one we actuate)

                    if OpticSysbefore.z_position == 0:
                        wavefrontupstream = wavefrontupstream * OpticSysbefore.EF_from_phase_and_ampl(
                            phase_abb=DM_phase_init[testbed.name_of_DMs.index(osname)], wavelengths=wavelength)
                    else:
                        wavefrontupstream = OpticSysbefore.prop_pup_to_DM_and_back(
                            wavefrontupstream, DM_phase_init[testbed.name_of_DMs.index(osname)], wavelength)

                    if dir_save_all_planes is not None:
                        # save phase on this DM
                        name_plane = 'Phase_init_on_' + osname + f'_wl{int(wavelength * 1e9)}'
                        save_plane_in_fits(dir_save_all_planes, name_plane,
                                           DM_phase_init[testbed.name_of_DMs.index(osname)])

                else:
                    wavefrontupstream = OpticSysbefore.EF_through(entrance_EF=wavefrontupstream, wavelength=wavelength)

                if dir_save_all_planes is not None:
                    # save PP plane after this subsystem
                    name_plane = 'EF_PP_after_' + osname + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefrontupstream)

            # then the DM we want to actuate !

            # if the DM is not in pupil plane, we can measure the first Fresnel transf only once
            if DM.z_position != 0:

                wavefrontupstreaminDM = crop_or_pad_image(
                    prop.prop_angular_spectrum(wavefrontupstream,
                                               wavelength,
                                               DM.z_position,
                                               DM.diam_pup_in_m / 2,
                                               DM.prad,
                                               dtype_complex=testbed.dtype_complex), DM.dim_overpad_pupil)

            if visu:
                plt.ion()
                plt.figure()

            # now we go throught the DM basis

            for i, phase_in_basis in enumerate(phasesBasis):
                # for i in range(DM.basis_size):

                if not silence:
                    if i % 10:
                        progress(i, DM.basis_size, status='')

                if MatrixType == 'perfect':
                    if DM.z_position == 0:

                        wavefront = wavefrontupstream * DM.EF_from_phase_and_ampl(
                            phase_abb=phase_in_basis + DM_phase_init[testbed.name_of_DMs.index(DM_name)],
                            wavelengths=wavelength)

                        if dir_save_all_planes is not None:
                            name_plane = 'Phase_on_' + DM_name + f'_wl{int(wavelength * 1e9)}'
                            save_plane_in_fits(dir_save_all_planes, name_plane,
                                               DM_phase_init[testbed.name_of_DMs.index(DM_name)])

                    else:
                        efDMplane = wavefrontupstreaminDM * DM.EF_from_phase_and_ampl(
                            phase_abb=phase_in_basis + DM_phase_init[testbed.name_of_DMs.index(DM_name)],
                            wavelengths=wavelength)
                        wavefront = crop_or_pad_image(
                            prop.prop_angular_spectrum(efDMplane,
                                                       wavelength,
                                                       -DM.z_position,
                                                       DM.diam_pup_in_m / 2,
                                                       DM.prad,
                                                       dtype_complex=testbed.dtype_complex), DM.dim_overpad_pupil)

                if MatrixType == 'smallphase':
                    # TODO we added a 1+ which was initially in Axel's code and that was
                    # removed. Need to be tested with and without on the testbed
                    if DM.z_position == 0:
                        wavefront = (1 + 1j * phase_in_basis) * wavefrontupstream * DM.EF_from_phase_and_ampl(
                            phase_abb=DM_phase_init[testbed.name_of_DMs.index(DM_name)], wavelengths=wavelength)
                    else:
                        efDMplane = wavefrontupstreaminDM * (1 + 1j * phase_in_basis) * DM.EF_from_phase_and_ampl(
                            phase_abb=DM_phase_init[testbed.name_of_DMs.index(DM_name)], wavelengths=wavelength)
                        wavefront = crop_or_pad_image(
                            prop.prop_angular_spectrum(efDMplane,
                                                       wavelength,
                                                       -DM.z_position,
                                                       DM.diam_pup_in_m / 2,
                                                       DM.prad,
                                                       dtype_complex=testbed.dtype_complex), DM.dim_overpad_pupil)

                if dir_save_all_planes is not None:
                    name_plane = 'EF_PP_after_' + DM_name + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, wavefront)
                # and finally we go through the subsystems after the DMs we want to actuate
                # (other DMs, coronagraph, etc). These ones we have to go through for each phase of the Basis
                for osname in OpticSysNameAfter:
                    OpticSysAfter: OpticalSystem = vars(testbed)[osname]
                    if osname != OpticSysNameAfter[-1]:

                        if isinstance(OpticSysAfter, DeformableMirror) and OpticSysAfter.active:

                            # this subsystem is an active DM but not the one we actuate now (located after the one we actuate)
                            if OpticSysAfter.z_position == 0:
                                wavefront = wavefront * OpticSysAfter.EF_from_phase_and_ampl(
                                    phase_abb=DM_phase_init[testbed.name_of_DMs.index(osname)], wavelengths=wavelength)
                            else:
                                wavefront = OpticSysAfter.prop_pup_to_DM_and_back(
                                    wavefront, DM_phase_init[testbed.name_of_DMs.index(osname)], wavelength)

                            if dir_save_all_planes is not None:
                                name_plane = 'Phase_init_on_' + osname + f'_wl{int(wavelength * 1e9)}'
                                save_plane_in_fits(dir_save_all_planes, name_plane,
                                                   DM_phase_init[testbed.name_of_DMs.index(osname)])

                        else:
                            wavefront = OpticSysAfter.EF_through(entrance_EF=wavefront, wavelength=wavelength)

                        if dir_save_all_planes is not None:
                            name_plane = 'EF_PP_after_' + osname + f'_wl{int(wavelength * 1e9)}'
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
                            OpticSysAfter.todetector(entrance_EF=wavefront, wavelength=wavelength, in_contrast=False) /
                            normalisation_testbed_EF_contrast, dimEstim)

                        if dir_save_all_planes is not None:
                            name_plane = 'FPAfterTestbed_' + osname + f'_wl{int(wavelength * 1e9)}'
                            save_plane_in_fits(dir_save_all_planes, name_plane, Gvector)

                # TODO Should we remove the intial FP field G0 in all casese ? For ideal
                # corono and flat DMs, this is 0, but it's not for non ideal coronagraph
                # or if we have a strong initial DM voltages. This needs
                # to be investigated, in simulation and on the testbed
                Gvector = Gvector - G0

                if dir_save_all_planes is not None:
                    name_plane = 'Gvector_in_matrix_' + osname + f'_wl{int(wavelength * 1e9)}'
                    save_plane_in_fits(dir_save_all_planes, name_plane, Gvector)

                if visu:
                    plt.clf()
                    plt.imshow(np.log10(np.abs(Gvector)**2), vmin=-10, vmax=-6)
                    print("Max contrast", np.log10(np.max(np.abs(Gvector)**2)))
                    plt.colorbar()
                    plt.pause(0.01)
                # We fill the interaction matrix:
                InterMat[:dimEstim**2, pos_in_matrix + i] = np.real(Gvector).flatten()
                InterMat[dimEstim**2:, pos_in_matrix + i] = np.imag(Gvector).flatten()
                # Note that we do not crop to DH. This is done after so that we can change DH more easily
                # without changeing the matrix

            if visu:
                plt.close()
                plt.ioff()
            # We save the interaction matrix:
            if (initial_DM_voltage == 0.).all():
                fits.writeto(os.path.join(matrix_dir, fileDirectMatrix + ".fits"),
                             InterMat[:, pos_in_matrix:pos_in_matrix + DM.basis_size],
                             header=header_expected,
                             overwrite=True)

            if not silence:
                print("")
                print("Time for interaction Matrix " + DM_name + " (s):", round(time.time() - start_time))
                print("")

        pos_in_matrix += DM.basis_size

    return InterMat


def name_header_efc_matrix(testbed: Testbed, DM: DeformableMirror, amplitudeEFC, MatrixType, dimEstim, wavelength,
                           matrix_dir):
    """ Create the name of the file and header of the EFC matrix that will be used to
    identify precisely parameters used in the calcul of this matrix.
    We then load a potential matrix and compare the header to the one existing and return a false boolean
    if they need to be recalculated.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    testbed : Testbed Optical_element
        a testbed with one or more DM
    DM : DM Optical_element
        the DM used to create the matrix

    amplitudeEFC: float
        amplitude of the EFC probe on the DM
    MatrixType: string
        'smallphase' (when applying modes on the DMs we, do a small phase assumption : exp(i phi) = 1+ i.phi)
        or 'perfect' (we keep exp(i phi)).
        in both case, if the DMs are not initially flat (non zero initial_DM_voltage),
        we do not make the small phase assumption for initial DM phase
    dimEstim: int
        size of the output image in teh estimator
    wavelength : float
        wavelength in m.
    matrix_dir : string
        path to directory to save all the matrices here

    Returns
    --------
    filePW : string
        file name of the efc matrix. If there is no identical matrix already measured in fits file, None.
    header : fits.Header() Object
        header of the efc matrix
    bool_already_existing_matrix :bool
        If there is no identical matrix already saved in fits file.
    """

    # Some string manips to name the matrix if we save it
    if MatrixType == 'perfect':
        headfile = "DirectMatPerf"
    elif MatrixType == 'smallphase':
        headfile = "DirectMatSP"
    else:
        raise ValueError("This Matrix type does not exist ([Correctionconfig]['MatrixType'] parameter).")

    if DM.basis_type == 'fourier':
        basis_type_str = 'Four'
        pass
    elif DM.basis_type == 'actuator':
        basis_type_str = 'Actu'
        headfile += "_EFCampl" + str(amplitudeEFC)
    else:
        raise ValueError("This basis type does not exist ([Correctionconfig]['DM_basis'] parameter).")

    DM_small_str = "_" + "_".join(DM.string_os.split("_")[3:])
    basis_str = DM_small_str + "_" + basis_type_str + "Basis" + str(DM.basis_size)
    fileDirectMatrix = headfile + basis_str + '_binEstim' + str(int(np.round(
        testbed.dimScience / dimEstim))) + testbed.string_testbed_without_DMS + "_resFP" + str(
            round(DM.Science_sampling / DM.wavelength_0 * wavelength, 2)) + '_wl' + str(int(wavelength * 1e9))

    header = fits.Header()
    header.insert(0, ('date_mat', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "matrix creation date"))
    # we load the configuration to save it in the fits
    if hasattr(testbed, 'config_file'):

        necessary_correc_param = dict()
        necessary_correc_param['DM4matrix'] = DM.Name_DM
        necessary_correc_param['DM_basis'] = DM.basis_type
        necessary_correc_param['MatrixType'] = MatrixType
        necessary_correc_param['amplitudeEFC'] = amplitudeEFC
        header = from_param_to_header(necessary_correc_param, header)

        necessary_estim_param = dict()
        necessary_estim_param['Estim_bin_factor'] = testbed.config_file["Estimationconfig"]["Estim_bin_factor"]
        necessary_estim_param['dimEstim'] = dimEstim
        necessary_estim_param['Estim_wl'] = wavelength * 1e9
        header = from_param_to_header(necessary_estim_param, header)

        additional_model_param = dict()
        additional_model_param['dtype_complex'] = testbed.dtype_complex
        header = from_param_to_header(additional_model_param, header)

        header = from_param_to_header(testbed.config_file["modelconfig"], header)

        necessary_dm_param = dict()
        for paramdmkey in testbed.config_file["DMconfig"].scalars:
            if DM.Name_DM in paramdmkey:
                necessary_dm_param[paramdmkey] = testbed.config_file["DMconfig"][paramdmkey]
        header = from_param_to_header(necessary_dm_param, header)

        header = from_param_to_header(testbed.config_file["Coronaconfig"], header)

        # Loading any existing matrix and comparing their headers to make sure they are created
        # using the same set of parameters
        if os.path.exists(os.path.join(matrix_dir, fileDirectMatrix + ".fits")):
            header_existing = fits.getheader(os.path.join(matrix_dir, fileDirectMatrix + ".fits"))
            # remove the basis kw created automatically  when saving the fits file
            for keyw in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2']:
                header_existing.remove(keyw)
            # we comapre the header (ignoring the date)
            bool_already_existing_matrix = fits.HeaderDiff(header_existing, header,
                                                           ignore_keywords=['DATE_MAT']).identical
        else:
            bool_already_existing_matrix = False

        if bool_already_existing_matrix:
            return fileDirectMatrix, header, bool_already_existing_matrix
        else:
            return fileDirectMatrix, header, bool_already_existing_matrix


def crop_interaction_matrix_to_dh(FullInteractionMatrix: np.ndarray, mask: np.ndarray):
    """
    Crop the  Interaction Matrix to the DH mask size.
    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    FullInteractionMatrix: Interaction matrix over the full focal plane
    mask : 2D numpy array
        a binary mask to delimitate the DH

    Returns
    --------
    DHInteractionMatrix : 2D numpy array
        matrix only inside the DH. first half is real part, second half is imag part

    """

    number_wl_matrix = len(FullInteractionMatrix)
    twice_size_FP_pix = FullInteractionMatrix[0].shape[0]
    number_actu = FullInteractionMatrix[0].shape[1]

    size_DH_pix = int(np.sum(mask))
    where_mask_flatten = np.where(mask.flatten() == 1.)
    DHInteractionMatrix = np.zeros((2 * size_DH_pix * number_wl_matrix, number_actu), dtype=float)

    for i in range(number_actu):
        for j_wave in range(number_wl_matrix):

            DHInteractionMatrix[2 * j_wave * int(size_DH_pix):(2 * j_wave + 1) * int(size_DH_pix),
                                i] = FullInteractionMatrix[j_wave][:int(twice_size_FP_pix) // 2, i][where_mask_flatten]
            DHInteractionMatrix[(2 * j_wave + 1) * int(size_DH_pix):(2 * j_wave + 2) * int(size_DH_pix),
                                i] = FullInteractionMatrix[j_wave][int(twice_size_FP_pix) // 2:, i][where_mask_flatten]

    return DHInteractionMatrix


def calc_efc_solution(mask, Result_Estimate, inversed_jacobian, testbed: Testbed):
    """Voltages to apply on the deformable mirrors in order to minimize the
    speckle intensity in the dark hole region.

    AUTHOR : Axel Potier

    Parameters
    ----------
    mask : 2D Binary mask
        Dark hole region
    Result_Estimate :  list of 2D complex array
        List is the number of wl in the estimation, usually 1 or testbed.nb_wav
        Each arrays are of size of sixe [dimEstim, dimEstim].
        estimation of focal plane EF
    inversed_jacobian :  2D array
        inverse of the jacobian matrix linking the estimation to the basis coefficient
    testbed : Testbed Optical_element
        a testbed with one or more DM

    Returns
    --------
    solution : 1D array
        voltage to apply on each deformable mirror actuator
    """
    EF_vector = np.zeros(2 * int(np.sum(mask)) * len(Result_Estimate))

    for i_wave, estimate_at_this_wave in enumerate(Result_Estimate):
        Resultat_cropdh = estimate_at_this_wave[np.where(mask == 1)]
        EF_vector[i_wave * 2 * int(np.sum(mask)):(i_wave * 2 + 1) *
                  int(np.sum(mask))] = np.real(Resultat_cropdh).flatten()
        EF_vector[(i_wave * 2 + 1) * int(np.sum(mask)):(i_wave * 2 + 2) *
                  int(np.sum(mask))] = np.imag(Resultat_cropdh).flatten()

    produit_mat = np.dot(inversed_jacobian, EF_vector)

    return testbed.basis_vector_to_act_vector(produit_mat)


def calc_em_solution(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed: Testbed):
    """Voltage to apply on the deformable mirror in order to minimize the
    speckle intensity in the dark hole region.

    AUTHOR : Axel Potier

    Parameters
    ----------
    mask : Binary mask
        Dark hole region.
    Result_Estimate : list of 2D complex array
        List is the number of wl in the estimation, usually 1 or testbed.nb_wav
        Each arrays are of size of sixe [dimEstim, dimEstim].
        estimation of focal plane EF.
    Hessian_Matrix : 2D array
        Hessian matrix of the DH energy.
    Jacobian : 2D array
        Jacobian matrix created linking the estimation to the basis coefficient.
    testbed : Testbed Optical_element
        Testbed with one or more DM.

    Returns
    --------
    solution : 1D array
        Voltage to apply on each deformable mirror actuator.
    """
    if len(Result_Estimate) > 1:
        raise ValueError("EM correction is not working in polychromatic mode.")

    # With notations from Potier PhD eq 4.74 p78:
    Eab = Result_Estimate[0][np.where(mask == 1)]
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Eab)).flatten()
    produit_mat = np.dot(Hessian_Matrix, realb0)

    return testbed.basis_vector_to_act_vector(produit_mat)


def calc_strokemin_solution(mask,
                            Result_Estimate,
                            Jacob_trans_Jacob,
                            Jacobian,
                            DesiredContrast,
                            last_best_alpha,
                            testbed: Testbed,
                            silence=False):
    """Voltage to apply on the deformable mirror in order to minimize the
    speckle intensity in the dark hole region in the stroke min solution See
    Axel Potier Phd for notation and Mazoyer et al. 2018a for alpha search
    improvement.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    mask : Binary mask
        Dark hole region.
    Result_Estimate : list of 2D complex array
        list is the number of wl in the estimation, usually 1 or testbed.nb_wav
        Each arrays are of size of sixe [dimEstim, dimEstim].
        estimation of focal plane EF.
    Jacob_trans_Jacob : 2D array
        Jabobian.Transpose(Jabobian) matrix.
    Jacobian : 2D array
        Jacobian matrix created linking the estimation to the basis coefficient.
    DesiredContrast : float
        The contrast value we wish to achieve.
    last_best_alpha : float
        Starting point for alpha.
    testbed : Testbed Optical_element
        Testbed with one or more DM.
    silence : boolean, default False.
        Whether to silence print outputs.

    Returns
    --------
    solution : 1D array
        Voltage to apply on each deformable mirror actuator.
    lasbestalpha : float
        The last best alpha. This avoid to recalculate the best alpha from scratch
        at each iteration since it's often a very close value.
    """

    pixel_in_mask = np.sum(mask)

    # With notations from Potier PhD eq 4.74 p78:
    Eab = np.zeros(int(np.sum(mask)) * len(Result_Estimate), dtype=testbed.dtype_complex)
    for i_wave, estimate_at_this_wave in enumerate(Result_Estimate):
        Eab[i_wave * int(np.sum(mask)):(i_wave + 1) * int(np.sum(mask))] = estimate_at_this_wave[np.where(mask == 1)]

    d0 = np.sum(np.abs(Eab)**2) / pixel_in_mask
    M0 = Jacob_trans_Jacob / pixel_in_mask
    realb0 = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Eab)).flatten() / pixel_in_mask
    Identity_M0size = np.identity(M0.shape[0])

    # we put this keyword to True to do at least 1 try
    TestSMfailed = True
    number_time_failed = 0

    while TestSMfailed:

        step_alpha = 1.3  # hard coded but can maybe be changed
        alpha = last_best_alpha * step_alpha**2

        # eq 4.79 Potier Phd
        DMSurfaceCoeff = np.dot(np.linalg.inv(M0 + alpha * Identity_M0size), realb0)

        ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0
        CurrentContrast = ResidualEnergy
        iteralpha = 0

        while CurrentContrast > DesiredContrast and alpha > 1e-12:

            # if this counter is not even incremented once, it means that our initial
            # alpha is probably too big
            iteralpha += 1

            alpha = alpha / step_alpha
            # LastDMSurfaceCoeff = DMSurfaceCoeff

            DMSurfaceCoeff = np.dot(np.linalg.inv(M0 + alpha * Identity_M0size), realb0)
            ResidualEnergy = np.dot(DMSurfaceCoeff, np.dot(M0, DMSurfaceCoeff)) - 2 * np.dot(realb0, DMSurfaceCoeff) + d0

            LastCurrentContrast = CurrentContrast
            CurrentContrast = ResidualEnergy

            if (CurrentContrast > 3 * LastCurrentContrast):
                # this step is to check if the SM is divergeing too quickly
                return "SMFailedTooManyTime", alpha

            if not silence:
                print(f"For alpha={np.log10(alpha):f}, " + f"Current Contrast:{np.log10(CurrentContrast):f}, " +
                      f"Last Contrast:{np.log10(LastCurrentContrast):f}, " +
                      f"Desired Contrast: {np.log10(DesiredContrast):f}")

        if iteralpha == 0:
            # we must do at least 1 iteration (the SM found a solution that dig the contrast)
            # or we fail !
            TestSMfailed = True
            number_time_failed += 1
            last_best_alpha *= 10
            if not silence:
                print("SM failed, we increase alpha 10 times")
            if number_time_failed > 20:
                return "SMFailedTooManyTime", alpha
        else:
            TestSMfailed = False
    if not silence:
        print(f"Number of iterations in this stroke min (number of tested alpha): {iteralpha:d}")
    return testbed.basis_vector_to_act_vector(DMSurfaceCoeff), alpha


def calc_steepest_solution(mask, Result_Estimate, Hessian_Matrix, Jacobian, testbed: Testbed):
    """Voltage to apply on the deformable mirror in order to minimize the
    speckle intensity in the dark hole region.

    AUTHOR : Axel Potier

    Parameters
    ----------
    mask : Binary mask
        Dark hole region.
    Result_Estimate : list of 2D complex array
        List is the number of wl in the estimation, usually 1 or testbed.nb_wav
        Each arrays are of size of sixe [dimEstim, dimEstim].
        estimation of focal plane EF.
    Hessian_Matrix : 2D array
        Hessian matrix of the DH energy.
    Jacobian : 2D array
        Inverse of the jacobian matrix linking the estimation to the basis coefficient.
    testbed : Testbed Optical_element
        Testbed with one or more DM.

    Returns
    --------
    solution : 1D array
        Voltage to apply on each deformable mirror actuator.
    """
    if len(Result_Estimate) > 1:
        raise ValueError("Steepest correction is not working in polychromatic mode.")

    Eab = np.zeros(int(np.sum(mask)))
    Resultat_cropdh = Result_Estimate[0][np.where(mask == 1)]
    Eab = np.real(np.dot(np.transpose(np.conjugate(Jacobian)), Resultat_cropdh)).flatten()
    pas = 2e3
    solution = pas * 2 * Eab

    return testbed.basis_vector_to_act_vector(solution)


# def concat_flat_real_imag(complex_arr):
#     """From a given complex numpy array, this function separates and flatten RE and IM parts and concatenates them.

#     AUTHORS: J Mazoyer

#     19/12/2022 : Introduction in asterix

#     Parameters
#     ----------
#     complex_arr : 2D complex numpy array
#         Initial array, must be complex

#     Returns
#     --------
#     flatten_concat_array : 1D real numpy array
#         zoomed out array
#     """

#     return np.concatenate((np.real(complex_arr).flatten(), np.imag(complex_arr).flatten()))
