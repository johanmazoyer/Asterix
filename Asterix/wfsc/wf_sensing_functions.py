import os
import time
import numpy as np

from astropy.io import fits

from Asterix.utils import resizing, invert_svd, save_plane_in_fits
from Asterix.optics import DeformableMirror, Testbed


def create_pw_matrix(testbed: Testbed,
                     amplitude,
                     posprobes,
                     dimEstim,
                     cutsvd,
                     matrix_dir,
                     polychrom,
                     wav_vec_estim=None,
                     silence=False,
                     **kwargs):
    """Build the nbwl times interaction matrix for pair-wise probing.

    AUTHOR : Johan Mazoyer

    Sept 2022 : created

    Parameters
    ----------
    testbed : Testbed Optical_element
        a testbed with one or more DM
    amplitude : float
        amplitude of the actuator pokes for pair(wise probing in nm
    posprobes : 1D-array
        index of the actuators to push and pull for pair-wise probing
    dimEstim :  int
        size of the output image after resizing in pixels
    cutsvd : float
        value not to exceed for the inverse eigeinvalues at each pixels
    matrix_dir : string
        path to directory to save all the matrices here
    polychrom : string
        For polychromatic estimation and correction:
        - 'singlewl': only a single wavelength is used for estimation / correction. 1 Interation Matrix
        - 'broadband_pwprobes': probes images PW are broadband but Matrices are at central wavelength: 1 PW Matrix and 1 Interation Matrix
        - 'multiwl': nb_wav images are used for estimation and there are nb_wav matrices of estimation and nb_wav matrices for correction
    wav_vec_estim : list of float, default None
        list of wavelengths to do the estimation, used in the case of polychrom == 'multiwl'
    silence : boolean, default False.
        Whether to silence print outputs.

    Returns
    --------
    PWVector : list of 2D array
                vector probe to be multiplied by the image difference
                matrix in order to retrieve the focal plane electric field
    """
    if wav_vec_estim is None:
        wav_vec_estim = testbed.wav_vec

    return_matrix = []

    if polychrom == 'singlewl':
        return_matrix.append(
            create_singlewl_pw_matrix(testbed,
                                      amplitude,
                                      posprobes,
                                      dimEstim,
                                      cutsvd,
                                      matrix_dir,
                                      wavelength=wav_vec_estim[0],
                                      silence=silence,
                                      **kwargs))

    elif polychrom == 'broadband_pwprobes':
        return_matrix.append(
            create_singlewl_pw_matrix(testbed,
                                      amplitude,
                                      posprobes,
                                      dimEstim,
                                      cutsvd,
                                      matrix_dir,
                                      wavelength=testbed.wavelength_0,
                                      silence=silence,
                                      **kwargs))

    elif polychrom == 'multiwl':
        for wave_i in wav_vec_estim:
            return_matrix.append(
                create_singlewl_pw_matrix(testbed,
                                          amplitude,
                                          posprobes,
                                          dimEstim,
                                          cutsvd,
                                          matrix_dir,
                                          wavelength=wave_i,
                                          silence=silence,
                                          **kwargs))
    else:
        raise ValueError(polychrom + "is not a valid value for [Estimationconfig]['polychromatic'] parameter.")

    return return_matrix


def create_singlewl_pw_matrix(testbed: Testbed,
                              amplitude,
                              posprobes,
                              dimEstim,
                              cutsvd,
                              matrix_dir,
                              wavelength,
                              silence=False,
                              **kwargs):
    """Build the interaction matrix for pair-wise probing at 1 WL.

    AUTHOR : Axel Potier
    Modified by Johan Mazoyer

    Sept 2022 : .fits file saving directly in the function to clean up estimator

    Parameters
    ----------
    testbed : Testbed Optical_element
        a testbed with one or more DM
    amplitude : float
        amplitude of the actuator pokes for pair(wise probing in nm
    posprobes : 1D-array
        index of the actuators to push and pull for pair-wise probing
    dimEstim : int
        size of the output image after resizing in pixels
    cutsvd : float
        value not to exceed for the inverse eigeinvalues at each pixels
    matrix_dir : string
        path to directory to save all the matrices here
    wavelength : float
        wavelength in m.
    silence : boolean, default False.
        Whether to silence print outputs.

    Returns
    --------
    PWVector : 2D array
        vector probe to be multiplied by the image difference
        matrix in order to retrieve the focal plane electric field
    """

    string_dims_PWMatrix = testbed.name_DM_to_probe_in_PW + "Prob" + "_".join(map(str, posprobes)) + "_PWampl" + str(
        int(amplitude)) + "_cut" + str(int(
            cutsvd // 1000)) + "k_dimEstim" + str(dimEstim) + testbed.string_os + "_resFP" + str(
                round(testbed.Science_sampling / testbed.wavelength_0 * wavelength, 2)) + '_wl' + str(
                    int(wavelength * 1e9))

    # Calculating and Saving PW matrix

    filePW = "MatPW_" + string_dims_PWMatrix
    if os.path.exists(os.path.join(matrix_dir, filePW + ".fits")):
        if not silence:
            print("")
            print("The PWmatrix " + filePW + " already exists")
        return fits.getdata(os.path.join(matrix_dir, filePW + ".fits"))

    start_time = time.time()
    if not silence:
        print("The PWmatrix " + filePW + " does not exists")
        print("Start PW matrix" + ' at ' + str(int(wavelength * 1e9)) + "nm (wait a few seconds)")

    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe, dimEstim, dimEstim), dtype=testbed.dtype_complex)
    probephase = np.zeros((numprobe, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil))
    matrix = np.zeros((numprobe, 2))
    PWMatrix = np.zeros((dimEstim**2, 2, numprobe))
    SVD = np.zeros((2, dimEstim, dimEstim))

    DM_probe: DeformableMirror = vars(testbed)[testbed.name_DM_to_probe_in_PW]

    psi0 = testbed.todetector(wavelength=wavelength, in_contrast=True)

    k = 0

    for i in posprobes:

        Voltage_probe = np.zeros(DM_probe.number_act)
        Voltage_probe[i] = amplitude
        probephase[k] = DM_probe.voltage_to_phase(Voltage_probe)

        # for PW the probes are not sent in the DM but at the entrance of the testbed.
        # with an hypothesis of small phase.
        # I tried to remove "1+"". It breaks the code
        # (coronagraph does not "remove the 1 exactly")
        # **kwarg is here to send dir_save_all_planes

        deltapsik[k] = resizing(
            testbed.todetector(entrance_EF=1 + 1j * probephase[k], wavelength=wavelength, in_contrast=True, **kwargs) -
            psi0, dimEstim)

        k = k + 1

    l_ind = 0
    for i in np.arange(dimEstim):
        for j in np.arange(dimEstim):
            matrix[:, 0] = np.real(deltapsik[:, i, j])
            matrix[:, 1] = np.imag(deltapsik[:, i, j])

            try:
                inversion = invert_svd(matrix, cutsvd, silence=True)
                SVD[:, i, j] = inversion[0]
                PWMatrix[l_ind] = inversion[2]
            except Exception:
                print("Careful: Error in invert_svd()! for l_ind=" + str(l_ind))
                SVD[:, i, j] = np.zeros(2)
                PWMatrix[l_ind] = np.zeros((2, numprobe))
            l_ind = l_ind + 1
    fits.writeto(os.path.join(matrix_dir, filePW + ".fits"), np.array(PWMatrix))
    visuPWMap = "EigenPW_" + string_dims_PWMatrix
    fits.writeto(os.path.join(matrix_dir, visuPWMap + ".fits"), np.array(SVD[1]))
    if not silence:
        print("Time for PW Matrix (s): ", np.round(time.time() - start_time))

    return PWMatrix


def calculate_pw_estimate(Difference, Vectorprobes, dimimages, dir_save_all_planes=None, dtype_complex='complex128'):
    """Calculate the focal plane electric field from the probe image
    differences and the modeled probe matrix.

    AUTHOR : Axel Potier

    Parameters
    ----------
    Difference : 3D array
        Cube with image difference for each probes.
    Vectorprobes : 2D array
        Model probe matrix for the same probe as for difference.
    dimimages : int
        Size of the output image after resizing in pixels.
    dir_save_all_planes : string or None, default None
        If not None, absolute directory to save all planes in fits for debugging purposes.
        This can generate a lot of fits especially if in a loop, use with caution.
    dtype_complex : string, default 'complex128'
        bit number for the complex arrays in the PW matrices.
        Can be 'complex128' or 'complex64'. The latter increases the speed of the mft but at the
        cost of lower precision.

    Returns
    --------
    EF_field : 2D array
        PWP estimation of the electrical field in the focal plane.
    """

    Difference_resized = np.zeros((Difference.shape[0], dimimages, dimimages))
    for i in range(Difference.shape[0]):
        Difference_resized[i] = resizing(Difference[i], dimimages)

    numprobe = len(Vectorprobes[0, 0])
    Differenceij = np.zeros((numprobe))
    Resultat = np.zeros((dimimages, dimimages), dtype=dtype_complex)
    l_ind = 0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:] = Difference_resized[:, i, j]
            Resultatbis = np.dot(Vectorprobes[l_ind], Differenceij)
            Resultat[i, j] = Resultatbis[0] + 1j * Resultatbis[1]

            l_ind = l_ind + 1

    if dir_save_all_planes is not None:
        name_plane = 'PW_Estimate'
        save_plane_in_fits(dir_save_all_planes, name_plane, Resultat / 4.)

    return Resultat / 4.


def simulate_pw_difference(input_wavefront,
                           testbed: Testbed,
                           posprobes,
                           amplitudePW,
                           voltage_vector=0.,
                           wavelengths=None,
                           **kwargs):
    """Simulate the acquisition of probe images using Pair-wise.

    and calculate the difference of images [I(+probe) - I(-probe)].
    we use testbed.name_DM_to_probe_in_PW to do the probes.

    AUTHOR : Axel Potier

    Parameters
    ----------
    input_wavefront : complex scalar or 2d complex array or 3d complex array, default is 1 (flat WF)
        Input wavefront in pupil plane.
    testbed : Testbed Optical_element
        Testbed with one or more DM.
    posprobes : 1D-array
        Index of the actuators to push and pull for pair-wise probing.
    amplitudePW : float
        PW probes amplitude in nm.
    voltage_vector : 1D float array or float, default 0
        Vector of voltages vectors for each DMs arounf which we do the difference.
    wavelengths : float, default None
        Wavelength of the estimation in m.

    Returns
    --------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing.
    """

    Difference = np.zeros((len(posprobes), testbed.dimScience, testbed.dimScience))

    for count, num_probe in enumerate(posprobes):

        Voltage_probe = np.zeros(testbed.number_act)
        indice_acum_number_act = 0

        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]

            if DM_name == testbed.name_DM_to_probe_in_PW:
                Voltage_probeDMprobe = np.zeros(DM.number_act)
                Voltage_probeDMprobe[num_probe] = amplitudePW
                Voltage_probe[indice_acum_number_act:indice_acum_number_act + DM.number_act] = Voltage_probeDMprobe

            indice_acum_number_act += DM.number_act

        # If we are in a polychromatic mode but we need monochromatic instensity
        # we have to be careful with the normalization, because
        # todetector_intensity is normalizing to polychromatic PSF by default
        if np.all(wavelengths == testbed.wav_vec):
            # easy case: we are monochromatic or polychromatic both for images and probes
            # It's either a monochromatic correction, or a polychromatic correction with
            # case polychromatic = 'broadband_pwprobes'
            Ikmoins = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                   voltage_vector=voltage_vector - Voltage_probe,
                                                   wavelengths=wavelengths,
                                                   **kwargs)

            Ikplus = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                  voltage_vector=voltage_vector + Voltage_probe,
                                                  wavelengths=wavelengths,
                                                  **kwargs)

        elif isinstance(wavelengths, (float, int)) and wavelengths in testbed.wav_vec:
            # hard case : we are monochromatic for the probes, but polychromatic for the rest of images
            # case polychromatic = 'singlewl' or polychromatic = 'multiwl'
            Ikmoins = testbed.todetector_intensity(
                entrance_EF=input_wavefront,
                voltage_vector=voltage_vector - Voltage_probe,
                wavelengths=wavelengths,
                in_contrast=False,
                **kwargs) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]

            Ikplus = testbed.todetector_intensity(
                entrance_EF=input_wavefront,
                voltage_vector=voltage_vector + Voltage_probe,
                wavelengths=wavelengths,
                in_contrast=False,
                **kwargs) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]

        else:
            raise ValueError(("You are trying to do a pw_difference with wavelength parameters I don't understand. "
                              "Code it yourself in simulate_pw_difference and be careful with the normalization"))

        Difference[count] = Ikplus - Ikmoins

    return Difference
