import os
import time
import numpy as np

from astropy.io import fits

from Asterix.utils import resizing, invert_svd, save_plane_in_fits
from Asterix.optics import DeformableMirror, Testbed


def create_pw_matrix(testbed: Testbed, amplitude, posprobes, dimEstim, cutsvd, matrix_dir, polychrom, **kwargs):
    """
    Build the nbwl times interaction matrix for pair-wise probing.

    AUTHOR : Johan Mazoyer

    Sept 2022 : created

    Parameters
    ----------
    testbed: Testbed Optical_element
        a testbed with one or more DM
    amplitude:  float
        amplitude of the actuator pokes for pair(wise probing in nm
    posprobes:  1D-array
        index of the actuators to push and pull for pair-wise probing
    dimEstim:  int
        size of the output image after resizing in pixels
    cutsvd:     float
        value not to exceed for the inverse eigeinvalues at each pixels
    matrix_dir : string
        path to directory to save all the matrices here
    polychrom: string
        For polychromatic estimation and correction:
        - 'centralwl': only the central wavelength is used for estimation / correction. 1 Interation Matrix
        - 'broadband_pwprobes': probes images PW are broadband but Matrices are at central wavelength: 1 PW Matrix and 1 Interation Matrix
        - 'multiwl': nb_wav images are used for estimation and there are nb_wav matrices of estimation and nb_wav matrices for correction

    Returns
    ------
    PWVector:   list of 2D array
                vector probe to be multiplied by the image difference
                matrix in order to retrieve the focal plane electric field
    """

    return_matrix = []

    if polychrom in ['centralwl', 'broadband_pwprobes']:
        return_matrix.append(
            create_singlewl_pw_matrix(testbed,
                                      amplitude,
                                      posprobes,
                                      dimEstim,
                                      cutsvd,
                                      matrix_dir,
                                      wavelength=testbed.wavelength_0,
                                      **kwargs))

    elif polychrom == 'multiwl':
        for wave_i in testbed.wav_vec:
            return_matrix.append(
                create_singlewl_pw_matrix(testbed,
                                          amplitude,
                                          posprobes,
                                          dimEstim,
                                          cutsvd,
                                          matrix_dir,
                                          wavelength=wave_i,
                                          **kwargs))
    else:
        raise ValueError(polychrom + " is not a valid polychromatic estimation/correction mode")

    return return_matrix


def create_singlewl_pw_matrix(testbed: Testbed, amplitude, posprobes, dimEstim, cutsvd, matrix_dir,
                              wavelength, **kwargs):
    """
    Build the interaction matrix for pair-wise probing at 1 WL

    AUTHOR : Axel Potier
    Modified by Johan Mazoyer

    Sept 2022 : .fits file saving directly in the function to clean up estimator

    Parameters
    ----------
    testbed: Testbed Optical_element
        a testbed with one or more DM
    amplitude:  float
        amplitude of the actuator pokes for pair(wise probing in nm
    posprobes:  1D-array
        index of the actuators to push and pull for pair-wise probing
    dimEstim:  int
        size of the output image after resizing in pixels
    cutsvd:     float
        value not to exceed for the inverse eigeinvalues at each pixels
    matrix_dir : string
        path to directory to save all the matrices here
    wavelength : float
        wavelength in m.

    Returns
    ------
    PWVector:   2D array
                vector probe to be multiplied by the image difference
                matrix in order to retrieve the focal plane electric field
    """

    string_dims_PWMatrix = testbed.name_DM_to_probe_in_PW + "Prob" + "_".join(map(
        str, posprobes)) + "_PWampl" + str(int(amplitude)) + "_cut" + str(int(
            cutsvd // 1000)) + "k_dimEstim" + str(dimEstim) + testbed.string_os + '_wl' + str(
                int(wavelength * 1e9))

    # Calculating and Saving PW matrix
    print("")
    filePW = "MatPW_" + string_dims_PWMatrix
    if os.path.exists(os.path.join(matrix_dir, filePW + ".fits")):
        print("The PWmatrix " + filePW + " already exists")
        return fits.getdata(os.path.join(matrix_dir, filePW + ".fits"))

    start_time = time.time()
    print("The PWmatrix " + filePW + " does not exists")
    print("Start PW matrix" + ' at ' + str(int(wavelength * 1e9)) + "nm (wait a few seconds)")

    numprobe = len(posprobes)
    deltapsik = np.zeros((numprobe, dimEstim, dimEstim), dtype=complex)
    probephase = np.zeros((numprobe, testbed.dim_overpad_pupil, testbed.dim_overpad_pupil))
    matrix = np.zeros((numprobe, 2))
    PWMatrix = np.zeros((dimEstim**2, 2, numprobe))
    SVD = np.zeros((2, dimEstim, dimEstim))

    DM_probe = vars(testbed)[testbed.name_DM_to_probe_in_PW]  # type: DeformableMirror

    #### TODO Careful. right now, you can only do estimation at the wl that are in
    ### the testbed.wav_vec vector. If you're not, you'll run into a bug in
    ### testbed.todetector(wavelength=wavelength)
    ### if we want to do AJ'trick where we optimize the wl to estimate for a given BW
    ### we have to change a bit the normalization
    psi0 = testbed.todetector(wavelength=wavelength)

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
            testbed.todetector(entrance_EF=1 + 1j * probephase[k], wavelength=wavelength, **kwargs) - psi0,
            dimEstim)

        k = k + 1

    l_ind = 0
    for i in np.arange(dimEstim):
        for j in np.arange(dimEstim):
            matrix[:, 0] = np.real(deltapsik[:, i, j])
            matrix[:, 1] = np.imag(deltapsik[:, i, j])

            try:
                inversion = invert_svd(matrix, cutsvd, visu=False)
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
    print("Time for PW Matrix (s): ", np.round(time.time() - start_time))

    return PWMatrix


def calculate_pw_estimate(Difference, Vectorprobes, dir_save_all_planes=None, **kwargs):
    """
    Calculate the focal plane electric field from the probe image
    differences and the modeled probe matrix.

    AUTHOR : Axel Potier

    Parameters
    ----------
    Difference: 3D array
        cube with image difference for each probes

    Vectorprobes: 2D array
        model probe matrix for the same probe as for difference

    dir_save_all_planes : default None
        If not None, directory to save all planes in fits for debugging purposes.
        This can generate a lot of fits especially if in a loop, use with caution

    Returns
    ------
    Difference: 3D array
        cube with image difference for each probes.
        Used for pair-wise probing

    """

    dimimages = len(Difference[0])
    numprobe = len(Vectorprobes[0, 0])
    Differenceij = np.zeros((numprobe))
    Resultat = np.zeros((dimimages, dimimages), dtype=complex)
    l_ind = 0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:] = Difference[:, i, j]
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
                           dimimages,
                           amplitudePW,
                           voltage_vector=0.,
                           wavelengths=None,
                           **kwargs):
    """
    Simulate the acquisition of probe images using Pair-wise
    and calculate the difference of images [I(+probe) - I(-probe)].
    we use testbed.name_DM_to_probe_in_PW to do the probes.

    AUTHOR : Axel Potier

    Parameters
    ----------
    input_wavefront: complex scalar or 2d complex array or 3d complex array. Default is 1 (flat WF)
        Input wavefront in pupil plane

    testbed: Testbed Optical_element
            a testbed with one or more DM

    posprobes : 1D-array
        Index of the actuators to push and pull for pair-wise probing

    dimimages : int
        Size of the output image after resizing in pixels

    amplitudePW: float
        PW probes amplitude in nm

    voltage_vector : 1D float array, default 0
            vector of voltages vectors for each DMs arounf which we do the difference

    wavelengths  :  float default None,
            wavelength of the estimation in m

    Returns
    ------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing
    """

    Difference = np.zeros((len(posprobes), dimimages, dimimages))

    for count, num_probe in enumerate(posprobes):

        Voltage_probe = np.zeros(testbed.number_act)
        indice_acum_number_act = 0

        for DM_name in testbed.name_of_DMs:
            DM = vars(testbed)[DM_name]  # type: DeformableMirror

            if DM_name == testbed.name_DM_to_probe_in_PW:
                Voltage_probeDMprobe = np.zeros(DM.number_act)
                Voltage_probeDMprobe[num_probe] = amplitudePW
                Voltage_probe[indice_acum_number_act:indice_acum_number_act +
                              DM.number_act] = Voltage_probeDMprobe

            indice_acum_number_act += DM.number_act

        # If we are in a polychromatic mode but we need monochromatic instensity
        # we have to be careful with the normalization, because
        # todetector_intensity is normalizing to polychromatic PSF by default
        # At some point, we will need to do estimate at different
        # WL than the one of the testbed (to do the AJ's trick where you scattered the WL
        # of correction the most efficiently in the BW). We will have to be careful with normalization here
        # and recode this part

        if isinstance(wavelengths, (float, int)) and wavelengths in testbed.wav_vec:
            # hard case : we are monochromatic for the probes, but polychromatic for the rest of images
            # polychromatic = 'centralwl' or polychromatic = 'multiwl'
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

        elif np.all(wavelengths == testbed.wav_vec):
            # easy case: we are monochromatic or polychromatic both for images and probes
            # It's either a monochromatic correction or a polychromatic correction with
            # polychromatic = 'broadband_pwprobes'
            Ikmoins = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                   voltage_vector=voltage_vector - Voltage_probe,
                                                   wavelengths=wavelengths,
                                                   **kwargs)

            Ikplus = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                  voltage_vector=voltage_vector + Voltage_probe,
                                                  wavelengths=wavelengths,
                                                  **kwargs)

        else:
            raise ValueError(
                ("You are trying to do a pw_difference with wavelength parameters I don't understand."
                 " Code it yourself in simulate_pw_difference and be careful with the normalization"))

        Difference[count] = resizing(Ikplus - Ikmoins, dimimages)

    return Difference
