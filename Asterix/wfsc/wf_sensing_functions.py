import os
import time
from datetime import datetime
import numpy as np

from astropy.io import fits

from Asterix.utils import resizing, invert_svd, save_plane_in_fits, from_param_to_header
from Asterix.optics import DeformableMirror, Testbed


def create_pw_matrix(testbed: Testbed,
                     voltage_probe,
                     dimEstim,
                     cutsvd,
                     matrix_dir,
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
        wav_vec_estim = np.array([testbed.wavelength_0])

    return_matrix = []

    for wave_i in wav_vec_estim:
        return_matrix.append(
            create_singlewl_pw_matrix(testbed,
                                      voltage_probe,
                                      dimEstim,
                                      cutsvd,
                                      matrix_dir,
                                      wavelength=wave_i,
                                      silence=silence,
                                      **kwargs))

    return return_matrix


def create_singlewl_pw_matrix(testbed: Testbed,
                              voltage_probe,
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

    filePW, header_expected, bool_already_existing_matrix = name_header_pwp_matrix(testbed, dimEstim, wavelength,
                                                                                   matrix_dir)

    if bool_already_existing_matrix:
        # there is already a really identical matrix calculated, we just load the old matrix fits file.
        if not silence:
            print("")
            print("The PWmatrix " + filePW + " already exists")

        return fits.getdata(os.path.join(matrix_dir, filePW + ".fits"))

    # there is no matrix measured with the same parameters, we recalculate
    start_time = time.time()
    if not silence:
        print("The PWmatrix " + filePW + " does not exists")
        print("Start PWP matrix" + ' at ' + str(int(wavelength * 1e9)) + "nm (wait a few seconds)")

    numprobe = len(voltage_probe)
    deltapsik = np.zeros((numprobe, dimEstim, dimEstim), dtype=testbed.dtype_complex)
    matrix = np.zeros((numprobe, 2))
    PWMatrix = np.zeros((dimEstim**2, 2, numprobe))
    SVD = np.zeros((2, dimEstim, dimEstim))

    psi0 = testbed.todetector(wavelength=wavelength, in_contrast=True)

    k = 0

    for voltage_probe in voltage_probe:
        # we find the DM used to probe
        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]
            DMvoltage = testbed.testbed_voltage_to_indiv_DM_voltage(voltage_probe, DM_name)
            if (DMvoltage == 0).all():
                continue
            else:
                probephase = DM.voltage_to_phase(DMvoltage)
                break

        # for PWP the probes are not sent in the DM but at the entrance of the testbed.
        # with an hypothesis of small phase.
        # I tried to remove "1+"". It breaks the code
        # (coronagraph does not "remove the 1 exactly")
        # **kwarg is here to send dir_save_all_planes
        deltapsik[k] = resizing(
            testbed.todetector(entrance_EF=1 + 1j * probephase, wavelength=wavelength, in_contrast=True, **kwargs) -
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

    fits.writeto(os.path.join(matrix_dir, filePW + ".fits"), np.array(PWMatrix), header_expected, overwrite=True)
    visuPWMap = filePW.replace("MatPW_", "EigenPW_")
    fits.writeto(os.path.join(matrix_dir, visuPWMap + ".fits"), np.array(SVD[1]), header_expected, overwrite=True)
    if not silence:
        print("Time for PWP Matrix (s): ", np.round(time.time() - start_time))

    return PWMatrix


def name_header_pwp_matrix(testbed: Testbed, dimEstim, wavelength, matrix_dir):
    """ Create the name of the file and header of the PW matrix that will be used to
    identify precisely parameter used in the calcul of this matrix and recalculate if need be.
    We then load a potential matrix and compare the header to the one existing.

    AUTHOR : Johan Mazoyer

    Parameters
    ----------
    testbed : Testbed Optical_element
        a testbed with one or more DM
    dimEstim : int
        size of the output image after resizing in pixels
    wavelength : float
        wavelength in m.
    matrix_dir : string
        path to directory to save all the matrices.

    Returns
    --------
    filePW : string
        file name of the pw/btp matrix. If there is no identical matrix already measured in fits file, None.
    header : fits.Header() Object
        header of the pw/btp matrix
    bool_already_existing_matrix :bool
        If there is no identical matrix already saved in fits file.
    """

    amplitude = testbed.config_file["Estimationconfig"]["amplitudePW"]
    posprobes = testbed.config_file["Estimationconfig"]["posprobes"]
    name_DM_to_probe_in_PW = testbed.config_file["Estimationconfig"]["name_DM_to_probe_in_PW"]
    cutsvd = testbed.config_file["Estimationconfig"]["cut"]

    string_dims_PWMatrix = name_DM_to_probe_in_PW + "Prob" + "_".join(map(str, posprobes)) + "_PWampl" + str(
        int(amplitude)) + "_cut" + str(int(
            cutsvd // 1000)) + "k_dimEstim" + str(dimEstim) + testbed.string_os + "_resFP" + str(
                round(testbed.Science_sampling / testbed.wavelength_0 * wavelength, 2)) + '_wl' + str(
                    int(wavelength * 1e9))

    filePW = "MatPW_" + string_dims_PWMatrix

    header = fits.Header()
    header.insert(0, ('date_mat', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "matrix creation date"))
    # we load the configuration to save it in the fits
    if hasattr(testbed, 'config_file'):

        necessary_estim_param = dict()
        necessary_estim_param['Estim_wl'] = wavelength * 1e9
        necessary_estim_param['name_DM_to_probe_in_PW'] = name_DM_to_probe_in_PW
        necessary_estim_param['dimEstim'] = dimEstim
        necessary_estim_param['Estim_bin_factor'] = testbed.config_file["Estimationconfig"]["Estim_bin_factor"]
        necessary_estim_param['amplitudePW'] = amplitude
        necessary_estim_param['posprobes'] = posprobes
        necessary_estim_param['cutsvd'] = cutsvd

        header = from_param_to_header(necessary_estim_param, header)

        additional_model_param = dict()
        additional_model_param['dtype_complex'] = testbed.dtype_complex
        header = from_param_to_header(additional_model_param, header)

        header = from_param_to_header(testbed.config_file["modelconfig"], header)

        necessary_dm_param = dict()
        for paramdmkey in testbed.config_file["DMconfig"].scalars:
            if name_DM_to_probe_in_PW in paramdmkey:
                necessary_dm_param[paramdmkey] = testbed.config_file["DMconfig"][paramdmkey]
        header = from_param_to_header(necessary_dm_param, header)

        header = from_param_to_header(testbed.config_file["Coronaconfig"], header)

    # Loading any existing matrix and comparing their headers to make sure they are created
    # using the same set of parameters
    if os.path.exists(os.path.join(matrix_dir, filePW + ".fits")):
        header_existing = fits.getheader(os.path.join(matrix_dir, filePW + ".fits"))
        # remove the basis kw created automatically  when saving the fits file
        for keyw in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3']:
            header_existing.remove(keyw)
        # we comapre the header (ignoring the date)
        bool_already_existing_matrix = fits.HeaderDiff(header_existing, header, ignore_keywords=['DATE_MAT']).identical
    else:
        bool_already_existing_matrix = False

    return filePW, header, bool_already_existing_matrix


def calculate_pw_estimate(Difference,
                          Vectorprobes,
                          dimimages,
                          dir_save_all_planes=None,
                          pwp_or_btp="pwp",
                          dtype_complex='complex128'):
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
    pwp_or_btp : string, default 'pwp'
        type of algorithm used, can be
            'pw' Pair Wise Probing
            'btp' Borde Traub Probing where the probe is pushed positevely
    dtype_complex : string, default 'complex128'
        bit number for the complex arrays in the PWP matrices.
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

    if pwp_or_btp in ['pw', "pwp", 'pairwise']:
        if dir_save_all_planes is not None:
            name_plane = 'PWP_Estimate'
            save_plane_in_fits(dir_save_all_planes, name_plane, Resultat / 4.)

        return Resultat / 4.

    elif pwp_or_btp in ['btp']:
        if dir_save_all_planes is not None:
            name_plane = 'BTP_Estimate'
            save_plane_in_fits(dir_save_all_planes, name_plane, Resultat / 2.)

        return Resultat / 2.

    else:
        raise ValueError("pwp_or_btp parameter can only take 2 values 'pwp' for Pair Wise Probing "
                         "or 'btp' for Borde Traub Probing")


def generate_actu_probe_voltages(testbed: Testbed, posprobes, amplitudePW, name_DM_to_probe_in_PW):
    """Generate the DM commands for each actuator probes.

    Parameters
    ----------
    testbed : Testbed Optical_element
        Testbed with one or more DM.
    posprobes : 1D-array
        list of index of the number actuators probes.
    amplitudePW : float
        PWP probes amplitude in nm.

    Returns
    --------
    voltage_probes : 2D float array of size [len(posprobes), testbed.number_actuators]
        Array of voltages vectors for each probes
    """

    voltage_probes = np.zeros((len(posprobes), testbed.number_act))
    DM_probe: DeformableMirror = vars(testbed)[name_DM_to_probe_in_PW]

    for count, num_probe in enumerate(posprobes):
        Voltage_probe_indiv_DM = np.zeros(DM_probe.number_act)
        Voltage_probe_indiv_DM[num_probe] = amplitudePW

        voltage_probes[count] = testbed.indiv_DM_voltage_to_testbed_voltage(Voltage_probe_indiv_DM,
                                                                            name_DM_to_probe_in_PW)

    return voltage_probes


def simulate_pw_probes(input_wavefront,
                       testbed: Testbed,
                       voltage_probes,
                       voltage_vector=0.,
                       wavelengths=None,
                       pwp_or_btp="pwp",
                       **kwargs):
    """Simulate the acquisition of probe images using Pair-wise.

    Parameters
    ----------
    input_wavefront : complex scalar or 2d complex array or 3d complex array, default is 1 (flat WF)
        Input wavefront in pupil plane.
    testbed : Testbed Optical_element
        Testbed with one or more DM.
    voltage_probes : 2D float array of size [len(posprobes), testbed.number_actuators]
        Array of voltages vectors for each probes
    voltage_vector : 1D float array or float, default 0
        Vector of voltages vectors for each DMs arounf which we do the difference.
    wavelengths : float, default None
        Wavelength of the estimation in m.
    pwp_or_btp : string, default 'pwp'
        type of algorithm used, can be
            'pw' Pair Wise Probing
            'btp' Borde Traub Probing where the probe is pushed positevely

    Returns
    --------
    Probed_images : 3D array
        Cube with all probed images. Use for pair-wise probing.
    """

    number_probes = len(voltage_probes)

    if pwp_or_btp == 'btp':
        Probed_images = np.zeros((1 + number_probes, testbed.dimimages, testbed.dimimages))

        # If we are in a polychromatic mode but we need monochromatic instensity
        # we have to be careful with the normalization, because
        # todetector_intensity is normalizing to polychromatic PSF by default
        if np.all(wavelengths == testbed.wav_vec):
            # easy case: we are monochromatic or polychromatic both for images and probes
            # It's either a monochromatic correction, or a polychromatic correction with
            # case polychromatic = 'broadband_pwprobes'
            Ik0 = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                               voltage_vector=voltage_vector,
                                               wavelengths=wavelengths,
                                               **kwargs)
        elif isinstance(wavelengths, (float, int)) and wavelengths in testbed.wav_vec:
            # hard case : we are monochromatic for the probes, but polychromatic for the rest of images
            # case polychromatic = 'singlewl' or polychromatic = 'multiwl'
            Ik0 = testbed.todetector_intensity(
                entrance_EF=input_wavefront,
                voltage_vector=voltage_vector,
                wavelengths=wavelengths,
                in_contrast=False,
                **kwargs) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]
        Probed_images[0] = Ik0

    if pwp_or_btp in ['pw', "pwp", 'pairwise']:
        Probed_images = np.zeros((2 * number_probes, testbed.dimScience, testbed.dimScience))
    else:
        raise ValueError("pwp_or_btp parameter can only take 2 values 'pwp', 'pairwise' for"
                         "Pair Wise Probing or 'btp' for Borde Traub Probing")

    for count, Voltage_probe in enumerate(voltage_probes):

        # If we are in a polychromatic mode but we need monochromatic instensity
        # we have to be careful with the normalization, because
        # todetector_intensity is normalizing to polychromatic PSF by default
        if np.all(wavelengths == testbed.wav_vec):
            # easy case: we are monochromatic or polychromatic both for images and probes
            # It's either a monochromatic correction, or a polychromatic correction with
            # case polychromatic = 'broadband_pwprobes'
            Ikplus = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                  voltage_vector=voltage_vector + Voltage_probe,
                                                  wavelengths=wavelengths,
                                                  **kwargs)

            if pwp_or_btp in ['pw', "pwp", 'pairwise']:
                Ikmoins = testbed.todetector_intensity(entrance_EF=input_wavefront,
                                                       voltage_vector=voltage_vector - Voltage_probe,
                                                       wavelengths=wavelengths,
                                                       **kwargs)

        elif isinstance(wavelengths, (float, int)) and wavelengths in testbed.wav_vec:
            # hard case : we are monochromatic for the probes, but polychromatic for the rest of images
            # case polychromatic = 'singlewl' or polychromatic = 'multiwl'
            Ikplus = testbed.todetector_intensity(
                entrance_EF=input_wavefront,
                voltage_vector=voltage_vector + Voltage_probe,
                wavelengths=wavelengths,
                in_contrast=False,
                **kwargs) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]

            if pwp_or_btp in ['pw', "pwp", 'pairwise']:
                Ikmoins = testbed.todetector_intensity(
                    entrance_EF=input_wavefront,
                    voltage_vector=voltage_vector - Voltage_probe,
                    wavelengths=wavelengths,
                    in_contrast=False,
                    **kwargs) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]

        else:
            raise ValueError(("You are trying to do a pw_difference with wavelength parameters I don't understand. "
                              "Code it yourself in simulate_pw_probes and be careful with the normalization"))

        if pwp_or_btp in ['pw', "pwp", 'pairwise']:
            Probed_images[2 * count] = Ikplus
            Probed_images[2 * count + 1] = Ikmoins

        if pwp_or_btp == 'btp':
            Probed_images[count + 1] = Ikplus

    return Probed_images


def btp_difference(Probed_images, testbed: Testbed, voltage_probes, wavelengths=None):
    """ make the difference [I(+probe) - I(0) - psi_k_model] for borde and traub probing.

    Parameters
    ----------
    testbed : Testbed Optical_element
        Testbed with one or more DM.
    voltage_probes : 2D float array of size [len(posprobes), testbed.number_actuators]
        Array of voltages vectors for each probes
    voltage_vector : 1D float array or float, default 0
        Vector of voltages vectors for each DMs arounf which we do the difference.
    wavelengths : float, default None
        Wavelength of the estimation in m.

    Returns
    --------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing.
    """

    Difference = np.zeros((len(voltage_probes), testbed.dimScience, testbed.dimScience))

    for count, voltage_probe in enumerate(voltage_probes):

        # we find the DM used to probe
        for DM_name in testbed.name_of_DMs:
            DM: DeformableMirror = vars(testbed)[DM_name]
            DMvoltage = testbed.testbed_voltage_to_indiv_DM_voltage(voltage_probe, DM_name)
            if (DMvoltage == 0).all():
                continue
            else:
                probephase = DM.voltage_to_phase(DMvoltage)
                break

        # If we are in a polychromatic mode but we need monochromatic instensity
        # we have to be careful with the normalization, because
        # todetector_intensity is normalizing to polychromatic PSF by default
        if np.all(wavelengths == testbed.wav_vec):
            # easy case: we are monochromatic or polychromatic both for images and probes
            # It's either a monochromatic correction, or a polychromatic correction with
            # case polychromatic = 'broadband_pwprobes'

            Int_fp_probe = testbed.todetector_intensity(entrance_EF=1 + 1j * probephase, wavelengths=wavelengths)

        elif isinstance(wavelengths, (float, int)) and wavelengths in testbed.wav_vec:
            # hard case : we are monochromatic for the probes, but polychromatic for the rest of images
            # case polychromatic = 'singlewl' or polychromatic = 'multiwl'

            Int_fp_probe = testbed.todetector_intensity(
                entrance_EF=1 + 1j * probephase, wavelengths=wavelengths,
                in_contrast=False) / testbed.norm_monochrom[testbed.wav_vec.tolist().index(wavelengths)]

        # Ikplus - Ik0 - psi_k_model
        Difference[count] = Probed_images[count + 1] - Probed_images[0] - Int_fp_probe

    return Difference


def pw_difference(Probed_images):
    """ Calculate the difference of images [I(+probe) - I(-probe)]

    Parameters
    ----------
    Probed_images

    Returns
    --------
    Difference : 3D array
        Cube with image difference for each probes. Use for pair-wise probing.
    """
    nb_probes = Probed_images.shape[0] // 2

    Difference = np.zeros((nb_probes, Probed_images.shape[1], Probed_images.shape[2]))

    for count in range(nb_probes):
        # kplus - Ikmoins
        Difference[count] = Probed_images[2 * count] - Probed_images[2 * count + 1]

    return Difference
