__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from configobj import ConfigObj
from validate import Validator

import Asterix.processing_functions as proc
import Asterix.propagation_functions as prop
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.WSC_functions as wsc

import Asterix.Optical_System_functions as OptSy
import Asterix.fits_functions as useful

__all__ = ["create_interaction_matrices", "correctionLoop"]

#######################################################
#######################################################
######## Interaction/control matrices for PW and EFC


def create_interaction_matrices(parameter_file,
                                NewMODELconfig={},
                                NewDMconfig={},
                                NewCoronaconfig={},
                                NewEstimationconfig={},
                                NewCorrectionconfig={},
                                NewSIMUconfig={}):

    Asterixroot = os.path.dirname(os.path.realpath(__file__))

    ### CONFIGURATION FILE
    configspec_file = Asterixroot + os.path.sep + "Param_configspec.ini"
    config = ConfigObj(parameter_file,
                       configspec=configspec_file,
                       default_encoding="utf8")
    vtor = Validator()
    checks = config.validate(vtor, copy=True)
    # copy=True for copying the comments

    if not os.path.exists(parameter_file):
        raise Exception("The parameter file " + parameter_file +
                        " cannot be found")

    if not os.path.exists(configspec_file):
        raise Exception("The parameter config file " + configspec_file +
                        " cannot be found")

    ### CONFIG
    Data_dir = config["Data_dir"]

    ##################
    ##################
    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

    #On bench or numerical simulation
    onbench = modelconfig["onbench"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)
    DM1_creating_pushact = DMconfig["DM1_creating_pushact"]
    DM3_creating_pushact = DMconfig["DM3_creating_pushact"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ##################
    ##################
    ### PW CONFIG
    Estimationconfig = config["Estimationconfig"]
    Estimationconfig.update(NewEstimationconfig)

    estimation = Estimationconfig["estimation"]
    amplitudePW = Estimationconfig["amplitudePW"]
    posprobes = Estimationconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = Estimationconfig["cut"]

    DH_sampling = Estimationconfig["DH_sampling"]

    ##################
    ##################
    ###EFC CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)
    DHshape = Correctionconfig["DHshape"]
    choosepix = Correctionconfig["choosepix"]
    choosepix = [int(i) for i in choosepix]

    DM1_otherbasis = Correctionconfig["DM1_otherbasis"]
    DM3_otherbasis = Correctionconfig["DM3_otherbasis"]
    Nbmodes = Correctionconfig["Nbmodes"]
    amplitudeEFC = Correctionconfig["amplitudeEFC"]
    regularization = Correctionconfig["regularization"]

    ##THEN DO
    model_dir = os.path.join(Asterixroot, "Model") + os.path.sep

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)

    if DM3_otherbasis is False:
        basistr = "actu"
    else:
        basistr = "fourier"

    # Initialize thd:
    pup_round = OptSy.pupil(modelconfig)
    DM1 = OptSy.deformable_mirror(modelconfig,
                                DMconfig,
                                load_fits=not DM1_creating_pushact,
                                save_fits=DM3_creating_pushact,
                                Name_DM='DM1',
                                model_dir=model_dir,
                                Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                DMconfig,
                                load_fits=not DM3_creating_pushact,
                                save_fits=DM3_creating_pushact,
                                Name_DM='DM3',
                                model_dir=model_dir,
                                Model_local_dir=Model_local_dir)

    # we also need to "clear" the apod plane because the THD2 is like that
    Coronaconfig.update({'filename_instr_apod': "ClearPlane"})
    corono = OptSy.coronagraph(modelconfig, Coronaconfig, model_dir=model_dir)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                                ["entrancepupil", "DM1", "DM3", "corono"])

    PSF = thd2.todetector_Intensity(center_on_pixel=True, noFPM=True)
    thd2.maxPSF = np.max(PSF)
    thd2.sumPSF = np.sum(PSF)

    #image size after binning
    dim_sampl = int(DH_sampling / thd2.science_sampling * thd2.dim_im / 2) * 2


    intermatrix_dir = os.path.join(Data_dir, "Interaction_Matrices",
                                   thd2.corono.corona_type)
    if thd2.corono.corona_type == 'fqpm':
        if thd2.corono.achrom_fqpm == True:
            intermatrix_dir = os.path.join(intermatrix_dir,
                                           "Achromatic_phase_mask")
        else:
            intermatrix_dir = os.path.join(intermatrix_dir,
                                           "Polychromatic_phase_mask")
    if thd2.corono.corona_type == 'knife':
        intermatrix_dir = os.path.join(
            intermatrix_dir, thd2.corono.coro_position,
            "offset_" + str(thd2.corono.knife_coro_offset) + "lop")

    intermatrix_dir = os.path.join(
        intermatrix_dir,
        str(int(thd2.wavelength_0 * 1e9)) + "nm",
        "p" + str(round(thd2.corono.diam_pup_in_m * 1e3, 2)) + "_l" +
        str(round(thd2.corono.diam_lyot_in_m * 1e3, 1)), "lop_" +
        str(round(thd2.science_sampling, 2)), "basis_" + basistr) + os.path.sep

    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    if onbench == True:
        Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    if thd2.DM1.active == True:
        nam2DM = "_2DM"
    else:
        nam2DM = ""

    string_dims_PWMatrix = "_".join(map(str, posprobes)) + "act_" + str(
        int(amplitudePW)) + "nm_" + str(
            int(cut)) + "cutsvd_dim_sampl_" + str(dim_sampl) + "_dim" + str(
                thd2.dim_im) + '_radpup' + str(thd2.prad)

    ####Calculating and Saving PW matrix
    if (estimation == "PairWise" or estimation == "pairwise"
            or estimation == "PW" or estimation == "pw"):
        filePW = "MatrixPW_" + string_dims_PWMatrix
        if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
            print("The matrix " + filePW + " already exists")
            vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
        else:
            print("Saving " + filePW + " ...")
            vectoressai, showsvd = wsc.createvectorprobes(
                thd2, amplitudePW, posprobes, dim_sampl, cut,
                thd2.wavelength_0)
            fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = "MapEigenvaluesPW" + string_dims_PWMatrix

        if os.path.exists(intermatrix_dir + visuPWMap + ".fits") is False:
            print("Saving " + visuPWMap + " ...")
            fits.writeto(intermatrix_dir + visuPWMap + ".fits", showsvd[1])

    # Saving PW matrices in Labview directory
    if onbench == True:
        probes = np.zeros((len(posprobes), thd2.DM3.number_act ),
                          dtype=np.float32)
        vectorPW = np.zeros((2, dim_sampl * dim_sampl * len(posprobes)),
                            dtype=np.float32)

        for i in np.arange(len(posprobes)):
            probes[i, posprobes[i]] = amplitudePW / 17
            vectorPW[0, i * dim_sampl * dim_sampl:(i + 1) * dim_sampl *
                     dim_sampl] = vectoressai[:, 0, i].flatten()
            vectorPW[1, i * dim_sampl * dim_sampl:(i + 1) * dim_sampl *
                     dim_sampl] = vectoressai[:, 1, i].flatten()
        fits.writeto(Labview_dir + "Probes_EFC_default.fits",
                     probes,
                     overwrite=True)
        fits.writeto(Labview_dir + "Matr_mult_estim_PW.fits",
                     vectorPW,
                     overwrite=True)

        ####Calculating and Saving EFC matrix
        if DHshape == "square":
            print("TO SET ON LABVIEW: ",
                  str(dim_sampl / 2 + np.array(np.fft.fftshift(choosepix))))

    # Creating WhichInPup.
    # if DM3_otherbasis = False, this is done inside the DM class
    # if not, I am a bit weary to put all DMX_otherbasis stuff which are clearly
    # EFC stuff inside the optical models class.
    # I think currently the name of the actuator inside the pupil is
    # used as the basis, which is not ideal at all, these are 2 different things.

    # DM1
    if DM1_otherbasis == True:
        thd2.DM1.WhichInPupil = np.arange(thd2.DM1.number_act )

    # DM3
    if DM3_otherbasis == True:
        thd2.DM1.WhichInPupil = np.arange(thd2.DM3.number_act )

    #measure the masks
    maskDH, _, string_dhshape = wsc.load_or_save_maskDH(
        intermatrix_dir, Correctionconfig, dim_sampl, DH_sampling, thd2.dim_im,
        thd2.science_sampling)

    #useful string
    string_dims_EFCMatrix = str(amplitudeEFC) + "nm_" + str(
        Nbmodes) + "modes_dim" + str(thd2.dim_im) + '_radpup' + str(
            thd2.prad) + nam2DM

    # Creating EFC control matrix
    fileEFCMatrix = "MatrixEFC" + string_dhshape + string_dims_EFCMatrix

    if os.path.exists(intermatrix_dir + fileEFCMatrix + ".fits") == True:
        print("The matrix " + fileEFCMatrix + " already exists")
        invertGDH = fits.getdata(intermatrix_dir + fileEFCMatrix + ".fits")
    else:

        # Actuator basis or another one?
        if DM3_otherbasis == True:
            DM3_basis = fits.getdata(Labview_dir + "Map_modes_DM3_foc.fits")
        else:
            DM3_basis = 0

        # Creating EFC Interaction matrix
        fileDirectMatrix = "DirectMatrix" + string_dhshape + string_dims_EFCMatrix

        if os.path.exists(intermatrix_dir + fileDirectMatrix +
                          ".fits") == True:
            print("The matrix " + fileDirectMatrix + " already exists")
            Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix +
                                   ".fits")
        else:

            # Creating EFC Interaction Matrix if does not exist
            print("Saving " + fileDirectMatrix + " ...")
            # this is typically the kind of stuff that would be more better
            # in the class
            if thd2.DM1.active == True:
                DM_pushact = np.concatenate(
                    (thd2.DM3.DM_pushact, thd2.DM1.DM_pushact_inpup))
                DM_WhichInPupil = np.concatenate(
                    (thd2.DM3.WhichInPupil,
                     thd2.DM3.number_act  + thd2.DM1.WhichInPupil))
            else:
                DM_pushact = thd2.DM3.DM_pushact
                DM_WhichInPupil = thd2.DM3.WhichInPupil

            Gmatrix = wsc.creatingCorrectionmatrix(
                thd2.entrancepupil.pup,
                thd2,
                dim_sampl,
                DM_pushact * amplitudeEFC * 2 * np.pi * 1e-9 /
                thd2.wavelength_0,
                maskDH,
                DM_WhichInPupil,
                otherbasis=DM3_otherbasis,
                basisDM3=DM3_basis,
            )

            fits.writeto(intermatrix_dir + fileDirectMatrix + ".fits", Gmatrix)

        # Calculating and saving EFC Control Matrix
        print("Saving " + fileEFCMatrix + " ...")
        SVD, SVD_trunc, invertGDH = wsc.invertSVD(
            Gmatrix,
            Nbmodes,
            goal="c",
            regul=regularization,
            otherbasis=DM3_otherbasis,
            basisDM3=DM3_basis,
            intermatrix_dir=intermatrix_dir)
        fits.writeto(intermatrix_dir + fileEFCMatrix + ".fits", invertGDH)

        plt.clf()
        plt.plot(SVD, "r.")
        plt.yscale("log")

        figSVDEFC = intermatrix_dir + "invertSVDEFC_square" + string_dhshape + string_dims_EFCMatrix + ".png"

        plt.savefig(figSVDEFC)

    if onbench == True:

        EFCmatrix_DM3 = np.zeros(
            (invertGDH.shape[1], thd2.DM3.number_act ),
            dtype=np.float32)
        for i in np.arange(len(thd2.DM3.WhichInPupil)):
            EFCmatrix_DM3[:, thd2.DM3.WhichInPupil[i]] = invertGDH[i, :]
        fits.writeto(Labview_dir + "Matrix_control_EFC_DM3_default.fits",
                     EFCmatrix_DM3,
                     overwrite=True)
        if thd2.DM1.active:
            EFCmatrix_DM1 = np.zeros(
                (invertGDH.shape[1], thd2.DM1.number_act ),
                dtype=np.float32)
            for i in np.arange(len(thd2.DM1.WhichInPupil)):
                EFCmatrix_DM1[:, thd2.DM1.WhichInPupil[i]] = invertGDH[
                    i + len(thd2.DM3.WhichInPupil), :]
            fits.writeto(Labview_dir + "Matrix_control_EFC_DM1_default.fits",
                         EFCmatrix_DM1,
                         overwrite=True)

    return 0


#######################################################
#######################################################
######## Simulation of a correction loop


def correctionLoop(parameter_file,
                   NewMODELconfig={},
                   NewDMconfig={},
                   NewCoronaconfig={},
                   NewEstimationconfig={},
                   NewCorrectionconfig={},
                   NewSIMUconfig={}):

    Asterixroot = os.path.dirname(os.path.realpath(__file__))

    ### CONFIGURATION FILE
    configspec_file = Asterixroot + os.path.sep + "Param_configspec.ini"
    config = ConfigObj(parameter_file,
                       configspec=configspec_file,
                       default_encoding="utf8")
    vtor = Validator()
    checks = config.validate(vtor, copy=True)
    # copy=True for copying the comments

    if not os.path.exists(parameter_file):
        raise Exception("The parameter file " + parameter_file +
                        " cannot be found")

    if not os.path.exists(configspec_file):
        raise Exception("The parameter config file " + configspec_file +
                        " cannot be found")

    ### CONFIG
    Data_dir = config["Data_dir"]

    ##################
    ##################
    ### MODEL CONFIG
    modelconfig = config["modelconfig"]
    modelconfig.update(NewMODELconfig)

    #On bench or numerical simulation
    onbench = modelconfig["onbench"]

    #Lambda over D in pixels
    wavelength_0 = modelconfig["wavelength_0"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM1_misregistration = DMconfig["DM1_misregistration"]
    DM3_misregistration = DMconfig["DM3_misregistration"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ##################
    ##################
    ### Estimation CONFIG
    Estimationconfig = config["Estimationconfig"]
    Estimationconfig.update(NewEstimationconfig)

    estimation = Estimationconfig["estimation"].lower()

    amplitudePW = Estimationconfig["amplitudePW"]
    posprobes = Estimationconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = Estimationconfig["cut"]
    DH_sampling = Estimationconfig["DH_sampling"]

    ##################
    ##################
    ###EFC CONFIG
    Correctionconfig = config["Correctionconfig"]
    Correctionconfig.update(NewCorrectionconfig)

    Nbmodes = Correctionconfig["Nbmodes"]
    DM1_otherbasis = Correctionconfig["DM1_otherbasis"]
    DM3_otherbasis = Correctionconfig["DM3_otherbasis"]

    amplitudeEFC = Correctionconfig["amplitudeEFC"]
    regularization = Correctionconfig["regularization"]

    ##################
    ##################
    ###SIMU CONFIG
    SIMUconfig = config["SIMUconfig"]
    SIMUconfig.update(NewSIMUconfig)
    Name_Experiment = SIMUconfig["Name_Experiment"]
    set_amplitude_abb = SIMUconfig["set_amplitude_abb"]
    ampl_abb_filename = SIMUconfig["ampl_abb_filename"]
    set_random_ampl = SIMUconfig["set_random_ampl"]
    ampl_rms = SIMUconfig["ampl_rms"]
    ampl_rhoc = SIMUconfig["ampl_rhoc"]
    ampl_slope = SIMUconfig["ampl_slope"]
    set_phase_abb = SIMUconfig["set_phase_abb"]
    set_random_phase = SIMUconfig["set_random_phase"]
    phase_rms = SIMUconfig["phase_rms"]
    phase_rhoc = SIMUconfig["phase_rhoc"]
    phase_slope = SIMUconfig["phase_slope"]
    phase_abb_filename = SIMUconfig["phase_abb_filename"]
    photon_noise = SIMUconfig["photon_noise"]
    nb_photons = SIMUconfig["nb_photons"]
    correction_algorithm = SIMUconfig["correction_algorithm"]
    Linearization = SIMUconfig["Linearization"]
    Nbiter_corr = SIMUconfig["Nbiter_corr"]
    Nbiter_corr = [int(i) for i in Nbiter_corr]
    Nbmode_corr = SIMUconfig["Nbmode_corr"]
    Nbmode_corr = [int(i) for i in Nbmode_corr]
    Linesearch = SIMUconfig["Linesearch"]
    Linesearchmode = SIMUconfig["Linesearchmode"]
    Linesearchmode = [int(i) for i in Linesearchmode]
    gain = SIMUconfig["gain"]

    ##THEN DO

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    ## Directories for saving the data
    model_dir = os.path.join(Asterixroot, "Model") + os.path.sep

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep

    # Initialize thd:
    pup_round = OptSy.pupil(modelconfig)
    DM1 = OptSy.deformable_mirror(modelconfig,
                                DMconfig,
                                Name_DM='DM1',
                                model_dir=model_dir,
                                Model_local_dir=Model_local_dir)

    DM3 = OptSy.deformable_mirror(modelconfig,
                                DMconfig,
                                Name_DM='DM3',
                                model_dir=model_dir,
                                Model_local_dir=Model_local_dir)

    # we also need to "clear" the apod plane because the THD2 is like that
    Coronaconfig.update({'filename_instr_apod': "ClearPlane"})
    corono = OptSy.coronagraph(modelconfig, Coronaconfig, model_dir=model_dir)
    # and then just concatenate
    thd2 = OptSy.Testbed([pup_round, DM1, DM3, corono],
                                ["entrancepupil", "DM1", "DM3", "corono"])

    # TODO Because the code is currently setup heavily on the
    # 'default testbed' beeing thd2 having those elements, VS code thinks
    # there is an error if it thinks they are not defined (although in practice
    # they are).
    thd2.entrancepupil = pup_round
    thd2.DM1 = DM1
    thd2.DM3 = DM3
    thd2.corono = corono
    # In practice this is done inside the Testbed initialization already !
    # and these lines are useless and I only put them to calm
    # To be removed when the correction and estimation class are done
    # in which case these class will be able when these things exists or not

    PSF = thd2.todetector_Intensity(center_on_pixel=True, noFPM=True)
    thd2.maxPSF = np.max(PSF)
    thd2.sumPSF = np.sum(PSF)

    #image size after binning
    dim_sampl = int(DH_sampling / thd2.science_sampling * thd2.dim_im / 2) * 2

    if onbench == True:
        Labview_dir = os.path.join(Data_dir, "Labview") + os.path.sep
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    result_dir = Labview_dir = os.path.join(Data_dir, "Results",
                                            Name_Experiment) + os.path.sep
    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

    ## Pair-wise probing directory
    if DM3_otherbasis == True:
        basistr = "fourier"
        DM3_basis = fits.getdata(Labview_dir + "Map_modes_DM3_foc.fits")
        thd2.DM3.WhichInPupil = np.arange(thd2.DM3.number_act )
    else:
        basistr = "actu"
        DM3_basis = 0

    # DM1
    if DM1_otherbasis == True:
        thd2.DM1.WhichInPupil = np.arange(thd2.DM1.number_act )

    intermatrix_dir = os.path.join(Data_dir, "Interaction_Matrices",
                                   thd2.corono.corona_type)
    if thd2.corono.corona_type == 'fqpm':
        if thd2.corono.achrom_fqpm == True:
            intermatrix_dir = os.path.join(intermatrix_dir,
                                           "Achromatic_phase_mask")
        else:
            intermatrix_dir = os.path.join(intermatrix_dir,
                                           "Polychromatic_phase_mask")
    if thd2.corono.corona_type == 'knife':
        intermatrix_dir = os.path.join(
            intermatrix_dir, thd2.corono.coro_position,
            "offset_" + str(thd2.corono.knife_coro_offset) + "lop")

    intermatrix_dir = os.path.join(
        intermatrix_dir,
        str(int(thd2.wavelength_0 * 1e9)) + "nm",
        "p" + str(round(thd2.diam_pup_in_m * 1e3, 2)) + "_l" +
        str(round(thd2.corono.diam_lyot_in_m * 1e3, 1)), "lop_" +
        str(round(thd2.science_sampling, 2)), "basis_" + basistr) + os.path.sep

    ##Load PW matrices
    if (estimation == "pairwise" or estimation == "pw"):

        string_dims_PWMatrix = "_".join(map(str, posprobes)) + "act_" + str(
            int(amplitudePW)) + "nm_" + str(int(
                cut)) + "cutsvd_dim_sampl_" + str(dim_sampl) + "_dim" + str(
                    thd2.dim_im) + '_radpup' + str(thd2.prad)

        filePW = ("MatrixPW_" + string_dims_PWMatrix)
        if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
            vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
        else:
            raise Exception("Please create PW matrix before correction")

    if thd2.DM1.active == True:
        nam2DM = "_2DM"
    else:
        nam2DM = ""

    #usefull string
    maskDH, maskDHcontrast, string_dhshape = wsc.load_or_save_maskDH(
        intermatrix_dir, Correctionconfig, dim_sampl, DH_sampling, thd2.dim_im,
        thd2.science_sampling)

    string_dims_EFCMatrix = str(amplitudeEFC) + "nm_" + str(
        Nbmodes) + "modes_dim" + str(thd2.dim_im) + '_radpup' + str(
            thd2.prad) + nam2DM

    ## Load Control matrix
    fileDirectMatrix = "DirectMatrix" + string_dhshape + string_dims_EFCMatrix

    if os.path.exists(intermatrix_dir + fileDirectMatrix + ".fits") == True:
        Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + ".fits")
    else:
        print(fileDirectMatrix)
        raise Exception("Please create Direct matrix before correction")

    if correction_algorithm == "EM" or correction_algorithm == "steepest":

        G = np.zeros((int(np.sum(maskDH)), Gmatrix.shape[1]), dtype=complex)
        G = (Gmatrix[0:int(Gmatrix.shape[0] / 2), :] +
             1j * Gmatrix[int(Gmatrix.shape[0] / 2):, :])
        transposecomplexG = np.transpose(np.conjugate(G))
        M0 = np.real(np.dot(transposecomplexG, G))

    ## Phase map and amplitude map for the static aberrations
    if set_phase_abb == True:
        if phase_abb_filename == '':
            phase_abb_filename = "phase_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                int(phase_rms * 1e9), int(phase_slope), phase_rhoc, thd2.prad)

        if set_random_phase is False and os.path.isfile(Model_local_dir +
                                                        phase_abb_filename +
                                                        ".fits") == True:
            phase_up = fits.getdata(Model_local_dir + phase_abb_filename +
                                    ".fits")

        else:
            phase_up = thd2.entrancepupil.random_phase_map(
                phase_rms, phase_rhoc, phase_slope)
            if set_random_phase is False:  # save it for next time
                fits.writeto(Model_local_dir + phase_abb_filename + ".fits",
                             phase_up, overwrite=True)

        phase_up = phase_up * 2 * np.pi / wavelength_0  # where should we do that ? here ?
    else:
        phase_up = 0

    if set_amplitude_abb == True:
        if ampl_abb_filename != '' and os.path.isfile(
                Model_local_dir + ampl_abb_filename +
                ".fits") == True and set_random_ampl is False:
            ampfinal = phase_ampl.scale_amplitude_abb(
                model_dir + ampl_abb_filename + ".fits", thd2.prad,
                thd2.entrancepupil.pup)
        else:
            ampl_abb_filename = "ampl_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                int(ampl_rms), int(ampl_slope), ampl_rhoc, thd2.prad)

            if set_random_ampl is False and os.path.isfile(Model_local_dir +
                                                           ampl_abb_filename +
                                                           ".fits") == True:
                ampfinal = fits.getdata(Model_local_dir + ampl_abb_filename +
                                        ".fits")
            else:
                ampfinal = thd2.entrancepupil.random_phase_map(
                    ampl_rms / 100., ampl_rhoc, ampl_slope)
            if set_random_ampl is False:  # save it for next time
                fits.writeto(Model_local_dir + ampl_abb_filename + ".fits",
                             ampfinal, overwrite=True)
    else:
        ampfinal = 0

    amplitude_abb_up = ampfinal
    phase_abb_up = phase_up

    ## To convert in photon flux
    contrast_to_photons = 1 / thd2.transmission() * nb_photons * thd2.maxPSF / thd2.sumPSF


    ## Adding error on the DM model?

    if DM3_misregistration == True:
        print("DM3 Misregistration!")
        pushactonDM3 = thd2.DM3.creatingpushact(DMconfig,
                                                load_fits=False,
                                                save_fits=False,
                                                model_dir=model_dir)
    else:
        pushactonDM3 = thd2.DM3.DM_pushact

    # not really defined for DM1 but it is there.
    if thd2.DM1.active == True:
        if DM1_misregistration == True:
            print("DM1 Misregistration!")
            pushactonDM1 = thd2.DM1.creatingpushact(DMconfig,
                                                    load_fits=False,
                                                    save_fits=False,
                                                    model_dir=model_dir)
        else:
            pushactonDM1 = thd2.DM1.DM_pushact

    ## Correction loop
    nbiter = len(modevector)
    imagedetector = np.zeros((nbiter + 1, thd2.dim_im, thd2.dim_im))
    phaseDM3 = np.zeros(
        (nbiter + 1, thd2.dim_overpad_pupil, thd2.dim_overpad_pupil))
    phaseDM1 = np.zeros(
        (nbiter + 1, thd2.dim_overpad_pupil, thd2.dim_overpad_pupil))
    meancontrast = np.zeros(nbiter + 1)

    voltage_DM1 = [0.] # initialize with no voltage
    voltage_DM3 = [0.] # initialize with no voltage

    # Initial wavefront in pupil plane
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=amplitude_abb_up)
    imagedetector[0] = thd2.todetector_Intensity(
        entrance_EF=input_wavefront) / thd2.maxPSF

    meancontrast[0] = np.mean(imagedetector[0][np.where(maskDHcontrast != 0)])
    print("Mean contrast in DH: ", meancontrast[0])
    if photon_noise == True:
        photondetector = np.zeros((nbiter + 1, thd2.dim_im, thd2.dim_im))
        photondetector[0] = np.random.poisson(imagedetector[0] *
                                              contrast_to_photons)

    plt.ion()
    plt.figure()
    previousmode = modevector[0]
    k = 0
    dim_pup = thd2.dim_overpad_pupil
    for mode in modevector:
        print("--------------------------------------------------")
        print("Iteration number: ", k, " EFC truncation: ", mode)
        if (estimation == "pairwise" or estimation == "pw"):
            Difference = wsc.createdifference(input_wavefront,
                                              posprobes,
                                              pushactonDM3,
                                              thd2,
                                              dim_sampl,
                                              amplitudePW,
                                              DM1phase=phaseDM1[k],
                                              DM3phase=phaseDM3[k],
                                              noise=photon_noise,
                                              numphot=nb_photons)

            resultatestimation = wsc.FP_PWestimate(Difference, vectoressai)

        elif estimation == "perfect":
            # If polychromatic, assume a perfect estimation at one wavelength
            # input_wavefront = thd2.EF_from_phase_and_ampl(
            #     phase_abb=phase_abb_up, ampl_abb=amplitude_abb_up)
            resultatestimation = thd2.todetector(
                entrance_EF=input_wavefront,
                DM1phase=phaseDM1[k],
                DM3phase=phaseDM3[k]) / np.sqrt(thd2.maxPSF)

            resultatestimation = proc.resampling(resultatestimation, dim_sampl)

        else:
            raise Exception("This estimation algorithm is not yet implemented")

        # Find the solution of actuator motions to apply
        if correction_algorithm == "EFC":

            if Linearization == True:

                # Calculate the control matrix for the current aberrations
                # (needed because of linearization of the problem?)
                # TODO concatenation should be done automatically in thd structure
                if thd2.DM1.active == True:
                    Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        thd2,
                        dim_sampl,
                        np.concatenate(
                            (thd2.DM3.DM_pushact, thd2.DM1.DM_pushact_inpup)) *
                        amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0,
                        maskDH,
                        np.concatenate((thd2.DM3.WhichInPupil,
                                        thd2.DM3.number_act  +
                                        thd2.DM1.WhichInPupil)),
                        otherbasis=DM3_otherbasis,
                        basisDM3=DM3_basis)
                else:
                    Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        thd2,
                        dim_sampl,
                        thd2.DM3.DM_pushact * amplitudeEFC * 2 * np.pi * 1e-9 /
                        wavelength_0,
                        maskDH,
                        thd2.DM3.WhichInPupil,
                        otherbasis=DM3_otherbasis,
                        basisDM3=DM3_basis)

            # Use the control matrix simulated for a null input aberration
            if Linesearch is False:
                if mode != previousmode or k == 0:
                    invertGDH = wsc.invertSVD(
                        Gmatrix,
                        mode,
                        goal="c",
                        visu=False,
                        regul=regularization,
                        otherbasis=DM3_otherbasis,
                        basisDM3=DM3_basis,
                        intermatrix_dir=intermatrix_dir,
                    )[2]

            else:
                # Look for the best number of modes for the control matrix
                # (uses the interaction matrix calculated for a null input aberration)
                meancontrasttemp = np.zeros(len(Linesearchmode))
                b = 0
                for mode in Linesearchmode:

                    SVD, SVD_trunc, invertGDH = wsc.invertSVD(
                        Gmatrix,
                        mode,
                        goal="c",
                        visu=False,
                        regul=regularization,
                        otherbasis=DM3_otherbasis,
                        basisDM3=DM3_basis,
                        intermatrix_dir=intermatrix_dir,
                    )
                    if thd2.DM1.active == True:

                        solution1 = wsc.solutionEFC(
                            maskDH, resultatestimation, invertGDH,
                            np.concatenate((thd2.DM3.WhichInPupil,
                                            thd2.DM3.number_act +
                                            thd2.DM1.WhichInPupil)),
                            thd2.DM3.number_act +
                            thd2.DM1.number_act)


                        apply_on_DM1 = solution1[DM3.number_act:]* (-gain * amplitudeEFC)
                        voltage_DM1_tmp = voltage_DM1[k] + apply_on_DM1
                        phaseDM1_tmp = thd2.DM1.voltage_to_phase(
                            voltage_DM1_tmp ,
                            wavelength=thd2.wavelength_0)

                    else:
                        solution1 = wsc.solutionEFC(
                            maskDH, resultatestimation, invertGDH,
                            thd2.DM3.WhichInPupil,
                            thd2.DM3.number_act)

                        phaseDM1_tmp = 0.


                    # Phase to apply on DM3

                    apply_on_DM3 = solution1[0:DM3.number_act]* (-gain * amplitudeEFC)
                    # Phase to apply on DM3
                    voltage_DM3_tmp = voltage_DM3[k] + apply_on_DM3
                    phaseDM3_tmp = thd2.DM3.voltage_to_phase(
                        voltage_DM3_tmp, wavelength=thd2.wavelength_0)

                    imagedetectortemp = thd2.todetector_Intensity(
                        entrance_EF=input_wavefront,
                        DM1phase=phaseDM1_tmp,
                        DM3phase=phaseDM3_tmp) / thd2.maxPSF

                    meancontrasttemp[b] = np.mean(
                        imagedetectortemp[np.where(maskDHcontrast != 0)])

                    print('contraste moyen avec regul ', mode, '=',
                          meancontrasttemp[b])

                    b = b + 1

                bestcontrast = np.amin(meancontrasttemp)
                bestregul = Linesearchmode[np.argmin(meancontrasttemp)]
                print('Meilleur contraste= ', bestcontrast, ' Best regul= ',
                      bestregul)

                invertGDH = wsc.invertSVD(
                    Gmatrix,
                    bestregul,
                    goal="c",
                    visu=False,
                    regul=regularization,
                    otherbasis=DM3_otherbasis,
                    basisDM3=DM3_basis,
                    intermatrix_dir=intermatrix_dir,
                )[2]

            if thd2.DM1.active == True:
                solution1 = wsc.solutionEFC(
                    maskDH, resultatestimation, invertGDH,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil, thd2.DM3.number_act  +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act  +
                    thd2.DM1.number_act )
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                            invertGDH, thd2.DM3.WhichInPupil,
                                            thd2.DM3.number_act )

        if correction_algorithm == "EM":
            if mode != previousmode or k == 0:
                invertM0 = wsc.invertSVD(
                    M0,
                    mode,
                    goal="c",
                    visu=False,
                    regul=regularization,
                    otherbasis=DM3_otherbasis,
                    basisDM3=DM3_basis,
                    intermatrix_dir=intermatrix_dir,
                )[2]

            # Concatenate can be done in the THD2 structure
            if thd2.DM1.active == True:
                solution1 = wsc.solutionEM(
                    maskDH, resultatestimation, invertM0, G,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil, thd2.DM3.number_act  +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act  +
                    thd2.DM1.number_act )
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEM(maskDH, resultatestimation,
                                           invertM0, G, thd2.DM3.WhichInPupil,
                                           thd2.DM3.number_act )

        if correction_algorithm == "steepest":
            if thd2.DM1.active == True:
                solution1 = wsc.solutionSteepest(
                    maskDH, resultatestimation, M0, G,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil, thd2.DM3.number_act  +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.number_act  +
                    thd2.DM1.number_act )
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionSteepest(maskDH, resultatestimation,
                                                 M0, G, thd2.DM3.WhichInPupil,
                                                 thd2.DM3.number_act)

        if thd2.DM1.active == True:
            # Phase to apply on DM1
            apply_on_DM1 = solution1[thd2.DM3.number_act :]* (-gain * amplitudeEFC)
            voltage_DM1.append(voltage_DM1[k] + apply_on_DM1)
            phaseDM1[k + 1] = thd2.DM1.voltage_to_phase(
                voltage_DM1[k+1] ,
                wavelength=thd2.wavelength_0)

        apply_on_DM3 = solution1[0:thd2.DM3.number_act ]* (-gain * amplitudeEFC)
        # Phase to apply on DM3
        voltage_DM3.append(voltage_DM3[k] + apply_on_DM3)
        phaseDM3[k + 1] = thd2.DM3.voltage_to_phase(
             voltage_DM3[k+1], wavelength=thd2.wavelength_0)

        imagedetector[k + 1] = thd2.todetector_Intensity(
            entrance_EF=input_wavefront,
            DM1phase=phaseDM1[k + 1],
            DM3phase=phaseDM3[k + 1]) / thd2.maxPSF

        meancontrast[k + 1] = np.mean(
            imagedetector[k + 1][np.where(maskDHcontrast != 0)])
        print("Mean contrast in DH: ", meancontrast[k + 1])
        if photon_noise == True:
            photondetector[k + 1] = np.random.poisson(imagedetector[k + 1] *
                                                      contrast_to_photons)
        plt.clf()
        plt.imshow(np.log10(imagedetector[k + 1]), vmin=-8, vmax=-5)
        plt.colorbar()
        plt.pause(0.01)
        previousmode = mode
        k = k + 1

    plt.show()

    ## SAVING...
    header = useful.from_param_to_header(config)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_Detector_Images" + ".fits",
                 imagedetector,
                 header,
                 overwrite=True)
    if thd2.DM1.active == True:
        fits.writeto(result_dir + current_time_str + "_Phase_on_DM1" + ".fits",
                     phaseDM1,
                     header,
                     overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Phase_on_DM3" + ".fits",
                 phaseDM3,
                 header,
                 overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Mean_Contrast_DH" + ".fits",
                 meancontrast,
                 header,
                 overwrite=True)
    config.filename = result_dir + current_time_str + "_Simulation_parameters" + ".ini"
    config.write()

    if photon_noise == True:
        fits.writeto(result_dir + current_time_str + "_Photon_counting" +
                     ".fits",
                     photondetector,
                     header,
                     overwrite=True)

    plt.clf()
    plt.plot(meancontrast)
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean contrast in Dark Hole")

    return input_wavefront, imagedetector