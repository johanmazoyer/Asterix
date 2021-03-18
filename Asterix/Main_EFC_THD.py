__author__ = 'Raphael Galicher, Johan Mazoyer, and Axel Potier'
print('test')
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

import Asterix.InstrumentSimu_functions as instr
import Asterix.fits_functions as useful

__all__ = ["create_interaction_matrices", "correctionLoop"]

#######################################################
#######################################################
######## Interaction/control matrices for PW and EFC


def create_interaction_matrices(parameter_file,
                                NewMODELconfig={},
                                NewDMconfig={},
                                NewCoronaconfig={},
                                NewPWconfig={},
                                NewEFCconfig={}):

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

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    ##################
    ##################
    ### PW CONFIG
    PWconfig = config["PWconfig"]
    PWconfig.update(NewPWconfig)
    amplitudePW = PWconfig["amplitudePW"]
    posprobes = PWconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = PWconfig["cut"]

    DH_sampling = PWconfig["DH_sampling"]

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)
    DHshape = EFCconfig["DHshape"]
    choosepix = EFCconfig["choosepix"]
    choosepix = [int(i) for i in choosepix]

    DM1_otherbasis = EFCconfig["DM1_otherbasis"]
    DM3_otherbasis = EFCconfig["DM3_otherbasis"]
    Nbmodes = EFCconfig["Nbmodes"]
    amplitudeEFC = EFCconfig["amplitudeEFC"]
    regularization = EFCconfig["regularization"]

    ##THEN DO
    model_dir = os.path.join(Asterixroot, "Model") + os.path.sep

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)

    if DM3_otherbasis == False:
        basistr = "actu"
    else:
        basistr = "fourier"

    # Initialize thd:
    thd2 = instr.THD2_testbed(modelconfig,
                              DMconfig,
                              Coronaconfig,
                              save_fits=True,
                              model_dir=model_dir,
                              Model_local_dir=Model_local_dir)

    #image size after binning
    dim_sampl = int(DH_sampling / thd2.science_sampling * thd2.dim_im / 2) * 2

    #for stability purose, but will be remove
    # corona_struct = thd2.corono

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
    filePW = "MatrixPW_" + string_dims_PWMatrix
    if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
        print("The matrix " + filePW + " already exist")
        vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
    else:
        print("Saving " + filePW + " ...")
        vectoressai, showsvd = wsc.createvectorprobes(thd2.wavelength_0, thd2,
                                                      amplitudePW, posprobes,
                                                      thd2.DM3.DM_pushact,
                                                      dim_sampl, cut)
        fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = "MapEigenvaluesPW" + string_dims_PWMatrix

        if os.path.exists(intermatrix_dir + visuPWMap + ".fits") == False:
            print("Saving " + visuPWMap + " ...")
            fits.writeto(intermatrix_dir + visuPWMap + ".fits", showsvd[1])

    # Saving PW matrices in Labview directory
    if onbench == True:
        probes = np.zeros((len(posprobes), thd2.DM3.DM_pushact.shape[0]),
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
        thd2.DM1.WhichInPupil = np.arange(thd2.DM1.DM_pushact.shape[0])

    # DM3
    if DM3_otherbasis == True:
        thd2.DM1.WhichInPupil = np.arange(thd2.DM3.DM_pushact.shape[0])

    #measure the masks
    maskDH, _, string_dhshape = wsc.load_or_save_maskDH(
        intermatrix_dir, EFCconfig, dim_sampl, DH_sampling, thd2.dim_im,
        thd2.science_sampling)

    #useful string
    string_dims_EFCMatrix = str(amplitudeEFC) + "nm_" + str(
        Nbmodes) + "modes_dim" + str(thd2.dim_im) + '_radpup' + str(
            thd2.prad) + nam2DM

    # Creating EFC control matrix
    fileEFCMatrix = "MatrixEFC" + string_dhshape + string_dims_EFCMatrix

    if os.path.exists(intermatrix_dir + fileEFCMatrix + ".fits") == True:
        print("The matrix " + fileEFCMatrix + " already exist")
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
            print("The matrix " + fileDirectMatrix + " already exist")
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
                     thd2.DM3.DM_pushact.shape[0] + thd2.DM1.WhichInPupil))
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
        # Save EFC control matrix in Labview directory
        #EFCmatrix = np.zeros((invertGDH.shape[1], DM_pushact.shape[0]),
        #                     dtype=np.float32)
        EFCmatrix_DM3 = np.zeros(
            (invertGDH.shape[1], thd2.DM3.DM_pushact.shape[0]),
            dtype=np.float32)
        for i in np.arange(len(thd2.DM3.WhichInPupil)):
            EFCmatrix_DM3[:, thd2.DM3.WhichInPupil[i]] = invertGDH[i, :]
        fits.writeto(Labview_dir + "Matrix_control_EFC_DM3_default.fits",
                     EFCmatrix_DM3,
                     overwrite=True)
        if thd2.DM1.active:
            EFCmatrix_DM1 = np.zeros(
                (invertGDH.shape[1], thd2.DM1.DM_pushact.shape[0]),
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
                   NewPWconfig={},
                   NewEFCconfig={},
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
    ### PW CONFIG
    PWconfig = config["PWconfig"]
    PWconfig.update(NewPWconfig)

    amplitudePW = PWconfig["amplitudePW"]
    posprobes = PWconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = PWconfig["cut"]
    DH_sampling = PWconfig["DH_sampling"]

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)

    Nbmodes = EFCconfig["Nbmodes"]
    DM1_otherbasis = EFCconfig["DM1_otherbasis"]
    DM3_otherbasis = EFCconfig["DM3_otherbasis"]

    amplitudeEFC = EFCconfig["amplitudeEFC"]
    regularization = EFCconfig["regularization"]

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
    estimation = SIMUconfig["estimation"]

    ##THEN DO

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    ## Directories for saving the data
    model_dir = os.path.join(Asterixroot, "Model") + os.path.sep

    Model_local_dir = os.path.join(Data_dir, "Model_local") + os.path.sep

    # Initialize thd:
    thd2 = instr.THD2_testbed(modelconfig,
                              DMconfig,
                              Coronaconfig,
                              load_fits=True,
                              model_dir=model_dir,
                              Model_local_dir=Model_local_dir)

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
        thd2.DM3.WhichInPupil = np.arange(thd2.DM3.DM_pushact.shape[0])
    else:
        basistr = "actu"
        DM3_basis = 0

    # DM1
    if DM1_otherbasis == True:
        thd2.DM1.WhichInPupil = np.arange(thd2.DM1.DM_pushact.shape[0])

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

    ##Load PW matrices
    if (estimation == "PairWise" or estimation == "pairwise"
            or estimation == "PW" or estimation == "pw"):

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
        intermatrix_dir, EFCconfig, dim_sampl, DH_sampling, thd2.dim_im,
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

        if set_random_phase == False and os.path.isfile(Model_local_dir +
                                                        phase_abb_filename +
                                                        ".fits") == True:
            phase_up = fits.getdata(Model_local_dir + phase_abb_filename +
                                    ".fits")

        else:
            phase_up = thd2.entrancepupil.random_phase_map(
                phase_rms, phase_rhoc, phase_slope)
            if set_random_phase == False:  # save it for next time
                fits.writeto(Model_local_dir + phase_abb_filename + ".fits",
                             phase_up)

        phase_up = phase_up * 2 * np.pi / wavelength_0  # where should we do that ? here ?
    else:
        phase_up = 0

    if set_amplitude_abb == True:
        if ampl_abb_filename != '' and os.path.isfile(
                Model_local_dir + ampl_abb_filename +
                ".fits") == True and set_random_ampl == False:
            ampfinal = phase_ampl.scale_amplitude_abb(
                model_dir + ampl_abb_filename + ".fits", thd2.prad,
                thd2.entrancepupil.pup)
        else:
            ampl_abb_filename = "ampl_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}".format(
                int(ampl_rms), int(ampl_slope), ampl_rhoc, thd2.prad)

            if set_random_ampl == False and os.path.isfile(Model_local_dir +
                                                           ampl_abb_filename +
                                                           ".fits") == True:
                ampfinal = fits.getdata(Model_local_dir + ampl_abb_filename +
                                        ".fits")
            else:
                ampfinal = thd2.entrancepupil.random_phase_map(
                    ampl_rms / 100., ampl_rhoc, ampl_slope)
                if set_random_ampl == False:  # save it for next time
                    fits.writeto(Model_local_dir + ampl_abb_filename + ".fits",
                                ampfinal)
    else:
        ampfinal = 0

    amplitude_abb_up = ampfinal
    phase_abb_up = phase_up

    ## To convert in photon flux
    # We can probably replace here by transmission !
    contrast_to_photons = (np.sum(thd2.entrancepupil.pup) /
                           np.sum(thd2.corono.lyot_pup.pup) * nb_photons *
                           thd2.maxPSF / thd2.sumPSF)

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

    # Initial wavefront in pupil plane
    input_wavefront = thd2.EF_from_phase_and_ampl(phase_abb=phase_abb_up,
                                                  ampl_abb=amplitude_abb_up)
    imagedetector[0] = thd2.todetector_Intensity(
        entrance_EF=input_wavefront) / thd2.maxPSF

    # useful.quickfits(imagedetector[0])
    # asd
    #     imagedetector[0] = (corona_struct.im_apodtodetector_chrom(
    # amplitude_abb_up, phase_abb_up)/
    #         corona_struct.maxPSF)

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
        if (estimation == "PairWise" or estimation == "pairwise"
                or estimation == "PW" or estimation == "pw"):

            Difference = wsc.createdifference(input_wavefront,
                                              posprobes,
                                              pushactonDM3 * amplitudePW *
                                              1e-9 * 2 * np.pi / wavelength_0,
                                              thd2,
                                              dim_sampl,
                                              DM1phase=phaseDM1[k],
                                              DM3phase=phaseDM3[k],
                                              noise=photon_noise,
                                              numphot=nb_photons)

            resultatestimation = wsc.FP_PWestimate(Difference, vectoressai)

        elif estimation == "Perfect":
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
                                        thd2.DM3.DM_pushact.shape[0] +
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
            if Linesearch == False:
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
                                            thd2.DM3.DM_pushact.shape[0] +
                                            thd2.DM1.WhichInPupil)),
                            thd2.DM3.DM_pushact.shape[0] +
                            thd2.DM1.DM_pushact.shape[0])

                        voltage_DM1 = solution1[pushactonDM3.shape[0]:]
                        # Phase to apply on DM1
                        apply_on_DM1 = thd2.DM1.voltage_to_phase(
                            voltage_DM1, wavelength=thd2.wavelength_0) * (
                                -gain * amplitudeEFC)

                        phaseDM1_tmp = phaseDM1[k] + proc.crop_or_pad_image(
                            apply_on_DM1, dim_pup)

                    else:
                        solution1 = wsc.solutionEFC(
                            maskDH, resultatestimation, invertGDH,
                            thd2.DM3.WhichInPupil,
                            thd2.DM3.DM_pushact.shape[0])

                        phaseDM1_tmp = 0.

                    # Phase to apply on DM3
                    voltage_DM3 = solution1[0:pushactonDM3.shape[0]]
                    apply_on_DM3 = thd2.DM3.voltage_to_phase(
                        voltage_DM3,
                        wavelength=thd2.wavelength_0) * (-gain * amplitudeEFC)

                    phaseDM3_tmp = phaseDM3[k] + proc.crop_or_pad_image(
                        apply_on_DM3, dim_pup)

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
                        (thd2.DM3.WhichInPupil, thd2.DM3.DM_pushact.shape[0] +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.DM_pushact.shape[0] +
                    thd2.DM1.DM_pushact.shape[0])
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                            invertGDH, thd2.DM3.WhichInPupil,
                                            thd2.DM3.DM_pushact.shape[0])

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
                        (thd2.DM3.WhichInPupil, thd2.DM3.DM_pushact.shape[0] +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.DM_pushact.shape[0] +
                    thd2.DM1.DM_pushact.shape[0])
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionEM(maskDH, resultatestimation,
                                           invertM0, G, thd2.DM3.WhichInPupil,
                                           thd2.DM3.DM_pushact.shape[0])

        if correction_algorithm == "steepest":
            if thd2.DM1.active == True:
                solution1 = wsc.solutionSteepest(
                    maskDH, resultatestimation, M0, G,
                    np.concatenate(
                        (thd2.DM3.WhichInPupil, thd2.DM3.DM_pushact.shape[0] +
                         thd2.DM1.WhichInPupil)),
                    thd2.DM3.DM_pushact.shape[0] +
                    thd2.DM1.DM_pushact.shape[0])
                # Concatenate should be done in the THD2 structure
            else:
                solution1 = wsc.solutionSteepest(maskDH, resultatestimation,
                                                 M0, G, thd2.DM3.WhichInPupil,
                                                 thd2.DM3.DM_pushact.shape[0])

        if thd2.DM1.active == True:

            voltage_DM1 = solution1[pushactonDM3.shape[0]:]
            # Phase to apply on DM1
            apply_on_DM1 = thd2.DM1.voltage_to_phase(
                voltage_DM1,
                wavelength=thd2.wavelength_0) * (-gain * amplitudeEFC)

            print(phaseDM1[k].shape)
            print(apply_on_DM1.shape)
            asd

            phaseDM1[k + 1] = phaseDM1[k] + proc.crop_or_pad_image(
                apply_on_DM1, dim_pup)

        voltage_DM3 = solution1[0:pushactonDM3.shape[0]]

        # Phase to apply on DM1
        apply_on_DM3 = thd2.DM3.voltage_to_phase(
            voltage_DM3, wavelength=thd2.wavelength_0) * (-gain * amplitudeEFC)

        phaseDM3[k + 1] = phaseDM3[k] + proc.crop_or_pad_image(
            apply_on_DM3, dim_pup)

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
    if thd2.DM1.active == True:
        cut_phaseDM1 = np.zeros((nbiter + 1, 2 * thd2.prad, 2 * thd2.prad))
        for it in np.arange(nbiter + 1):
            cut_phaseDM1[it] = proc.crop_or_pad_image(phaseDM1[it],
                                                      2 * thd2.prad)
    cut_phaseDM3 = np.zeros((nbiter + 1, 2 * thd2.prad, 2 * thd2.prad))
    for it in np.arange(nbiter + 1):
        cut_phaseDM3[it] = proc.crop_or_pad_image(phaseDM3[it], 2 * thd2.prad)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_Detector_Images" + ".fits",
                 imagedetector,
                 header,
                 overwrite=True)
    if thd2.DM1.active == True:
        fits.writeto(result_dir + current_time_str + "_Phase_on_DM1" + ".fits",
                     cut_phaseDM1,
                     header,
                     overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Phase_on_DM3" + ".fits",
                 cut_phaseDM3,
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