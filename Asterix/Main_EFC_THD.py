__author__ = 'Axel Potier'

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
from Asterix.InstrumentSimu_functions import coronagraph
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

    #Image
    dim_im = modelconfig["dim_im"]  #image size on detector

    #Lambda over D in pixels
    wavelength_0 = modelconfig["wavelength_0"]
    science_sampling = modelconfig["science_sampling"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM3_creating_pushact = DMconfig["DM3_creating_pushact"]
    DM1_active = DMconfig["DM1_active"]
    DM1_creating_pushact = DMconfig["DM1_creating_pushact"]
    DM1_z_position = DMconfig["DM1_z_position"]

    DMconfig[
        "DM1_misregistration"] = False  # initially no misregistration, only in the correction part
    DMconfig[
        "DM3_misregistration"] = False  # initially no misregistration, only in the correction part

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

    #image size after binning
    dim_sampl = int(DH_sampling / science_sampling * dim_im / 2) * 2

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig["MinimumSurfaceRatioInThePupil"]
    DHshape = EFCconfig["DHshape"]
    choosepix = EFCconfig["choosepix"]
    DM1_otherbasis = EFCconfig["DM1_otherbasis"]
    DM3_otherbasis = EFCconfig["DM3_otherbasis"]
    Nbmodes = EFCconfig["Nbmodes"]
    amplitudeEFC = EFCconfig["amplitudeEFC"]
    regularization = EFCconfig["regularization"]

    ##THEN DO
    model_dir = Asterixroot + os.path.sep + "Model" + os.path.sep

    if DM3_otherbasis == False:
        basistr = "actu"
    else:
        basistr = "fourier"

    # Initialize coronagraphs:
    corona_struct = coronagraph(modelconfig, Coronaconfig, model_dir)

    # Directories for saving data
    intermatrix_dir = (Data_dir + "Interaction_Matrices/" +
                       corona_struct.corona_type + "/")
    if corona_struct.corona_type == 'fqpm':
        if corona_struct.achrom_fqpm == True:
            intermatrix_dir = intermatrix_dir + "Achromatic_phase_mask/"
        else:
            intermatrix_dir = intermatrix_dir + "Polychromatic_phase_mask/"
    if corona_struct.corona_type == 'knife':
        intermatrix_dir = (intermatrix_dir + corona_struct.coro_position +
                           "/offset_" + str(corona_struct.knife_coro_offset) +
                           "lop/")
    intermatrix_dir = (intermatrix_dir + str(int(wavelength_0 * 1e9)) +
                       "nm/p" +
                       str(round(corona_struct.diam_pup_in_m * 1e3, 2)) +
                       "_l" +
                       str(round(corona_struct.diam_lyot_in_m * 1e3, 1)) +
                       "/lop_" + str(round(science_sampling, 2)) + "/basis_" +
                       basistr + "/")

    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    if onbench == True:
        Labview_dir = Data_dir + "Labview/"
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    Model_local_dir = Data_dir + "Model_local/"
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)
    tmp_nam = "_dimpuparray" + str(int(corona_struct.dim_overpad_pupil))

    # if corona_struct.prop_apod2lyot == "mft":
    #     tmp_nam = "_dimpuparray"+str(
    #         int(corona_struct.dim_overpad_pupil))
    # else:
    #     tmp_nam=""

    if DM1_active == True:
        nam2DM = "_2DM"

        # DM influence functions  # ARGH ! Not to be hardcoded here !!!
        dx, dxout = prop.prop_fresnel(corona_struct.prad * 2 * 1.25,
                                      wavelength_0,
                                      DM1_z_position,
                                      corona_struct.diam_pup_in_m / 2,
                                      corona_struct.prad,
                                      retscale=1)
        corona_struct.pradDM1 = corona_struct.prad * dx / dxout

        if DM1_creating_pushact == True:
            if dx > 2 * dxout:
                print(dx, dxout)
                raise Exception(
                    "Need to enhance the pupil size in pixel for Fresnel propagation"
                )
            # Influence functions of DM1 in DM1 plane
            DM1_pushact = instr.creatingpushact(
                model_dir,
                corona_struct.diam_pup_in_m,
                2 * corona_struct.pradDM1,
                DMconfig,
                corona_struct.dim_overpad_pupil,
                Name_DM='DM1')
            fits.writeto(Model_local_dir + "DM1_PushActInPup_ray" +
                         str(int(corona_struct.pradDM1)) + tmp_nam + ".fits",
                         DM1_pushact,
                         overwrite=True)

            # Influence functions of DM1 in pupil plane
            # used to create the EFC matrix ()
            DM1_pushact_inpup = instr.creatingpushact_inpup(
                DM1_pushact, wavelength_0, corona_struct,
                DMconfig["DM1_z_position"])
            fits.writeto(Model_local_dir + "DM1_PushActInPup_ray" +
                         str(int(corona_struct.pradDM1)) + tmp_nam +
                         "_inPup_real.fits",
                         DM1_pushact_inpup.real,
                         overwrite=True)
            fits.writeto(Model_local_dir + "DM1_PushActInPup_ray" +
                         str(int(corona_struct.pradDM1)) + tmp_nam +
                         "_inPup_imaginary.fits",
                         DM1_pushact_inpup.imag,
                         overwrite=True)
        else:
            # Influence functions of DM1 in pupil plane
            # used to create the EFC matrix ()
            DM1_pushact_inpup = fits.getdata(
                Model_local_dir + "DM1_PushActInPup_ray" +
                str(int(corona_struct.pradDM1)) + tmp_nam + "_inPup_real.fits"
            ) + 1j * fits.getdata(Model_local_dir + "DM1_PushActInPup_ray" +
                                  str(int(corona_struct.pradDM1)) + tmp_nam +
                                  "_inPup_imaginary.fits")
    else:
        nam2DM = ""

    if DM3_creating_pushact == True:
        DM3_pushact = instr.creatingpushact(model_dir,
                                            corona_struct.diam_pup_in_m,
                                            2 * corona_struct.prad,
                                            DMconfig,
                                            corona_struct.dim_overpad_pupil,
                                            Name_DM='DM3')
        fits.writeto(Model_local_dir + "DM3_PushActInPup_ray" +
                     str(int(corona_struct.prad)) + tmp_nam + ".fits",
                     DM3_pushact,
                     overwrite=True)
    else:
        DM3_pushact = fits.getdata(Model_local_dir + "DM3_PushActInPup_ray" +
                                   str(int(corona_struct.prad)) + tmp_nam +
                                   ".fits")

    ####Calculating and Recording PW matrix
    filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
              "_".join(map(str, posprobes)) + "act_" + str(int(amplitudePW)) +
              "nm_" + str(int(cut)) + "cutsvd_dim" +
              str(corona_struct.dim_im) + '_raypup' + str(corona_struct.prad))
    if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
        print("The matrix " + filePW + " already exist")
        vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
    else:
        print("Recording " + filePW + " ...")
        vectoressai, showsvd = wsc.createvectorprobes(wavelength_0,
                                                      corona_struct,
                                                      amplitudePW, posprobes,
                                                      DM3_pushact, dim_sampl,
                                                      cut)
        fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = ("MapEigenvaluesPW" + "_" + "_".join(map(str, posprobes)) +
                     "act_" + str(int(amplitudePW)) + "nm_dim" +
                     str(corona_struct.dim_im) + '_raypup' +
                     str(corona_struct.prad))
        if os.path.exists(intermatrix_dir + visuPWMap + ".fits") == False:
            print("Recording " + visuPWMap + " ...")
            fits.writeto(intermatrix_dir + visuPWMap + ".fits", showsvd[1])

    # Saving PW matrices in Labview directory
    if onbench == True:
        probes = np.zeros((len(posprobes), DM3_pushact.shape[0]),
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

        ####Calculating and Recording EFC matrix
        if DHshape == "square":
            print("TO SET ON LABVIEW: ",
                  str(dim_sampl / 2 + np.array(np.fft.fftshift(choosepix))))

    # Creating WhichInPup
    # DM1
    if DM1_active == True:
        DM1_fileWhichInPup = "DM1_Whichactfor" + str(
            MinimumSurfaceRatioInThePupil) + '_raypup' + str(
                corona_struct.prad)
        if os.path.exists(intermatrix_dir + DM1_fileWhichInPup +
                          ".fits") == True:
            print("The matrix " + DM1_fileWhichInPup + " already exist")
            DM1_WhichInPupil = fits.getdata(intermatrix_dir +
                                            DM1_fileWhichInPup + ".fits")
        else:
            print("Recording" + DM1_fileWhichInPup + " ...")

            if DM1_otherbasis == False:
                DM1_WhichInPupil = wsc.creatingWhichinPupil(
                    DM1_pushact_inpup, corona_struct.entrancepupil,
                    MinimumSurfaceRatioInThePupil)
            else:
                DM1_WhichInPupil = np.arange(DM1_pushact_inpup.shape[0])
            fits.writeto(intermatrix_dir + DM1_fileWhichInPup + ".fits",
                         DM1_WhichInPupil)

    # DM3
    DM3_fileWhichInPup = "DM3_Whichactfor" + str(
        MinimumSurfaceRatioInThePupil) + '_raypup' + str(corona_struct.prad)

    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        print("The matrix " + DM3_fileWhichInPup + " already exist")
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup +
                                        ".fits")
    else:
        print("Recording" + DM3_fileWhichInPup + " ...")

        if DM3_otherbasis == False:
            DM3_WhichInPupil = wsc.creatingWhichinPupil(
                DM3_pushact, corona_struct.entrancepupil.pup,
                MinimumSurfaceRatioInThePupil)
        else:
            DM3_WhichInPupil = np.arange(DM3_pushact.shape[0])
        fits.writeto(intermatrix_dir + DM3_fileWhichInPup + ".fits",
                     DM3_WhichInPupil)

    # Creating EFC control matrix
    string_dhshape = wsc.string_DHshape(EFCconfig)
    fileEFCMatrix = "MatrixEFC" + string_dhshape

    fileEFCMatrix = fileEFCMatrix + str(amplitudeEFC) + "nm_" + str(
        Nbmodes) + "modes_dim" + str(corona_struct.dim_im) + '_raypup' + str(
            corona_struct.prad) + nam2DM

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
        string_dhshape = wsc.string_DHshape(EFCconfig)
        fileDirectMatrix = "DirectMatrix" + string_dhshape

        fileDirectMatrix = fileDirectMatrix + str(
            amplitudeEFC) + "nm_dim" + str(
                corona_struct.dim_im) + '_raypup' + str(
                    corona_struct.prad) + nam2DM

        if os.path.exists(intermatrix_dir + fileDirectMatrix +
                          ".fits") == True:
            print("The matrix " + fileDirectMatrix + " already exist")
            Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix +
                                   ".fits")
        else:
            maskDH, _ = wsc.load_or_save_maskDH(intermatrix_dir, EFCconfig,
                                                dim_sampl, DH_sampling, dim_im,
                                                science_sampling)

            # Creating EFC Interaction Matrix if does not exist
            print("Recording " + fileDirectMatrix + " ...")

            if DM1_active == True:
                DM_pushact = np.concatenate((DM3_pushact, DM1_pushact_inpup))
                DM_WhichInPupil = np.concatenate(
                    (DM3_WhichInPupil,
                     DM3_pushact.shape[0] + DM1_WhichInPupil))
            else:
                DM_pushact = DM3_pushact
                DM_WhichInPupil = DM3_WhichInPupil

            Gmatrix = wsc.creatingCorrectionmatrix(
                corona_struct.entrancepupil.pup,
                corona_struct,
                dim_sampl,
                DM_pushact * amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0,
                maskDH,
                DM_WhichInPupil,
                otherbasis=DM3_otherbasis,
                basisDM3=DM3_basis,
            )

            fits.writeto(intermatrix_dir + fileDirectMatrix + ".fits", Gmatrix)

        # Calculating and recording EFC Control Matrix
        print("Recording " + fileEFCMatrix + " ...")
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

        string_dhshape = wsc.string_DHshape(EFCconfig)
        figSVDEFC = "invertSVDEFC_square" + string_dhshape

        figSVDEFC = figSVDEFC + str(amplitudeEFC) + "nm_dim" + str(
            corona_struct.dim_im) + '_raypup' + str(
                corona_struct.prad) + nam2DM + ".png"
        plt.savefig(figSVDEFC)

    if onbench == True:
        # Save EFC control matrix in Labview directory
        EFCmatrix = np.zeros((invertGDH.shape[1], DM_pushact.shape[0]),
                             dtype=np.float32)
        for i in np.arange(len(DM_WhichInPupil)):
            EFCmatrix[:, DM_WhichInPupil[i]] = invertGDH[i, :]
        fits.writeto(Labview_dir + "Matrix_control_EFC_DM3_default.fits",
                     EFCmatrix,
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

    #Image
    dim_im = modelconfig["dim_im"]  #image size on detector

    #Lambda over D in pixels
    wavelength_0 = modelconfig["wavelength_0"]
    science_sampling = modelconfig["science_sampling"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)
    DM1_active = DMconfig["DM1_active"]
    DM1_z_position = DMconfig["DM1_z_position"]
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

    #image size after binning
    dim_sampl = int(DH_sampling / science_sampling * dim_im / 2) * 2

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig["MinimumSurfaceRatioInThePupil"]
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
    amplitude_abb = SIMUconfig["amplitude_abb"]
    set_phase_abb = SIMUconfig["set_phase_abb"]
    set_random_phase = SIMUconfig["set_random_phase"]
    phaserms = SIMUconfig["phaserms"]
    rhoc_phase = SIMUconfig["rhoc_phase"]
    slope_phase = SIMUconfig["slope_phase"]
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
    model_dir = Asterixroot + os.path.sep + "Model" + os.path.sep
    # model_dir = Data_dir + os.path.sep + "Model" + os.path.sep

    # Initialize coronagraphs:
    corona_struct = coronagraph(modelconfig, Coronaconfig, model_dir)

    if onbench == True:
        Labview_dir = Data_dir + "Labview/"
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    Model_local_dir = Data_dir + "Model_local/"
    if not os.path.exists(Model_local_dir):
        print("Creating directory " + Model_local_dir + " ...")
        os.makedirs(Model_local_dir)

    result_dir = Data_dir + "Results/" + Name_Experiment + "/"
    if not os.path.exists(result_dir):
        print("Creating directory " + result_dir + " ...")
        os.makedirs(result_dir)

    ## Pair-wise probing directory
    if DM3_otherbasis == True:
        basistr = "fourier"
        DM3_basis = fits.getdata(Labview_dir + "Map_modes_DM3_foc.fits")
    else:
        basistr = "actu"
        DM3_basis = 0

    intermatrix_dir = (Data_dir + "Interaction_Matrices/" +
                       corona_struct.corona_type + "/")
    if corona_struct.corona_type == 'fqpm':
        if corona_struct.achrom_fqpm == True:
            intermatrix_dir = intermatrix_dir + "Achromatic_phase_mask/"
        else:
            intermatrix_dir = intermatrix_dir + "Polychromatic_phase_mask/"
    if corona_struct.corona_type == 'knife':
        intermatrix_dir = (intermatrix_dir + corona_struct.coro_position +
                           "/offset_" + str(corona_struct.knife_coro_offset) +
                           "lop/")
    intermatrix_dir = (intermatrix_dir + str(int(wavelength_0 * 1e9)) +
                       "nm/p" +
                       str(round(corona_struct.diam_pup_in_m * 1e3, 2)) +
                       "_l" +
                       str(round(corona_struct.diam_lyot_in_m * 1e3, 1)) +
                       "/lop_" + str(round(science_sampling, 2)) + "/basis_" +
                       basistr + "/")

    tmp_nam = "_dimpuparray" + str(int(corona_struct.dim_overpad_pupil))
    # if corona_struct.prop_apod2lyot == "mft":
    #     tmp_nam = "_dimpuparray"+str(
    #         int(corona_struct.dim_overpad_pupil))
    # else:
    #     tmp_nam=""

    ##Load PW matrices
    if (estimation == "PairWise" or estimation == "pairwise"
            or estimation == "PW" or estimation == "pw"):
        filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
                  "_".join(map(str, posprobes)) + "act_" +
                  str(int(amplitudePW)) + "nm_" + str(int(cut)) +
                  "cutsvd_dim" + str(corona_struct.dim_im) + '_raypup' +
                  str(corona_struct.prad))
        if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
            vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
        else:
            raise Exception("Please create PW matrix before correction")

    #Load DM3 actuator functions
    if os.path.exists(Model_local_dir + "DM3_PushActInPup_ray" +
                      str(int(corona_struct.prad)) + tmp_nam +
                      ".fits") == True:
        DM3_pushact = fits.getdata(Model_local_dir + "DM3_PushActInPup_ray" +
                                   str(int(corona_struct.prad)) + tmp_nam +
                                   ".fits")
    else:
        raise Exception("Please create DM3_PushActInPup before correction")

    # List of DM3 actuators that are inside the pupil
    DM3_fileWhichInPup = "DM3_Whichactfor" + str(
        MinimumSurfaceRatioInThePupil) + '_raypup' + str(corona_struct.prad)
    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup +
                                        ".fits")
    else:
        raise Exception(
            "Please create DM3 Whichactfor matrix before correction")

    if DM1_active == True:
        nam2DM = "_2DM"

        #Load DM1 actuator functions
        if os.path.exists(Model_local_dir + "DM1_PushActInPup_ray" +
                          str(int(corona_struct.pradDM1)) + tmp_nam +
                          ".fits") == True:
            DM1_pushact = fits.getdata(Model_local_dir +
                                       "DM1_PushActInPup_ray" +
                                       str(int(corona_struct.pradDM1)) +
                                       tmp_nam + ".fits")
        else:
            raise Exception("Please create DM1_PushActInPup before correction")
        tmp = Model_local_dir + "DM1_PushActInPup_ray" + str(
            int(corona_struct.pradDM1)) + tmp_nam
        if os.path.exists(tmp + "_inPup_real.fits") == True and os.path.exists(
                tmp + "_inPup_imaginary.fits") == True:
            DM1_pushact_inpup = fits.getdata(
                tmp + "_inPup_real.fits"
            ) + 1j * fits.getdata(tmp + "_inPup_imaginary.fits")

        # List of DM1 actuators that are inside the pupil
        DM1_fileWhichInPup = "DM1_Whichactfor" + str(
            MinimumSurfaceRatioInThePupil) + '_raypup' + str(
                corona_struct.prad)
        if os.path.exists(intermatrix_dir + DM1_fileWhichInPup +
                          ".fits") == True:
            DM1_WhichInPupil = fits.getdata(intermatrix_dir +
                                            DM1_fileWhichInPup + ".fits")
        else:
            raise Exception(
                "Please create DM1 Whichactfor matrix before correction")
    else:
        nam2DM = ""

    ## Load Control matrix
    string_dhshape = wsc.string_DHshape(EFCconfig)
    fileDirectMatrix = "DirectMatrix" + string_dhshape

    fileDirectMatrix = fileDirectMatrix + str(amplitudeEFC) + "nm_dim" + str(
        corona_struct.dim_im) + '_raypup' + str(corona_struct.prad) + nam2DM

    if os.path.exists(intermatrix_dir + fileDirectMatrix + ".fits") == True:
        Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + ".fits")
    else:
        raise Exception("Please create Direct matrix before correction")

    maskDH, maskDHcontrast = wsc.load_or_save_maskDH(intermatrix_dir,
                                                     EFCconfig, dim_sampl,
                                                     DH_sampling, dim_im,
                                                     science_sampling)

    if correction_algorithm == "EM" or correction_algorithm == "steepest":

        G = np.zeros((int(np.sum(maskDH)), Gmatrix.shape[1]), dtype=complex)
        G = (Gmatrix[0:int(Gmatrix.shape[0] / 2), :] +
             1j * Gmatrix[int(Gmatrix.shape[0] / 2):, :])
        transposecomplexG = np.transpose(np.conjugate(G))
        M0 = np.real(np.dot(transposecomplexG, G))

    ## Phase map and amplitude map for the static aberrations
    if set_phase_abb == True:
        if set_random_phase == True:
            print("Random phase aberrations upstream from coronagraph")
            phase_up = phase_ampl.random_phase_map(
                corona_struct.dim_overpad_pupil, phaserms, rhoc_phase,
                slope_phase, corona_struct.entrancepupil)
        else:
            if phase_abb_filename == '':
                phase_abb_filename = "phase_{:d}rms_spd{:d}_rhoc{:.1f}_rad{:d}.fits".format(
                    int(phaserms * 1e9), int(slope_phase), rhoc_phase,
                    corona_struct.prad)
            if os.path.isfile(Model_local_dir + phase_abb_filename + ".fits"):
                phase_up = fits.getdata(Model_local_dir + phase_abb_filename +
                                        ".fits")
            else:
                print(
                    "Fixed phase aberrations upstream from coronagraph, file do not exist yet, generated and saved in "
                    + phase_abb_filename + ".fits")
                phase_up = phase_ampl.random_phase_map(
                    corona_struct.dim_overpad_pupil, phaserms, rhoc_phase,
                    slope_phase, corona_struct.entrancepupil.pup)
                fits.writeto(Model_local_dir + phase_abb_filename + ".fits",
                             phase_up)
            print(
                "Fixed phase aberrations upstream from coronagraph, loaded from: "
                + phase_abb_filename + ".fits")

        phase_up = phase_up * 2 * np.pi / wavelength_0
    else:
        phase_up = 0

    if set_amplitude_abb == True:
        ampfinal = phase_ampl.scale_amplitude_abb(
            model_dir + amplitude_abb + ".fits", corona_struct.prad,
            corona_struct.entrancepupil.pup)
    else:
        ampfinal = 0

    amplitude_abb_up = ampfinal
    phase_abb_up = phase_up

    ## To convert in photon flux
    contrast_to_photons = (np.sum(corona_struct.entrancepupil.pup) /
                           np.sum(corona_struct.lyot_pup.pup) * nb_photons *
                           corona_struct.maxPSF / corona_struct.sumPSF)

    ## Adding error on the DM model?
    if DM3_misregistration == True:
        print("DM Misregistration!")
        pushactonDM3 = instr.creatingpushact(model_dir,
                                             corona_struct.diam_pup_in_m,
                                             2 * corona_struct.prad,
                                             DMconfig,
                                             corona_struct.dim_overpad_pupil,
                                             Name_DM='DM3')
    else:
        pushactonDM3 = DM3_pushact

    ## Correction loop
    nbiter = len(modevector)
    imagedetector = np.zeros(
        (nbiter + 1, corona_struct.dim_im, corona_struct.dim_im))
    phaseDM3 = np.zeros((nbiter + 1, corona_struct.dim_overpad_pupil,
                         corona_struct.dim_overpad_pupil))
    phaseDM1 = np.zeros((nbiter + 1, corona_struct.dim_overpad_pupil,
                         corona_struct.dim_overpad_pupil))
    meancontrast = np.zeros(nbiter + 1)

    # Initial wavefront in pupil plane
    imagedetector[0] = (corona_struct.entrancetodetector_Intensity(
        amplitude_abb_up, phase_abb_up) / corona_struct.maxPSF)

    # TODO Not good. We should do the creation of a WF from phase + abb at one place only,
    # either inside entrancetodetector fucntion or in a separate functions.
    # I would adovcate doing it in a separate functions becasue we use it at different places
    input_wavefront = corona_struct.entrancepupil.pup * (
        1 + amplitude_abb_up) * np.exp(1j * phase_abb_up)

    meancontrast[0] = np.mean(imagedetector[0][np.where(maskDHcontrast != 0)])
    print("Mean contrast in DH: ", meancontrast[0])
    if photon_noise == True:
        photondetector = np.zeros(
            (nbiter + 1, corona_struct.dim_im, corona_struct.dim_im))
        photondetector[0] = np.random.poisson(imagedetector[0] *
                                              contrast_to_photons)

    plt.ion()
    plt.figure()
    previousmode = modevector[0]
    k = 0
    dim_pup = corona_struct.dim_overpad_pupil
    for mode in modevector:
        print("--------------------------------------------------")
        print("Iteration number: ", k, " EFC truncation: ", mode)
        if (estimation == "PairWise" or estimation == "pairwise"
                or estimation == "PW" or estimation == "pw"):

            Difference = instr.createdifference(
                input_wavefront,
                posprobes,
                pushactonDM3 * amplitudePW * 1e-9 * 2 * np.pi / wavelength_0,
                corona_struct,
                dim_sampl,
                noise=photon_noise,
                numphot=nb_photons)

            resultatestimation = wsc.FP_PWestimate(Difference, vectoressai)

        elif estimation == "Perfect":
            # If polychromatic, assume a perfect estimation at one wavelength
            resultatestimation = (corona_struct.entrancetodetector(
                amplitude_abb_up,
                phase_abb_up,
                DM1_active=DM1_active,
                phaseDM1=phaseDM1[k],
                DM3_active=True,
                phaseDM3=phaseDM3[k],
                DM1_z_position=DM1_z_position) / np.sqrt(corona_struct.maxPSF))

            resultatestimation = proc.resampling(resultatestimation, dim_sampl)

        else:
            raise Exception("This estimation algorithm is not yet implemented")

        # Find the solution of actuator motions to apply
        if correction_algorithm == "EFC":

            if Linearization == True:

                # Calculate the control matrix for the current aberrations
                # (needed because of linearization of the problem?)

                if DM1_active == True:
                    Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        corona_struct,
                        dim_sampl,
                        np.concatenate((DM3_pushact, DM1_pushact_inpup)) *
                        amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0,
                        maskDH,
                        np.concatenate(
                            (DM3_WhichInPupil,
                             DM3_pushact.shape[0] + DM1_WhichInPupil)),
                        otherbasis=DM3_otherbasis,
                        basisDM3=DM3_basis)
                else:
                    Gmatrix = wsc.creatingCorrectionmatrix(
                        input_wavefront,
                        corona_struct,
                        dim_sampl,
                        DM3_pushact * amplitudeEFC * 2 * np.pi * 1e-9 /
                        wavelength_0,
                        maskDH,
                        DM3_WhichInPupil,
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
                    tmp_input_wavefront = input_wavefront

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
                    if DM1_active == True:

                        solution1 = wsc.solutionEFC(
                            maskDH, resultatestimation, invertGDH,
                            np.concatenate(
                                (DM3_WhichInPupil,
                                 DM3_pushact.shape[0] + DM1_WhichInPupil)),
                            DM3_pushact.shape[0] + DM1_pushact.shape[0])
                        # Phase to apply on DM1
                        apply_on_DM1 = wsc.apply_on_DM(
                            solution1[pushactonDM3.shape[0]:],
                            DM1_pushact) * (-gain * amplitudeEFC * 2 * np.pi *
                                            1e-9 / wavelength_0)

                        tmp_input_wavefront = instr.prop_pup_DM1_DM3(
                            tmp_input_wavefront, apply_on_DM1, wavelength_0,
                            DM1_z_position, corona_struct.diam_pup_in_m / 2,
                            corona_struct.prad)
                    else:
                        solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                                    invertGDH,
                                                    DM3_WhichInPupil,
                                                    DM3_pushact.shape[0])

                # Phase to apply on DM3
                    apply_on_DM3 = wsc.apply_on_DM(
                        solution1[0:pushactonDM3.shape[0]],
                        pushactonDM3) * (-gain * amplitudeEFC * 2 * np.pi *
                                         1e-9 / wavelength_0)

                    # Propagation in DM1 plane, add DM1 phase,
                    # propagate to next pupil plane (DM3 plane),
                    # add DM3 phase and propagate to detector
                    # imagedetectortemp=(corona_struct.entrancetodetector_Intensity(
                    #             amplitude_abb_up, phase_abb_up,
                    #             DM3_active = True, phaseDM3 = apply_on_DM3,
                    #             DM1_active=DM1_active,phaseDM1=apply_on_DM1,
                    #             DM1_z_position=DM1_z_position)/corona_struct.maxPSF)

                    # # Add DM3 phase
                    tmp_input_wavefront = tmp_input_wavefront * np.exp(
                        1j * proc.crop_or_pad_image(
                            apply_on_DM3, corona_struct.dim_overpad_pupil))

                    imagedetectortemp = (
                        abs(corona_struct.todetector(tmp_input_wavefront))**2 /
                        corona_struct.maxPSF)

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

            if DM1_active == True:
                solution1 = wsc.solutionEFC(
                    maskDH, resultatestimation, invertGDH,
                    np.concatenate((DM3_WhichInPupil,
                                    DM3_pushact.shape[0] + DM1_WhichInPupil)),
                    DM3_pushact.shape[0] + DM1_pushact.shape[0])
            else:
                solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                            invertGDH, DM3_WhichInPupil,
                                            DM3_pushact.shape[0])

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

            if DM1_active == True:
                solution1 = wsc.solutionEM(
                    maskDH, resultatestimation, invertM0, G,
                    np.concatenate((DM3_WhichInPupil,
                                    DM3_pushact.shape[0] + DM1_WhichInPupil)),
                    DM3_pushact.shape[0] + DM1_pushact.shape[0])
            else:
                solution1 = wsc.solutionEM(maskDH, resultatestimation,
                                           invertM0, G, DM3_WhichInPupil,
                                           DM3_pushact.shape[0])

        if correction_algorithm == "steepest":
            if DM1_active == True:
                solution1 = wsc.solutionSteepest(
                    maskDH, resultatestimation, M0, G,
                    np.concatenate((DM3_WhichInPupil,
                                    DM3_pushact.shape[0] + DM1_WhichInPupil)),
                    DM3_pushact.shape[0] + DM1_pushact.shape[0])
            else:
                solution1 = wsc.solutionSteepest(maskDH, resultatestimation,
                                                 M0, G, DM3_WhichInPupil,
                                                 DM3_pushact.shape[0])

        if DM1_active == True:
            # Phase to apply on DM1
            apply_on_DM1 = wsc.apply_on_DM(
                solution1[pushactonDM3.shape[0]:], DM1_pushact) * (
                    -gain * amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0)
            phaseDM1[k + 1] = phaseDM1[k] + proc.crop_or_pad_image(
                apply_on_DM1, dim_pup)

            # # Propagation in DM1 plane, add DM1 phase
            # # and propagate to next pupil plane (DM3 plane)
            # TODO Not good, 2 very different stuff (wavefront before and after the DM system) have the same name
            input_wavefront = instr.prop_pup_DM1_DM3(
                input_wavefront, apply_on_DM1, wavelength_0, DM1_z_position,
                corona_struct.diam_pup_in_m / 2., corona_struct.prad)

        # Phase to apply on DM3
        apply_on_DM3 = wsc.apply_on_DM(
            solution1[0:pushactonDM3.shape[0]], pushactonDM3) * (
                -gain * amplitudeEFC * 2 * np.pi * 1e-9 / wavelength_0)

        phaseDM3[k + 1] = phaseDM3[k] + proc.crop_or_pad_image(
            apply_on_DM3, dim_pup)

        # TODO Not good, 2  different stuff have the same name
        input_wavefront = input_wavefront * np.exp(
            1j * proc.crop_or_pad_image(apply_on_DM3, dim_pup))

        # imagedetector[k + 1] = (
        #     abs(corona_struct.apodtodetector(input_wavefront))**2 /
        #     corona_struct.maxPSF)

        # Propagation in DM1 plane, add DM1 phase,
        # propagate to next pupil plane (DM3 plane),
        # add DM3 phase and propagate to detector
        imagedetector[k + 1] = (corona_struct.entrancetodetector_Intensity(
            amplitude_abb_up,
            phase_abb_up,
            DM1_active=DM1_active,
            phaseDM1=phaseDM1[k + 1],
            DM3_active=True,
            phaseDM3=phaseDM3[k + 1],
            DM1_z_position=DM1_z_position) / corona_struct.maxPSF)

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
    if DM1_active == True:
        cut_phaseDM1 = np.zeros(
            (nbiter + 1, 2 * corona_struct.prad, 2 * corona_struct.prad))
        for it in np.arange(nbiter + 1):
            cut_phaseDM1[it] = proc.crop_or_pad_image(phaseDM1[it],
                                                      2 * corona_struct.prad)
    cut_phaseDM3 = np.zeros(
        (nbiter + 1, 2 * corona_struct.prad, 2 * corona_struct.prad))
    for it in np.arange(nbiter + 1):
        cut_phaseDM3[it] = proc.crop_or_pad_image(phaseDM3[it],
                                                  2 * corona_struct.prad)

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_Detector_Images" + ".fits",
                 imagedetector,
                 header,
                 overwrite=True)
    if DM1_active == True:
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