__author__ = 'Axel Potier'

import os
import sys
import datetime
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import skimage.transform

from configobj import ConfigObj
from validate import Validator

import Asterix.processing_functions as proc
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
    dim_sampl = modelconfig["dim_sampl"]  #image size after binning

    #Lambda over D in pixels
    wavelength = modelconfig["wavelength"]
    science_sampling = modelconfig["science_sampling"]

    #pupil and Lyot stop
    pdiam = modelconfig["pdiam"]
    lyotdiam = modelconfig["lyotdiam"]
    filename_instr_pup = modelconfig["filename_instr_pup"]
    filename_instr_lyot = modelconfig["filename_instr_lyot"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM3_pitch = DMconfig["DM3_pitch"]
    DM3_creating_pushact = DMconfig["DM3_creating_pushact"]
    DM3_x309 = DMconfig["DM3_x309"]
    DM3_y309 = DMconfig["DM3_y309"]
    DM3_xy309 = [DM3_x309, DM3_y309]
    DM3_filename_actu309 = DMconfig["DM3_filename_actu309"]
    DM3_filename_grid_actu = DMconfig["DM3_filename_grid_actu"]
    DM3_filename_actu_infl_fct = DMconfig["DM3_filename_actu_infl_fct"]

    ##################
    ##################
    ### coronagraph CONFIG
    coroconfig = config["coroconfig"]
    coroconfig.update(Newcoroconfig)
    corona_type = coroconfig["coronagraph"]

    ##################
    ##################
    ### PW CONFIG
    PWconfig = config["PWconfig"]
    PWconfig.update(NewPWconfig)
    amplitudePW = PWconfig["amplitudePW"]
    posprobes = PWconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = PWconfig["cut"]

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig["MinimumSurfaceRatioInThePupil"]
    DHshape = EFCconfig["DHshape"]
    choosepix = EFCconfig["choosepix"]
    choosepix = [int(i) for i in choosepix]
    circ_rad = EFCconfig["circ_rad"]
    circ_rad = [int(i) for i in circ_rad]
    circ_side = EFCconfig["circ_side"]
    circ_offset = EFCconfig["circ_offset"]
    circ_angle = EFCconfig["circ_angle"]
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

    # Directories for saving data
    intermatrix_dir = (Data_dir + "Interaction_Matrices/" + corona_type + "/" +
                       str(int(wavelength * 1e9)) + "nm/p" +
                       str(round(pdiam * 1e3, 2)) + "_l" +
                       str(round(lyotdiam * 1e3, 1)) + "/ldp_" +
                       str(round(science_sampling, 2)) + "/basis_" + basistr +
                       "/")

    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    if onbench == True:
        Labview_dir = Data_dir + "Labview/"
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    # Initialize coronagraphs:
    corona_struct = coronagraph(model_dir, modelconfig, coroconfig)

    # DM influence functions
    if DM3_creating_pushact == True:
        DM3_pushact = instr.creatingpushact(
            model_dir,
            dim_im,
            pdiam,
            corona_struct.prad,
            DM3_xy309,
            pitchDM=DM3_pitchDM,
            filename_actu309=DM3_filename_actu309,
            filename_grid_actu=DM3_filename_grid_actu,
            filename_actu_infl_fct=DM3_filename_actu_infl_fct)
        fits.writeto(model_dir + "PushActInPup" + str(int(dim_im)) + ".fits",
                     DM3_pushact,
                     overwrite=True)
    else:
        if os.path.exists(model_dir + "PushActInPup" + str(int(dim_im)) +
                          ".fits") == False:
            print("Extracting data from zip file...")
            ZipFile(model_dir + "PushActInPup" + str(int(dim_im)) + ".zip",
                    "r").extractall(model_dir)

        DM3_pushact = fits.getdata(model_dir + "PushActInPup" + str(int(dim_im)) +
                               ".fits")

    # Initialize coronagraphs:

    ## transmission of the phase mask (exp(i*phase))
    ## centered on pixel [0.5,0.5]
    if coronagraph == "fqpm":
        Coronaconfig.FPmsk = instr.FQPM(dim_im, err=err_fqpm)
        Coronaconfig.perfect_coro = True

    elif coronagraph == "knife":
        Coronaconfig.FPmsk = instr.KnifeEdgeCoro(
            dim_im, coro_position, knife_coro_offset,
            science_sampling * lyotdiam / pdiam)
        Coronaconfig.perfect_coro = False
    elif coronagraph == "vortex":
        phasevortex = 0  # to be defined
        Coronaconfig.FPmsk = np.exp(1j * phasevortex)
        Coronaconfig.perfect_coro = True

    ##############
    ##AV
    ##############
    ## Binary entrance pupil
    entrancepupil = instr.create_binary_pupil(
                    model_dir, filename_instr_pup, dim_im, prad)

    ##############
    ##AV
    ##############
    ## Binary Lyot stop
    Coronaconfig.lyot_pup = instr.create_binary_pupil(
                    model_dir, filename_instr_lyot, dim_im, lyotrad)

    if Coronaconfig.perfect_coro:
        Coronaconfig.perfect_Lyot_pupil = instr.pupiltolyot(entrancepupil,Coronaconfig)

    ####Calculating and Recording PW matrix
    filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
              "_".join(map(str, posprobes)) + "act_" + str(int(amplitudePW)) +
              "nm_" + str(int(cut)) + "cutsvd")
    if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
        print("The matrix " + filePW + " already exist")
        vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
    else:
        print("Recording " + filePW + " ...")
        vectoressai, showsvd = wsc.createvectorprobes(
            wavelength, corona_struct.entrancepupil, corona_struct,
            amplitudePW, posprobes, DM3_pushact, dim_sampl, cut)
        fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = ("MapEigenvaluesPW" + "_" + "_".join(map(str, posprobes)) +
                     "act_" + str(int(amplitudePW)) + "nm")
        if os.path.exists(intermatrix_dir + visuPWMap + ".fits") == False:
            print("Recording " + visuPWMap + " ...")
            fits.writeto(intermatrix_dir + visuPWMap + ".fits", showsvd[1])

    # Saving PW matrices in Labview directory
    if onbench == True:
        probes = np.zeros((len(posprobes), DM3_pushact.shape[0]), dtype=np.float32)
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
    # Creating WhichInPup?
    DM3_fileWhichInPup = "DM3_Whichactfor" + str(MinimumSurfaceRatioInThePupil)

    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        print("The matrix " + DM3_fileWhichInPup + " already exist")
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup + ".fits")
    else:
        print("Recording" + DM3_fileWhichInPup + " ...")

        if DM3_otherbasis == False:
            DM3_WhichInPupil = wsc.creatingWhichinPupil(
                DM3_pushact, entrancepupil,
                MinimumSurfaceRatioInThePupil)
        else:
            DM3_WhichInPupil = np.arange(DM3_pushact.shape[0])
        fits.writeto(intermatrix_dir + DM3_fileWhichInPup + ".fits", DM3_WhichInPupil)

    # Creating EFC control matrix?
    if DHshape == "square":
        fileEFCMatrix = ("MatrixEFC_square_" + "_".join(map(str, choosepix)) +
                         "pix_" + str(amplitudeEFC) + "nm_" + str(Nbmodes) +
                         "modes")
    else:
        fileEFCMatrix = ("MatrixEFC_circle_" + "_".join(map(str, circ_rad)) +
                         "pix_" + str(circ_side) + '_' + str(circ_offset) +
                         'pix_' + str(circ_angle) + 'deg_' +
                         str(amplitudeEFC) + "nm_" + str(Nbmodes) + "modes")

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
        if DHshape == "square":
            fileDirectMatrix = ("DirectMatrix_square_" +
                                "_".join(map(str, choosepix)) + "pix_" +
                                str(amplitudeEFC) + "nm_")
        else:
            fileDirectMatrix = ("DirectMatrix_circle_" +
                                "_".join(map(str, circ_rad)) + "pix_" +
                                str(circ_side) + '_' + str(circ_offset) +
                                'pix_' + str(circ_angle) + 'deg_' +
                                str(amplitudeEFC) + "nm_")
        if os.path.exists(intermatrix_dir + fileDirectMatrix +
                          ".fits") == True:
            print("The matrix " + fileDirectMatrix + " already exist")
            Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix +
                                   ".fits")
        else:

            # Creating MaskDH?
            fileMaskDH = ("MaskDH_" + str(dim_sampl))
            if DHshape == "square":
                fileMaskDH = fileMaskDH + ("x" + str(dim_sampl) + "_square_" +
                                           "_".join(map(str, choosepix)) +
                                           "pix")
            else:
                fileMaskDH = fileMaskDH + (
                    "r" + str(dim_sampl) + "_circle_" +
                    "_".join(map(str, circ_rad)) + 'pix_' + str(circ_side) +
                    '_' + str(circ_offset) + 'pix_' + str(circ_angle) + 'deg')

            if os.path.exists(intermatrix_dir + fileMaskDH + ".fits") == True:
                print("Mask of DH " + fileMaskDH + " already exist")
                maskDH = fits.getdata(intermatrix_dir + fileMaskDH + ".fits")
            else:
                print("Recording " + fileMaskDH + " ...")
                maskDH = wsc.creatingMaskDH(dim_sampl,
                                            DHshape,
                                            choosepixDH=choosepix,
                                            circ_rad=circ_rad,
                                            circ_side=circ_side,
                                            circ_offset=circ_offset,
                                            circ_angle=circ_angle)
                fits.writeto(intermatrix_dir + fileMaskDH + ".fits", maskDH)

            # Creating EFC Interaction Matrix if does not exist
            print("Recording " + fileDirectMatrix + " ...")
            ## Non coronagraphic PSF
            PSF = np.abs(
                corona_struct.lyottodetector(corona_struct.entrancepupil *
                                             corona_struct.lyot_pup))**2
            maxPSF = np.amax(PSF)

            Gmatrix = wsc.creatingCorrectionmatrix(
                corona_struct.entrancepupil,
                0,
                0,
                corona_struct,
                dim_sampl,
                wavelength,
                amplitudeEFC,
                DM3_pushact,
                maskDH,
                DM3_WhichInPupil,
                maxPSF,
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
        if DHshape == "square":
            plt.savefig(intermatrix_dir + "invertSVDEFC_square_" +
                        "_".join(map(str, choosepix)) + "pix_" +
                        str(amplitudeEFC) + "nm_.png")
        else:
            plt.savefig(intermatrix_dir + "invertSVDEFC_circle_" +
                        "_".join(map(str, circ_rad)) + "pix_" +
                        str(circ_side) + '_' + str(circ_offset) + 'pix_' +
                        str(circ_angle) + 'deg_' + str(amplitudeEFC) +
                        "nm_.png")

    if onbench == True:
        # Save EFC control matrix in Labview directory
        EFCmatrix = np.zeros((invertGDH.shape[1], DM3_pushact.shape[0]),
                             dtype=np.float32)
        for i in np.arange(len(DM3_WhichInPupil)):
            EFCmatrix[:, DM3_WhichInPupil[i]] = invertGDH[i, :]
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
    dim_sampl = modelconfig["dim_sampl"]  #image size after binning

    #Lambda over D in pixels
    wavelength = modelconfig["wavelength"]
    science_sampling = modelconfig["science_sampling"]

    #pupil and Lyot stop
    pdiam = modelconfig["pdiam"]
    lyotdiam = modelconfig["lyotdiam"]
    filename_instr_pup = modelconfig["filename_instr_pup"]
    filename_instr_lyot = modelconfig["filename_instr_lyot"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM3_pitch = DMconfig["DM3_pitch"]
    DM3_x309 = DMconfig["DM3_x309"]
    DM3_y309 = DMconfig["DM3_y309"]
    DM3_xy309 = [DM3_x309, DM3_y309]
    DM3_filename_actu309 = DMconfig["DM3_filename_actu309"]
    DM3_filename_grid_actu = DMconfig["DM3_filename_grid_actu"]
    DM3_filename_actu_infl_fct = DMconfig["DM3_filename_actu_infl_fct"]

    ##################
    ##################
    ### coronagraph CONFIG
    coroconfig = config["coroconfig"]
    coroconfig.update(Newcoroconfig)

    corona_type = coroconfig["coronagraph"]

    ##################
    ##################
    ### PW CONFIG
    PWconfig = config["PWconfig"]
    PWconfig.update(NewPWconfig)
    amplitudePW = PWconfig["amplitudePW"]
    posprobes = PWconfig["posprobes"]
    posprobes = [int(i) for i in posprobes]
    cut = PWconfig["cut"]

    ##################
    ##################
    ###EFC CONFIG
    EFCconfig = config["EFCconfig"]
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig["MinimumSurfaceRatioInThePupil"]
    DHshape = EFCconfig["DHshape"]
    choosepix = EFCconfig["choosepix"]
    choosepix = [int(i) for i in choosepix]
    circ_rad = EFCconfig["circ_rad"]
    circ_rad = [int(i) for i in circ_rad]
    circ_side = EFCconfig["circ_side"]
    circ_offset = EFCconfig["circ_offset"]
    circ_angle = EFCconfig["circ_angle"]
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
    xerror = SIMUconfig["xerror"]
    yerror = SIMUconfig["yerror"]
    angerror = SIMUconfig["angerror"]
    gausserror = SIMUconfig["gausserror"]
    estimation = SIMUconfig["estimation"]

    ##THEN DO

    ## Number of modes that is used as a function of the iteration cardinal
    modevector = []
    for i in np.arange(len(Nbiter_corr)):
        modevector = modevector + [Nbmode_corr[i]] * Nbiter_corr[i]

    ## Directories for saving the data
    model_dir = Asterixroot + os.path.sep + "Model" + os.path.sep

    if onbench == True:
        Labview_dir = Data_dir + "Labview/"
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

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
############
## AV
############
#RG Version
#intermatrix_dir = (Data_dir + "Interaction_Matrices/" + coronagraph + "/" +
#JMa
intermatrix_dir = (Data_dir + "Interaction_Matrices/" + corona_type + "/" +
                       str(int(wavelength * 1e9)) + "nm/p" +
                       str(round(pdiam * 1e3, 2)) + "_l" +
                       str(round(lyotdiam * 1e3, 1)) + "/ldp_" +
                       str(round(science_sampling, 2)) + "/basis_" + basistr +
                       "/")

    ## DM influence functions
    if os.path.exists(model_dir + "PushActInPup" + str(int(dim_im)) +
                      ".fits") == False:
        print("Extracting data from zip file...")
        ZipFile(model_dir + "PushActInPup" + str(int(dim_im)) + ".zip",
                "r").extractall(model_dir)

    DM3_pushact = fits.getdata(model_dir + "PushActInPup" + str(int(dim_im)) +
                           ".fits")

    # Initialize coronagraphs:
    corona_struct = coronagraph(model_dir, modelconfig, coroconfig)

    ## Non coronagraphic PSF
    # PSF = np.abs(instr.pupiltodetector(entrancepupil , 1, corona_struct.lyot_pup))**2
    PSF = np.abs(
        corona_struct.lyottodetector(corona_struct.entrancepupil *
                                     corona_struct.lyot_pup))**2
    maxPSF = np.amax(PSF)

    ##Load PW matrices
    if (estimation == "PairWise" or estimation == "pairwise"
            or estimation == "PW" or estimation == "pw"):
        filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
                  "_".join(map(str, posprobes)) + "act_" +
                  str(int(amplitudePW)) + "nm_" + str(int(cut)) + "cutsvd")
        if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
            vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
        else:
            print("Please create PW matrix before correction")
            sys.exit()

    DM3_fileWhichInPup = "DM3_Whichactfor" + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup + ".fits")
    else:
        print("Please create DM3 Whichactfor matrix before correction")
        sys.exit()

    ## Load Control matrix
    if DHshape == "square":
        fileDirectMatrix = ("DirectMatrix_square_" +
                            "_".join(map(str, choosepix)) + "pix_" +
                            str(amplitudeEFC) + "nm_")
    else:
        fileDirectMatrix = ("DirectMatrix_circle_" +
                            "_".join(map(str, circ_rad)) + "pix_" +
                            str(circ_side) + '_' + str(circ_offset) + 'pix_' +
                            str(circ_angle) + 'deg_' + str(amplitudeEFC) +
                            "nm_")
    if os.path.exists(intermatrix_dir + fileDirectMatrix + ".fits") == True:
        Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + ".fits")
    else:
        print("Please create Direct matrix before correction")
        sys.exit()

    fileMaskDH = ("MaskDH_" + str(dim_sampl))
    if DHshape == "square":
        fileMaskDH = fileMaskDH + ("x" + str(dim_sampl) + "_square_" +
                                   "_".join(map(str, choosepix)) + 'pix')
    else:
        fileMaskDH = fileMaskDH + ("r" + str(dim_sampl) + "_circle_" +
                                   "_".join(map(str, circ_rad)) + 'pix_' +
                                   str(circ_side) + '_' + str(circ_offset) +
                                   'pix_' + str(circ_angle) + 'deg')
    if os.path.exists(intermatrix_dir + fileMaskDH + ".fits") == True:
        maskDH = fits.getdata(intermatrix_dir + fileMaskDH + ".fits")
    else:
        print("Please create MaskDH matrix before correction")
        sys.exit()

    if correction_algorithm == "EM" or correction_algorithm == "steepest":
        G = np.zeros((int(np.sum(maskDH)), len(DM3_WhichInPupil)), dtype=complex)
        G = (Gmatrix[0:int(Gmatrix.shape[0] / 2), :] +
             1j * Gmatrix[int(Gmatrix.shape[0] / 2):, :])
        transposecomplexG = np.transpose(np.conjugate(G))
        M0 = np.real(np.dot(transposecomplexG, G))

    ## Phase map and amplitude map for the static aberrations
    ## TODO Load aberration maps (A checker, Amplitude sans doute a refaire proprement!!!)
    if set_phase_abb == True:
        if set_random_phase == True:
            print("Random phase aberrations upstream from coronagraph")
            phase_up = instr.random_phase_map(dim_im, phaserms, rhoc_phase,
                                           slope_phase)
        else:
            print("FITS file for phase aberrations upstream from coronagraph")
            phase_up = fits.getdata(model_dir + phase_abb_filename + ".fits")

        phase_up = phase_up * 2 * np.pi / wavelength
    else:
        phase_up = 0

    if set_amplitude_abb == True:
        #File with amplitude aberrations in amplitude (not intensity)
        # centered on the pixel dim/2+1, dim/2 +1 with dim = 2*[dim/2]
        # diameter of the pupil is 148 pixels in this image
        amp = np.fft.fftshift(fits.getdata(model_dir + amplitude_abb + ".fits"))

        #Rescale to the pupil size
        amp1 = skimage.transform.rescale(amp,
                                         2*prad/148,
                                         preserve_range=True,
                                         anti_aliasing=True,
                                         multichannel=False)
        # Shift to center between 4 pixels
        tmp_phase_ramp=np.fft.fftshift(instr.shift_phase_ramp(amp1.shape[0],-.5,-.5))
        amp1 = np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifft2(amp1)*tmp_phase_ramp)))

        # Create the array with same size as the pupil
        ampfinal = np.zeros((dim_im, dim_im))
        ampfinal[int(dim_im / 2 - len(amp1) / 2):
                int(dim_im / 2 + len(amp1) / 2),
                 int(dim_im / 2 - len(amp1) / 2):
                int(dim_im / 2 + len(amp1) / 2), ] = amp1
    
        #Set the average to 0 inside entrancepupil
        ampfinal = (ampfinal / np.mean(ampfinal[np.where(entrancepupil != 0)])
                     - np.ones((dim_im, dim_im))) * entrancepupil  # /10
    else:
        ampfinal = 0

    amplitude_abb_up = ampfinal
    phase_abb_up = phase_up

    ## To convert in photon flux
    contrast_to_photons = (np.sum(corona_struct.entrancepupil) /
                           np.sum(corona_struct.lyot_pup) * nb_photons *
                           maxPSF / np.sum(PSF))

    ## Adding error on the DM model?
    if xerror == 0 and yerror == 0 and angerror == 0 and gausserror == 0:
        pushactonDM = DM3_pushact
    else:
        print("Misregistration!")
        pushactonDM = instr.creatingpushact(
            model_dir,
            dim_im,
            pdiam,
            corona_struct.prad,
            DM3_xy309,
            pitchDM=DM3_pitchDM,
            filename_actu309=DM3_filename_actu309,
            filename_grid_actu=DM3_filename_grid_actu,
            filename_actu_infl_fct=DM3_filename_actu_infl_fct,
            xerror=xerror,
            yerror=yerror,
            angerror=angerror,
            gausserror=gausserror)

    ## Correction loop
    nbiter = len(modevector)
    imagedetector = np.zeros((nbiter + 1, dim_im, dim_im))
    phaseDM = np.zeros((nbiter + 1, dim_im, dim_im))
    meancontrast = np.zeros(nbiter + 1)
    if os.path.exists(intermatrix_dir + fileMaskDH + "_contrast.fits") == True:
        maskDHcontrast = fits.getdata(intermatrix_dir + fileMaskDH +
                                      "_contrast.fits")
    else:
        maskDHcontrast = wsc.creatingMaskDH(
            dim_im,
            DHshape,
            choosepixDH=[
                element * dim_im / dim_sampl for element in choosepix
            ],
            circ_rad=[element * dim_im / dim_sampl for element in circ_rad],
            circ_side=circ_side,
            circ_offset=circ_offset * dim_im / dim_sampl,
            circ_angle=circ_angle)
        fits.writeto(intermatrix_dir + fileMaskDH + "_contrast.fits",
                     maskDHcontrast)

    input_wavefront = corona_struct.entrancepupil * (
        1 + amplitude_abb_up) * np.exp(1j * phase_abb_up)

    imagedetector[0] = (
        abs(corona_struct.pupiltodetector(input_wavefront))**2 / maxPSF)
    meancontrast[0] = np.mean(imagedetector[0][np.where(maskDHcontrast != 0)])
    print("Mean contrast in DH: ", meancontrast[0])
    if photon_noise == True:
        photondetector = np.zeros((nbiter + 1, dim_im, dim_im))
        photondetector[0] = np.random.poisson(imagedetector[0] *
                                              contrast_to_photons)

    plt.ion()
    plt.figure()
    previousmode = 0
    k = 0
    for mode in modevector:
        print("--------------------------------------------------")
        print("Iteration number: ", k, " EFC truncation: ", mode)
        if (estimation == "PairWise" or estimation == "pairwise"
                or estimation == "PW" or estimation == "pw"):
            Difference = instr.createdifference(amplitude_abb_up,
                                                phase_abb_up,
                                                posprobes,
                                                pushactonDM,
                                                amplitudePW,
                                                corona_struct.entrancepupil,
                                                corona_struct,
                                                PSF,
                                                dim_sampl,
                                                wavelength,
                                                noise=photon_noise,
                                                numphot=nb_photons)
            resultatestimation = wsc.FP_PWestimate(Difference, vectoressai)

        elif estimation == "Perfect":
            resultatestimation = corona_struct.pupiltodetector(
                input_wavefront) / np.sqrt(maxPSF)

            resultatestimation = proc.resampling(resultatestimation, dim_sampl)

        else:
            print("This estimation algorithm is not yet implemented")
            sys.exit()

        if correction_algorithm == "EFC":

            if Linearization == True:

                Gmatrix = wsc.creatingCorrectionmatrix(
                    corona_struct.entrancepupil,
                    amplitude_abb_up,
                    phase_abb_up,
                    corona_struct,
                    dim_sampl,
                    wavelength,
                    amplitudeEFC,
                    DM3_pushact,
                    maskDH,
                    DM3_WhichInPupil,
                    maxPSF,
                    otherbasis=DM3_otherbasis,
                    basisDM3=DM3_basis,
                )

            if Linesearch == False:
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

                    solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                                invertGDH, DM3_WhichInPupil,
                                                DM3_pushact.shape[0])

                    apply_on_DM = (-gain * amplitudeEFC * np.dot(
                        solution1, pushactonDM.reshape(
                            1024, dim_im * dim_im)).reshape(dim_im, dim_im) *
                                   2 * np.pi * 1e-9 / wavelength)

                    input_wavefront = corona_struct.entrancepupil * (
                        1 + amplitude_abb_up) * np.exp(1j *
                                                    (phase_abb_up + apply_on_DM))

                    imagedetectortemp = (abs(
                        corona_struct.pupiltodetector(input_wavefront))**2 /
                                         maxPSF)

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

            solution1 = wsc.solutionEFC(maskDH, resultatestimation, invertGDH,
                                        DM3_WhichInPupil, DM3_pushact.shape[0])

        if correction_algorithm == "EM":

            if mode != previousmode:
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

            solution1 = wsc.solutionEM(maskDH, resultatestimation, invertM0, G,
                                       DM3_WhichInPupil, DM3_pushact.shape[0])

        if correction_algorithm == "steepest":
            solution1 = wsc.solutionSteepest(maskDH, resultatestimation, M0, G,
                                             DM3_WhichInPupil, DM3_pushact.shape[0])

        apply_on_DM = (-gain * amplitudeEFC * np.dot(
            solution1, pushactonDM.reshape(
                DM3_pushact.shape[0], dim_im * dim_im)).reshape(dim_im, dim_im) *
                       2 * np.pi * 1e-9 / wavelength)
        phaseDM[k + 1] = phaseDM[k] + apply_on_DM
        phase_abb_up = phase_abb_up + apply_on_DM
        input_wavefront = corona_struct.entrancepupil * (
            1 + amplitude_abb_up) * np.exp(1j * phase_abb_up)

        imagedetector[k + 1] = (
            abs(corona_struct.pupiltodetector(input_wavefront))**2 / maxPSF)
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
    cut_phaseDM = np.zeros(
        (nbiter + 1, 2 * corona_struct.prad, 2 * corona_struct.prad))
    for it in np.arange(nbiter + 1):
        cut_phaseDM[it] = proc.cropimage(phaseDM[it], 200, 200,
                                         2 * corona_struct.prad)
        # plt.clf()
        # plt.figure(figsize=(3, 3))
        # plt.imshow(np.log10(imagedetector[it,100:300,100:300]),vmin=-8,vmax=-5,cmap='Blues_r')#CMRmap
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig(result_dir+'image-'+str(2*it+1)+'.jpeg')
        # plt.close()

    current_time_str = datetime.datetime.today().strftime("%Y%m%d_%Hh%Mm%Ss")
    fits.writeto(result_dir + current_time_str + "_Detector_Images" + ".fits",
                 imagedetector,
                 header,
                 overwrite=True)
    fits.writeto(result_dir + current_time_str + "_Phase_on_DM2" + ".fits",
                 cut_phaseDM,
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

    return phase_abb_up, imagedetector
