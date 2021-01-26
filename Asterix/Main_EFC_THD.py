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
import Asterix.InstrumentSimu_functions as instr
import Asterix.fits_functions as AsFit

__all__ = ["create_interaction_matrices", "correctionLoop"]

#######################################################
#######################################################
######## Interaction/control matrices for PW and EFC


def create_interaction_matrices(parameter_file,
                                NewMODELconfig={},
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

    #DM model
    pitchDM = modelconfig["pitchDM"]
    creating_pushact = modelconfig["creating_pushact"]
    x309 = modelconfig["x309"]
    y309 = modelconfig["y309"]
    xy309 = [x309, y309]
    filename_actu309 = modelconfig["filename_actu309"]
    filename_grid_actu = modelconfig["filename_grid_actu"]
    filename_actu_infl_fct = modelconfig["filename_actu_infl_fct"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    coronagraph = Coronaconfig["coronagraph"]
    coro_position = Coronaconfig["coro_position"]
    knife_coro_offset = Coronaconfig["knife_coro_offset"]
    err_fqpm = Coronaconfig["err_fqpm"]
    prop_lyot2science = Coronaconfig["prop_lyot2science"]

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
    otherbasis = EFCconfig["otherbasis"]
    Nbmodes = EFCconfig["Nbmodes"]
    amplitudeEFC = EFCconfig["amplitudeEFC"]
    regularization = EFCconfig["regularization"]

    ##THEN DO

    model_dir = Asterixroot + os.path.sep + "Model" + os.path.sep

    if otherbasis == False:
        basistr = "actu"
    else:
        basistr = "fourier"

    # Directories for saving data
    intermatrix_dir = (Data_dir + "Interaction_Matrices/" + coronagraph + "/" +
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

    # Pupil and Lyot radii in pixels
    lyotrad = dim_im / 2 / science_sampling
    prad = int(np.ceil(lyotrad * pdiam / lyotdiam))
    lyotrad = int(np.ceil(lyotrad))

    # DM influence functions
    if creating_pushact == True:
        pushact = instr.creatingpushact(
            model_dir,
            dim_im,
            pdiam,
            prad,
            xy309,
            pitchDM=pitchDM,
            filename_actu309=filename_actu309,
            filename_grid_actu=filename_grid_actu,
            filename_actu_infl_fct=filename_actu_infl_fct)
        fits.writeto(model_dir + "PushActInPup" + str(int(dim_im)) + ".fits",
                     pushact,
                     overwrite=True)
    else:
        if os.path.exists(model_dir + "PushActInPup" + str(int(dim_im)) +
                          ".fits") == False:
            print("Extracting data from zip file...")
            ZipFile(model_dir + "PushActInPup" + str(int(dim_im)) + ".zip",
                    "r").extractall(model_dir)

        pushact = fits.getdata(model_dir + "PushActInPup" + str(int(dim_im)) +
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

    ## Entrance pupil and Lyot stop
    if filename_instr_pup != "" and filename_instr_lyot != "":
        entrancepupil = fits.getdata(model_dir +
                                                  filename_instr_pup)
        Coronaconfig.lyot_pup = fits.getdata(model_dir + filename_instr_lyot)
    else:
        entrancepupil = instr.roundpupil(dim_im, prad)
        Coronaconfig.lyot_pup = instr.roundpupil(dim_im, lyotrad)

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
            wavelength, entrancepupil, Coronaconfig, amplitudePW,
            posprobes, pushact, dim_sampl, cut)
        fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = ("MapEigenvaluesPW" + "_" + "_".join(map(str, posprobes)) +
                     "act_" + str(int(amplitudePW)) + "nm")
        if os.path.exists(intermatrix_dir + visuPWMap + ".fits") == False:
            print("Recording " + visuPWMap + " ...")
            fits.writeto(intermatrix_dir + visuPWMap + ".fits", showsvd[1])

    # Saving PW matrices in Labview directory
    if onbench == True:
        probes = np.zeros((len(posprobes), pushact.shape[0]), dtype=np.float32)
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
    fileWhichInPup = "Whichactfor" + str(MinimumSurfaceRatioInThePupil)

    if os.path.exists(intermatrix_dir + fileWhichInPup + ".fits") == True:
        print("The matrix " + fileWhichInPup + " already exist")
        WhichInPupil = fits.getdata(intermatrix_dir + fileWhichInPup + ".fits")
    else:
        print("Recording" + fileWhichInPup + " ...")

        if otherbasis == False:
            WhichInPupil = wsc.creatingWhichinPupil(
                pushact, entrancepupil,
                MinimumSurfaceRatioInThePupil)
        else:
            WhichInPupil = np.arange(pushact.shape[0])
        fits.writeto(intermatrix_dir + fileWhichInPup + ".fits", WhichInPupil)

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
        if otherbasis == True:
            basisDM3 = fits.getdata(Labview_dir + "Map_modes_DM3_foc.fits")
        else:
            basisDM3 = 0

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
            PSF = np.abs(instr.lyottodetector(entrancepupil * Coronaconfig.lyot_pup, Coronaconfig))**2
            maxPSF = np.amax(PSF)

            Gmatrix = wsc.creatingCorrectionmatrix(
                entrancepupil,
                0,
                0,
                Coronaconfig,
                dim_sampl,
                wavelength,
                amplitudeEFC,
                pushact,
                maskDH,
                WhichInPupil,
                maxPSF,
                otherbasis=otherbasis,
                basisDM3=basisDM3,
            )

            fits.writeto(intermatrix_dir + fileDirectMatrix + ".fits", Gmatrix)

        # Calculating and recording EFC Control Matrix
        print("Recording " + fileEFCMatrix + " ...")
        SVD, SVD_trunc, invertGDH = wsc.invertSVD(
            Gmatrix,
            Nbmodes,
            goal="c",
            regul=regularization,
            otherbasis=otherbasis,
            basisDM3=basisDM3,
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
        EFCmatrix = np.zeros((invertGDH.shape[1], pushact.shape[0]),
                             dtype=np.float32)
        for i in np.arange(len(WhichInPupil)):
            EFCmatrix[:, WhichInPupil[i]] = invertGDH[i, :]
        fits.writeto(Labview_dir + "Matrix_control_EFC_DM3_default.fits",
                     EFCmatrix,
                     overwrite=True)

    return 0


#######################################################
#######################################################
######## Simulation of a correction loop


def correctionLoop(parameter_file,
                   NewMODELconfig={},
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

    #DM model
    pitchDM = modelconfig["pitchDM"]
    x309 = modelconfig["x309"]
    y309 = modelconfig["y309"]
    xy309 = [x309, y309]
    filename_actu309 = modelconfig["filename_actu309"]
    filename_grid_actu = modelconfig["filename_grid_actu"]
    filename_actu_infl_fct = modelconfig["filename_actu_infl_fct"]

    ##################
    ##################
    ### coronagraph CONFIG
    Coronaconfig = config["Coronaconfig"]
    Coronaconfig.update(NewCoronaconfig)

    coronagraph = Coronaconfig["coronagraph"]
    coro_position = Coronaconfig["coro_position"]
    knife_coro_offset = Coronaconfig["knife_coro_offset"]
    err_fqpm = Coronaconfig["err_fqpm"]
    prop_lyot2science = Coronaconfig["prop_lyot2science"]

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
    otherbasis = EFCconfig["otherbasis"]
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
    phase_abb = SIMUconfig["phase_abb"]
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
    if otherbasis == False:
        basistr = "actu"
    else:
        basistr = "fourier"
    intermatrix_dir = (Data_dir + "Interaction_Matrices/" + coronagraph + "/" +
                       str(int(wavelength * 1e9)) + "nm/p" +
                       str(round(pdiam * 1e3, 2)) + "_l" +
                       str(round(lyotdiam * 1e3, 1)) + "/ldp_" +
                       str(round(science_sampling, 2)) + "/basis_" + basistr +
                       "/")

    if otherbasis == True:
        basisDM3 = fits.getdata(Labview_dir + "Map_modes_DM3_foc.fits")
    else:
        basisDM3 = 0

    ## DM influence functions
    if os.path.exists(model_dir + "PushActInPup" + str(int(dim_im)) +
                      ".fits") == False:
        print("Extracting data from zip file...")
        ZipFile(model_dir + "PushActInPup" + str(int(dim_im)) + ".zip",
                "r").extractall(model_dir)

    pushact = fits.getdata(model_dir + "PushActInPup" + str(int(dim_im)) +
                           ".fits")

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

    ## Entrance pupil and Lyot stop
    lyotrad = dim_im / 2 / science_sampling
    prad = int(np.ceil(lyotrad * pdiam / lyotdiam))
    lyotrad = int(np.ceil(lyotrad))

    if filename_instr_pup != "" and filename_instr_lyot != "":
        entrancepupil = fits.getdata(model_dir +
                                                  filename_instr_pup)
        Coronaconfig.lyot_pup = fits.getdata(model_dir + filename_instr_lyot)
    else:
        entrancepupil = instr.roundpupil(dim_im, prad)
        if prop_lyot2science == 'fft':
            Coronaconfig.lyot_pup = instr.roundpupil(dim_im, lyotrad)
        if prop_lyot2science == 'mft':
            Coronaconfig.lyot_pup = instr.roundpupil(2*lyotrad, dim_im//2)

    if Coronaconfig.perfect_coro:
        Coronaconfig.perfect_Lyot_pupil = instr.pupiltolyot(entrancepupil,Coronaconfig)

    ## Non coronagraphic PSF 
    # PSF = np.abs(instr.pupiltodetector(entrancepupil , 1, Coronaconfig.lyot_pup))**2
    PSF = np.abs(instr.lyottodetector(entrancepupil * Coronaconfig.lyot_pup, Coronaconfig))**2
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

    fileWhichInPup = "Whichactfor" + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + fileWhichInPup + ".fits") == True:
        WhichInPupil = fits.getdata(intermatrix_dir + fileWhichInPup + ".fits")
    else:
        print("Please create Whichactfor matrix before correction")
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
        G = np.zeros((int(np.sum(maskDH)), len(WhichInPupil)), dtype=complex)
        G = (Gmatrix[0:int(Gmatrix.shape[0] / 2), :] +
             1j * Gmatrix[int(Gmatrix.shape[0] / 2):, :])
        transposecomplexG = np.transpose(np.conjugate(G))
        M0 = np.real(np.dot(transposecomplexG, G))

    ## Phase map and amplitude map for the static aberrations
    ## TODO Load aberration maps (A checker, Amplitude sans doute a refaire proprement!!!)
    if set_phase_abb == True:
        if set_random_phase == True:
            phase = instr.random_phase_map(dim_im, phaserms, rhoc_phase,
                                           slope_phase)
        else:
            phase = fits.getdata(model_dir + phase_abb + ".fits")

        phase = phase * 2 * np.pi / wavelength
    else:
        phase = 0

    if set_amplitude_abb == True:
        oui = fits.getdata(model_dir + amplitude_abb +
                           ".fits")  # *roundpupil(dim_im,prad)
        moy = np.mean(oui[np.where(oui != 0)])
        amp = oui / moy
        amp1 = skimage.transform.rescale(amp,
                                         int(2 * prad / 148 * 400) /
                                         amp.shape[0],
                                         preserve_range=True,
                                         anti_aliasing=True,
                                         multichannel=False)
        ampfinal = np.zeros((dim_im, dim_im))
        ampfinal[int(dim_im / 2 - len(amp1) / 2) +
                 1:int(dim_im / 2 + len(amp1) / 2) + 1,
                 int(dim_im / 2 - len(amp1) / 2) +
                 1:int(dim_im / 2 + len(amp1) / 2) + 1, ] = amp1
        ampfinal = (ampfinal) * instr.roundpupil(dim_im, prad - 1)
        moy = np.mean(ampfinal[np.where(ampfinal != 0)])
        ampfinal = (ampfinal / moy - np.ones(
            (dim_im, dim_im))) * instr.roundpupil(dim_im, prad - 1)  # /10
    else:
        ampfinal = 0

    amplitude_abb = ampfinal
    phase_abb = phase

    ## To convert in photon flux
    contrast_to_photons = (np.sum(entrancepupil) /
                           np.sum(Coronaconfig.lyot_pup) * nb_photons *
                           maxPSF / np.sum(PSF))

    ## Adding error on the DM model?
    if xerror == 0 and yerror == 0 and angerror == 0 and gausserror == 0:
        pushactonDM = pushact
    else:
        print("Misregistration!")
        pushactonDM = instr.creatingpushact(
            model_dir,
            dim_im,
            pdiam,
            prad,
            xy309,
            pitchDM=pitchDM,
            filename_actu309=filename_actu309,
            filename_grid_actu=filename_grid_actu,
            filename_actu_infl_fct=filename_actu_infl_fct,
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

    input_wavefront = entrancepupil * (
        1 + amplitude_abb) * np.exp(1j * phase_abb)

    imagedetector[0] = (
        abs(instr.pupiltodetector(input_wavefront, Coronaconfig))**2 / maxPSF)
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
            Difference = instr.createdifference(amplitude_abb,
                                                phase_abb,
                                                posprobes,
                                                pushactonDM,
                                                amplitudePW,
                                                entrancepupil,
                                                Coronaconfig,
                                                PSF,
                                                dim_sampl,
                                                wavelength,
                                                noise=photon_noise,
                                                numphot=nb_photons)
            resultatestimation = wsc.FP_PWestimate(Difference, vectoressai)

        elif estimation == "Perfect":
            resultatestimation = instr.pupiltodetector(
                input_wavefront, Coronaconfig) / np.sqrt(maxPSF)

            resultatestimation = proc.resampling(resultatestimation, dim_sampl)

        else:
            print("This estimation algorithm is not yet implemented")
            sys.exit()

        if correction_algorithm == "EFC":

            if Linearization == True:

                Gmatrix = wsc.creatingCorrectionmatrix(
                    entrancepupil,
                    amplitude_abb,
                    phase_abb,
                    Coronaconfig,
                    dim_sampl,
                    wavelength,
                    amplitudeEFC,
                    pushact,
                    maskDH,
                    WhichInPupil,
                    maxPSF,
                    otherbasis=otherbasis,
                    basisDM3=basisDM3,
                )

            if Linesearch == False:
                invertGDH = wsc.invertSVD(
                    Gmatrix,
                    mode,
                    goal="c",
                    visu=False,
                    regul=regularization,
                    otherbasis=otherbasis,
                    basisDM3=basisDM3,
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
                        otherbasis=otherbasis,
                        basisDM3=basisDM3,
                        intermatrix_dir=intermatrix_dir,
                    )

                    solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                                invertGDH, WhichInPupil,
                                                pushact.shape[0])

                    apply_on_DM = (-gain * amplitudeEFC * np.dot(
                        solution1, pushactonDM.reshape(
                            1024, dim_im * dim_im)).reshape(dim_im, dim_im) *
                                   2 * np.pi * 1e-9 / wavelength)

                    input_wavefront = entrancepupil * (
                        1 + amplitude_abb) * np.exp(1j *
                                                    (phase_abb + apply_on_DM))

                    imagedetectortemp = (abs(
                        instr.pupiltodetector(input_wavefront, Coronaconfig))**
                                         2 / maxPSF)

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
                    otherbasis=otherbasis,
                    basisDM3=basisDM3,
                    intermatrix_dir=intermatrix_dir,
                )[2]

            solution1 = wsc.solutionEFC(maskDH, resultatestimation, invertGDH,
                                        WhichInPupil, pushact.shape[0])

        if correction_algorithm == "EM":

            if mode != previousmode:
                invertM0 = wsc.invertSVD(
                    M0,
                    mode,
                    goal="c",
                    visu=False,
                    regul=regularization,
                    otherbasis=otherbasis,
                    basisDM3=basisDM3,
                    intermatrix_dir=intermatrix_dir,
                )[2]

            solution1 = wsc.solutionEM(maskDH, resultatestimation, invertM0, G,
                                       WhichInPupil, pushact.shape[0])

        if correction_algorithm == "steepest":
            solution1 = wsc.solutionSteepest(maskDH, resultatestimation, M0, G,
                                             WhichInPupil, pushact.shape[0])

        apply_on_DM = (-gain * amplitudeEFC * np.dot(
            solution1, pushactonDM.reshape(
                pushact.shape[0], dim_im * dim_im)).reshape(dim_im, dim_im) *
                       2 * np.pi * 1e-9 / wavelength)
        phaseDM[k + 1] = phaseDM[k] + apply_on_DM
        phase_abb = phase_abb + apply_on_DM
        input_wavefront = entrancepupil * (
            1 + amplitude_abb) * np.exp(1j * phase_abb)
        imagedetector[k + 1] = (
            abs(instr.pupiltodetector(input_wavefront, Coronaconfig))**2 /
            maxPSF)
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
    header = AsFit.from_param_to_header(config)
    cut_phaseDM = np.zeros((nbiter + 1, 2 * prad, 2 * prad))
    for it in np.arange(nbiter + 1):
        cut_phaseDM[it] = proc.cropimage(phaseDM[it], 200, 200, 2 * prad)
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

    return phase_abb, imagedetector
