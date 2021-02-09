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

    #Lambda over D in pixels
    wavelength = modelconfig["wavelength"]
    science_sampling = modelconfig["science_sampling"]

    #image size after binning
    dim_sampl = int(modelconfig["DH_sampling"]/science_sampling*dim_im/2)*2
    
    #pupil and Lyot stop
    diam_pup_in_m = modelconfig["diam_pup_in_m"]
    diam_lyot_in_m = modelconfig["diam_lyot_in_m"]

    ##################
    ##################
    ### DM CONFIG
    DMconfig = config["DMconfig"]
    DMconfig.update(NewDMconfig)

    DM3_creating_pushact = DMconfig["DM3_creating_pushact"]
    DM1_creating_pushact = DMconfig["DM1_creating_pushact"]
    DM1_z_position = DMconfig["DM1_z_position"]

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

    # Initialize coronagraphs:
    corona_struct = coronagraph(model_dir, modelconfig, Coronaconfig)

    # Directories for saving data
    intermatrix_dir = (Data_dir + "Interaction_Matrices/" +
                       corona_struct.corona_type + "/" +
                       str(int(wavelength * 1e9)) + "nm/p" +
                       str(round(diam_pup_in_m * 1e3, 2)) + "_l" +
                       str(round(diam_lyot_in_m * 1e3, 1)) + "/ldp_" +
                       str(round(science_sampling, 2)) + "/basis_" +
                       basistr + "/")

    if not os.path.exists(intermatrix_dir):
        print("Creating directory " + intermatrix_dir + " ...")
        os.makedirs(intermatrix_dir)

    if onbench == True:
        Labview_dir = Data_dir + "Labview/"
        if not os.path.exists(Labview_dir):
            print("Creating directory " + Labview_dir + " ...")
            os.makedirs(Labview_dir)

    # DM influence functions
    dx,dxout = instr.prop_fresnel(dim_im,wavelength,DM1_z_position,
    diam_pup_in_m/2,corona_struct.prad,retscale=1)
#    print('Ici')
#    print(dx/dxout)
#    print(corona_struct.diam_pup_in_m/corona_struct.prad/2* corona_struct.diam_pup_in_m/2/ (wavelength*DM1_z_position) * corona_struct.dim_im/corona_struct.prad)
#    print(rad/prad* rad/ (lam*z) * dim_im/prad)

    if DM1_creating_pushact == True:
        DM1_pushact = instr.creatingpushactv2(
         model_dir,diam_pup_in_m,corona_struct.prad,#*dx/dxout,
         DMconfig,which_DM=1,
         xerror=0,yerror=0,angerror=0,gausserror=0)
        fits.writeto(model_dir + "DM1_PushActInPup_ray" +
                 str(int(corona_struct.prad))+".fits",DM1_pushact,overwrite=True)
    else:
        if os.path.exists(model_dir + "DM1_PushActInPup_ray" +
                 str(int(corona_struct.prad))+".fits") == False:
            print("Extracting data from zip file...")
            ZipFile(model_dir + "DM1_PushActInPup_ray" +
                 str(int(corona_struct.prad)) + ".zip", "r").extractall(model_dir)

        DM1_pushact = fits.getdata(model_dir + "DM1_PushActInPup_ray" +
             str(int(corona_struct.prad))+".fits")


    if DM3_creating_pushact == True:
        DM3_pushact = instr.creatingpushactv2(
         model_dir,diam_pup_in_m,corona_struct.prad,
         DMconfig,which_DM=3,
         xerror=0,yerror=0,angerror=0,gausserror=0)
        fits.writeto(model_dir + "DM3_PushActInPup_ray"  +
                 str(int(corona_struct.prad))+".fits",DM3_pushact,overwrite=True)
    else:
        if os.path.exists(model_dir + "DM3_PushActInPup_ray" +
                 str(int(corona_struct.prad))+".fits") == False:
            print("Extracting data from zip file...")
            ZipFile(model_dir + "DM3_PushActInPup_ray"+
                 str(int(corona_struct.prad)) + ".zip", "r").extractall(model_dir)

        DM3_pushact = fits.getdata(model_dir + "DM3_PushActInPup_ray" +
                str(int(corona_struct.prad))+".fits")

    ####Calculating and Recording PW matrix
    filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
              "_".join(map(str, posprobes)) + "act_" + str(int(amplitudePW)) +
              "nm_" + str(int(cut)) + "cutsvd_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
    if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
        print("The matrix " + filePW + " already exist")
        vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
    else:
        print("Recording " + filePW + " ...")
        vectoressai, showsvd = wsc.createvectorprobes(
            wavelength, corona_struct,
            amplitudePW, posprobes, DM3_pushact, dim_sampl, cut)
        fits.writeto(intermatrix_dir + filePW + ".fits", vectoressai)

        visuPWMap = ("MapEigenvaluesPW" + "_" + "_".join(map(str, posprobes)) +
                     "act_" + str(int(amplitudePW)) + "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
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
    DM3_fileWhichInPup = "DM3_Whichactfor" + str(MinimumSurfaceRatioInThePupil)+'_raypup'+str(corona_struct.prad)

    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        print("The matrix " + DM3_fileWhichInPup + " already exist")
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup + ".fits")
    else:
        print("Recording" + DM3_fileWhichInPup + " ...")

        if DM3_otherbasis == False:
            DM3_WhichInPupil = wsc.creatingWhichinPupil(
                DM3_pushact, corona_struct.entrancepupil,
                MinimumSurfaceRatioInThePupil)
        else:
            DM3_WhichInPupil = np.arange(DM3_pushact.shape[0])
        fits.writeto(intermatrix_dir + DM3_fileWhichInPup + ".fits", DM3_WhichInPupil)

    # Creating EFC control matrix?
    if DHshape == "square":
        fileEFCMatrix = ("MatrixEFC_square_" + "_".join(map(str, choosepix)) +
                         "pix_" + str(amplitudeEFC) + "nm_" + str(Nbmodes) +
                         "modes_dim"+str(dim_im)+
                         '_raypup'+str(corona_struct.prad))
    else:
        fileEFCMatrix = ("MatrixEFC_circle_" + "_".join(map(str, circ_rad)) +
                         "pix_" + str(circ_side) + '_' + str(circ_offset) +
                         'pix_' + str(circ_angle) + 'deg_' +
                         str(amplitudeEFC) + "nm_" + str(Nbmodes) + "modes_dim"+
                         str(dim_im)+'_raypup'+str(corona_struct.prad))

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
                                str(amplitudeEFC) + "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
        else:
            fileDirectMatrix = ("DirectMatrix_circle_" +
                                "_".join(map(str, circ_rad)) + "pix_" +
                                str(circ_side) + '_' + str(circ_offset) +
                                'pix_' + str(circ_angle) + 'deg_' +
                                str(amplitudeEFC) + "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
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
                                           "pix_dim"+
                    str(dim_im)+'_raypup'+str(corona_struct.prad))
            else:
                fileMaskDH = fileMaskDH + (
                    "r" + str(dim_sampl) + "_circle_" +
                    "_".join(map(str, circ_rad)) + 'pix_' + str(circ_side) +
                    '_' + str(circ_offset) + 'pix_' + str(circ_angle) + 'deg_dim'+
                    str(dim_im)+'_raypup'+str(corona_struct.prad))

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
                        str(amplitudeEFC) + "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad)+".png")
        else:
            plt.savefig(intermatrix_dir + "invertSVDEFC_circle_" +
                        "_".join(map(str, circ_rad)) + "pix_" +
                        str(circ_side) + '_' + str(circ_offset) + 'pix_' +
                        str(circ_angle) + 'deg_' + str(amplitudeEFC) +
                        "nm_dim"+str(dim_im)+
                        '_raypup'+str(corona_struct.prad)+".png")

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

    #Lambda over D in pixels
    wavelength = modelconfig["wavelength"]
    science_sampling = modelconfig["science_sampling"]

    #image size after binning
    dim_sampl = int(modelconfig["DH_sampling"]/science_sampling*dim_im/2)*2

    #pupil and Lyot stop
    diam_pup_in_m = modelconfig["diam_pup_in_m"]
    diam_lyot_in_m = modelconfig["diam_lyot_in_m"]

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


    # Initialize coronagraphs:
    corona_struct = coronagraph(model_dir, modelconfig, Coronaconfig)

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

    intermatrix_dir = (Data_dir + "Interaction_Matrices/" +
                       corona_struct.corona_type
                     + "/" + str(int(wavelength * 1e9)) + "nm/p" +
                       str(round(diam_pup_in_m * 1e3, 2)) + "_l" +
                       str(round(diam_lyot_in_m * 1e3, 1)) + "/ldp_" +
                       str(round(science_sampling, 2)) + "/basis_" + basistr +
                       "/")

    ## DM influence functions
    if os.path.exists(model_dir + "DM3_PushActInPup_ray"  +
                 str(int(corona_struct.prad))+ ".fits") == False:
        print("Extracting data from zip file...")
        ZipFile(model_dir + "DM3_PushActInPup_ray" +
                 str(int(corona_struct.prad)) + ".zip", "r").extractall(model_dir)

    DM3_pushact = fits.getdata(model_dir + "DM3_PushActInPup_ray" +
                str(int(corona_struct.prad))+".fits")

    ## Non coronagraphic PSF with no aberrations
    PSF = np.abs(
        corona_struct.lyottodetector(corona_struct.entrancepupil *
                                     corona_struct.lyot_pup))**2

    maxPSF = np.amax(PSF)

    ##Load PW matrices
    if (estimation == "PairWise" or estimation == "pairwise"
            or estimation == "PW" or estimation == "pw"):
        filePW = ("MatrixPW_" + str(dim_sampl) + "x" + str(dim_sampl) + "_" +
                  "_".join(map(str, posprobes)) + "act_" +
                  str(int(amplitudePW)) + "nm_" + str(int(cut)) + "cutsvd_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
        if os.path.exists(intermatrix_dir + filePW + ".fits") == True:
            vectoressai = fits.getdata(intermatrix_dir + filePW + ".fits")
        else:
            print("Please create PW matrix before correction")
            sys.exit()

    DM3_fileWhichInPup = "DM3_Whichactfor" + str(MinimumSurfaceRatioInThePupil)+'_raypup'+str(corona_struct.prad)
    if os.path.exists(intermatrix_dir + DM3_fileWhichInPup + ".fits") == True:
        DM3_WhichInPupil = fits.getdata(intermatrix_dir + DM3_fileWhichInPup + ".fits")
    else:
        print("Please create DM3 Whichactfor matrix before correction")
        sys.exit()

    ## Load Control matrix
    if DHshape == "square":
        fileDirectMatrix = ("DirectMatrix_square_" +
                            "_".join(map(str, choosepix)) + "pix_" +
                            str(amplitudeEFC) + "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
    else:
        fileDirectMatrix = ("DirectMatrix_circle_" +
                            "_".join(map(str, circ_rad)) + "pix_" +
                            str(circ_side) + '_' + str(circ_offset) + 'pix_' +
                            str(circ_angle) + 'deg_' + str(amplitudeEFC) +
                            "nm_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
    if os.path.exists(intermatrix_dir + fileDirectMatrix + ".fits") == True:
        Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + ".fits")
    else:
        print("Please create Direct matrix before correction")
        sys.exit()

    fileMaskDH = ("MaskDH_" + str(dim_sampl))
    if DHshape == "square":
        fileMaskDH = fileMaskDH + ("x" + str(dim_sampl) + "_square_" +
                      "_".join(map(str, choosepix)) + "pix_dim"+
                      str(dim_im)+'_raypup'+str(corona_struct.prad))
    else:
        fileMaskDH = fileMaskDH + (
                    "r" + str(dim_sampl) + "_circle_" +
                    "_".join(map(str, circ_rad)) + 'pix_' + str(circ_side) +
                    '_' + str(circ_offset) + 'pix_' + str(circ_angle) + 'deg_dim'+
                    str(dim_im)+'_raypup'+str(corona_struct.prad))

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
            print("Fixed phase aberrations upstream from coronagraph, loaded from: " + phase_abb_filename + ".fits")
            if os.path.isfile(model_dir + phase_abb_filename + ".fits"):
                phase_up = fits.getdata(model_dir + phase_abb_filename + ".fits")
            else:
                print("Fixed phase aberrations upstream from coronagraph, file do not exist yet, generate and save in "+ phase_abb_filename + ".fits")
                phase_up = instr.random_phase_map(dim_im, phaserms, rhoc_phase,
                                           slope_phase)
                fits.writeto(model_dir + phase_abb_filename + ".fits", phase_up)

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
                                         2*corona_struct.prad/148*1.03,
                                         preserve_range=True,
                                         anti_aliasing=True,
                                         multichannel=False)
        # Shift to center between 4 pixels
        #tmp_phase_ramp=np.fft.fftshift(instr.shift_phase_ramp(amp1.shape[0],-.5,-.5))
        #bidouille entre le grandissement 1.03 à la ligne au-dessus et le -1,-1 au lieu
        #de -.5,-.5 C'est pour éviter un écran d'amplitude juste plus petit que la pupille
        tmp_phase_ramp=np.fft.fftshift(instr.shift_phase_ramp(amp1.shape[0],-1.,-1.))
        amp1 = np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifft2(amp1)*tmp_phase_ramp)))

        # Create the array with same size as the pupil
        ampfinal = np.zeros((dim_im, dim_im))
        ampfinal[int(dim_im / 2 - len(amp1) / 2):
                int(dim_im / 2 + len(amp1) / 2),
                 int(dim_im / 2 - len(amp1) / 2):
                int(dim_im / 2 + len(amp1) / 2), ] = amp1
    
        #Set the average to 0 inside entrancepupil
        ampfinal = (ampfinal / np.mean(ampfinal[np.where(corona_struct.entrancepupil != 0)])
                     - np.ones((dim_im, dim_im))) * corona_struct.entrancepupil  # /10
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
        pushactonDM3 = DM3_pushact
    else:
        print("Misregistration!")
        pushactonDM3 = instr.creatingpushactv2(
         model_dir,diam_pup_in_m,corona_struct.prad,
         DMconfig,which_DM=3,
         xerror=xerror,yerror=yerror,angerror=angerror,gausserror=gausserror)

    ## Correction loop
    nbiter = len(modevector)
    imagedetector = np.zeros((nbiter + 1, dim_im, dim_im))
    phaseDM3 = np.zeros((nbiter + 1, dim_im, dim_im))
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
                                                pushactonDM3,
                                                amplitudePW,
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

                    print('voir EFC main l807')
                    solution1 = wsc.solutionEFC(maskDH, resultatestimation,
                                                invertGDH, DM3_WhichInPupil,
                                                DM3_pushact.shape[0])

                    apply_on_DM3 = (-gain * amplitudeEFC * np.dot(
                        solution1, pushactonDM3.reshape(
                            1024, dim_im * dim_im)).reshape(dim_im, dim_im) *
                                   2 * np.pi * 1e-9 / wavelength)

                    input_wavefront = corona_struct.entrancepupil * (
                        1 + amplitude_abb_up) * np.exp(1j *
                                                    (phase_abb_up + apply_on_DM3))

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

        apply_on_DM3 = (-gain * amplitudeEFC * np.dot(
            solution1, pushactonDM3.reshape(
                DM3_pushact.shape[0], pushactonDM3.shape[1] * pushactonDM3.shape[2])).reshape(pushactonDM3.shape[1], pushactonDM3.shape[2]) *
                       2 * np.pi * 1e-9 / wavelength)
        phaseDM3[k + 1,
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2),
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2)]= phaseDM3[k,
           int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2),
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2)]+ apply_on_DM3
        phase_abb_up[int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2),
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2)] = phase_abb_up[
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2),
          int(dim_im/2-pushactonDM3.shape[1]/2):int(dim_im/2+pushactonDM3.shape[1]/2)] + apply_on_DM3
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
    cut_phaseDM3 = np.zeros(
        (nbiter + 1, 2 * corona_struct.prad, 2 * corona_struct.prad))
    for it in np.arange(nbiter + 1):
        cut_phaseDM3[it] = proc.cropimage(phaseDM3[it], 200, 200,
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

    return phase_abb_up, imagedetector
