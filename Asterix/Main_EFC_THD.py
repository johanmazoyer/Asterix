#Version 18/12/2019 A. Potier
#Correction 26/12/2019
#Correction 29/01/2020
#Correction 22/01/2020
#Creates interaction matrices for PW+EFC on the THD2

import os
import datetime
from astropy.io import fits

from Asterix.shortcuts import *
import Asterix.processing_functions as proc
import Asterix.display_functions as disp
import Asterix.fits_functions as fi
import Asterix.WSC_functions as wsc
import Asterix.InstrumentSimu_functions as instr


from configobj import ConfigObj
from validate import Validator

__all__ = ['create_interaction_matrices', 'correctionLoop']

def create_interaction_matrices(parameter_file,NewMODELconfig={},NewPWconfig={},NewEFCconfig={}):
    
    
    Asterixroot = os.path.dirname(os.path.realpath(__file__))
    
    ### CONFIGURATION FILE
    configspec_file   = Asterixroot + os.path.sep+ 'Test_param_configspec.ini'
    config            = ConfigObj(parameter_file,configspec=configspec_file, default_encoding='utf8')
    vtor              = Validator()
    checks            = config.validate(vtor,copy=True) # copy=True for copying the comments     
    print(os.path.exists(configspec_file))
    
    print(parameter_file)
    if not os.path.exists(parameter_file):
        raise Exception('The parameter file '+ parameter_file + ' cannot be found')

    if not os.path.exists(configspec_file):
        raise Exception('The parameter config file '+ configspec_file + ' cannot be found')

    ### CONFIG                         
    Data_dir = config['Data_dir']

    ### MODEL CONFIG
    modelconfig = config['modelconfig']
    modelconfig.update(NewMODELconfig)
    isz   = modelconfig['isz']
    wavelength = modelconfig['wavelength']
    pdiam = modelconfig['pdiam']
    lyotdiam  = modelconfig['lyotdiam']
    set_ldp_manually = modelconfig['set_ldp_manually']
    ld_p= modelconfig['ld_p']
    coronagraph = modelconfig['coronagraph']
    coro_position = modelconfig['coro_position']
    dimimages= modelconfig['dimimages']
    obstructed_pupil= modelconfig['obstructed_pupil']
    creating_pushact= modelconfig['creating_pushact']
    findxy309byhand= modelconfig['findxy309byhand']
    y309= modelconfig['y309']
    x309= modelconfig['x309']


    ### PW CONFIG
    PWconfig=config['PWconfig']
    PWconfig.update(NewPWconfig)
    amplitudePW = PWconfig['amplitudePW']
    posprobes = PWconfig['posprobes']
    posprobes=[int(i) for i in posprobes]
    cut= PWconfig['cut']

    ###EFC CONFIG
    EFCconfig=config['EFCconfig']
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig['MinimumSurfaceRatioInThePupil']
    choosepix = EFCconfig['choosepix']
    choosepix=[int(i) for i in choosepix]
    otherbasis= EFCconfig['otherbasis']
    Nbmodes = EFCconfig['Nbmodes']
    amplitudeEFC = EFCconfig['amplitudeEFC']
    regularization = EFCconfig['regularization']
    
    ##THEN DO

    # model_dir = os.getcwd()+'/'+'Model/'
    model_dir = Asterixroot + os.path.sep + 'Model'+ os.path.sep 

    lyot=lyotdiam/pdiam
    ld_p=ld_p*lyot
    mperpix = 6.5e-6 #Size pixel on detector in meter
    distance = (198.29+301.64)*10**(-3) #Distance Lyot detector in meters
    ld_m = distance*wavelength/lyotdiam #Resolution element in meter
    if set_ldp_manually == False:
        ld_p = ld_m/mperpix  # lambda/D (in pixels)
    
    
    prad0=isz/2/ld_p
    lyotrad=prad0*lyot
    prad=np.ceil(prad0)
    lyotrad=np.ceil(lyotrad)
    prad=int(prad)
    lyotrad=int(lyotrad)
    
    if otherbasis == False:
        basistr = 'actu'
    else:
        basistr = 'fourier'
    intermatrix_dir = Data_dir + 'Interaction_Matrices/' + coronagraph+ '/'+ str(int(wavelength*1e9)) +'nm/p' + str(round(pdiam*1e3 , 2)) + '_l' + str(round(lyotdiam*1e3,1)) + '/ldp_'+str(round(ld_p/lyot,2))+'/basis_' + basistr+'/'
    
    if not os.path.exists(intermatrix_dir):
        print('Creating directory ' + intermatrix_dir + ' ...')
        os.makedirs(intermatrix_dir)
        
    
    Labview_dir = Data_dir+'Labview/'
    if not os.path.exists(Labview_dir):
        print('Creating directory ' + Labview_dir + ' ...')
        os.makedirs(Labview_dir)

    if creating_pushact==True:
        pushact=instr.creatingpushact(model_dir,file309,x309,y309)
        
        fits.writeto(model_dir+'PushActInPup400.fits', pushact)
    else:
        if os.path.exists(model_dir+'PushActInPup400.fits') == False:
            print('Extracting data from zip file...')
            ZipFile(model_dir+'PushActInPup400.zip', 'r').extractall(model_dir)
            
        pushact = fits.getdata(model_dir+'PushActInPup400.fits')
    
    ## transmission of the phase mask (exp(i*phase))
    ## centered on pixel [0.5,0.5]
    if coronagraph =='fqpm':
        coro = fits.getdata(model_dir+'FQPM.fits')
        perfect_coro=True
    elif coronagraph=='knife':
        coro = instr.KnifeEdgeCoro(isz,coro_position,1.2,ld_p)
        perfect_coro=False
    elif coronagraph =='vortex':
        phasevortex=0# to be defined
        coro = np.exp(1j*phasevortex)
        perfect_coro=True
        
    if obstructed_pupil==True:
        #Does not exist yet!!!
        entrancepupil=fits.getdata(model_dir+'instrument_pupil.fits')
        lyot=fits.getdata(model_dir+'lyot_pupil.fits')
        
    else:
        entrancepupil=instr.roundpupil(isz,prad)
        lyot = instr.roundpupil(isz,lyotrad)
    
    perfect_entrance_pupil=entrancepupil
    
    ####Calculating and Recording PW matrix
    filePW = 'MatrixPW_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm_'  + str(int(cut)) + 'cutsvd'
    if os.path.exists(intermatrix_dir + filePW + '.fits') == True:
        print('The matrix ' + filePW + ' already exist')
        vectoressai=fits.getdata(intermatrix_dir + filePW + '.fits')
    else:
        print('Recording ' + filePW + ' ...')
        vectoressai,showsvd = wsc.createvectorprobes(wavelength,entrancepupil,coro,lyot,amplitudePW,posprobes,pushact,dimimages,cut)
        fits.writeto(intermatrix_dir + filePW + '.fits', vectoressai)
    
        visuPWMap='MapEigenvaluesPW'+ '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm'
        if os.path.exists(intermatrix_dir + visuPWMap + '.fits' ) == False:
            print('Recording ' + visuPWMap + ' ...')
            fits.writeto(intermatrix_dir + visuPWMap+'.fits', showsvd[1])
                
    #Saving PW matrices in Labview directory
    probes=np.zeros((len(posprobes),1024),dtype=np.float32)
    vectorPW=np.zeros((2,dimimages*dimimages*len(posprobes)),dtype=np.float32)
    
    for i in np.arange(len(posprobes)):
        probes[i,posprobes[i]]=amplitudePW/17
        vectorPW[0,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,0,i].flatten()
        vectorPW[1,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,1,i].flatten()
    fits.writeto(Labview_dir + 'Probes_EFC_default.fits', probes, overwrite=True)
    fits.writeto(Labview_dir + 'Matr_mult_estim_PW.fits', vectorPW, overwrite=True)
            
    
    ####Calculating and Recording EFC matrix
    print('TO SET ON LABVIEW: ',str(150/2+np.array(shift(choosepix))))    
    #Creating WhichInPup?
    fileWhichInPup = 'Whichactfor' + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + fileWhichInPup + '.fits') == True:
        print('The matrix ' + fileWhichInPup + ' already exist')
        WhichInPupil = fits.getdata(intermatrix_dir + fileWhichInPup + '.fits')
    else:
        print('Recording' + fileWhichInPup + ' ...')
        if otherbasis==False:
            WhichInPupil = wsc.creatingWhichinPupil(pushact,entrancepupil,MinimumSurfaceRatioInThePupil)
        else:
            WhichInPupil = np.arange(1024)
        fits.writeto(intermatrix_dir + fileWhichInPup+'.fits', WhichInPupil)

        
        
    #Creating EFC matrix?
    fileEFCMatrix = 'MatrixEFC_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_' + str(Nbmodes) + 'modes'
                
    if os.path.exists(intermatrix_dir + fileEFCMatrix + '.fits') == True:
        print('The matrix ' + fileEFCMatrix + ' already exist')
        invertGDH = fits.getdata(intermatrix_dir + fileEFCMatrix + '.fits')
    else:
        
    #Actuator basis or another one?
        if otherbasis == True:
            basisDM3 = fits.getdata(Labview_dir+'Map_modes_DM3_foc.fits')
        else:
            basisDM3=0
        
        #Creating Direct matrix?
        fileDirectMatrix = 'DirectMatrix_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_'
        if os.path.exists(intermatrix_dir + fileDirectMatrix + '.fits') == True:
            print('The matrix ' + fileDirectMatrix + ' already exist')
            Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + '.fits')
        else:
            
            
            #Creating MaskDH?
            fileMaskDH = 'MaskDH_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, choosepix))
            if os.path.exists(intermatrix_dir + fileMaskDH + '.fits') == True:
                print('The matrix ' + fileMaskDH + ' already exist')
                maskDH = fits.getdata(intermatrix_dir+fileMaskDH+'.fits')
            else:
                print('Recording ' + fileMaskDH + ' ...')
                maskDH = wsc.creatingMaskDH(dimimages,'square',choosepix)
                #maskDH = wsc.creatingMaskDH(dimimages,'circle',inner=0 , outer=35 , xdecay= 8)
                fits.writeto(intermatrix_dir + fileMaskDH+'.fits', maskDH)

            
            #Creating Direct Matrix if does not exist
            print('Recording ' + fileDirectMatrix + ' ...')
            Gmatrix = wsc.creatingCorrectionmatrix(entrancepupil,coro,lyot,dimimages,wavelength,amplitudeEFC,pushact,maskDH,WhichInPupil,otherbasis=otherbasis,basisDM3=basisDM3)
            fits.writeto(intermatrix_dir + fileDirectMatrix+'.fits', Gmatrix)

        
        
        #Recording EFC Matrix
        print('Recording ' + fileEFCMatrix + ' ...')
        SVD,SVD_trunc,invertGDH = wsc.invertSVD(Gmatrix,Nbmodes,goal='c',regul=regularization,otherbasis=otherbasis,basisDM3=basisDM3,intermatrix_dir=intermatrix_dir,)
        fits.writeto(intermatrix_dir + fileEFCMatrix+'.fits', invertGDH)

        plt.clf()
        plt.plot(SVD,'r.')
        plt.yscale('log')
        plt.savefig(intermatrix_dir+'invertSVDEFC_'+ '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_.png') 
    
    
    
    #Save EFC matrix in Labview directory
    EFCmatrix = np.zeros((invertGDH.shape[1],1024),dtype=np.float32)
    for i in np.arange(len(WhichInPupil)):
        EFCmatrix[:,WhichInPupil[i]]=invertGDH[i,:]
    fits.writeto(Labview_dir + 'Matrix_control_EFC_DM3_default.fits', EFCmatrix, overwrite=True)

    return 0
    
    
    
    

def correctionLoop(parameter_file,NewMODELconfig={},NewPWconfig={},NewEFCconfig={},NewSIMUconfig={}): 
    
    Asterixroot = os.path.dirname(os.path.realpath(__file__))

    ### CONFIGURATION FILE
    configspec_file   = Asterixroot + os.path.sep+'Test_param_configspec.ini'
    config            = ConfigObj(parameter_file,configspec=configspec_file, default_encoding='utf8')
    vtor              = Validator()
    checks            = config.validate(vtor,copy=True) # copy=True for copying the comments     
    
    ### CONFIG                         
    Data_dir = config['Data_dir']

    ### MODEL CONFIG
    modelconfig = config['modelconfig']
    modelconfig.update(NewMODELconfig)
    isz   = modelconfig['isz']
    wavelength = modelconfig['wavelength']
    pdiam = modelconfig['pdiam']
    lyotdiam  = modelconfig['lyotdiam']
    set_ldp_manually = modelconfig['set_ldp_manually']
    ld_p= modelconfig['ld_p']
    coronagraph = modelconfig['coronagraph']
    coro_position = modelconfig['coro_position']
    dimimages= modelconfig['dimimages']
    obstructed_pupil= modelconfig['obstructed_pupil']
    creating_pushact= modelconfig['creating_pushact']
    findxy309byhand= modelconfig['findxy309byhand']
    y309= modelconfig['y309']
    x309= modelconfig['x309']

    ### PW CONFIG
    PWconfig=config['PWconfig']
    PWconfig.update(NewPWconfig)
    amplitudePW = PWconfig['amplitudePW']
    posprobes = PWconfig['posprobes']
    posprobes=[int(i) for i in posprobes]
    cut= PWconfig['cut']

    ###EFC CONFIG
    EFCconfig=config['EFCconfig']
    EFCconfig.update(NewEFCconfig)
    MinimumSurfaceRatioInThePupil = EFCconfig['MinimumSurfaceRatioInThePupil']
    choosepix = EFCconfig['choosepix']
    choosepix=[int(i) for i in choosepix]
    otherbasis= EFCconfig['otherbasis']
    Nbmodes = EFCconfig['Nbmodes']
    amplitudeEFC = EFCconfig['amplitudeEFC']
    regularization = EFCconfig['regularization']
    
    ###SIMU CONFIG
    SIMUconfig = config['SIMUconfig']
    SIMUconfig.update(NewSIMUconfig)
    Name_Experiment = SIMUconfig['Name_Experiment']
    set_amplitude_abb=SIMUconfig['set_amplitude_abb']
    amplitude_abb = SIMUconfig['amplitude_abb']
    set_phase_abb=SIMUconfig['set_phase_abb']
    set_random_phase=SIMUconfig['set_random_phase']
    phaserms = SIMUconfig['phaserms']
    rhoc_phase = SIMUconfig['rhoc_phase']
    slope_phase = SIMUconfig['slope_phase']
    phase_abb = SIMUconfig['phase_abb']
    photon_noise=SIMUconfig['photon_noise']
    nb_photons=SIMUconfig['nb_photons']
    correction_algorithm=SIMUconfig['correction_algorithm']
    Nbiter=SIMUconfig['Nbiter']
    Nbiter=[int(i) for i in Nbiter]
    Nbmode=SIMUconfig['Nbmode']
    Nbmode=[int(i) for i in Nbmode]
    gain = SIMUconfig['gain']
    errormodel = SIMUconfig['errormodel']
    error = SIMUconfig['error']
    estimation = SIMUconfig['estimation']

    modevector=[]
    for i in np.arange(len(Nbiter)):
        modevector=modevector+[Nbmode[i]]*Nbiter[i]
    
    ##THEN DO
    
    model_dir = Asterixroot + os.path.sep + 'Model' + os.path.sep 
    
    Labview_dir = Data_dir+'Labview/'
    if not os.path.exists(Labview_dir):
        print('Creating directory ' + Labview_dir + ' ...')
        os.makedirs(Labview_dir)

    result_dir = Data_dir + 'Results/' + Name_Experiment + '/'
    if not os.path.exists(result_dir):
        print('Creating directory ' + result_dir + ' ...')
        os.makedirs(result_dir)

    lyot=lyotdiam/pdiam
    ld_p=ld_p*lyot
    mperpix = 6.5e-6 #Size pixel on detector in meter
    distance = (198.29+301.64)*10**(-3) #Distance Lyot detector in meters
    ld_m = distance*wavelength/lyotdiam #Resolution element in meter
    if set_ldp_manually == False:
        ld_p = ld_m/mperpix  # lambda/D (in pixels)
    
    
    prad0=isz/2/ld_p
    lyotrad=prad0*lyot
    prad=np.ceil(prad0)
    lyotrad=np.ceil(lyotrad)
    prad=int(prad)
    lyotrad=int(lyotrad)
    
    
    if otherbasis == False:
        basistr = 'actu'
    else:
        basistr = 'fourier'
    intermatrix_dir = Data_dir + 'Interaction_Matrices/' + coronagraph+ '/'+ str(int(wavelength*1e9)) +'nm/p' + str(round(pdiam*1e3 , 2)) + '_l' + str(round(lyotdiam*1e3,1)) + '/ldp_'+str(round(ld_p/lyot,2))+'/basis_' + basistr+'/'
    
        
    if otherbasis == True:
        basisDM3 = fits.getdata(Labview_dir+'Map_modes_DM3_foc.fits')
        basisDM3 = fits.getdata(Labview_dir+'Map_modes_DM3_foc.fits')
    else:
        basisDM3=0
    
    if os.path.exists(model_dir+'PushActInPup400.fits') == False:
        print('Extracting data from zip file...')
        ZipFile(model_dir+'PushActInPup400.zip', 'r').extractall(model_dir)
        
    pushact = fits.getdata(model_dir+'PushActInPup400.fits')
    
    ## transmission of the phase mask (exp(i*phase))
    ## centered on pixel [0.5,0.5]
    if coronagraph =='fqpm':
        coro = fits.getdata(model_dir+'FQPM.fits')
        perfect_coro=True
    elif coronagraph=='knife':
        coro = instr.KnifeEdgeCoro(isz,coro_position,1.2,ld_p)
        perfect_coro=False
    elif coronagraph =='vortex':
        phasevortex=0# to be defined
        coro = np.exp(1j*phasevortex)
        perfect_coro=True
        
    if obstructed_pupil==True:
        #Does not exist yet!!!
        entrancepupil=fits.getdata(model_dir+'instrument_pupil.fits')
        lyot=fits.getdata(model_dir+'lyot_pupil.fits')
        
    else:
        entrancepupil=instr.roundpupil(isz,prad)
        lyot = instr.roundpupil(isz,lyotrad)
    
    perfect_entrance_pupil=entrancepupil
    PSF=np.abs(instr.pupiltodetector(entrancepupil,1,lyot))
    squaremaxPSF=np.amax(PSF)
    
    ##Load matrices
    if estimation=='PairWise' or estimation=='pairwise' or estimation=='PW' or estimation=='pw':
        filePW = 'MatrixPW_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm_'  + str(int(cut)) + 'cutsvd'
        if os.path.exists(intermatrix_dir + filePW + '.fits') == True:
            vectoressai=fits.getdata(intermatrix_dir + filePW + '.fits')
        else:
            print('Please create PW matrix before correction')
            sys.exit()
        
        
        
        
    fileWhichInPup = 'Whichactfor' + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + fileWhichInPup + '.fits') == True:
        WhichInPupil = fits.getdata(intermatrix_dir + fileWhichInPup + '.fits')
    else:
        print('Please create Whichactfor matrix before correction')
        sys.exit()
        
    
    
    
    fileDirectMatrix = 'DirectMatrix_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_'
    if os.path.exists(intermatrix_dir + fileDirectMatrix + '.fits') == True:
        Gmatrix = fits.getdata(intermatrix_dir + fileDirectMatrix + '.fits')
    else:
        print('Please create Direct matrix before correction')
        sys.exit()
        
    
    
    
    fileMaskDH = 'MaskDH_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, choosepix))
    if os.path.exists(intermatrix_dir + fileMaskDH + '.fits') == True:
        maskDH = fits.getdata(intermatrix_dir+fileMaskDH+'.fits')
    else:
        print('Please create MaskDH matrix before correction')
        sys.exit()
        
        
        
    if (correction_algorithm == 'EM' or correction_algorithm == 'steepest'):
        G=np.zeros((int(np.sum(maskDH)),len(WhichInPupil)),dtype=complex)
        G=Gmatrix[0:int(Gmatrix.shape[0]/2),:]+1j*Gmatrix[int(Gmatrix.shape[0]/2):,:]
        transposecomplexG=np.transpose(np.conjugate(G))
        M0=np.real(np.dot(transposecomplexG,G))
        
        
    ##Load aberration maps (A refaire proprement!!!)
    if set_phase_abb==True:
        if set_random_phase==True:
            phase=instr.random_phase_map(isz,phaserms,rhoc_phase,slope_phase)
        else:    
            phase=fits.getdata(model_dir+phase_abb+'.fits')
        
        phase=phase*2*np.pi/wavelength
    else:
        phase=0

    if set_amplitude_abb==True:
        oui=fits.getdata(model_dir+amplitude_abb+'.fits')#*roundpupil(isz,prad)
        moy=np.mean(oui[np.where(oui!=0)])
        amp=oui/moy
        amp1=cv2.resize(amp, dsize=(int(2*prad/148*400),int(2*prad/148*400)),interpolation=cv2.INTER_AREA)
        ampfinal=np.zeros((isz,isz))
        ampfinal[int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1,int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1]=amp1
        ampfinal=(ampfinal)*instr.roundpupil(isz,prad-1)
        moy=np.mean(ampfinal[np.where(ampfinal!=0)])
        ampfinal=(ampfinal/moy-np.ones((isz,isz)))*instr.roundpupil(isz,prad-1)#/10
    else:
        ampfinal=0
    
    amplitude_abb=ampfinal
    phase_abb=phase

    
    ## SIMU
    contrast_to_photons=np.sum(entrancepupil)/np.sum(lyot)*nb_photons*squaremaxPSF**2/np.sum(PSF)**2
    
    if error==0:
        pushactonDM=pushact
    else:
        print('Misregistration!')
        file309=0
        pushactonDM=instr.creatingpushact(model_dir,file309,x309,y309,findxy309byhand,isz,prad,pdiam,modelerror=errormodel,error=error)
    
    nbiter=len(modevector)
    imagedetector=np.zeros((nbiter+1,isz,isz))
    phaseDM=np.zeros((nbiter+1,isz,isz))
    meancontrast=np.zeros(nbiter+1)
    maskDHisz=wsc.creatingMaskDH(isz,'square',choosepixDH=[element*isz/dimimages for element in choosepix])
    input_wavefront=entrancepupil*(1+amplitude_abb)*np.exp(1j*phase_abb)
    imagedetector[0]=abs(instr.pupiltodetector(input_wavefront,coro,lyot,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF)**2
    meancontrast[0]=np.mean(imagedetector[0][np.where(maskDHisz!=0)])
    print('Mean contrast in DH: ', meancontrast[0])
    if photon_noise==True:
        photondetector=np.zeros((nbiter+1,isz,isz))
        photondetector[0]=np.random.poisson(imagedetector[0]*contrast_to_photons)

    plt.ion()
    plt.figure()
    previousmode=0
    k=0
    for mode in modevector:
        print('--------------------------------------------------')
        print('Iteration number: ', k , ' EFC truncation: ' , mode )
        if estimation=='PairWise' or estimation=='pairwise' or estimation=='PW' or estimation=='pw':
            Difference=instr.createdifference(amplitude_abb,phase_abb,posprobes,pushactonDM,amplitudePW,entrancepupil,coro,lyot,PSF,dimimages,wavelength,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil,noise=photon_noise,numphot=nb_photons)
            resultatestimation=wsc.FP_PWestimate(Difference,vectoressai)
            
        elif estimation=='Perfect':
            resultatestimation=instr.pupiltodetector(input_wavefront,coro,lyot,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF
            resultatestimation=proc.resampling(resultatestimation,dimimages)
            
        else:
            print('This estimation algorithm is not yet implemented')
            sys.exit()
            
        if correction_algorithm == 'EFC':
            
            if mode!=previousmode:
                invertGDH=wsc.invertSVD(Gmatrix,mode,goal='c',visu=False,regul=regularization,otherbasis=otherbasis,basisDM3=basisDM3,intermatrix_dir=intermatrix_dir)[2]
            
            solution1=wsc.solutionEFC(maskDH,resultatestimation,invertGDH,WhichInPupil)
            
            
        if correction_algorithm == 'EM':
        
            if mode!=previousmode:
                invertM0=wsc.invertSVD(M0,mode,goal='c',visu=False,regul=regularization,otherbasis=otherbasis,basisDM3=basisDM3,intermatrix_dir=intermatrix_dir)[2]
            
            solution1=wsc.solutionEM(maskDH,resultatestimation,invertM0,G,WhichInPupil)
        
        
        if correction_algorithm == 'steepest':
            solution1=wsc.solutionSteepest(maskDH,resultatestimation,M0,G,WhichInPupil)
            
            

        apply_on_DM=-gain*amplitudeEFC*np.dot(solution1,pushactonDM.reshape(1024,isz*isz)).reshape(isz,isz)*2*np.pi*1e-9/wavelength
        phaseDM[k+1]=phaseDM[k]+apply_on_DM
        phase_abb=phase_abb+apply_on_DM
        input_wavefront=entrancepupil*(1+amplitude_abb)*np.exp(1j*phase_abb)
        imagedetector[k+1]=abs(instr.pupiltodetector(input_wavefront,coro,lyot,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF)**2
        meancontrast[k+1]=np.mean(imagedetector[k+1][np.where(maskDHisz!=0)])
        print('Mean contrast in DH: ',meancontrast[k+1])
        if photon_noise==True:
            photondetector[k+1]=np.random.poisson(imagedetector[k+1]*contrast_to_photons)

            
        plt.clf()
        plt.imshow(np.log10(imagedetector[k+1]),vmin=-8,vmax=-5)
        plt.colorbar()
        plt.pause(0.01)
        previousmode=mode
        k=k+1
        
    plt.show()


    ## SAVING...
    header=fi.from_param_to_header(config)
    cut_phaseDM=np.zeros((nbiter+1,2*prad,2*prad))
    for it in np.arange(nbiter+1):
        cut_phaseDM[it]=proc.cropimage(phaseDM[it],200,200,2*prad)
        # plt.clf()
        # plt.figure(figsize=(3, 3))
        # plt.imshow(np.log10(imagedetector[it,100:300,100:300]),vmin=-8,vmax=-5,cmap='Blues_r')#CMRmap
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig(result_dir+'image-'+str(2*it+1)+'.jpeg')
        # plt.close()

    current_time_str = datetime.datetime.today().strftime('%Y%m%d_%Hh%Mm%Ss')
    fits.writeto(result_dir + 'Detector_Images_' +current_time_str+ '.fits', imagedetector, header, overwrite=True)
    fits.writeto(result_dir + 'Phase_on_DM2_' +current_time_str+ '.fits', cut_phaseDM, header, overwrite=True)
    fits.writeto(result_dir + 'Mean_Contrast_DH_' +current_time_str+ '.fits', meancontrast, header, overwrite=True)
    config.filename = result_dir + 'Simulation_parameters_' +current_time_str+ '.ini'
    config.write()
    

    if photon_noise==True:
        fits.writeto(result_dir + 'Photon_counting_' +current_time_str+ '.fits', photondetector, header, overwrite=True)
    
    plt.clf()    
    plt.plot(meancontrast)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean contrast in Dark Hole')




    return phase_abb,imagedetector

