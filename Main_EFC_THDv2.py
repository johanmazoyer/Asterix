#Version 18/12/2019 A. Potier
#Correction 26/12/2019
#Correction 29/01/2019
#Creates interaction matrices for PW+EFC on the THD2

from shortcuts import *



import processing_functions as proc
import display_functions as disp
import fits_functions as fi
import WSC_functions as wsc
import InstrumentSimu_functions as instr



#from Parameters_EFC_THD import *

#from Function_EFC_THD_v3 import *


# from processing_functions import *
# from display_functions import *
# from fits_functions import *
# from WSC_functions import *
# from InstrumentSimu_functions import *


from configobj import ConfigObj
from validate import Validator



def create_interaction_matrices(parameter_file):
    
    
    
    
    ### CONFIGURATION FILE
    configspec_file   = 'Essai_param_configspec.ini'
    config            = ConfigObj(parameter_file,configspec=configspec_file)
    vtor              = Validator()
    checks            = config.validate(vtor,copy=True) # copy=True for copying the comments     
    
    ### CONFIG                         
    
    main_dir = config['main_dir']
    Labview_dir= config['Labview_dir']

    ### MODEL CONFIG
    modelconfig = config['modelconfig']
    isz   = modelconfig['isz']
    wavelength = modelconfig['wavelength']
    pdiam = modelconfig['pdiam']
    lyotdiam  = modelconfig['lyotdiam']
    set_ldp_manually = modelconfig['set_ldp_manually']
    ld_p= modelconfig['ld_p']
    coronagraph = modelconfig['coronagraph']
    coro_position = modelconfig['coro_position']
    new= modelconfig['new']
    dimimages= modelconfig['dimimages']
    obstructed_pupil= modelconfig['obstructed_pupil']
    creating_pushact= modelconfig['creating_pushact']
    findxy309byhand= modelconfig['findxy309byhand']
    y309= modelconfig['y309']
    x309= modelconfig['x309']


    ### PW CONFIG
    PWconfig=config['PWconfig']
    amplitudePW = PWconfig['amplitudePW']
    posprobes = PWconfig['posprobes']
    posprobes=[int(i) for i in posprobes]
    cut= PWconfig['cut']

    ###EFC CONFIG
    EFCconfig=config['EFCconfig']
    MinimumSurfaceRatioInThePupil = EFCconfig['MinimumSurfaceRatioInThePupil']
    choosepix = EFCconfig['choosepix']
    choosepix=[int(i) for i in choosepix]
    otherbasis= EFCconfig['otherbasis']
    Nbmodes = EFCconfig['Nbmodes']
    amplitudeEFC = EFCconfig['amplitudeEFC']
    regularization = EFCconfig['regularization']
    
    ##THEN DO

    model_dir = main_dir+'Model/'
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
    intermatrix_dir = main_dir + 'Interaction_Matrices/' + coronagraph+ '/'+ str(int(wavelength*1e9)) +'nm/p' + str(round(pdiam*1e3 , 2)) + '_l' + str(round(lyotdiam*1e3,1)) + '/ldp_'+str(round(ld_p/lyot,2))+'/basis_' + basistr+'/'
    
    if not os.path.exists(intermatrix_dir):
        print('Creating directory ' + intermatrix_dir + ' ...')
        os.makedirs(intermatrix_dir)
    
    reechpup = isz
    
    if creating_pushact==True:
        pushact=instr.creatingpushact(model_dir,file309,x309,y309)
    
        fi.SaveFits(pushact,['',0],model_dir,'PushActInPup400')
    else:
        if os.path.exists(model_dir+'PushActInPup400.fits') == False:
            print('Extracting data from zip file...')
            ZipFile(model_dir+'PushActInPup400.zip', 'r').extractall(model_dir)
            
        pushact = fi.LoadImageFits(model_dir+'PushActInPup400.fits')
    
    print(main_dir)
    ## transmission of the phase mask (exp(i*phase))
    ## centered on pixel [0.5,0.5]
    if coronagraph =='fqpm':
        coro = fi.LoadImageFits(model_dir+'FQPM.fits')
        perfect_coro=True
    elif coronagraph=='knife':
        coro = instr.KnifeEdgeCoro(isz,coro_position,1.2,ld_p)
        perfect_coro=False
    elif coronagraph =='vortex':
        phasevortex=0# to be defined
        coro = exp(1j*phasevortex)
        perfect_coro=True
        
    if obstructed_pupil==True:
        #Does not exist yet!!!
        entrancepupil=fi.LoadImageFits(model_dir+'instrument_pupil.fits')
        lyot=fi.LoadImageFits(model_dir+'lyot_pupil.fits')
        
    else:
        entrancepupil=instr.roundpupil(isz,prad)
        lyot = instr.roundpupil(isz,lyotrad)
    
    perfect_entrance_pupil=entrancepupil
    
    ####Calculating and Recording PW matrix
    filePW = 'MatrixPW_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm_'  + str(int(cut)) + 'cutsvd'
    if os.path.exists(intermatrix_dir + filePW + '.fits') == True:
        print('The matrix ' + filePW + ' already exist')
        vectoressai=fi.LoadImageFits(intermatrix_dir + filePW + '.fits')
    else:
        print('Recording ' + filePW + ' ...')
        vectoressai,showsvd = wsc.createvectorprobes(wavelength,entrancepupil,coro,lyot,amplitudePW,posprobes,pushact,reechpup,new,dimimages,cut)
        fi.SaveFits(vectoressai , [' ',0] , intermatrix_dir , filePW)
    
        visuPWMap='MapEigenvaluesPW'+ '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm'
        if os.path.exists(intermatrix_dir + visuPWMap + '.fits' ) == False:
            print('Recording ' + visuPWMap + ' ...')
            #vectoressai,showsvd = createvectorprobes(wavelength,entrancepupil,coro,lyot,amplitudePW,posprobes,pushact,reechpup,new,dimimages,1e100)
            fi.SaveFits(showsvd[1] , [' ',0] , intermatrix_dir , visuPWMap)
                
    #Saving PW matrices in Labview directory
    probes=np.zeros((len(posprobes),1024),dtype=np.float32)
    vectorPW=np.zeros((2,dimimages*dimimages*len(posprobes)),dtype=np.float32)
    
    for i in np.arange(len(posprobes)):
        probes[i,posprobes[i]]=amplitudePW/17
        vectorPW[0,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,0,i].flatten()
        vectorPW[1,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,1,i].flatten()
    fi.SaveFits(probes, [' ',0] , Labview_dir , 'Probes_EFC_default' , replace= True )
    fi.SaveFits(vectorPW, [' ',0] , Labview_dir , 'Matr_mult_estim_PW' , replace= True )
            
    
    
    ####Calculating and Recording EFC matrix
    print('TO SET ON LABVIEW: ',str(150/2+np.array(shift(choosepix))))    
    #Creating WhichInPup?
    fileWhichInPup = 'Whichactfor' + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + fileWhichInPup + '.fits') == True:
        print('The matrix ' + fileWhichInPup + ' already exist')
        WhichInPupil = fi.LoadImageFits(intermatrix_dir + fileWhichInPup + '.fits')
    else:
        print('Recording' + fileWhichInPup + ' ...')
        if otherbasis==False:
            WhichInPupil = wsc.creatingWhichinPupil(pushact,entrancepupil,MinimumSurfaceRatioInThePupil)
        else:
            WhichInPupil = np.arange(1024)
        fi.SaveFits(WhichInPupil,[' ',0],intermatrix_dir,fileWhichInPup)
        
        
    #Creating EFC matrix?
    fileEFCMatrix = 'MatrixEFC_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_' + str(Nbmodes) + 'modes'
                
    if os.path.exists(intermatrix_dir + fileEFCMatrix + '.fits') == True:
        print('The matrix ' + fileEFCMatrix + ' already exist')
        invertGDH = fi.LoadImageFits(intermatrix_dir + fileEFCMatrix + '.fits')
    else:
        
    #Actuator basis or another one?
        if otherbasis == True:
            basisDM3 = fi.LoadImageFits(Labview_dir+'Map_modes_DM3_foc.fits')
        else:
            basisDM3=0
        
        #Creating Direct matrix?
        fileDirectMatrix = 'DirectMatrix_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_'
        if os.path.exists(intermatrix_dir + fileDirectMatrix + '.fits') == True:
            print('The matrix ' + fileDirectMatrix + ' already exist')
            Gmatrix = fi.LoadImageFits(intermatrix_dir + fileDirectMatrix + '.fits')
        else:
            
            
            #Creating MaskDH?
            fileMaskDH = 'MaskDH_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, choosepix))
            if os.path.exists(intermatrix_dir + fileMaskDH + '.fits') == True:
                print('The matrix ' + fileMaskDH + ' already exist')
                maskDH = fi.LoadImageFits(intermatrix_dir+fileMaskDH+'.fits')
            else:
                print('Recording ' + fileMaskDH + ' ...')
                maskDH = wsc.creatingMaskDH(dimimages,'square',choosepix)
                fi.SaveFits(maskDH,[' ',0],intermatrix_dir,fileMaskDH)
            
            #Creating Direct Matrix if does not exist
            print('Recording ' + fileDirectMatrix + ' ...')
            Gmatrix = wsc.creatingCorrectionmatrix(entrancepupil,coro,lyot,reechpup,new,wavelength,amplitudeEFC,pushact,maskDH,WhichInPupil,otherbasis=otherbasis,basisDM3=basisDM3)
            fi.SaveFits(Gmatrix,[' ',0],intermatrix_dir,fileDirectMatrix)
            
        
        
        #Recording EFC Matrix
        print('Recording ' + fileEFCMatrix + ' ...')
        SVD,SVD_trunc,invertGDH = wsc.invertSVD(Gmatrix,Nbmodes,goal='c',regul=regularization,otherbasis=otherbasis,basisDM3=basisDM3,intermatrix_dir=intermatrix_dir,)
        fi.SaveFits(invertGDH,[' ',0],intermatrix_dir,fileEFCMatrix)
        plt.clf()
        plt.plot(SVD,'r.')
        plt.yscale('log')
        plt.savefig(intermatrix_dir+'invertSVDEFC_'+ '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_.png') 
    
    
    
    #Save EFC matrix in Labview directory
    EFCmatrix = np.zeros((invertGDH.shape[1],1024),dtype=np.float32)
    for i in np.arange(len(WhichInPupil)):
        EFCmatrix[:,WhichInPupil[i]]=invertGDH[i,:]
    fi.SaveFits(EFCmatrix, [' ',0] , Labview_dir , 'Matrix_control_EFC_DM3_default' , replace= True )

    return 0
    
    
    
    

def CorrectionLoop(parameter_file): 
    
    
    
    
    ### CONFIGURATION FILE
    configspec_file   = 'Essai_param_configspec.ini'
    config            = ConfigObj(parameter_file,configspec=configspec_file)
    vtor              = Validator()
    checks            = config.validate(vtor,copy=True) # copy=True for copying the comments     
    
    ### CONFIG                         
    
    main_dir = config['main_dir']
    Labview_dir= config['Labview_dir']

    ### MODEL CONFIG
    modelconfig = config['modelconfig']
    isz   = modelconfig['isz']
    wavelength = modelconfig['wavelength']
    pdiam = modelconfig['pdiam']
    lyotdiam  = modelconfig['lyotdiam']
    set_ldp_manually = modelconfig['set_ldp_manually']
    ld_p= modelconfig['ld_p']
    coronagraph = modelconfig['coronagraph']
    coro_position = modelconfig['coro_position']
    new= modelconfig['new']
    dimimages= modelconfig['dimimages']
    obstructed_pupil= modelconfig['obstructed_pupil']
    creating_pushact= modelconfig['creating_pushact']
    findxy309byhand= modelconfig['findxy309byhand']
    y309= modelconfig['y309']
    x309= modelconfig['x309']


    ### PW CONFIG
    PWconfig=config['PWconfig']
    amplitudePW = PWconfig['amplitudePW']
    posprobes = PWconfig['posprobes']
    posprobes=[int(i) for i in posprobes]
    cut= PWconfig['cut']

    ###EFC CONFIG
    EFCconfig=config['EFCconfig']
    MinimumSurfaceRatioInThePupil = EFCconfig['MinimumSurfaceRatioInThePupil']
    choosepix = EFCconfig['choosepix']
    choosepix=[int(i) for i in choosepix]
    otherbasis= EFCconfig['otherbasis']
    Nbmodes = EFCconfig['Nbmodes']
    amplitudeEFC = EFCconfig['amplitudeEFC']
    regularization = EFCconfig['regularization']
    
    ###SIMU CONFIG
    SIMUconfig = config['SIMUconfig']
    set_amplitude_abb=SIMUconfig['set_amplitude_abb']
    amplitude_abb = SIMUconfig['amplitude_abb']
    set_phase_abb=SIMUconfig['set_phase_abb']
    phase_abb = SIMUconfig['phase_abb']
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
    
    model_dir = main_dir+'Model/'
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
    intermatrix_dir = main_dir + 'Interaction_Matrices/' + coronagraph+ '/'+ str(int(wavelength*1e9)) +'nm/p' + str(round(pdiam*1e3 , 2)) + '_l' + str(round(lyotdiam*1e3,1)) + '/ldp_'+str(round(ld_p/lyot,2))+'/basis_' + basistr+'/'
    
        
    if otherbasis == True:
        basisDM3 = fi.LoadImageFits(Labview_dir+'Map_modes_DM3_foc.fits')
    else:
        basisDM3=0
    
    reechpup = isz
    
    if os.path.exists(model_dir+'PushActInPup400.fits') == False:
        print('Extracting data from zip file...')
        ZipFile(model_dir+'PushActInPup400.zip', 'r').extractall(model_dir)
        
    pushact = fi.LoadImageFits(model_dir+'PushActInPup400.fits')
    
    ## transmission of the phase mask (exp(i*phase))
    ## centered on pixel [0.5,0.5]
    if coronagraph =='fqpm':
        coro = fi.LoadImageFits(model_dir+'FQPM.fits')
        perfect_coro=True
    elif coronagraph=='knife':
        coro = instr.KnifeEdgeCoro(isz,coro_position,1.2,ld_p)
        perfect_coro=False
    elif coronagraph =='vortex':
        phasevortex=0# to be defined
        coro = exp(1j*phasevortex)
        perfect_coro=True
        
    if obstructed_pupil==True:
        #Does not exist yet!!!
        entrancepupil=fi.LoadImageFits(model_dir+'instrument_pupil.fits')
        lyot=fi.LoadImageFits(model_dir+'lyot_pupil.fits')
        
    else:
        entrancepupil=instr.roundpupil(isz,prad)
        lyot = instr.roundpupil(isz,lyotrad)
    
    perfect_entrance_pupil=entrancepupil
    squaremaxPSF=np.amax(np.abs(instr.pupiltodetector(entrancepupil,1,lyot,reechpup,new)))
    
    ##Load matrices
    if estimation=='PairWise':
        filePW = 'MatrixPW_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm_'  + str(int(cut)) + 'cutsvd'
        if os.path.exists(intermatrix_dir + filePW + '.fits') == True:
            vectoressai=fi.LoadImageFits(intermatrix_dir + filePW + '.fits')
        else:
            print('Please create PW matrix before correction')
            sys.exit()
        
        
        
        
    fileWhichInPup = 'Whichactfor' + str(MinimumSurfaceRatioInThePupil)
    if os.path.exists(intermatrix_dir + fileWhichInPup + '.fits') == True:
        WhichInPupil = fi.LoadImageFits(intermatrix_dir + fileWhichInPup + '.fits')
    else:
        print('Please create Whichactfor matrix before correction')
        sys.exit()
        
    
    
    
    fileDirectMatrix = 'DirectMatrix_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_'
    if os.path.exists(intermatrix_dir + fileDirectMatrix + '.fits') == True:
        Gmatrix = fi.LoadImageFits(intermatrix_dir + fileDirectMatrix + '.fits')
    else:
        print('Please create Direct matrix before correction')
        sys.exit()
        
    
    
    
    fileMaskDH = 'MaskDH_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, choosepix))
    if os.path.exists(intermatrix_dir + fileMaskDH + '.fits') == True:
        maskDH = fi.LoadImageFits(intermatrix_dir+fileMaskDH+'.fits')
    else:
        print('Please create MaskDH matrix before correction')
        sys.exit()
        
        
    ##Load aberration maps (A refaire proprement!!!)
    if set_phase_abb==True:
        phase=fi.LoadImageFits(model_dir+phase_abb+'.fits')
        phase=phase*2*np.pi/wavelength
    else:
        phase=0

    if set_amplitude_abb==True:
        oui=fi.LoadImageFits(model_dir+amplitude_abb+'.fits')#*roundpupil(isz,prad)
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
    
    
    if error==0:
        pushactonDM=pushact
    else:
        print('Misregistration!')
        pushactonDM=instr.creatingpushact(model_dir,file309,x309,y309,findxy309byhand,modelerror=errormodel)
    
    nbiter=len(modevector)
    imagedetector=np.zeros((nbiter+1,isz,isz))
    input_wavefront=entrancepupil*(1+amplitude_abb)*np.exp(1j*phase_abb)
    imagedetector[0]=abs(instr.pupiltodetector(input_wavefront,coro,lyot,reechpup,isz,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF)**2
    plt.ion()
    plt.figure()
    previousmode=0
    k=0
    for mode in modevector:
        print(k,mode)
        if estimation=='PairWise':
            Difference=instr.createdifference(amplitude_abb,phase_abb,posprobes,pushactonDM,amplitudeEFC,entrancepupil,coro,lyot,reechpup,new,wavelength,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil,noise=False)
            resultatestimation=wsc.FP_PWestimate(Difference,vectoressai)
            
        elif estimation=='Perfect':
            resultatestimation=instr.pupiltodetector(input_wavefront,coro,lyot,reechpup,new,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF
            
        else:
            print('This estimation algorithm is not yet implemented')
            sys.exit()
            
        if mode!=previousmode:
            invertGDH=wsc.invertSVD(Gmatrix,mode,goal='c',visu=False,regul=regularization,otherbasis=otherbasis,basisDM3=basisDM3,intermatrix_dir=intermatrix_dir)[2]
            
            
        solution1=wsc.solutiontocorrect(maskDH,resultatestimation,invertGDH,WhichInPupil)
        phase_abb=phase_abb-gain*amplitudeEFC*np.dot(solution1,pushactonDM.reshape(1024,isz*isz)).reshape(isz,isz)*2*np.pi*1e-9/wavelength
        input_wavefront=entrancepupil*(1+amplitude_abb)*np.exp(1j*phase_abb)
        imagedetector[k+1]=abs(instr.pupiltodetector(input_wavefront,coro,lyot,reechpup,isz,perfect_coro=perfect_coro,perfect_entrance_pupil=perfect_entrance_pupil)/squaremaxPSF)**2
        plt.clf()
        plt.imshow(np.log10(imagedetector[k+1]),vmin=-8,vmax=-5)
        plt.colorbar()
        plt.pause(0.01)
        previousmode=mode
        k=k+1
        
    plt.show(block=True)    
    fi.SaveFits(imagedetector,[' ',0],main_dir,'EssaiCorrection',replace=True)
        
    return phase_abb,imagedetector

