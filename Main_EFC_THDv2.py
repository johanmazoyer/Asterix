#Version 18/12/2019 A. Potier
#Correction 26/12/2019
#Creates interaction matrices for PW+EFC on the THD2

from Function_EFC_THD_v3 import *
    #from Function_EFC_THD import *



####Calculating and Recording PW matrix
filePW = 'MatrixPW_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm_'  + str(int(cut)) + 'cutsvd'
if os.path.exists(intermatrix_dir + filePW + '.fits') == True:
    print('The matrix ' + filePW + ' already exist')
    vectoressai=LoadImageFits(intermatrix_dir + filePW + '.fits')
else:
    print('Recording ' + filePW + ' ...')
    vectoressai,showsvd = createvectorprobes(amplitudePW,posprobes,cut)
    SaveFits(vectoressai , [' ',0] , intermatrix_dir , filePW)

    visuPWMap='MapEigenvaluesPW'+ '_' + '_'.join(map(str, posprobes)) + 'act_' + str(int(amplitudePW)) + 'nm'
    if os.path.exists(intermatrix_dir + visuPWMap + '.fits' ) == False:
        print('Recording ' + visuPWMap + ' ...')
        vectoressai,showsvd = createvectorprobes(amplitudePW,posprobes,1e100)
        SaveFits(showsvd[1] , [' ',0] , intermatrix_dir , visuPWMap)
            
#Saving PW matrices in Labview directory
probes=np.zeros((len(posprobes),1024),dtype=np.float32)
vectorPW=np.zeros((2,dimimages*dimimages*len(posprobes)),dtype=np.float32)

for i in np.arange(len(posprobes)):
    probes[i,posprobes[i]]=amplitudePW/17
    vectorPW[0,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,0,i].flatten()
    vectorPW[1,i*dimimages*dimimages:(i+1)*dimimages*dimimages]=vectoressai[:,1,i].flatten()
SaveFits(probes, [' ',0] , Labview_dir , 'Probes_EFC_default' , replace= True )
SaveFits(vectorPW, [' ',0] , Labview_dir , 'Matr_mult_estim_PW' , replace= True )
        


####Calculating and Recording EFC matrix
    
#Creating WhichInPup?
fileWhichInPup = 'Whichactfor' + str(MinimumSurfaceRatioInThePupil)
if os.path.exists(intermatrix_dir + fileWhichInPup + '.fits') == True:
    print('The matrix ' + fileWhichInPup + ' already exist')
    WhichInPupil = LoadImageFits(intermatrix_dir + fileWhichInPup + '.fits')
else:
    print('Recording' + fileWhichInPup + ' ...')
    if otherbasis==False:
        WhichInPupil = creatingWhichinPupil(MinimumSurfaceRatioInThePupil)
    else:
        WhichInPupil = np.arange(1024)
    SaveFits(WhichInPupil,[' ',0],intermatrix_dir,fileWhichInPup)
    
    
#Creating EFC matrix?
fileEFCMatrix = 'MatrixEFC_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_' + str(Nbmodes) + 'modes'
            
if os.path.exists(intermatrix_dir + fileEFCMatrix + '.fits') == True:
    print('The matrix ' + fileEFCMatrix + ' already exist')
    invertGDH = LoadImageFits(intermatrix_dir + fileEFCMatrix + '.fits')
else:
    
   #Actuator basis or another one?
    if otherbasis == True:
        basisDM3 = LoadImageFits(Labview_dir+'Map_modes_DM3_foc.fits')
    else:
        basisDM3=0
    
    #Creating Direct matrix?
    fileDirectMatrix = 'DirectMatrix_' + '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_'
    if os.path.exists(intermatrix_dir + fileDirectMatrix + '.fits') == True:
        print('The matrix ' + fileDirectMatrix + ' already exist')
        Gmatrix = LoadImageFits(intermatrix_dir + fileDirectMatrix + '.fits')
    else:
        
        
        #Creating MaskDH?
        fileMaskDH = 'MaskDH_' + str(dimimages) + 'x' + str(dimimages) + '_' + '_'.join(map(str, choosepix))
        if os.path.exists(intermatrix_dir + fileMaskDH + '.fits') == True:
            print('The matrix ' + fileMaskDH + ' already exist')
            maskDH = LoadImageFits(intermatrix_dir+fileMaskDH+'.fits')
        else:
            print('Recording ' + fileMaskDH + ' ...')
            maskDH = creatingMaskDH(choosepix)
            SaveFits(maskDH,[' ',0],intermatrix_dir,fileMaskDH)
        
        #Creating Direct Matrix if does not exist
        print('Recording ' + fileDirectMatrix + ' ...')
        Gmatrix = creatingCorrectionmatrix(amplitudeEFC,maskDH,WhichInPupil,otherbasis=otherbasis,basisDM3=basisDM3)
        SaveFits(Gmatrix,[' ',0],intermatrix_dir,fileDirectMatrix)
        
    
    
    #Recording EFC Matrix
    print('Recording ' + fileEFCMatrix + ' ...')
    invertGDH = invertDSCC(Gmatrix,Nbmodes,goal='c',regul='tikhonov',otherbasis=otherbasis,basisDM3=basisDM3)[1]
    SaveFits(invertGDH,[' ',0],intermatrix_dir,fileEFCMatrix)



#Save EFC matrix in Labview directory
EFCmatrix = np.zeros((invertGDH.shape[1],1024),dtype=np.float32)
for i in np.arange(len(WhichInPupil)):
    EFCmatrix[:,WhichInPupil[i]]=invertGDH[i,:]
SaveFits(EFCmatrix, [' ',0] , Labview_dir , 'Matrix_control_EFC_DM3_default' , replace= True )



