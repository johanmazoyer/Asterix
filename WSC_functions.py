#Version 29 Janvier 2020
from shortcuts import *
import InstrumentSimu_functions as instr
import matplotlib.pyplot as plt
import processing_functions as proc


def invertSVD(matrix_to_invert,cut,goal='e',regul='truncation',visu=True,otherbasis=False,basisDM3=0,intermatrix_dir=''):
    ''' --------------------------------------------------
    Invert a matrix after a Singular Value Decomposition. The inversion can be regularized.
    
    Parameters:
    ----------
    matrix_to_invert: 
    cut: 
    goal: string, can be 'e' or 'c'
          if 'e': the cut set the inverse singular value not to exceed
          if 'c': the cut set the number of modes to take into account (keep the lowest inverse singular values)
    regul: string, can be 'truncation' or 'tikhonov'
          if 'truncation': when goal is set to 'c', the modes with the highest inverse singular values are truncated
          if 'tikhonov': when goal is set to 'c', the modes with the highest inverse singular values are smooth (low pass filter)
    visu: boolean, if True, plot and save the crescent inverse singular values , before regularization
    otherbasis: boolean, 
    basisDM3: goes with other basis
    
    Return:
    ------
    np.diag(InvS): Inverse eigenvalues of the input matrix
    np.diag(InvS_truncated): Inverse eigenvalues of the input matrix after regularization
    pseudoinverse: Regularized inverse of the input matrix
    -------------------------------------------------- '''
    U, s, V = np.linalg.svd(matrix_to_invert, full_matrices=False)
    #print(s)
    S = np.diag(s)
    InvS=np.linalg.inv(S)
    InvS_truncated=np.linalg.inv(S)
    #print(InvS)
    if(visu==True):
        plt.plot(np.diag(InvS),'r.')
        plt.yscale('log')
    #     plt.savefig(intermatrix_dir+'invertSVDEFC_'+ '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_.png')  
    
    if(goal=='e'):
        InvS_truncated[np.where(InvS_truncated>cut)]=0
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS_truncated),np.transpose(U))
      
    if(goal=='c'):
        if regul=='truncation':
            InvS_truncated[cut:]=0
        if regul=='tikhonov':
            InvS_truncated=np.diag(s/(s**2+s[cut]**2))
            if(visu==True):
                plt.plot(np.diag(InvS_truncated),'b.')
                plt.yscale('log')
                #plt.show()
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS_truncated),np.transpose(U))
    
    if (otherbasis==True):
        pseudoinverse=np.dot(np.transpose(basisDM3),pseudoinverse)
        
    return [np.diag(InvS),np.diag(InvS_truncated),pseudoinverse]

    
    

def createvectorprobes(wavelength,entrancepupil,coro_mask,lyot_mask,amplitude,posprobes,pushact,dimimages,cutsvd):
    ''' --------------------------------------------------
    Build the interaction matrix for pair-wise probing.
    
    Parameters:
    ----------
    wavelength: float, wavelength of the  incoming flux in meter
    entrancepupil: 2D-array, entrance pupil shape
    coro_mask: 2D array, can be complex. coronagraphic mask
    lyot_mask: 2D array, lyot mask
    amplitude: float, amplitude of the actuator pokes for pair(wise probing in nm
    posprobes: 1D-array, index of the actuators to push and pull for pair-wise probing
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    dimimages: int, size of the output image after resampling in pixels
    cutsvd: float, value not to exceed for the inverse eigeinvalues at each pixels
    
    
    Return:
    ------
    PWVector: 2D array, vector probe to be multiplied by the image difference matrix in order to retrieve the focal plane electric field
    SVD: 2D array, map of the inverse singular values for each pixels and before regularization
    -------------------------------------------------- '''
    isz=len(pushact[0])
    numprobe=len(posprobes)
    deltapsik=np.zeros((numprobe,dimimages,dimimages),dtype=complex)
    probephase=np.zeros((numprobe,isz,isz))
    matrix=np.zeros((numprobe,2))
    PWVector=np.zeros((dimimages**2,2,numprobe))
    SVD=np.zeros((2,dimimages,dimimages))
    squaremaxPSF=np.amax(np.abs(instr.pupiltodetector(entrancepupil,1,lyot_mask)))
    k=0
    for i in posprobes:
        probephase[k]=amplitude*pushact[i]
        probephase[k]=2*np.pi*(probephase[k])*1e-9/wavelength
        inputwavefront=entrancepupil*(1+1j*probephase[k])
        deltapsikbis=instr.pupiltodetector(inputwavefront,coro_mask,lyot_mask,perfect_coro=True,perfect_entrance_pupil=entrancepupil)/squaremaxPSF
        deltapsik[k]=proc.resampling(deltapsikbis,dimimages)
        k=k+1

    l=0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            matrix[:,0]=real(deltapsik[:,i,j])
            matrix[:,1]=im(deltapsik[:,i,j])
    
            try:
                inversion=invertSVD(matrix,cutsvd,visu=False)
                SVD[:,i,j]=inversion[0]
                PWVector[l]=inversion[2]
            except:
                print('Careful: Error! for l='+str(l))
                SVD[:,i,j]=np.zeros(2)
                PWVector[l]=np.zeros((2,numprobe))
            l=l+1  
    return [PWVector,SVD]
   
    
    
def creatingWhichinPupil(pushact,entrancepupil,cutinpupil):
    ''' --------------------------------------------------
    Create a vector with the index of all the actuators located in the entrance pupil
    
    Parameters:
    ----------
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    entrancepupil: 2D-array, entrance pupil shape
    cutinpupil: float, minimum surface of an actuator inside the pupil to be taken into account (between 0 and 1, in ratio of an actuator perfectly centered in the entrance pupil)
    
    Return:
    ------
    WhichInPupil: 1D array, index of all the actuators located inside the pupil
    -------------------------------------------------- '''
    WhichInPupil = []
    for i in np.arange(int(1024)):
        Psivector=pushact[i]
        cut=cutinpupil*np.sum(Psivector)
        if(np.sum(Psivector*entrancepupil)>cut):
            WhichInPupil.append(i)
    
    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil
    


def creatingMaskDH(dimimages,shape,choosepixDH=[0,0,0,0], inner=0, outer=0, xdecay=0):
    ''' --------------------------------------------------
    Create a binary mask.
    
    Parameters:
    ----------
    dimimages: int, size of the output squared mask
    shape: string, can be 'square' or 'circle' , define the shape of the binary mask.
    choosepixDH: 1D array, if shape is 'square', define the edges of the binary mask in pixels.
    inner: float, if shape is 'circle', define the inner edge of the binary mask
    outer: float, if shape is 'circle', define the outer edge of the binary mask
    xdecay: float, if shape is 'circle', can define to keep only one side of the circle
    
    Return:
    ------
    maskDH: 2D array, binary mask
    -------------------------------------------------- '''
    xx, yy = np.meshgrid(np.arange(dimimages)-(dimimages)/2, np.arange(dimimages)-(dimimages)/2)
    rr     = np.hypot(yy, xx)
    if shape=='square':
        maskDH=np.ones((dimimages,dimimages))
        maskDH[xx<choosepixDH[0]]=0
        maskDH[xx>choosepixDH[1]]=0
        maskDH[yy<choosepixDH[2]]=0
        maskDH[yy>choosepixDH[3]]=0
    if shape=='circle':
        maskDH=np.ones((dimimages,dimimages))
        maskDH[rr>=outer]=0
        maskDH[rr<inner]=0
        maskDH[xx<xdecay]=0
    return maskDH



def creatingCorrectionmatrix(entrancepupil,coro_mask,lyot_mask,dimimages,wavelength,amplitude,pushact,mask,Whichact,otherbasis=False,basisDM3=0):
    ''' --------------------------------------------------
    Create the jacobian matrix for Electric Field Conjugation
    
    Parameters:
    ----------
    entrancepupil: 2D-array, entrance pupil shape
    coro_mask: 2D array, can be complex. coronagraphic mask
    lyot_mask: 2D array, lyot mask
    dimimages: int, size of the output image after resampling in pixels
    wavelength: float, wavelength of the  incoming flux in meter
    amplitude: float, amplitude of the actuator pokes for pair(wise probing in nm
    pushact: 3D-array, opd created by the pokes of all actuators in the DM
    mask: 2D array, binary mask whose pixel=1 will be taken into account
    Whichact: 1D array, index of the actuators taken into account to create the jacobian matrix
    otherbasis:
    basisDM3:
    
    Return:
    ------
    Gmatrixbis: 2D array, jacobian matrix for Electric Field Conjugation
    -------------------------------------------------- '''
    #change basis if needed
    if (otherbasis == True):
        nb_fct=basisDM3.shape[0]#number of functions in the basis
        tmp=pushact.reshape(pushact.shape[0],pushact.shape[1]*pushact.shape[2])
        bas_fct=np.dot(basisDM3,tmp).reshape(nb_fct,pushact.shape[1],pushact.shape[2])
    else:
        bas_fct = np.array([pushact[ind] for ind in Whichact])
        nb_fct=len(Whichact)
    squaremaxPSF=np.amax(np.abs(instr.pupiltodetector(entrancepupil,1,lyot_mask)))
    print('Start EFC')
    Gmatrixbis=np.zeros((2*int(np.sum(mask)),nb_fct))
    k=0
    for i in range(nb_fct):
        if i%100 == 0:
            print(i)
        Psivector=amplitude*bas_fct[i]
        Psivector=2*np.pi*(Psivector)*1e-9/wavelength
        inputwavefront=entrancepupil*(1+1j*Psivector)
        Gvector=instr.pupiltodetector(inputwavefront,coro_mask,lyot_mask,perfect_coro=True,perfect_entrance_pupil=entrancepupil)/squaremaxPSF
        Gvector=proc.resampling(Gvector,dimimages)
        Gmatrixbis[0:int(np.sum(mask)),k]=real(Gvector[np.where(mask==1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,k]=im(Gvector[np.where(mask==1)]).flatten()
        k=k+1
    print('End EFC')
    return Gmatrixbis
    


def solutionEFC(mask,Result_Estimate,inversed_jacobian,WhichInPupil):
    ''' --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle intensity in the dark hole region
    
    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    inversed_jacobian: 2D array, inverse of the jacobian matrix created with all the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account to create the jacobian matrix
    
    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- '''
    Eab=np.zeros(2*int(np.sum(mask)))
    Resultatbis=(Result_Estimate[np.where(mask==1)])
    Eab[0:int(np.sum(mask))]=real(Resultatbis).flatten()     
    Eab[int(np.sum(mask)):]=im(Resultatbis).flatten()
    cool=np.dot(inversed_jacobian,Eab)
        
    
    solution=np.zeros(1024)
    solution[WhichInPupil]=cool
    return solution
    
    
    
    
def solutionEM(mask,Result_Estimate,Hessian_Matrix,Jacobian,WhichInPupil):
    ''' --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle intensity in the dark hole region
    
    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    Hessian_Matrix: 2D array , Hessian matrix of the DH energy
    Jacobian: 2D array, inverse of the jacobian matrix created with all the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account to create the jacobian matrix
    
    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- '''
    
    
    Eab=np.zeros(int(np.sum(mask)))
    Resultatbis=(Result_Estimate[np.where(mask==1)])
    Eab=np.real(np.dot(np.transpose(np.conjugate(Jacobian)),Resultatbis)).flatten()
    cool=np.dot(Hessian_Matrix,Eab)
        
    
    solution=np.zeros(1024)
    solution[WhichInPupil]=cool
    return solution
    
    
    
    
def solutionSteepest(mask,Result_Estimate,Hessian_Matrix,Jacobian,WhichInPupil):
    ''' --------------------------------------------------
    Voltage to apply on the deformable mirror in order to minimize the speckle intensity in the dark hole region
    
    Parameters:
    ----------
    mask: Binary mask corresponding to the dark hole region
    Result_Estimate: 2D array can be complex, focal plane electric field
    Hessian_Matrix: 2D array , Hessian matrix of the DH energy
    Jacobian: 2D array, inverse of the jacobian matrix created with all the actuators in WhichInPupil
    WhichInPupil: 1D array, index of the actuators taken into account to create the jacobian matrix
    
    Return:
    ------
    solution: 1D array, voltage to apply on each deformable mirror actuator
    -------------------------------------------------- '''
    
    Eab=np.zeros(int(np.sum(mask)))
    Resultatbis=(Result_Estimate[np.where(mask==1)])
    Eab=np.real(np.dot(np.transpose(np.conjugate(Jacobian)),Resultatbis)).flatten()
    pas=2e3
    #cool=2*(np.dot(M0,sol)+np.real(np.dot(np.transpose(np.conjugate(G)),Resultatbis))).flatten()
    cool=pas*2*Eab
        
    
    solution=np.zeros(1024)
    solution[WhichInPupil]=cool
    return solution


    
    
    
def FP_PWestimate(Difference,Vectorprobes):
    ''' --------------------------------------------------
    Calculate the focal plane electric field from the prone image differences and the modeled probe matrix
    
    Parameters:
    ----------
    Difference: 3D array, cube with image difference for each probes
    Vectorprobes: 2D array, model probe matrix for the same probe as for difference
    
    Return:
    ------
    Difference: 3D array, cube with image difference for each probes. Use for pair-wise probing
    -------------------------------------------------- '''
    dimimages=len(Difference[0])
    numprobe=len(Vectorprobes[0,0])
    Differenceij=np.zeros((numprobe))
    Resultat=np.zeros((dimimages,dimimages),dtype=complex)
    l=0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:]=Difference[:,i,j]
            Resultatbis=np.dot(Vectorprobes[l],Differenceij)
            Resultat[i,j]=Resultatbis[0]+1j*Resultatbis[1]
            
            l=l+1  
    return Resultat/4