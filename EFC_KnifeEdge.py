#10 Janvier 2019
#Creation de la fonction ElectricFieldConjugation qui permet la visualisation directe des itérations.
    
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as spsignal
import scipy.optimize as opt
import scipy.ndimage as nd
import skimage.transform
from astropy.io import fits
import glob
import os
import cv2
#plt.rcParams['image.interpolation'] = 'nearest'
# qsdqsd

def custom_plot(pup, img):
    ''' --------------------------------------------------
    Plots two images next to each other.
    -------------------------------------------------- '''
    f1 = plt.figure(1, figsize=(10,5))
    f1.clf()
    ax1 = f1.add_subplot(121)
    ax1.imshow(pup, cmap="hot")
    ax2 = f1.add_subplot(122)
    ax2.imshow(img, cmap="hot")
    f1.tight_layout()
    

def four_plot(img1, img2, img3, img4):
    ''' --------------------------------------------------
    Plots four images next to each other.
    -------------------------------------------------- '''
    f1 = plt.figure(1, figsize=(22,11))
    f1.clf()
    ax1 = f1.add_subplot(141)
    ax1.imshow(img1, cmap="hot")
    ax2 = f1.add_subplot(142)
    ax2.imshow(img2, cmap="hot")
    ax3 = f1.add_subplot(143)
    ax3.imshow(img3, cmap="hot")
    ax4 = f1.add_subplot(144)
    ax4.imshow(img4, cmap="hot")
    f1.tight_layout()



def translationFFT(a,b):
    maskx=np.zeros((isz,isz))
    masky=np.zeros((isz,isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            maskx[i,j]=j*np.pi*2*a/isz-a*np.pi
            masky[i,j]=i*np.pi*2*b/isz-b*np.pi
    masktot=np.exp(-1j*maskx)*np.exp(-1j*masky)
    return masktot
    

    
def detectortopupil2(Resultat,prad1):
    phaseretrouvee=shift(shift(ifft((Resultat)))*roundpupil(78,prad1)*masktot2)
    phaseretrouvee=fft(phaseretrouvee)*maskcorono2
    phaseretrouvee=ifft((phaseretrouvee))/shift(masktot2)
    return phaseretrouvee



def translationFFT(isz,a,b):
    maskx=np.zeros((isz,isz))
    masky=np.zeros((isz,isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            maskx[i,j]=j*np.pi*2*a/isz-a*np.pi
            masky[i,j]=i*np.pi*2*b/isz-b*np.pi
    masktot=np.exp(-1j*maskx)*np.exp(-1j*masky)
    return masktot
    
def FQPM2():
    phase= np.zeros((dimGDH,dimGDH))
    for i in np.arange(dimGDH):
        for j in np.arange(dimGDH):
            if(i<dimGDH/2 and j<dimGDH/2): phase[i,j]=np.pi
            if(i>=dimGDH/2 and j>=dimGDH/2): phase[i,j]=np.pi
        
    FQPM=np.exp(1j*phase)
    return FQPM


def FQPM():
    phase= np.zeros((isz,isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            if(i<isz/2 and j<isz/2): phase[i,j]=np.pi
            if(i>=isz/2 and j>=isz/2): phase[i,j]=np.pi
        
    FQPM=np.exp(1j*phase)
    return FQPM
    
def KnifeEdgeCoro(position,shiftinldp):
    Knife=np.zeros((isz,isz))
    for i in np.arange(isz):
        if position=='left':
            if (i>isz/2+shiftinldp*ld_p): Knife[:,i]=1
        if position=='right':
            if (i<isz/2-shiftinldp*ld_p): Knife[:,i]=1
    return Knife
 

    
def roundpupil(nbpix,prad1):
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy, xx)
    pupilnormal=np.zeros((nbpix,nbpix))
    pupilnormal[rr<=prad1]=1.
    return pupilnormal


def pupilforcorono(prad1):
    
    xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
    #rr     = np.hypot(yy+1/2, xx+1/2)
    rr     = np.hypot(yy, xx)
    
    #pupil1
    pupil1=roundpupil(isz,prad1)
    #Focale1
    focal1=fft(shift(pupil1*masktot))


    #Pupil2:Intensité nulle dans le Lyot
    pupil2=focal1*maskcorono
    pupil2=ifft(pupil2)/shift(masktot)
    pupil2=shift(pupil2)
    pupil2[rr<=prad1]=0
    pupil2=shift(pupil2)

    #Focale 1 pour coronographe parfait
    pupilcomplex=fft(pupil2*shift(masktot))
    pupilcomplex=pupilcomplex*maskcorono

    #Pupille pour coronographe parfait
    pupil1bis=ifft(pupilcomplex)
    pupil1bis2=shift(pupil1bis)/masktot
    return pupil1bis2
    
    
def pupiltodetector1(aberrationamp,aberrationphase,prad1,prad2):
    #Focale1        
    #focal1end=shift(pupilforcorono(prad1)*(1+aberrationamp)*np.exp(1j*aberrationphase*roundpupil(isz,prad1))*masktot)   
    focal1end=shift(roundpupil(isz,prad1)*(1+aberrationamp)*np.exp(1j*aberrationphase*roundpupil(isz,prad1))*masktot)   
    
    focal1end=fft(focal1end)
    
    #Pupille2
    pupil2end=focal1end*maskcorono
    pupil2end=ifft(pupil2end)/shift(masktot)
    
    #Intensité en sortie de SCC
    focal2end=pupil2end*shift(roundpupil(isz,prad2))
    focal2end=fft(focal2end)
    return focal2end
    
    
    
def pupiltodetector2(aberrationphase,prad1,prad2):
    #Focale1        
    #focal1end=shift(pupilforcorono(prad1)*(1+1j*aberrationphase*roundpupil(isz,prad1))*masktot)
    focal1end=shift(roundpupil(isz,prad1)*(1+1j*aberrationphase*roundpupil(isz,prad1))*masktot)
    
    focal1end=fft(focal1end)
    
    #Pupille2
    pupil2end=focal1end*maskcorono
    pupil2end=ifft(pupil2end)/shift(masktot)
    
    #Intensité en sortie de SCC
    focal2end=pupil2end*shift(roundpupil(isz,prad2))
    focal2end=fft(focal2end)
    return focal2end
    


def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, theta,h):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+h
    return g
    

# def pushact(which,coeff,grilleact, actshapeinpupilresized):
#     coefforactshape=np.round(1.98*6*coeff)
#     xact=coeff*(grilleact[0,which]+(x309-grilleact[0,309]))
#     yact=coeff*(grilleact[1,which]+(y309-grilleact[1,309]))
#     Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-coefforactshape/2,xact-coefforactshape/2))
#     Psivector[np.where(Psivector<1e-4)]=0
#     return Psivector
    
    
def pushact_function(which,coeff,grilleact, actshapeinpupilresized,devx,changeactfunc=False):
    coefforactshape=np.round(1.98*6*coeff)
    xact=coeff*(grilleact[0,which]+(x309-grilleact[0,309]))
    yact=coeff*(grilleact[1,which]+(y309-grilleact[1,309]))
    Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-coefforactshape/2+np.sqrt(devx**2/2),xact-coefforactshape/2+np.sqrt(devx**2/2)))
    #Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-coefforactshape/2,xact-coefforactshape/2+devx))
    #Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-coefforactshape/2,xact-coefforactshape/2))
    Psivector[np.where(Psivector<1e-4)]=0
    #Psivector=rotate_frame(Psivector,devx, interpolation = 'nearest', cyx=None)
    if (changeactfunc==True):
        x, y = np.mgrid[0:isz, 0:isz]
        xy=(x,y)
        xo,yo=np.unravel_index(Psivector.argmax(), Psivector.shape)
        Psivector=twoD_Gaussian(xy, 1, devx,devx , xo,yo, 0,0)
    return Psivector
    
    
    
    
def rotate_frame(img, angle, interpolation = 'lanczos4', cyx=None):
    """
    Rotates the input frame by the given angle (in degrees), around the center
    of the frame by default, or around the given cxy coordinates.
    
    Parameters
    ----------
    img: 2D-array
        input frame
    angle: float
        angle for the rotation
    interpolation: string
        interpolation method. Can be:
        'cubic', 'linear', 'nearest', 'area', 'lanczos4'
        'lanczos4' is the default as it seems to produce the least artifacts.
    cyx: tuple (cy,cx)
        center of the rotation. If None is given, rotation is performed around
        the center of the frame.
        
    Returns
    -------
    rotated_img: 2D-array
        Rotated frame, same dimensions as inpute frame.
    """
    ny, nx = img.shape
    
    cx = nx / 2.
    cy = ny / 2. 

    if not cyx:
          cx = nx / 2.
          cy = ny / 2. 
  
        
    # interpolation type
    if interpolation == 'cubic':
        intp        = cv2.INTER_CUBIC 
    elif interpolation == 'linear':
        intp        = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        intp        = cv2.INTER_NEAREST
    elif interpolation == 'area':
        intp        = cv2.INTER_AREA
    elif interpolation == 'lanczos4':
        intp        = cv2.INTER_LANCZOS4
    
    if abs(angle) > 0:
        M           = cv2.getRotationMatrix2D((cx,cy), angle, 1)
        rotated_img = cv2.warpAffine(img.astype(np.float32), M, (nx, ny), flags=intp)
    else:
        rotated_img = img
    
    return rotated_img
    
    
    
    
#DETECTORTOPUPIL A REVOIR!!!!    
def detectortopupil(Resultat,prad1):
    phaseretrouvee=shift(shift(ifft((Resultat)))*roundpupil(isz,prad1)*masktot)
    phaseretrouvee=fft(phaseretrouvee)*maskcorono
    phaseretrouvee=ifft((phaseretrouvee))/shift(masktot)
    return phaseretrouvee

    

def butterworth(isz,image,order,length):
    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rr = np.hypot(yy, xx)
    butt=1/(1+(np.sqrt(2)-1)*(rr/length)**(2*order))
    return image*butt
    

    
np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=1000)
def invertDSCC(interact,coupe,goal='e',visu=False):
    U, s, V = np.linalg.svd(interact, full_matrices=False)
    #print(s)
    S = np.diag(s)
    InvS=np.linalg.inv(S)
    #print(InvS)
    if(visu==True):
        plt.plot(np.diag(InvS),'r.')
        plt.yscale('log')
        plt.show()
        
    if(goal=='e'):
        InvS[np.where(InvS>coupe)]=0
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
        return [np.diag(InvS),pseudoinverse]
      
    if(goal=='c'):
        InvS[coupe:]=0
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
        return [np.diag(InvS),pseudoinverse]


def LoadImageFits(docs_dir):
    openfits=fits.open(docs_dir)
    image=openfits[0].data
    return image
    

def CubeFits(docs_dir):
    image_list = []
    for filename in glob.glob(docs_dir+'*.fits'):
        image=LoadImageFits(filename)
        image_list.append(image)
        
    image_array = np.array(image_list)
    return image_array
    

def MoyenneFits(docs_dir):
    Cubeimage=CubeFits(docs_dir)
    Sommeimage=0
    for i in np.arange((Cubeimage.shape[0])):
        Sommeimage=Sommeimage+Cubeimage[i]
    imagemoyenne=Sommeimage/Cubeimage.shape[0]
    return imagemoyenne
    
def SaveFits(image,head,doc_dir2,name):
    hdu=fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr=hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits')
    
    
def GetFluxmetreValue(fitspath):
    openfits=fits.open(fitspath)
    hdu=openfits[0].header
    fluxmetre=hdu['FLUX_W']
    return fluxmetre


def Reech(image,reechpup,new):
    Gvectorbis=image[int(isz/2-reechpup):int(isz/2+reechpup),int(isz/2-reechpup):int(isz/2+reechpup)] #Image 400x400-8
    Gvectorbis=ishift(Gvectorbis)  
    Gvectorbis=ifft(Gvectorbis)
    Gvectorbis=shift(Gvectorbis)
    Gvector=Gvectorbis[int(reechpup-new/2):int(reechpup+new/2),int(reechpup-new/2):int(reechpup+new/2)] #Pup 160x160-3
    Gvector=ishift(Gvector)
    Gvector=fft(Gvector) #Image 160x160-3
    return Gvector


def shotnoise(nbpix,num_photons):
    mu_p        = num_photons * np.ones((nbpix,nbpix))
    seed       = np.random.random_integers(1000)
    rs         = np.random.RandomState(seed)
    shot_noise = rs.poisson(num_photons, (nbpix, nbpix))
    #return 0
    return (shot_noise-mu_p)/num_photons
    

def createvectorprobes(posprobes,cutsvd, pushact):
    numprobe=len(posprobes)
    deltapsik=np.zeros((numprobe,dimimages,dimimages),dtype=complex)
    probephase=np.zeros((numprobe,isz,isz))
    matrix=np.zeros((numprobe,2))
    Vecteurenvoi=np.zeros((dimimages**2,2,numprobe))
    SVD=np.zeros((2,dimimages,dimimages))
    whennoabb=pupiltodetector2(0,prad,lyotrad)
    k=0
    for i in posprobes:
        probephase[k]=amplitude*pushact[i]
        #probephase[k]=amplitude*pushact(i,coeff,grille,actshapeinpupil,1,changeactfunc=True)
        deltapsikbis=pupiltodetector2(2*np.pi*(probephase[k])*1e-9/wavelength,prad,lyotrad)-whennoabb
        deltapsikbis=Reech(shift(deltapsikbis),reechpup,new)/squaremaxPSF
        deltapsik[k]=(shift(deltapsikbis)[int(new/2)-int(dimimages/2):int(new/2)+int(dimimages/2),int(new/2)-int(dimimages/2):int(new/2)+int(dimimages/2)])
        k=k+1

    l=0
    for i in np.arange(dimimages):
    
        for j in np.arange(dimimages):
            matrix[:,0]=real(deltapsik[:,i,j])
            matrix[:,1]=im(deltapsik[:,i,j])
    
            try:
                SVD[:,i,j]=invertDSCC(matrix,cutsvd,visu=False)[0]
                Vecteurenvoi[l]=invertDSCC(matrix,cutsvd,visu=False)[1]
            except:
                print('Careful: Error! for l='+str(l))
                SVD[:,i,j]=np.zeros(2)
                Vecteurenvoi[l]=np.zeros((2,numprobe))
            l=l+1  
    return [Vecteurenvoi,SVD]
   

def estimateEab(Difference,Vecteurprobes):
    numprobe=len(Vecteurprobes[0,0])
    Differenceij=np.zeros((numprobe))
    Resultat=np.zeros((dimimages,dimimages),dtype=complex)
    l=0
    for i in np.arange(dimimages):
        for j in np.arange(dimimages):
            Differenceij[:]=Difference[:,i,j]
            Resultatbis=np.dot(Vecteurprobes[l],Differenceij)
            Resultat[i,j]=Resultatbis[0]+1j*Resultatbis[1]
            
            l=l+1  
    return Resultat/4
    
    
def creatingWhichinPupil(cutinpupil,pushact):
    WhichInPupil = []
    for i in np.arange(int(1024)):
        Psivector=pushact[i]
        cut=cutinpupil*np.sum(actshapeinpupil)
        if(np.sum(Psivector*roundpupil(isz,lyotrad*coeff))>=cut):
            WhichInPupil.append(i)
    
    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil
    


def creatingMaskDH(shape,choosepixDH=[0,0,0,0], inner=0, outer=0, xdecay=0):
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



def creatingCorrectionmatrix(mask,Whichact,pushact):
    Gmatrixbis=np.zeros((2*int(np.sum(mask)),len(Whichact)))
    whennoabb=pupiltodetector2(0,prad,lyotrad)   
    k=0
    for i in Whichact:
        Psivector=amplitude*pushact[i]
        #Psivector=amplitude*pushact(i,coeff,grille,actshapeinpupil,1,changeactfunc=True)
        # Gvectorbisbis=shift(pupiltodetector1(0,0+2*np.pi*(Psivector)*1e-9/wavelength,prad,lyotrad))-shift(pupiltodetector1(0,0-2*np.pi*(Psivector)*1e-9/wavelength,prad,lyotrad))
        # Gvectorbisbis=Gvectorbisbis/2
        
        Gvectorbisbis=shift(pupiltodetector2(2*np.pi*(Psivector)*1e-9/wavelength,prad,lyotrad))-shift(whennoabb)
    
        Gvector=Reech(Gvectorbisbis,reechpup,new)/squaremaxPSF
        
        Gmatrixbis[0:int(np.sum(mask)),k]=real(shift(Gvector)[np.where(mask==1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,k]=im(shift(Gvector)[np.where(mask==1)]).flatten()
        k=k+1
    return Gmatrixbis


def solutiontocorrect(mask,ResultatEstimate,invertG):
    Eab=np.zeros(2*int(np.sum(mask)))
    Resultatbis=(ResultatEstimate[np.where(mask==1)])
    Eab[0:int(np.sum(mask))]=real(Resultatbis).flatten()     
    Eab[int(np.sum(mask)):]=im(Resultatbis).flatten()
    cool=np.dot(invertG,Eab)
    
    solution=np.zeros(1024)
    solution[WhichInPupil]=cool
    return solution
    
    
    
def createdifference(aberramp,aberrphase,posprobes,pushact,noise=False):
    Ikmoins=np.zeros((isz,isz))
    Ikplus=np.zeros((isz,isz))
    Difference=np.zeros((len(posprobes),dimimages,dimimages))
    shot1=1
    shot2=1
    numphot=2e2
    
    if noise==True:
        shot1=(1+shotnoise(int(isz),numphot))
        shot2=(1+shotnoise(int(isz),numphot))
    
    
    k=0
    for i in posprobes:
        #probephase=amplitude*pushact(i,coeff,grille,actshapeinpupil,devx)
        probephase=amplitude*pushact[i]
        probephase=2*np.pi*probephase*1e-9/wavelength
        Ikmoins=shot1*np.abs(pupiltodetector1(aberramp,aberrphase-1*probephase,coeff*prad,coeff*lyotrad))**2/squaremaxPSF**2 #Image 800x800-8
        
        Ikplus=shot2*np.abs(pupiltodetector1(aberramp,aberrphase+1*probephase,coeff*prad,coeff*lyotrad))**2/squaremaxPSF**2 #Image 800x800-8
        
        Difference1=shift(Ikplus-Ikmoins)
        Difference[k]=shift(Reech(Difference1,reechpup,new))
        k=k+1
        
    return Difference
    
    
def determinecontrast(image,chiffre):
    xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
    rr     = np.hypot(yy, xx)
    contrast=np.zeros(int(isz/2/chiffre))
    for i in chiffre*np.arange(int(isz/2/chiffre)):
        whereimage=np.zeros((isz,isz))
        whereimage[np.where(rr>=i)]=1
        whereimage[np.where(rr>=i+chiffre)]=0
        whereimage[np.where(xx<=30)]=0
        #whereimage[np.where(abs(yy)<10)]=0
        imagebis=(abs(image))*whereimage
        contrast[int(i/chiffre)]=np.std(imagebis[np.where(whereimage!=0)])
    return contrast





date1=20180906
date2=20180920
date2=20190115

docs_dir_in=os.path.expanduser('/home/apotier/Documents/Recherche/DonneesTHD/EFC/'+str(date2)+'_EFC/')
docs_dir_out=os.path.expanduser('/home/apotier/Documents/Recherche/DonneesTHD/EFC/'+str(date2)+'_EFC/iter0/')

docs_dir_estimate=os.path.expanduser('/home/apotier/Documents/Recherche/DonneesTHD/EFC/'+str(date1)+'_EFC/SCC_Data2_updated/')
docs_dir_matrix_act=os.path.expanduser('/home/apotier/Documents/Recherche/DonneesTHD/EFC/')


#Raccourcis FFT
fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift
ishift=np.fft.ifftshift

abs=np.abs
im=np.imag
real=np.real
mean=np.mean
dot=np.dot
amax=np.amax

#Raccourcis conversions angles
dtor    = np.pi / 180.0 # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor


isz    = 400  # image size (in pixels)
wavelength = 783.25e-9 # wavelength for image (in meters)
mperpix=6.5e-6 #Size pixel on detector in meter

lyotdiam  = 8.0e-3  # lyot diameter (in meters)
lyotdiam = 7.9e-3
distance=(198.29+301.64)*10**(-3) #Distance Lyot detector in meters
ld_m=distance*wavelength/lyotdiam #Resolution element in meter
ld_p = ld_m/mperpix  # lambda/D (in pixels)

lyotrad = np.round(isz / ld_p / 2.0) #Radius pupil to get this resolution

pdiam=8.23e-3
prad=np.round(pdiam*lyotrad/lyotdiam)

masktot=translationFFT(isz,0.5,0.5)
maskcorono=shift(KnifeEdgeCoro('left',1.2))


coeff=1
isz=isz*coeff
reechpup=isz/2/coeff

grille=LoadImageFits(docs_dir_matrix_act+'Grid_actu.fits')
actshape=LoadImageFits(docs_dir_matrix_act+'Actu_DM32_field=6x6pitch_pitch=22pix.fits')

act309=np.zeros((isz,isz))
act309[int(isz/2-108/2):int(isz/2+108/2),int(isz/2-108/2):int(isz/2+108/2)]=LoadImageFits(docs_dir_estimate+'Probe309/Phase_estim.fits')
y309,x309=np.unravel_index(act309.argmax(), act309.shape)



coefforactshape=np.round(1.98*6*coeff)
resizeactshape=skimage.transform.resize(actshape, (coefforactshape,coefforactshape), order=1,preserve_range=True)
resizeactshape=nd.interpolation.shift(resizeactshape,(0.3,0.3)) ### ATTENTION, ceci a été fait de manière arbitraire!

actshapeinpupil=np.zeros((isz,isz))
actshapeinpupil[0:int(coefforactshape),0:int(coefforactshape)]=resizeactshape


#A modifier
new=150     #Taille réechantillonage
dimimages=150  #Taille DH (découpage des images)

numprobe=3
orderbutt=50
lengthbutt=100

squaremaxPSF=np.amax(np.abs(fft(roundpupil(new,lyotrad))))
##
pushact=np.zeros((1024,isz,isz))
for i in np.arange(1024):
    pushact[i]=pushact_function(i,coeff,grille,actshapeinpupil,0)
#SaveFits(pushact,['',0],'/home/apotier/Documents/Recherche/SimuPython/EFC/Knife_Edge_Corono/','PushActInPup')

##
amplitude=50
posprobes=[466,498]
#posprobes=[309,495,659]
cut=1e5
vectoressai,showsvd=createvectorprobes(posprobes,cut,pushact)
##
plt.imshow(showsvd[1])
plt.show()

##
#WhichInPupil=creatingWhichinPupil(0.1,pushact)
#WhichInPupil=creatingWhichinPupil(0.5,pushact)
#WhichInPupil=creatingWhichinPupil(0.25)
WhichInPupil=creatingWhichinPupil(0,pushact)

print('WhichInPupil is ok')

##    
#choosepix=[-20,20,-20,-5]
#choosepix=[5,35,-35,35]
#choosepix=[-35,35,-35,35]
#choosepix=[11,29,-29,29]
#choosepix=[5,23,-23,23]
#choosepix=[10,35,-35,35]
choosepix=[4,20,-8,8]
#maskDH=creatingMaskDH(choosepix)
maskDH=creatingMaskDH('square',choosepixDH=choosepix,inner=5,outer=35,xdecay=5)
Gmatrix=creatingCorrectionmatrix(maskDH,WhichInPupil,pushact)
print('Gmatrix is ok')
##
invertGDH=invertDSCC(Gmatrix,500,goal='c',visu=True)[1]

##Result image for one error
error=0
docs_dir=os.path.expanduser('/home/apotier/Documents/Recherche/SimuPython/EFC/RobustnessStudy2/')
phase=LoadImageFits(docs_dir+'phase20rmsfm3.fits')
phase=phase*2*np.pi/wavelength
#phase=0
# amp=LoadImageFits(docs_dir+'amp13rmsfm2.fits')
# amp=amp/np.std(amp)*5e-9
# amp=amp*2*np.pi/wavelength
#amp=0
oui=LoadImageFits('/home/apotier/Documents/Recherche/DonneesTHD/EFC/Amplitudebanc')#*roundpupil(isz,prad)
moy=np.mean(oui[np.where(oui!=0)])
amp=oui/moy
amp1=cv2.resize(amp, dsize=(int(2*prad/148*400),int(2*prad/148*400)),interpolation=cv2.INTER_AREA)
ampfinal=np.zeros((isz,isz))
ampfinal[int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1,int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1]=amp1
ampfinal=(ampfinal)*roundpupil(isz,prad-1)
moy=np.mean(ampfinal[np.where(ampfinal!=0)])
ampfinal=(ampfinal/moy-np.ones((isz,isz)))*roundpupil(isz,prad-1)#/10
phase=0
ampfinal=0


def ElectricFieldConjugation(amplitude_abb,phase_abb,modevector,gain,error,PW=True):
    
    if error==0:
        pushactonDM=pushact
    else:
        print('Misregistration!')
        pushactonDM=np.zeros((1024,isz,isz))
        for i in np.arange(1024):
            pushactonDM[i]=pushact_function(i,coeff,grille,actshapeinpupil,error)
    
    nbiter=len(modevector)
    imagedetector=np.zeros((nbiter+1,isz,isz))
    imagedetector[0]=abs(shift(pupiltodetector1(amplitude_abb,phase_abb,prad,lyotrad)))**2/squaremaxPSF**2
    plt.ion()
    plt.figure()
    previousmode=0
    k=0
    for mode in modevector:
        print(k,mode)
        if PW==True:
            Difference=createdifference(amplitude_abb,phase_abb,posprobes,pushactonDM,noise=False)
            resultatestimation=estimateEab(Difference,vectoressai)
        else:
            resultatestimation=shift(Reech(shift(pupiltodetector1(amplitude_abb,phase_abb,prad,lyotrad)),reechpup,new))/squaremaxPSF
            
        if mode!=previousmode:
            invertGDH=invertDSCC(Gmatrix,mode,goal='c',visu=False)[1]
            
        solution1=solutiontocorrect(maskDH,resultatestimation,invertGDH)
        phase_abb=phase_abb-gain*amplitude*np.dot(solution1,pushactonDM.reshape(1024,isz*isz)).reshape(isz,isz)*2*np.pi*1e-9/wavelength
        imagedetector[k+1]=abs(shift(pupiltodetector1(amplitude_abb,phase_abb,prad,lyotrad)))**2/squaremaxPSF**2
        plt.clf()
        plt.imshow(np.log10(imagedetector[k+1]),vmin=-8,vmax=-5)
        plt.colorbar()
        plt.pause(0.01)
        previousmode=mode
        k=k+1
    return phase_abb,imagedetector

##
phase=LoadImageFits(docs_dir+'phase20rmsfm3.fits')
phase=phase*2*np.pi/wavelength
#phase=0
# amp=LoadImageFits(docs_dir+'amp13rmsfm2.fits')
# amp=amp/np.std(amp)*5e-9
# amp=amp*2*np.pi/wavelength
#amp=0
oui=LoadImageFits('/home/apotier/Documents/Recherche/DonneesTHD/EFC/Amplitudebanc')#*roundpupil(isz,prad)
moy=np.mean(oui[np.where(oui!=0)])
amp=oui/moy
amp1=cv2.resize(amp, dsize=(int(2*prad/148*400),int(2*prad/148*400)),interpolation=cv2.INTER_AREA)
ampfinal=np.zeros((isz,isz))
ampfinal[int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1,int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1]=amp1
ampfinal=(ampfinal)*roundpupil(isz,prad-1)
moy=np.mean(ampfinal[np.where(ampfinal!=0)])
ampfinal=(ampfinal/moy-np.ones((isz,isz)))*roundpupil(isz,prad-1)#/10

# phase=0
# amp=0
modevec=[75]*10+[100]*20+[110]*20+[120]*20+[130]*30+[140]*30+[150]*80#+[160]*80
modevec=[110]*25+[120]*15+[130]*50
gain=0.1
error=0
ouais,image=ElectricFieldConjugation(amp,phase,modevec,gain,error,PW=False)

##
SaveFits(image,[' ',0],'/home/apotier/Documents/Recherche/SimuPython/EFC/Knife_Edge_Corono/','EssaiSmallDHAberr150modes')


##
error=0
docs_dir=os.path.expanduser('/home/apotier/Documents/Recherche/SimuPython/EFC/RobustnessStudy2/')
phase=LoadImageFits(docs_dir+'phase20rmsfm3.fits')
phase=phase*2*np.pi/wavelength
#phase=0
# amp=LoadImageFits(docs_dir+'amp13rmsfm2.fits')
# amp=amp/np.std(amp)*5e-9
# amp=amp*2*np.pi/wavelength
#amp=0
oui=LoadImageFits('/home/apotier/Documents/Recherche/DonneesTHD/EFC/Amplitudebanc')#*roundpupil(isz,prad)
moy=np.mean(oui[np.where(oui!=0)])
amp=oui/moy
amp1=cv2.resize(amp, dsize=(int(2*prad/148*400),int(2*prad/148*400)),interpolation=cv2.INTER_AREA)
ampfinal=np.zeros((isz,isz))
ampfinal[int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1,int(isz/2-len(amp1)/2)+1:int(isz/2+len(amp1)/2)+1]=amp1
ampfinal=(ampfinal)*roundpupil(isz,prad-1)
moy=np.mean(ampfinal[np.where(ampfinal!=0)])
ampfinal=(ampfinal/moy-np.ones((isz,isz)))*roundpupil(isz,prad-1)#/10

gain=0.5
k=0
imagedetector=np.zeros((50,isz,isz),dtype=complex)
imagedetector[0]=shift(pupiltodetector1(ampfinal,phase,prad,lyotrad))
for modes in [100]*9 + [200]*10 + [300]*10 + [400]*10 + [500]*10:
    print(modes)
    invertGDH=invertDSCC(Gmatrix,modes,goal='c',visu=False)[1]
    Difference=createdifference(ampfinal,phase,posprobes,error,noise=False)
    resultatestimation=estimateEab(Difference,vectoressai)
    solution1=solutiontocorrect(maskDH,resultatestimation,invertGDH)
    ph=np.zeros((isz,isz))
    for i in WhichInPupil:
        ph=ph+solution1[i]*amplitude*pushact(i,coeff,grille,actshapeinpupil,error)*2*np.pi*1e-9/wavelength
        phase=phase-gain*solution1[i]*amplitude*pushact(i,coeff,grille,actshapeinpupil,error)*2*np.pi*1e-9/wavelength
    imagedetector[k+1]=shift(pupiltodetector1(ampfinal,phase,prad,lyotrad))
    k=k+1
    ##
SaveFits(((abs(imagedetector)**2/squaremaxPSF**2)),[' ',0],'/home/apotier/Documents/Recherche/SimuPython/EFC/Knife_Edge_Corono/','RegularizedEFC_100_200_300_400_500_80Lyot_5pix')
    
##

font = {'color':  'black',
        'weight': 'bold',
        'size': 12,
        }

Choosetitle=['Initial phase','IterationsFDH', None, 'FinalFDH', 'IterationsHDH', None, 'FinalHDH']
colored=['k','b:','b:','b','r:','r:','r']
xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
rr     = np.hypot(yy, xx)
chiffre=5
chiffre=1
contrast=np.zeros(int(isz/2/chiffre))

#result=LoadImageFits('/home/apotier/Documents/Recherche/SimuPython/EFC/Probes/SSIm466act0.99Which.fits')
fig, ax = plt.subplots(1,1)

for k in np.arange(10):
    for i in chiffre*np.arange(int(isz/2/chiffre)):
    #for i in np.arange(80):
        #print(i)
        whereimage=np.zeros((isz,isz))
        whereimage[np.where(rr>=i)]=1
        whereimage[np.where(rr>=i+chiffre)]=0
        whereimage[np.where(xx<=23)]=0
        whereimage[np.where(abs(yy)<10)]=0
        imagebis=(abs(imagedetector[k])**2/squaremaxPSF**2)*whereimage
        contrast[int(i/chiffre)]=np.std(imagebis[np.where(whereimage!=0)])

    ax.plot(chiffre*np.arange(int(isz/2/chiffre))/ld_p,contrast)#,colored[k],label=Choosetitle[k])
plt.yscale('log')
plt.xlim(0,20)
#plt.ylim(8e-9,2e-5)
(xmin, xmax) = ax.xaxis.get_view_interval()
(ymin, ymax) = ax.yaxis.get_view_interval()
ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax),color = 'black', linewidth = 5))
ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin),color = 'black', linewidth = 6))
ax.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax),color = 'black', linewidth = 5))
ax.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax),color = 'black', linewidth = 6))

ax.xaxis.set_tick_params(length = 5,width=2,labelsize = 12,which = 'major')
ax.yaxis.set_tick_params(length = 5,width=2,labelsize = 12)
ax.set_xlabel(r'Angular separation ($\lambda$/D)', fontdict=font)
ax.set_ylabel('Contrast RMS', fontdict=font)
#ax.legend()

plt.show()

