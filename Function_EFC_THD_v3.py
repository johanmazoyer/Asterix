#Version 26/12/19

def translationFFT(isz,a,b):
    maskx=np.zeros((isz,isz))
    masky=np.zeros((isz,isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            maskx[i,j]=j*np.pi*2*a/isz-a*np.pi
            masky[i,j]=i*np.pi*2*b/isz-b*np.pi
    masktot=np.exp(-1j*maskx)*np.exp(-1j*masky)
    return masktot


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
        if position=='top':
            if (i>isz/2+shiftinldp*ld_p): Knife[i,:]=1
        if position=='bottom':
            if (i<isz/2-shiftinldp*ld_p): Knife[i,:]=1
    return shift(Knife)
    
def roundpupil(nbpix,prad1):
    xx, yy = np.meshgrid(np.arange(nbpix)-(nbpix)/2, np.arange(nbpix)-(nbpix)/2)
    rr     = np.hypot(yy+1/2, xx+1/2)
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
    pupil2=focal1*coro
    pupil2=ifft(pupil2)/shift(masktot)
    pupil2=shift(pupil2)
    pupil2[rr<=prad1]=0
    pupil2=shift(pupil2)

    #Focale 1 pour coronographe parfait
    pupilcomplex=fft(pupil2*shift(masktot))
    pupilcomplex=pupilcomplex*coro

    #Pupille pour coronographe parfait
    pupil1bis=ifft(pupilcomplex)
    pupil1bis2=shift(pupil1bis)/masktot
    return pupil1bis2
    
    
    
    
    
def pupiltodetector(input_wavefront,coro_mask,lyot_mask): #aberrationphase,prad1,prad2
    #Focale1  
    focal1end=shift(input_wavefront*masktot)
    focal1end=fft(focal1end)
    
    #Pupille2
    pupil2end=focal1end*coro_mask
    pupil2end=ifft(pupil2end)#/shift(masktot)
    
    #Intensité en sortie de SCC
    focal2end=pupil2end*shift(lyot_mask)
    focal2end=fft(focal2end)
    
    sqrtimage=Reech(shift(focal2end),reechpup,new)
    return shift(sqrtimage)
    


def pushact_function(which,grilleact, actshapeinpupilresized,xycent):
    xact=(grilleact[0,which]+(x309-grilleact[0,309]))
    yact=(grilleact[1,which]+(y309-grilleact[1,309]))
    Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent,xact-xycent))
    Psivector[np.where(Psivector<1e-4)]=0
    #Psivector=np.rot90(Psivector,2)
    return Psivector


def invertDSCC(interact,coupe,goal='e',regul='truncation',visu=True,otherbasis=False,basisDM3=0):
    U, s, V = np.linalg.svd(interact, full_matrices=False)
    #print(s)
    S = np.diag(s)
    InvS=np.linalg.inv(S)
    #print(InvS)
    if(visu==True):
        plt.plot(np.diag(InvS),'r.')
        plt.yscale('log')
        plt.savefig(intermatrix_dir+'invertSVDEFC_'+ '_'.join(map(str, choosepix)) + 'pix_' + str(amplitudeEFC) + 'nm_.png')
        #plt.show()
        
    if(goal=='e'):
        InvS[np.where(InvS>coupe)]=0
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
      
    if(goal=='c'):
        if regul=='truncation':
            InvS[coupe:]=0
        if regul=='tikhonov':
            InvS=np.diag(s/(s**2+s[coupe]**2))
            if(visu==True):
                plt.plot(np.diag(InvS),'b.')
                plt.yscale('log')
        pseudoinverse=np.dot(np.dot(np.transpose(V),InvS),np.transpose(U))
    
    if (otherbasis==True):
        pseudoinverse=np.dot(np.transpose(basisDM3),pseudoinverse)
        
    return [np.diag(InvS),pseudoinverse]


def LoadImageFits(docs_dir):
    openfits=fits.open(docs_dir)
    image=openfits[0].data
    return image
    
    
def SaveFits(image,head,doc_dir2,name,replace=False):
    hdu=fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdr=hdul[0].header
    hdr.set(head[0],head[1])
    hdu.writeto(doc_dir2+name+'.fits', overwrite=replace )
    

def cropimage(img,ctr_x,ctr_y,newsizeimg):
    lenimg=len(img)
    imgs2=lenimg/2
    newimgs2=newsizeimg/2
    return img[int(ctr_x-newimgs2):int(ctr_x+newimgs2),int(ctr_y-newimgs2):int(ctr_y+newimgs2)]
    
    
def Reech(image,reechpup,new):
    Gvectorbis=cropimage(image,isz/2,isz/2,reechpup)
    Gvectorbis=ishift(Gvectorbis)  
    Gvectorbis=ifft(Gvectorbis)
    Gvectorbis=shift(Gvectorbis)
    Gvector=cropimage(Gvectorbis,reechpup/2,reechpup/2,new)
    Gvector=ishift(Gvector)
    Gvector=fft(Gvector) #Image 160x160-3
    return Gvector
    

def createvectorprobes(amplitude,posprobes,cutsvd):
    numprobe=len(posprobes)
    deltapsik=np.zeros((numprobe,dimimages,dimimages),dtype=complex)
    probephase=np.zeros((numprobe,isz,isz))
    matrix=np.zeros((numprobe,2))
    Vecteurenvoi=np.zeros((dimimages**2,2,numprobe))
    SVD=np.zeros((2,dimimages,dimimages))
    
    k=0
    for i in posprobes:
        probephase[k]=amplitude*pushact[i]
        probephase[k]=2*np.pi*(probephase[k])*1e-9/wavelength
        inputwavefront=entrancepupil*(1+1j*probephase[k]*roundpupil(isz,prad))
        deltapsikbis=pupiltodetector(inputwavefront,coro,lyot)/squaremaxPSF-pupiltodetector(entrancepupil,coro,lyot)/squaremaxPSF
        deltapsik[k]=cropimage(deltapsikbis,new/2,new/2,dimimages)
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
   
    
    
def creatingWhichinPupil(cutinpupil):
    WhichInPupil = []
    for i in np.arange(int(1024)):
        Psivector=pushact[i]
        cut=cutinpupil*np.sum(actshapeinpupil)
        if(np.sum(Psivector*roundpupil(isz,lyotrad))>cut):
            WhichInPupil.append(i)
    
    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil
    


def creatingMaskDH(choosepixDH):
    xx, yy = np.meshgrid(np.arange(dimimages)-(dimimages)/2, np.arange(dimimages)-(dimimages)/2)
    rr     = np.hypot(yy, xx)
    maskDH=np.ones((dimimages,dimimages))
    maskDH[xx<choosepixDH[0]]=0
    maskDH[xx>choosepixDH[1]]=0
    maskDH[yy<choosepixDH[2]]=0
    maskDH[yy>choosepixDH[3]]=0
    return maskDH



def creatingCorrectionmatrix(amplitude,mask,Whichact,otherbasis=False,basisDM3=0):
    #change basis if needed
    if (otherbasis == True):
        nb_fct=basisDM3.shape[0]#number of functions in the basis
        tmp=pushact.reshape(pushact.shape[0],pushact.shape[1]*pushact.shape[2])
        bas_fct=np.dot(basisDM3,tmp).reshape(nb_fct,pushact.shape[1],pushact.shape[2])
    else:
        bas_fct = np.array([pushact[ind] for ind in Whichact])
        nb_fct=len(Whichact)

    print('Start EFC')
    Gmatrixbis=np.zeros((2*int(np.sum(mask)),nb_fct))
    k=0
    for i in range(nb_fct):
        if i%100 == 0:
            print(i)
        Psivector=amplitude*bas_fct[i]
        Psivector=2*np.pi*(Psivector)*1e-9/wavelength
        inputwavefront=entrancepupil*(1+1j*Psivector*roundpupil(isz,prad))
        Gvector=(pupiltodetector(inputwavefront,coro,lyot))/squaremaxPSF-pupiltodetector(entrancepupil,coro,lyot)/squaremaxPSF
        Gmatrixbis[0:int(np.sum(mask)),k]=real(Gvector[np.where(mask==1)]).flatten()
        Gmatrixbis[int(np.sum(mask)):,k]=im(Gvector[np.where(mask==1)]).flatten()
        k=k+1
    print('End EFC')
    return Gmatrixbis

def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, h):
    x=xy[0]
    y=xy[1]
    xo = float(xo)
    yo = float(yo)
    theta=0
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))+h
    return g.flatten()

def gauss2Dfit(data):
 #2D-Gaussian fit
    popt=np.zeros(8)
    w,h = data.shape
    x, y = np.mgrid[0:w, 0:h]
    xy=(x,y)

    #Fit 2D Gaussian with fixed parameters
    initial_guess = (np.amax(data), 1, 1,len(data)/2,len(data)/2, 0)

    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, xy, data.flatten(), p0=initial_guess)
    except RuntimeError:
        print("Error - curve_fit failed")

    return popt[3],popt[4]



from Parameters_EFC_THD import *

#Raccourcis FFT
fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift
ishift=np.fft.ifftshift

#Raccourcis généraux
abs=np.abs
im=np.imag
real=np.real
mean=np.mean
dot=np.dot
amax=np.amax

#Raccourcis conversions angles
dtor    = np.pi / 180.0 # degree to radian conversion factor
rad2mas = 3.6e6 / dtor  # radian to milliarcsecond conversion factor



model_dir = main_dir+'Model/'
mperpix = 6.5e-6 #Size pixel on detector in meter
distance = (198.29+301.64)*10**(-3) #Distance Lyot detector in meters
ld_m = distance*wavelength/lyotdiam #Resolution element in meter
if set_ldp_manually == False:
    ld_p = ld_m/mperpix  # lambda/D (in pixels)

# lyotrad = np.round(isz / ld_p / 2.0) #Radius pupil to get this resolution
# prad = np.round(pdiam*lyotrad/lyotdiam)


prad0=isz/2/ld_p
lyotrad=prad0*lyot
prad=np.ceil(prad0)
lyotrad=np.ceil(lyotrad)
# prad=np.round(prad)
# lyotrad=np.round(lyotrad)
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


lyot = roundpupil(isz,lyotrad)
masktot = LoadImageFits(model_dir+'Translation_Half_Pix.fits')[0]+1j*LoadImageFits(model_dir+'Translation_Half_Pix.fits')[1]
reechpup = isz

grille = LoadImageFits(model_dir+'Grid_actu.fits')
actshape = LoadImageFits(model_dir+'Actu_DM32_field=6x6pitch_pitch=22pix.fits')
file309='Phase_estim_Act309_v20200107.fits'
im309size = len(LoadImageFits(model_dir+file309))
act309 = np.zeros((isz,isz))
act309[int(isz/2-im309size/2):int(isz/2+im309size/2),int(isz/2-im309size/2):int(isz/2+im309size/2)] = LoadImageFits(model_dir+file309)
#y309,x309 = np.unravel_index(act309.argmax(), act309.shape)
y309,x309 = np.unravel_index(act309.argmin(), act309.shape)
#print(y309-200+54,x309-200+54)
resizeactshape = skimage.transform.rescale(actshape, 2*prad0/pdiam*.3e-3/22, order=1,preserve_range=True,anti_aliasing=True)

#Gauss2Dfit for centering the rescaled influence function
dx,dy = gauss2Dfit(resizeactshape)
xycent = len(resizeactshape/2)
resizeactshape=nd.interpolation.shift(resizeactshape,(xycent-dx,xycent-dy))
#Put the centered influence function inside a larger array (400x400)
actshapeinpupil = np.zeros((isz,isz))
actshapeinpupil[0:len(resizeactshape),0:len(resizeactshape)] = resizeactshape/np.amax(resizeactshape)

#Center of actuator 309 (found by SCC on master for the moment)
y309=212.65#68.5+200-54
x309=211.18#63.5+200-54

#shift by (0.5,0.5) pixel because the pupil is centerd between pixels
y309=y309-0.5
x309=x309-0.5

pushact = LoadImageFits(model_dir+'PushActInPup400.fits')
#pushact=np.zeros((1024,isz,isz))
#for i in np.arange(1024):
#    pushact[i]=pushact_function(i,grille,actshapeinpupil,xycent)
#SaveFits(pushact,['',0],model_dir,'PushActInPup400')

## transmission of the phase mask (exp(i*phase))
## centered on pixel [0.5,0.5]
if coronagraph =='fqpm':
    coro = LoadImageFits(model_dir+'FQPM.fits')
elif coronagraph=='knife':
    coro = KnifeEdgeCoro('top',1.2)
elif coronagraph =='vortex':
    phasevortex=0# to be defined
    coro = exp(1j*phasevortex)
entrancepupil=roundpupil(isz,prad)

#Flux normalization
#inputwavefront = entrancepupil*translationFFT(isz,30,30)
#squaremaxPSF = np.amax(shift(abs(pupiltodetector(inputwavefront,coro,lyot))))

maskreech=translationFFT(new,0.5,0.5)
squaremaxPSF=np.amax(np.abs(fft(shift(roundpupil(new,lyotrad)*maskreech))))


# phaseaberr=LoadImageFits('/home/apotier/Documents/Recherche/SimuPython/COFFEE/Tests20190911/PhiUp.fits')
# phase=np.zeros((isz,isz))
# phase[int(isz/2)-int(len(phaseaberr)/2):int(isz/2)+int(len(phaseaberr)/2),int(isz/2)-int(len(phaseaberr)/2):int(isz/2)+int(len(phaseaberr)/2)]=phaseaberr
# 
# inputwavefront=entrancepupil*(1+1j*phase*roundpupil(isz,prad))
# 
# 
# imfinale=(abs(pupiltodetector(inputwavefront,coro,lyot)-pupiltodetector(entrancepupil,coro,lyot))**2)#[100:300,100:300]
# 
# imfinale=imfinale/squaremaxPSF**2
# #SaveFits(imfinale,['',0],model_dir,'ImageNewCode')