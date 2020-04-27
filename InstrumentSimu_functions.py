#Version 29 Janvier 2020
from shortcuts import *
import fits_functions as fi
import processing_functions as proc

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


    
def translationFFT(isz,a,b):
    ''' --------------------------------------------------
    Create a phase ramp of size (isz,isz) which can be multiply by an image to shift its fourier transform by (a,b) pixels
    
    Parameters:
    ----------
    isz: int, size of the phase ramp (in pixels)
    a: float, shift desired in the x direction (in pixels)
    b: float, shift desired in the y direction (in pixels)
    
    Return:
    ------
    masktot: 2D array, phase ramp
    -------------------------------------------------- '''
    #Verify this function works
    maska=np.linspace(-np.pi*a,np.pi*a,isz)
    maskb=np.linspace(-np.pi*b,np.pi*b,isz)
    xx,yy=np.meshgrid(maska,maskb)
    masktot=np.exp(-1j*xx)*np.exp(-1j*yy)
    return masktot

    

def FQPM(isz):
    ''' --------------------------------------------------
    Create a perfect Four Quadrant Phase Mask coronagraph of size (isz,isz)
    
    Parameters:
    ----------
    isz: int, size of the coronagraph (in pixels)
    
    Return:
    ------
    FQPM: 2D array, four quadrant phase mask coronagraph
    -------------------------------------------------- '''
    phase= np.zeros((isz,isz))
    for i in np.arange(isz):
        for j in np.arange(isz):
            if(i<isz/2 and j<isz/2): phase[i,j]=np.pi
            if(i>=isz/2 and j>=isz/2): phase[i,j]=np.pi
        
    FQPM=np.exp(1j*phase)
    return FQPM
 
def KnifeEdgeCoro(isz,position,shiftinldp,ld_p):
    ''' --------------------------------------------------
    Create a Knife edge coronagraph of size (isz,isz)
    
    Parameters:
    ----------
    isz: int, size of the coronagraph (in pixels)
    position: string, can be 'left', 'right', 'top' or 'bottom' to define the orientation of the coronagraph
    shiftinldp: Position of the edge, with respect to the image center, in number of pixels per resolution element
    ld_p: Number of pixels per resolution element
    
    Return:
    ------
    shift(Knife): 2D array, Knife edge coronagraph, located at the four edges of the image
    -------------------------------------------------- '''
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
    
def roundpupil(isz,prad1):
    ''' --------------------------------------------------
    Create a circular pupil. The center of the pupil is located between 4 pixels.
    
    Parameters:
    ----------
    isz: int, size of the image (in pixels)
    prad1: float, size of the pupil radius (in pixels)
    
    Return:
    ------
    pupilnormal: output circular pupil
    -------------------------------------------------- '''
    xx, yy = np.meshgrid(np.arange(isz)-(isz)/2, np.arange(isz)-(isz)/2)
    rr     = np.hypot(yy+1/2, xx+1/2)
    pupilnormal=np.zeros((isz,isz))
    pupilnormal[rr<=prad1]=1.
    return pupilnormal
    
    
    
def pupiltodetector(input_wavefront,coro_mask,lyot_mask,perfect_coro=False,perfect_entrance_pupil=0): #aberrationphase,prad1,prad2
    ''' --------------------------------------------------
    Propagate a wavefront in a pupil plane through a high-contrast imaging instrument, until the science detector.
    The image is then cropped and resampled.
    
    Parameters:
    ----------
    input_wavefront: 2D array, can be complex. Input wavefront
    coro_mask: 2D array, can be complex. coronagraphic mask
    lyot_mask: 2D array, lyot mask
    perfect_coro: bool, set if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    perfect_entrance_pupil: 2D array, entrance pupil which should be nulled by the used coronagraph
    
    Return:
    ------
    shift(sqrtimage): 2D array, focal plane electric field created by the input wavefront through the HCI instrument.
    -------------------------------------------------- '''
    isz=len(input_wavefront)
    masktot=translationFFT(isz,0.5,0.5)
    #Focale1 
    focal1end=shift(input_wavefront*masktot)
    if perfect_coro==True:
        focal1end=focal1end-shift(perfect_entrance_pupil*masktot)
    
    focal1end=fft(focal1end)
    
    #Pupille2
    pupil2end=focal1end*coro_mask
    pupil2end=ifft(pupil2end)#/shift(masktot)
    
    #Intensité en sortie de SCC
    focal2end=pupil2end*shift(lyot_mask)
    focal2end=fft(focal2end)
    focal2end=shift(focal2end)
    
    return focal2end
    
    
    
def pushact_function(which,grilleact, actshapeinpupilresized,xycent,xy309,modelerror='NoError',error=0):
    ''' --------------------------------------------------
    Push the desired DM actuator in the pupil
    
    Parameters:
    ----------
    which: int, index of the individual actuator to push
    grilleact: 2D array, x and y position of all the DM actuator in the pupil
    actshapeinpupilresized: well-sampled actuator to be translated to its true position
    xycent: numpy array, position of the actuator in actshapeinpupilresized before translation
    xy309: numpy array, position of the actuator 309 in pixels
    modelerror: string, can be 'NoError', 'translationxy', 'translationx', 'translationy', 'rotation', 'influence_function'
    error: size of the error in pixels or in degrees
    
    Return:
    ------
    Psivector: 2D array, pupil plane phase with the opd created by the poke of the desired actuator
    -------------------------------------------------- '''
    isz=len(actshapeinpupilresized)
    x309=xy309[0]
    y309=xy309[1]
    xact=(grilleact[0,which]+(x309-grilleact[0,309]))
    yact=(grilleact[1,which]+(y309-grilleact[1,309]))
    
    if modelerror=='NoError':
        Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent,xact-xycent))
    if modelerror=='translationxy':
        Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent+np.sqrt(error**2/2),xact-xycent+np.sqrt(error**2/2)))
    if modelerror=='translationx':
        Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent,xact-xycent+error))
    if modelerror=='translationy':
        Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent+error,xact-xycent))
    if modelerror=='rotation':
        Psivector=proc.rotate_frame(Psivector,devx, interpolation = 'nearest', cyx=None)
    if modelerror=='influence_function':
        Psivector=nd.interpolation.shift(actshapeinpupilresized,(yact-xycent,xact-xycent))
        x, y = np.mgrid[0:isz, 0:isz]
        xy=(x,y)
        xo,yo=np.unravel_index(Psivector.argmax(), Psivector.shape)
        Psivector=twoD_Gaussian(xy, 1, 1+devx,1+devx , xo,yo, 0,0)
        
    Psivector[np.where(Psivector<1e-4)]=0
    
    return Psivector
    
    
def creatingpushact(model_dir,file309,x309,y309,findxy309byhand,isz,prad,pdiam,modelerror='NoError',error=0):
    ''' --------------------------------------------------
    Push the desired DM actuator in the pupil
    
    Parameters:
    ----------
    model_dir:
    file309:
    x309:
    y309:
    findxy309byhand:
    modelerror:
    error:
    
    Return:
    ------
    pushact: 
    -------------------------------------------------- '''
    # It may not work at the moment
    if findxy309byhand==False:
        file309='Phase_estim_Act309_v20200107.fits'
        im309size = len(fi.LoadImageFits(model_dir+file309))
        act309 = np.zeros((isz,isz))
        act309[int(isz/2-im309size/2):int(isz/2+im309size/2),int(isz/2-im309size/2):int(isz/2+im309size/2)] = fi.LoadImageFits(model_dir+file309)
        y309,x309 = np.unravel_index(abs(act309).argmax(), act309.shape)
    
    #shift by (0.5,0.5) pixel because the pupil is centerd between pixels
    y309=y309-0.5
    x309=x309-0.5
    xy309=[x309,y309]
    
    
    grille = fi.LoadImageFits(model_dir+'Grid_actu.fits')
    actshape = fi.LoadImageFits(model_dir+'Actu_DM32_field=6x6pitch_pitch=22pix.fits')
    resizeactshape = skimage.transform.rescale(actshape, 2*prad/pdiam*.3e-3/22, order=1,preserve_range=True,anti_aliasing=True,multichannel=False)
    
    #Gauss2Dfit for centering the rescaled influence function
    dx,dy = proc.gauss2Dfit(resizeactshape)
    xycent = len(resizeactshape)/2
    resizeactshape=nd.interpolation.shift(resizeactshape,(xycent-dx,xycent-dy))
    
    #Put the centered influence function inside a larger array (400x400)
    actshapeinpupil = np.zeros((isz,isz))
    actshapeinpupil[0:len(resizeactshape),0:len(resizeactshape)] = resizeactshape/np.amax(resizeactshape)

    pushact=np.zeros((1024,isz,isz))
    for i in np.arange(1024):
        pushact[i]=pushact_function(i,grille,actshapeinpupil,xycent,xy309,modelerror,error)
    return pushact
    
    
    
    
def createdifference(aberramp,aberrphase,posprobes,pushact,amplitude,entrancepupil,coro_mask,lyot_mask,PSF,dimimages,wavelength,perfect_coro=False,perfect_entrance_pupil=0,noise=False,numphot=1e30):
    ''' --------------------------------------------------
    Simulate the acquisition of probe images (actuator pokes) and create their differences
    
    Parameters:
    ----------
    aberramp: 0 or 2D-array, upstream amplitude aberration
    aberrphase: 0 or 2D-array, upstream phase aberration
    posprobes: 1D-array, index of the actuators to push and pull for pair-wise probing
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    amplitude: float, amplitude of the actuator pokes for pair(wise probing in nm
    entrancepupil: 2D-array, entrance pupil shape
    coro_mask: 2D array, can be complex. coronagraphic mask
    lyot_mask: 2D array, lyot mask
    dimimages: int, size of the output image after resampling in pixels
    wavelength: float, wavelength of the  incoming flux in meter
    perfect_coro: bool, set if you want sqrtimage to be 0 when input_wavefront==perfect_entrance_pupil
    perfect_entrance_pupil: 2D array, entrance pupil which should be nulled by the used coronagraph
    noise: boolean, if True, add photon noise. WARNING, the function shotnoise has to be rewritten before use
    numphot: int, number of photons entering the pupil
    
    Return:
    ------
    Difference: 3D array, cube with image difference for each probes. Use for pair-wise probing
    -------------------------------------------------- '''
    isz=len(entrancepupil)
    Ikmoins=np.zeros((isz,isz))
    Ikplus=np.zeros((isz,isz))
    Difference=np.zeros((len(posprobes),dimimages,dimimages))
    
    squaremaxPSF=np.amax(PSF)
    
    contrast_to_photons=np.sum(entrancepupil)/np.sum(lyot_mask)*numphot*squaremaxPSF**2/np.sum(PSF)**2
    
    k=0
    for i in posprobes:
        probephase=amplitude*pushact[i]
        probephase=2*np.pi*probephase*1e-9/wavelength
        input_wavefront=entrancepupil*(1+aberramp)*np.exp(1j*(aberrphase-1*probephase))
        Ikmoins=np.abs(pupiltodetector(input_wavefront,coro_mask,lyot_mask,perfect_coro,perfect_entrance_pupil))**2/squaremaxPSF**2 
        input_wavefront=entrancepupil*(1+aberramp)*np.exp(1j*(aberrphase+1*probephase))
        Ikplus=np.abs(pupiltodetector(input_wavefront,coro_mask,lyot_mask,perfect_coro,perfect_entrance_pupil))**2/squaremaxPSF**2

        if noise==True:
            Ikplus=np.random.poisson(Ikplus*contrast_to_photons)/contrast_to_photons
            Ikmoins=np.random.poisson(Ikmoins*contrast_to_photons)/contrast_to_photons
            
        Ikplus=abs(proc.resampling(Ikplus,dimimages))
        Ikmoins=abs(proc.resampling(Ikmoins,dimimages))
        #print(np.sum(contrast_to_photons*Ikplus*1e-9),np.sum(entrancepupil)/np.sum(lyot_mask))

        Difference[k]=(Ikplus-Ikmoins)
        k=k+1

    return Difference




def random_phase_map(isz,phaserms,rhoc,slope):
    ''' --------------------------------------------------
    Create a random phase map, whose PSD decrease in f^(-slope)
    
    Parameters:
    ----------
    isz: integer, size of the generated phase map
    phaserms: float, level of aberration
    rhoc: float, see Bordé et Traub 2006
    slope: float, slope of the PSD
    
    Return:
    ------
    phase: 2D array, static random phase map (or OPD) generated 
    -------------------------------------------------- '''
    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rho=np.hypot(yy, xx)
    PSD0=1
    PSD=PSD0/(1+(rho/rhoc)**slope)
    sqrtPSD=np.sqrt(2*PSD)
    randomphase=2*np.pi * (np.random.rand(isz, isz) - 0.5)
    product=shift(sqrtPSD*np.exp(1j*randomphase))
    phase=np.real(ifft(product))
    phase=phase/np.std(phase)*phaserms
    return phase

















    
    
