import os
import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
import skimage.transform

import Asterix.propagation_functions as prop
import Asterix.processing_functions as proc
import Asterix.phase_amplitude_functions as phase_ampl
import Asterix.fits_functions as useful


##############################################
##############################################
### Optical_System
class Optical_System:
    """ --------------------------------------------------
    Super class Optical_System allows to pass parameters to all sub class. We can then creat blocks inside this super class
    An Optical_System start and end in the pupil plane. 
    The entrance and exit pupil plane must always of the same size (dim_overpad_pupil)    
    With these convention, they can be easily assemble to create complex optical systems. 
    
    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, modelconfig):
        """ --------------------------------------------------
        Initialize Optical_System objects 
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        #pupil in pixel
        self.prad = int(modelconfig["diam_pup_in_pix"] / 2)

        # 1.25 is hard coded for now. TODO Fix that.
        # All pupils in the code must have this dimensions, so that the blocks can be easily switch
        self.dim_overpad_pupil = int(self.prad * 1.25) * 2

        #Lambda over D in pixels in the focal plane
        # at the reference wavelength
        #image size and resolution in detector
        self.dim_im = modelconfig["dim_im"]
        self.science_sampling = modelconfig["science_sampling"]

        #pupil in meters
        self.diam_pup_in_m = modelconfig["diam_pup_in_m"]

        # Exit pupil radius
        self.exitpup_rad = self.prad
        # this is the exit pupil radius, that is used to define the L/D
        # in self.todetector function.
        # by default this is the entrance pupil rad. of course, this can be changed

        # wavelength
        self.Delta_wav = modelconfig["Delta_wav"]
        self.nb_wav = modelconfig["nb_wav"]
        self.wavelength_0 = modelconfig["wavelength_0"]

        if self.Delta_wav != 0:
            if (self.nb_wav % 2 == 0) or self.nb_wav < 2:
                raise Exception(
                    "please set nb_wav parameter to an odd number > 1")

            self.wav_vec = np.linspace(self.wavelength_0 - self.Delta_wav / 2,
                                       self.wavelength_0 + self.Delta_wav / 2,
                                       num=self.nb_wav,
                                       endpoint=True)
        else:
            self.wav_vec = np.array([self.wavelength_0])

    #We define functions that all Optical_System object can use.
    # These can be overwritten for a subclass if need be

    def EF_through(self, entrance_EF=1., **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance pupil to exit pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 

        **kwargs: other kw parameters can be passed

        NEED TO BE DEFINED FOR ALL Optical_System

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if isinstance(entrance_EF, float) or isinstance(entrance_EF, np.float):
            entrance_EF = np.full(
                (self.dim_overpad_pupil, self.dim_overpad_pupil),
                np.float(entrance_EF))

        if isinstance(entrance_EF, np.ndarray) == False:
            raise Exception(
                "entrance_EF should be a float of a numpy array of floats")

        exit_EF = entrance_EF
        return exit_EF

    def todetector(self,
                   entrance_EF=1.,
                   wavelength=None,
                   center_on_pixel=False,
                   **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system and then
        to Science focal plane.
        TODO can be made polychromatic but maybe not because people 
        will do bad things with it like the intensity of summed EF :-)
        
        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF 
        or during self.EF_through

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        ef_focal_plane : 2D array of size [self.dim_im, self.dim_im]
        Electric field in the focal plane.
            the lambda / D is defined such as self.wavelength_0 /  (2*self.exitpup_rad) = self.science_sampling pixels
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if center_on_pixel == True:
            Psf_offset = (0, 0)
        else:
            Psf_offset = (-0.5, -0.5)

        if wavelength == None:
            wavelength = self.wavelength_0

        lambda_ratio = wavelength / self.wavelength_0

        exit_EF = self.EF_through(entrance_EF=entrance_EF,
                                  wavelength=wavelength,
                                  **kwargs)

        focal_plane_EF = prop.mft(exit_EF,
                                  self.exitpup_rad * 2,
                                  self.dim_im,
                                  self.dim_im / self.science_sampling *
                                  lambda_ratio,
                                  xshift=Psf_offset[0],
                                  yshift=Psf_offset[1],
                                  inv=1)

        return focal_plane_EF

    def todetector_Intensity(self,
                             entrance_EF=1.,
                             wavelengths=None,
                             center_on_pixel=False,
                             **kwargs):
        """ --------------------------------------------------
        Propagate the electric field from entrance plane through the system, then
        to Science focal plane and measure intensity

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelengths : float or float array of wavelength in m. Default is all wavelenthg in self.wav_vec

        center_on_pixel : bool Default False
            If True, the PSF will be centered on a pixel
            If False, the PSF will be centered between 4 pixels
        This of course assume that no tip-tilt have been introduced in the entrance_EF 
        or during self.EF_through

        **kwargs: other kw parameters can be passed direclty to self.EF_through function

        Returns
        ------
        focal_plane_Intensity : 2D array of size [self.dim_im, self.dim_im]
        Intensity in the focal plane.
            the lambda / D is defined such as self.wavelength_0 /  (2*self.exitpup_rad) = self.science_sampling pixels
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        if wavelengths == None:
            wavelength_vec = self.wav_vec

        elif isinstance(wavelengths, float) or isinstance(
                wavelengths, np.float):
            wavelength_vec = [wavelengths]
        else:
            wavelength_vec = wavelengths
        #TODO check if wavelengths have good format float array and raise
        # exception if not

        focal_plane_Intensity = np.zeros((self.dim_im, self.dim_im))

        for wav in wavelength_vec:
            focal_plane_Intensity += np.abs(
                self.todetector(entrance_EF=entrance_EF,
                                wavelength=wav,
                                center_on_pixel=center_on_pixel,
                                **kwargs))**2

        return focal_plane_Intensity

    def transmission(self, **kwargs):
        """
        measure ratio of photons lost when
        crossing the system compared to a clear aperture of radius self.prad

        **kwargs: other kw parameters can be passed direclty to self.EF_through function
        
        Returns
        ------ 
        float is the ratio exit flux  / clear entrance pupil flux
    
        AUTHOR : Johan Mazoyer

        """
        clear_entrance_pupil = phase_ampl.roundpupil(self.dim_overpad_pupil,
                                                     self.prad)

        exit_EF = self.EF_through(**kwargs)

        throughput_loss = np.sum(
            np.abs(exit_EF)) / np.sum(clear_entrance_pupil)

        return throughput_loss

    def EF_from_phase_and_ampl(self,
                               phase_abb=0.,
                               ampl_abb=0.,
                               wavelength=None):
        """ --------------------------------------------------
        Create an electrical field from an phase and amplitude aberrations
        
        Parameters
        ----------
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no phase aberration (default)  
        
        phase_abb : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil]. real
            if 0, no amplitude aberration (default)  
            

        wavelength : float. Default is the reference self.wavelength_0
             current wavelength in m. 


        Returns
        ------
        EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system
        
        AUTHOR : Johan Mazoyer
        """
        if np.iscomplexobj(phase_abb) or np.iscomplexobj(ampl_abb):
            raise Exception(
                "phase_abb and ampl_abb must be real arrays, not complex")

        if (phase_abb == 0.).all() and (ampl_abb == 0).all():
            return 1.
        else:
            if wavelength == None:
                wavelength = self.wavelength_0
            lambda_ratio = wavelength / self.wavelength_0

            return (1 + ampl_abb) * np.exp(1j * phase_abb / lambda_ratio)


##############################################
##############################################
### PUPIL
class pupil(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of single pupil
    pupil is a sub class of Optical_System.  
    
    Obviously you can define your pupil
    withot that with 2d arrray multiplication (this is a fairly simple object).
    
    The main advantage of defining them using Optical_System is that you can 
    use default Optical_System functions to obtain PSF, transmission, etc... 
    and concatenate them with other elements
    
    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 prad,
                 directory="",
                 filename="",
                 noPup=False):
        """ --------------------------------------------------
        Initialize a pupil object.


        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions) 
                        to initialize Optical_System class

        prad : int
            radius in pixels of the round pupil mask
        
        directory : string (default "")
            name of the directory where filename is

        filename : string (default "")
            name of the .fits file
            The pupil .fits files should be be 2D and square([dim_fits,dim_fits]) 
            and assumed to be centered between 4 pixels.
            if dim_fits < dim_overpad_pupil then the pupil is zero-padded
            if dim_fits > dim_overpad_pupil we raise an Exception

            TODO: include here function random_phase_map, scale_amplitude_abb, shift_phase_ramp 
            TODO: include an SCC Lyot pupil function here !
            TODO: for now pupil .fits are monochromatic but the pupil propagation EF_through 
            use wavelenght as a parameter

        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.exitpup_rad = prad
        self.radius = prad

        self.pup = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                           1.)

        if filename != "":

            # we start by a bunch of tests to check
            # that pupil has a certain acceptable form.
            print("we load the pupil: " + filename)
            print("we assume center between 4 pixels")
            pup_fits = fits.getdata(os.path.join(directory, filename))

            if len(pup_fits.shape) != 2:
                raise Exception("file " + filename + " should be a 2D array")

            if pup_fits.shape[0] != pup_fits.shape[1]:
                raise Exception("file " + filename +
                                " appears to be not square")

            if pup_fits.shape[0] == self.dim_overpad_pupil:
                self.pup = pup_fits

            elif pup_fits.shape[0] > self.dim_overpad_pupil:
                raise Exception(
                    "file " + filename +
                    " size ({} pix)  is larger".format(pup_fits.shape[0]) +
                    "than the expected size of pupil size arrays ({} pix)".
                    format(self.dim_overpad_pupil))
            else:
                print("we pad the pupil to be at the correct size")
                self.pup = proc.crop_or_pad_image(pup_fits,
                                                  self.dim_overpad_pupil)
        else:  # no filename
            if noPup == False:
                self.pup = phase_ampl.roundpupil(self.dim_overpad_pupil, prad)

    def EF_through(self, entrance_EF=1., wavelength=None):
        """ --------------------------------------------------
        Propagate the electric field through the pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if len(self.pup.shape) == 2:
            return entrance_EF * self.pup

        elif len(self.pup.shape) == 3:
            if self.pup.shape != self.nb_wav:
                raise Exception(
                    "I'm confused, your pupil seem to be polychromatic" +
                    "(pup.shape=3) but the # of WL (pup.shape[0]={}) ".format(
                        self.pup.shape[0]) +
                    "is different from the system # of WL (nb_wav={})".format(
                        self.nb_wav))
            else:
                return entrance_EF * self.pup[self.wav_vec.tolist().index(
                    wavelength)]
        else:
            raise Exception("pupil dimension are not acceptable")


##############################################
##############################################
### CORONAGRAPHS
class coronagraph(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a coronagraph system (from apod plane to the Lyot plane)
    coronagraph is a sub class of Optical_System.  
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self, modelconfig, coroconfig, model_dir):
        """ --------------------------------------------------
        Initialize a coronograph object
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        coroconfig : coronagraph parameters
        model_dir : if needed, we load the mask in this directory
        
        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        #pupil and Lyot stop in m
        self.diam_lyot_in_m = modelconfig["diam_lyot_in_m"]

        #Lyot stop in pixel
        self.lyotrad = int(self.prad * self.diam_lyot_in_m /
                           self.diam_pup_in_m)

        self.exitpup_rad = self.lyotrad

        #coronagraph
        self.corona_type = coroconfig["corona_type"].lower()

        if self.corona_type == "fqpm" or self.corona_type == "knife":
            # dim_fp_fft definition only use if prop_apod2lyot == 'fft'
            self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
            for i, wav in enumerate(self.wav_vec):
                self.dim_fp_fft[i] = int(
                    np.ceil(self.prad * self.science_sampling *
                            self.diam_lyot_in_m / self.diam_pup_in_m *
                            self.wavelength_0 / wav)) * 2
                # we take the ceil to be sure that we measure at least the good resolution
                # We do not need to be exact, the mft in science_focal_plane will be

        if self.corona_type == "fqpm":
            self.prop_apod2lyot = 'mft'
            self.err_fqpm = coroconfig["err_fqpm"]
            self.achrom_fqpm = coroconfig["achrom_fqpm"]
            self.FPmsk = self.FQPM()
            self.perfect_coro = True

        elif self.corona_type == "classiclyot":
            self.prop_apod2lyot = 'mft-babinet'
            self.Lyot_fpm_sampling = 30  # hard coded for now, this is very internal cooking
            self.rad_lyot_fpm = coroconfig["rad_lyot_fpm"]
            self.FPmsk = self.ClassicalLyot()
            self.perfect_coro = False

        elif self.corona_type == "knife":
            self.prop_apod2lyot = 'fft'
            self.coro_position = coroconfig["coro_position"].lower()
            self.knife_coro_offset = coroconfig["knife_coro_offset"]
            self.FPmsk = self.KnifeEdgeCoro()
            self.perfect_coro = False

        elif self.corona_type == "vortex":
            self.prop_apod2lyot = 'fft'
            phasevortex = 0  # to be defined
            self.FPmsk = np.exp(1j * phasevortex)
            self.perfect_coro = True

        # We need a pupil only to measure the response
        # of the coronograph to a clear pupil to remove it
        # if perfect corono. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        self.clearpup = pupil(modelconfig, self.prad)

        # Plane at the entrance of the coronagraph. In THD2, this is an empty plane.
        # In Roman this is where is the apodiser
        self.apod_pup = pupil(modelconfig,
                              self.prad,
                              directory=model_dir,
                              filename=modelconfig["filename_instr_apod"],
                              noPup=True)

        self.lyot_pup = pupil(modelconfig,
                              self.lyotrad,
                              directory=model_dir,
                              filename=modelconfig["filename_instr_lyot"])

        if self.perfect_coro == True:
            # do a propagation once with self.perfect_Lyot_pupil = 0 to
            # measure the Lyot pupil that will be removed after
            self.perfect_Lyot_pupil = 0
            self.perfect_Lyot_pupil = self.EF_through(
                entrance_EF=self.clearpup.EF_through())

    def FQPM(self):
        """ --------------------------------------------------
        Create a Four Quadrant Phase Mask coronagraph 
        

        Returns
        ------
        FQPM : list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            FQPM mask
        
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
        -------------------------------------------------- """

        if self.prop_apod2lyot == "fft":
            maxdimension_array_fpm = np.max(self.dim_fp_fft)
        else:
            maxdimension_array_fpm = self.dim_im
            print("you should really not simulate FQPM with MFT")

            self.dim_fp_fft = np.zeros(len(self.wav_vec), dtype=np.int)
            for i, wav in enumerate(self.wav_vec):
                self.dim_fp_fft[i] = int(
                    np.ceil(self.prad * self.science_sampling *
                            self.diam_lyot_in_m / self.diam_pup_in_m *
                            self.wavelength_0 / wav)) * 2

        xx, yy = np.meshgrid(
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2,
            np.arange(maxdimension_array_fpm) - (maxdimension_array_fpm) / 2)

        fqpm_thick_vert = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_vert[np.where(xx < 0)] = 1
        fqpm_thick_hor = np.zeros(
            (maxdimension_array_fpm, maxdimension_array_fpm))
        fqpm_thick_hor[np.where(yy >= 0)] = 1
        fqpm_thick = fqpm_thick_vert - fqpm_thick_hor

        fqpm = list()
        for i, wav in enumerate(self.wav_vec):
            if self.prop_apod2lyot == "fft":
                dim_fp = self.dim_fp_fft[i]
            else:
                dim_fp = self.dim_im
                print("really you should not do that")

            phase = np.zeros((dim_fp, dim_fp))
            fqpm_thick_cut = proc.crop_or_pad_image(fqpm_thick, dim_fp)
            phase[np.where(fqpm_thick_cut != 0)] = (np.pi + self.err_fqpm)
            if self.achrom_fqpm == False:
                phase = phase * self.wavelength_0 / wav

            fqpm.append(np.exp(1j * phase))

        return fqpm

    def KnifeEdgeCoro(self):
        """ --------------------------------------------------
        Create a Knife edge coronagraph of size (dim_im,dim_im)
    
        Returns
        ------
        shift(Knife) :list of len(self.wav_vec) 2D arrays giving the complex transmission of the
            Knife edge coronagraph mask. TODO A CHECKER YA PTET UN SOUCIS DE SHIFT
        
        AUTHOR : Axel Potier
        Modified by Johan Mazoyer
        -------------------------------------------------- """

        if len(self.wav_vec) == 1:
            raise Exception(
                "KnifeEdgeCoro only working in monochromatic as of now")
        maxdimension_array_fpm = np.max(self.dim_fp_fft)

        # self.coro_position can be 'left', 'right', 'top' or 'bottom'
        # to define the orientation of the coronagraph

        #  Number of pixels per resolution element at central wavelength
        ld_p_0 = self.science_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        xx, yy = np.meshgrid(np.arange(maxdimension_array_fpm),
                             np.arange(maxdimension_array_fpm))

        Knife = np.zeros((maxdimension_array_fpm, maxdimension_array_fpm))
        if self.coro_position == "right":
            Knife[np.where(xx > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "left":
            Knife[np.where(xx < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "bottom":
            Knife[np.where(yy > (maxdimension_array_fpm / 2 +
                                 self.knife_coro_offset * ld_p_0))] = 1
        if self.coro_position == "top":
            Knife[np.where(yy < (maxdimension_array_fpm / 2 -
                                 self.knife_coro_offset * ld_p_0))] = 1

        return [Knife]

    def ClassicalLyot(self):
        """ --------------------------------------------------
        Create a classical Lyot coronagraph of radius rad_LyotFP 0

        rad_LyotFP : int, radius of the Lyot focal plane
    
        Returns
        ------
         classical Lyot : 2D array

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        ld_p = self.Lyot_fpm_sampling * self.diam_lyot_in_m / self.diam_pup_in_m

        rad_LyotFP_pix = self.rad_lyot_fpm * ld_p

        self.dim_fpm = 2 * int(rad_LyotFP_pix)
        ClassicalLyotstop = 1 - phase_ampl.roundpupil(self.dim_fpm,
                                                      rad_LyotFP_pix)

        return [ClassicalLyotstop]

    ##############################################
    ##############################################
    ### Propagation through coronagraph

    def EF_through(self, entrance_EF=1., wavelength=None, noFPM=False):
        """ --------------------------------------------------
        Propagate the electric field from apod plane before the apod
        pupil to Lyot plane after Lyot pupil

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF


        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if noFPM:
            FPmsk = 1.
        else:
            FPmsk = self.FPmsk[self.wav_vec.tolist().index(wavelength)]

        lambda_ratio = wavelength / self.wavelength_0

        input_wavefront_after_apod = self.apod_pup.EF_through(
            entrance_EF=entrance_EF, wavelength=wavelength)

        if self.prop_apod2lyot == "fft":
            dim_fp_fft_here = self.dim_fp_fft[self.wav_vec.tolist().index(
                wavelength)]
            input_wavefront_pad = proc.crop_or_pad_image(
                entrance_EF, dim_fp_fft_here)
            # Phase ramp to center focal plane between 4 pixels

            maskshifthalfpix = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, 0.5, 0.5)
            maskshifthalfpix_invert = phase_ampl.shift_phase_ramp(
                dim_fp_fft_here, -0.5, -0.5)

            #Apod plane to focal plane
            corono_focal_plane = np.fft.fft2(
                np.fft.fftshift(input_wavefront_pad * maskshifthalfpix))

            # Focal plane to Lyot plane
            lyotplane_before_lyot = np.fft.fftshift(
                np.fft.ifft2(
                    corono_focal_plane * FPmsk)) * maskshifthalfpix_invert
            # we take the convention that when we are in pupil plane the field must be
            # "non-shifted". Because the shift in pupil plane is resolution dependent
            # which depend on the method (fft is not exactly science resolution because
            # of rounding issues, mft-babinet does not use the science resolution, etc).
            # these shift in both direction should be included in apod and pup multiplication
            # to save time

        elif self.prop_apod2lyot == "mft-babinet":
            #Apod plane to focal plane

            # in babinet, we have to shift also the initial pupil.
            # this is a bit absurd but this is the only way if we want
            # focal plane to be always centered between 4 pixels
            # if we only centered duing the PSF then in the subtraction in babinet's trick
            # we introduce a shift. Also this is not a perfect shift
            # We need to code a anti-shift for mft-1 !

            maskshifthalfpix_fpm = phase_ampl.shift_phase_ramp(
                self.dim_overpad_pupil,
                0.5 / self.Lyot_fpm_sampling * lambda_ratio,
                0.5 / self.Lyot_fpm_sampling * lambda_ratio)

            maskshifthalfpix_fpm_inverse = phase_ampl.shift_phase_ramp(
                self.dim_overpad_pupil,
                -0.5 / self.Lyot_fpm_sampling * lambda_ratio,
                -0.5 / self.Lyot_fpm_sampling * lambda_ratio)

            input_wavefront_after_apod_shifted = input_wavefront_after_apod * maskshifthalfpix_fpm

            corono_focal_plane = prop.mft(
                input_wavefront_after_apod_shifted,
                self.dim_overpad_pupil,
                self.dim_fpm,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                inv=1)

            # Focal plane to Lyot plane
            lyotplane_before_lyot_central_part = prop.mft(
                corono_focal_plane * (1 - FPmsk),
                self.dim_fpm,
                self.dim_overpad_pupil,
                self.dim_fpm / self.Lyot_fpm_sampling * lambda_ratio,
                inv=-1)

            # Babinet's trick
            lyotplane_before_lyot = (input_wavefront_after_apod_shifted -
                                     lyotplane_before_lyot_central_part
                                     ) * maskshifthalfpix_fpm_inverse
            # this is ugly as sh*t but it works to be coherent with other convention in the code

        elif self.prop_apod2lyot == "mft":
            # Apod plane to focal plane
            # We need to code a anti-shift in mft-1 !

            maskshifthalfpix_fpm_inverse = phase_ampl.shift_phase_ramp(
                self.dim_overpad_pupil,
                -0.5 / self.science_sampling * lambda_ratio,
                -0.5 / self.science_sampling * lambda_ratio)

            corono_focal_plane = prop.mft(input_wavefront_after_apod,
                                          2 * self.prad,
                                          self.dim_im,
                                          self.dim_im / self.science_sampling *
                                          lambda_ratio,
                                          xshift=-0.5,
                                          yshift=-0.5,
                                          inv=1)

            # Focal plane to Lyot plane
            lyotplane_before_lyot = proc.crop_or_pad_image(
                prop.mft(corono_focal_plane * FPmsk,
                         self.dim_im,
                         2 * self.prad,
                         self.dim_im / self.science_sampling * lambda_ratio,
                         inv=-1),
                self.dim_overpad_pupil) * maskshifthalfpix_fpm_inverse

        else:
            raise Exception(
                self.prop_apod2lyot +
                " is not a known prop_apod2lyot propagation mehtod")

        # crop to the dim_overpad_pupil expeted size
        lyotplane_before_lyot_crop = proc.crop_or_pad_image(
            lyotplane_before_lyot, self.dim_overpad_pupil)

        # Field after filtering by Lyot stop
        lyotplane_after_lyot = self.lyot_pup.EF_through(
            entrance_EF=lyotplane_before_lyot_crop, wavelength=wavelength)

        if (self.perfect_coro == True) & (noFPM == False):
            lyotplane_after_lyot = lyotplane_after_lyot - self.perfect_Lyot_pupil

        return lyotplane_after_lyot


##############################################
##############################################
### Deformable mirror
class deformable_mirror(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of a deformable mirror 
    (in pupil plane or out of pupil plane)
    coronagraph is a sub class of Optical_System.  
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 DMconfig,
                 Name_DM='DM3',
                 Measure_and_save=True,
                 model_dir='',
                 Model_local_dir=''):
        """ --------------------------------------------------
        Initialize a deformable mirror object
        TODO handle misregistration that is currently not working 
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        DMconfig : DM parameters
        Name_DM : the name of the DM, which allows to find it in the parameter file
        Measure_and_save : bool, default = True if true, we measure and save the pushact functions
        model_dir: directory to find Measured positions for each actuator in pixel and 
                    influence fun. ie Things you cannot measure yourself and need to be given
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.exitpup_rad = self.prad

        #radius of the pupil in pixel in the DM plane.
        # by default the size of the pupil, changed if z_position != 0
        self.pradDM = self.prad

        self.z_position = DMconfig[Name_DM + "_z_position"]
        self.active = DMconfig[Name_DM + "_active"]
        self.MinimumSurfaceRatioInThePupil = DMconfig[
            "MinimumSurfaceRatioInThePupil"]

        if self.active == False:
            return

        # We need a pupil in creatingpushact_inpup() and for
        # which in pup. THIS IS NOT THE ENTRANCE PUPIL,
        # this is a clear pupil of the same size
        self.clearpup = pupil(modelconfig, self.prad)

        #define self.pradDM and check that the pupil is large enough
        # for out of pupil propagation
        if self.z_position != 0:
            dx, dxout = prop.prop_fresnel(self.dim_overpad_pupil,
                                          self.wavelength_0,
                                          self.z_position,
                                          self.diam_pup_in_m / 2,
                                          self.prad,
                                          retscale=1)
            #radius of the pupil in pixel in the DM plane
            self.pradDM = self.prad * dx / dxout

            if dx > 2 * dxout:
                print(dx, dxout)
                raise Exception(
                    "Need to enhance the pupil size in pixel for Fresnel propagation"
                )

        Name_pushact = Name_DM + "_PushActInPup_ray" + str(int(
            self.pradDM)) + "_dimpuparray" + str(int(self.dim_overpad_pupil))

        Name_WhichInPup = Name_DM + "_Whichactfor" + str(
            self.MinimumSurfaceRatioInThePupil) + '_raypup' + str(self.prad)

        if Measure_and_save == True:
            # in this case the pushact inpup are
            # measured and saved

            # DM_pushact is always in the DM plane
            self.DM_pushact = self.creatingpushact(model_dir,
                                                   DMconfig,
                                                   Name_DM=Name_DM)
            fits.writeto(Model_local_dir + Name_pushact + '.fits',
                         self.DM_pushact,
                         overwrite=True)

            if self.z_position != 0:
                # DM_pushact_inpup is always in the pupil plane
                self.DM_pushact_inpup = self.creatingpushact_inpup()

                fits.writeto(Model_local_dir + Name_pushact +
                             '_inPup_real.fits',
                             np.real(self.DM_pushact_inpup),
                             overwrite=True)
                fits.writeto(Model_local_dir + Name_pushact +
                             '_inPup_imag.fits',
                             np.imag(self.DM_pushact_inpup),
                             overwrite=True)

                self.DM_pushact_inpup = self.creatingpushact_inpup()

            else:
                # if the DM is in the pupil plane
                self.DM_pushact_inpup = self.DM_pushact
                # This is a repetition but coherent and
                # allows you to easily concatenate wherever are the DMs

            self.WhichInPupil = creatingWhichinPupil(
                self.DM_pushact_inpup, self.clearpup.pup,
                self.MinimumSurfaceRatioInThePupil)

            fits.writeto(Model_local_dir + Name_WhichInPup + '.fits',
                         self.WhichInPupil,
                         overwrite=True)

        else:
            # in this case the pushact inpup and WhichInPup are loaded

            self.WhichInPupil = useful.check_and_load_fits(
                Model_local_dir, Name_WhichInPup)

            self.DM_pushact = useful.check_and_load_fits(
                Model_local_dir, Name_pushact)

            if self.z_position != 0:
                DM_pushact_inpup_real = useful.check_and_load_fits(
                    Model_local_dir, Name_pushact + '_inPup_real')
                DM_pushact_inpup_imag = useful.check_and_load_fits(
                    Model_local_dir, Name_pushact + '_inPup_imag')

                self.DM_pushact_inpup = DM_pushact_inpup_real + 1j * DM_pushact_inpup_imag

    def EF_through(self, entrance_EF=1., wavelength=None, DMphase=0.):
        """ --------------------------------------------------
        Propagate the electric field through the DM.
        if z_DM = 0, then it's just a phase multiplication
        if z_DM != 0, this is where we do the fresnel

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        DMphase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                    Can also be a float scalar in which case DM_phase is constant
                    default is 0. 
                    
        TODO: Maybe we can put the electrical field directly ? 
        TODO: do all the annoying saving and loading of fits in the initialization here


        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system

        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check
        # and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if isinstance(DMphase, float) or isinstance(DMphase, np.float):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                              np.float(DMphase))

        # if the DM is not active or if the surface is 0
        # we save some time : the EF is not modified
        if self.active == False or (DMphase == 0.).all():
            return entrance_EF

        if isinstance(DMphase, float) or isinstance(DMphase, np.float):
            DMphase = np.full((self.dim_overpad_pupil, self.dim_overpad_pupil),
                              np.float(DMphase))

        lambda_ratio = wavelength / self.wavelength_0

        if self.z_position == 0:
            return entrance_EF * np.exp(1j * DMphase / lambda_ratio)
        else:
            return self.prop_pup_to_DM_and_back(entrance_EF,
                                                DMphase / lambda_ratio,
                                                wavelength)

    def creatingpushact(self, model_dir, DMconfig, Name_DM='DM1'):
        """ --------------------------------------------------
        OPD map induced in the DM plane for each actuator

        Parameters
        ----------
        model_dir : directory where the influence function of the DM is saved
        diam_pup_in_m : diameter of the pupil in meter
        diam_pup_in_pix : radius of the pupil in pixels
        DMconfig : structure with all information on DMs
        dim_array : dimension of the output array

        Error on the model of the DM

        This function needs to be severely cleaned (names and comment). 
        I do it quickly to show how the  DM class is a good idea but leave 
        the cleaning for later (it's late and saturday night)
            
        Returns
        ------
        pushact : 
        -------------------------------------------------- """

        diam_pup_in_pix = 2 * self.prad
        diam_pup_in_m = self.diam_pup_in_m
        dim_array = self.dim_overpad_pupil

        pitchDM = DMconfig[Name_DM + "_pitch"]
        filename_ActuN = DMconfig[Name_DM + "_filename_ActuN"]
        filename_grid_actu = DMconfig[Name_DM + "_filename_grid_actu"]
        filename_actu_infl_fct = DMconfig[Name_DM + "_filename_actu_infl_fct"]
        ActuN = DMconfig[Name_DM + "_ActuN"]
        y_ActuN = DMconfig[Name_DM + "_y_ActuN"]
        x_ActuN = DMconfig[Name_DM + "_x_ActuN"]
        xy_ActuN = [x_ActuN, y_ActuN]

        misregistration = DMconfig[Name_DM + "_misregistration"]
        if misregistration == True:
            xerror = DMconfig[Name_DM + "_xerror"]
            yerror = DMconfig[Name_DM + "_yerror"]
            angerror = DMconfig[Name_DM + "_angerror"]
            gausserror = DMconfig[Name_DM + "_gausserror"]
        else:
            xerror = 0.
            yerror = 0.
            angerror = 0.
            gausserror = 0.

        #Measured positions for each actuator in pixel
        measured_grid = fits.getdata(model_dir + filename_grid_actu)
        #Ratio: pupil radius in the measured position over
        # pupil radius in the numerical simulation
        sampling_simu_over_meaasured = diam_pup_in_pix / 2 / fits.getheader(
            model_dir + filename_grid_actu)['PRAD']
        if filename_ActuN != "":
            im_ActuN = fits.getdata(model_dir + filename_ActuN)
            im_ActuN_dim = proc.crop_or_pad_image(im_ActuN, dim_array)

            ytmp, xtmp = np.unravel_index(
                np.abs(im_ActuN_dim).argmax(), im_ActuN_dim.shape)
            # shift by (0.5,0.5) pixel because the pupil is
            # centered between pixels
            xy_ActuN = [xtmp - 0.5, ytmp - 0.5]

        #Position for each actuator in pixel for the numerical simulation
        simu_grid = actuator_position(measured_grid, xy_ActuN, ActuN,
                                      sampling_simu_over_meaasured)
        # Influence function and the pitch in pixels
        actshape = fits.getdata(model_dir + filename_actu_infl_fct)
        pitch_actshape = fits.getheader(model_dir +
                                        filename_actu_infl_fct)['PITCH']

        # Scaling the influence function to the desired dimension
        # for numerical simulation
        resizeactshape = skimage.transform.rescale(
            actshape,
            diam_pup_in_pix / diam_pup_in_m * pitchDM / pitch_actshape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
            multichannel=False)

        # Gauss2Dfit for centering the rescaled influence function
        Gaussian_fit_param = proc.gauss2Dfit(resizeactshape)
        dx = Gaussian_fit_param[3]
        dy = Gaussian_fit_param[4]
        xycent = len(resizeactshape) / 2
        resizeactshape = nd.interpolation.shift(resizeactshape,
                                                (xycent - dx, xycent - dy))

        # Put the centered influence function inside an array (2*prad x 2*prad)
        actshapeinpupil = np.zeros((dim_array, dim_array))
        if len(resizeactshape) < dim_array:
            actshapeinpupil[0:len(resizeactshape),
                            0:len(resizeactshape
                                  )] = resizeactshape / np.amax(resizeactshape)
            xycenttmp = len(resizeactshape) / 2
        else:
            actshapeinpupil = resizeactshape[
                0:dim_array, 0:dim_array] / np.amax(resizeactshape)
            xycenttmp = dim_array / 2

        # Fill an array with the influence functions of all actuators
        pushact = np.zeros((simu_grid.shape[1], dim_array, dim_array))
        for i in np.arange(pushact.shape[0]):
            if gausserror == 0:
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp + yerror,
                     simu_grid[0, i] + dim_array / 2 - xycenttmp + xerror))

                # Add an error on the orientation of the grid
                if angerror != 0:
                    Psivector = nd.rotate(Psivector,
                                          angerror,
                                          order=5,
                                          cval=0,
                                          reshape=False)[0:dim_array,
                                                         0:dim_array]
            else:
                # Add an error on the sizes of the influence functions
                Psivector = nd.interpolation.shift(
                    actshapeinpupil,
                    (simu_grid[1, i] + dim_array / 2 - xycenttmp,
                     simu_grid[0, i] + dim_array / 2 - xycenttmp))

                xo, yo = np.unravel_index(Psivector.argmax(), Psivector.shape)
                x, y = np.mgrid[0:dim_array, 0:dim_array]
                xy = (x, y)
                Psivector = proc.twoD_Gaussian(xy,
                                               1,
                                               1 + gausserror,
                                               1 + gausserror,
                                               xo,
                                               yo,
                                               0,
                                               0,
                                               flatten=False)
            Psivector[np.where(Psivector < 1e-4)] = 0

            pushact[i] = Psivector

        return pushact

    def creatingpushact_inpup(self):
        """ --------------------------------------------------
        OPD map induced by out-of-pupil DM in the pupil plane for each actuator
        ## Create the influence functions of an out-of-pupil DM
        ## in the pupil plane

        TODO: Can use prop_pup_to_DM_and_back function to avoid repetition

        Parameters
        ----------
        DM_pushact : OPD map induced by the DM in the DM plane

        Returns
        ------
        pushact_inpup : Map of the complex phase induced in pupil plane
        -------------------------------------------------- """

        dim_entrancepupil = self.dim_overpad_pupil
        # do we really need a pupil here ?!? seems to me it would be more general
        # with all the actuators and not really shorter.
        EF_inDMplane, dxout = prop.prop_fresnel(self.clearpup.pup,
                                                self.wavelength_0,
                                                self.z_position,
                                                self.diam_pup_in_m / 2,
                                                self.prad)
        pushact_inpup = np.zeros(
            (self.DM_pushact.shape[0], dim_entrancepupil, dim_entrancepupil),
            dtype=complex)

        for i in np.arange(self.DM_pushact.shape[0]):
            EF_back_in_pup_plane, dxpup = prop.prop_fresnel(
                EF_inDMplane *
                proc.crop_or_pad_image(self.DM_pushact[i], dim_entrancepupil),
                self.wavelength_0, -self.z_position, self.diam_pup_in_m / 2,
                self.prad)
            pushact_inpup[i] = EF_back_in_pup_plane

        return pushact_inpup

    def prop_pup_to_DM_and_back(self, entrance_EF, phase_DM, wavelength):
        """ --------------------------------------------------
        Propagate the field towards an out-of-pupil plane (DM1 plane),
        add the DM1 phase, and propagate to the next pupil plane (DM3 plane)

        Parameters
        ----------
        pupil_wavefront : 2D array (float, double or complex)
                    Wavefront in the pupil plane

        phase_DM1 : 2D array
                    Phase introduced by DM1
        
        wavelength : float
                    wavelength in m
        
        Returns
        ------
        UDM3 : 2D array (complex)
                Wavefront in the pupil plane after DM1
                (corresponds to DM3 plane on THD2 bench)

        AUTHOR : Raphaël Galicher

        REVISION HISTORY :
        Revision 1.1  2021-02-10 Raphaël Galicher
            Initial revision
        Revision 2.0 2021-02-28 Johan Mazoyer 
            Make it more general for all DMs, put in the struc
        -------------------------------------------------- """

        # Propagation in DM plane out of pupil
        EF_inDMplane, dxout = prop.prop_fresnel(entrance_EF, wavelength,
                                                self.z_position,
                                                self.diam_pup_in_m / 2.,
                                                self.prad)

        # Add DM phase
        EF_inDMplane_after_DM = EF_inDMplane * np.exp(1j * phase_DM)
        # and propagate to next pupil plane
        EF_back_in_pup_plane, dxpup = prop.prop_fresnel(
            EF_inDMplane_after_DM, wavelength, -self.z_position,
            self.diam_pup_in_m / 2., self.prad)
        return EF_back_in_pup_plane


class THD2_testbed(Optical_System):
    """ --------------------------------------------------
    initialize and describe the behavior of the THD testbed 
    

    AUTHOR : Johan Mazoyer
    -------------------------------------------------- """
    def __init__(self,
                 modelconfig,
                 DMconfig,
                 coroconfig,
                 Measure_and_save=True,
                 model_dir='',
                 Model_local_dir=''):
        """ --------------------------------------------------
        Initialize a the DM system and the coronagraph
        
        Parameters
        ----------
        modelconfig : general configuration parameters (sizes and dimensions)
        DMconfig : DM parameters
        coroconfig : coronagraph parameters
        Measure_and_save : bool, default = True 
                if true, we measure and save the long stuff
                if false, we just load it
        model_dir: directory to find Measured positions for each actuators and 
                    influence fun and complex corona mask. 
                    ie all the things you cannot measure yourself and need to be given
        Model_local_dir: directory to save things you can measure yourself 
                    and can save to save time
        Mesaure

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """

        # Initialize the Optical_System class and inherit properties
        super().__init__(modelconfig)

        self.entrancepupil = pupil(modelconfig,
                                   self.prad,
                                   directory=model_dir,
                                   filename=modelconfig["filename_instr_pup"])
        self.DM1 = deformable_mirror(modelconfig,
                                     DMconfig,
                                     Name_DM='DM1',
                                     Measure_and_save=Measure_and_save,
                                     model_dir=model_dir,
                                     Model_local_dir=Model_local_dir)
        self.DM3 = deformable_mirror(modelconfig,
                                     DMconfig,
                                     Measure_and_save=Measure_and_save,
                                     Name_DM='DM3',
                                     model_dir=model_dir,
                                     Model_local_dir=Model_local_dir)
        self.corono = coronagraph(modelconfig, coroconfig, model_dir=model_dir)

        self.exitpup_rad = self.corono.lyotrad

        # Measure the PSF and store max and Sum
        self.maxPSF, self.sumPSF = self.max_sum_PSF()

    def EF_through(self,
                   entrance_EF=1.,
                   wavelength=None,
                   DM1phase=0.,
                   DM3phase=0.,
                   noFPM=False):
        """ --------------------------------------------------
        Propagate the electric field through the 2 DMs and the coronograph

        Parameters
        ----------
        entrance_EF :   2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                        Can also be a float scalar in which case entrance_EF is constant
                        default is 1.
        Electric field in the pupil plane a the entrance of the system. 

        wavelength : float. Default is self.wavelength_0 the reference wavelength
                current wavelength in m. 
        
        DM1phase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
                    Can also be a float scalar in which case DM1phase is constant
                    default is 0. 
        
        DM3phase : 2D array of size [self.dim_overpad_pupil, self.dim_overpad_pupil],can be complex.  
            Can also be a float scalar in which case DM3phase is constant
            default is 0. 
        
        noFPM : bool (default: False)
            if True, remove the FPM if one want to measure a un-obstructed PSF
                    

        Returns
        ------
        exit_EF : 2D array, of size [self.dim_overpad_pupil, self.dim_overpad_pupil] 
            Electric field in the pupil plane a the exit of the system


        AUTHOR : Johan Mazoyer 
        -------------------------------------------------- """

        # call the Optical_System super function to check and format the variable entrance_EF
        entrance_EF = super().EF_through(entrance_EF=entrance_EF)

        if wavelength == None:
            wavelength = self.wavelength_0

        if isinstance(DM1phase, float) or isinstance(DM1phase, np.float):
            DM1phase = np.full(
                (self.dim_overpad_pupil, self.dim_overpad_pupil),
                np.float(DM1phase))

        if isinstance(DM3phase, float) or isinstance(DM3phase, np.float):
            DM3phase = np.full(
                (self.dim_overpad_pupil, self.dim_overpad_pupil),
                np.float(DM3phase))

        EF_afterentrancepup = self.entrancepupil.EF_through(
            entrance_EF=entrance_EF, wavelength=wavelength)

        EF_afterDM1 = self.DM1.EF_through(entrance_EF=EF_afterentrancepup,
                                          wavelength=wavelength,
                                          DMphase=DM1phase)
        EF_afterDM3 = self.DM3.EF_through(entrance_EF=EF_afterDM1,
                                          wavelength=wavelength,
                                          DMphase=DM3phase)
        EF_afterCorono = self.corono.EF_through(entrance_EF=EF_afterDM3,
                                                wavelength=wavelength,
                                                noFPM=noFPM)
        return EF_afterCorono

    def max_sum_PSF(self):
        """ --------------------------------------------------
        Measure the non-coronagraphic PSF with no focal plane mask
        and with flat DMs and return max and sum
        
        Returns
        ------
        np.amax(PSF): max of the non-coronagraphic PSF
        np.sum(PSF): sum of the non-coronagraphic PSF

        AUTHOR : Johan Mazoyer
        -------------------------------------------------- """
        PSF = self.todetector_Intensity(center_on_pixel=True, noFPM=True)

        return np.amax(PSF), np.sum(PSF)


def actuator_position(measured_grid, measured_ActuN, ActuN,
                      sampling_simu_over_measured):
    """ --------------------------------------------------
    Convert the measred positions of actuators to positions for numerical simulation
    Parameters
    ----------
    measured_grid : 2D array (float) of shape is 2 x Nb_actuator
                    x and y measured positions for each actuator (unit = pixel)
    measured_ActuN: 1D array (float) of shape 2
                    x and y positions of actuator ActuN same unit as measured_grid
    ActuN:          int
                    Index of the actuator ActuN (corresponding to measured_ActuN) 
    sampling_simu_over_meaasured : float
                    Ratio of sampling in simulation grid over sampling in measured grid 
    Returns
    ------
    simu_grid : 2D array of shape is 2 x Nb_actuator
                x and y positions of each actuator for simulation
                same unit as measured_ActuN
    -------------------------------------------------- """
    simu_grid = measured_grid * 0
    for i in np.arange(measured_grid.shape[1]):
        simu_grid[:, i] = measured_grid[:, i] - measured_grid[:, int(
            ActuN)] + measured_ActuN
    simu_grid = simu_grid * sampling_simu_over_measured
    return simu_grid


def creatingWhichinPupil(pushact, entrancepupil, cutinpupil):
    """ --------------------------------------------------
    Create a vector with the index of all the actuators located in the entrance pupil
    
    Parameters:
    ----------
    pushact: 3D-array, opd created by the pokes of all actuators in the DM.
    entrancepupil: 2D-array, entrance pupil shape
    cutinpupil: float, minimum surface of an actuator inside the pupil to be taken into account (between 0 and 1, in ratio of an actuator perfectly centered in the entrance pupil)
    
    Return:
    ------
    WhichInPupil: 1D array, index of all the actuators located inside the pupil
    -------------------------------------------------- """
    WhichInPupil = []
    tmp_entrancepupil = proc.crop_or_pad_image(entrancepupil, pushact.shape[2])

    for i in np.arange(pushact.shape[0]):
        Psivector = pushact[i]
        cut = cutinpupil * np.sum(np.abs(Psivector))

        if np.sum(Psivector * tmp_entrancepupil) > cut:
            WhichInPupil.append(i)

    WhichInPupil = np.array(WhichInPupil)
    return WhichInPupil
